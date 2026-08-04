"""End-to-end 학습 / End-to-end training orchestrator.

DB에서 historical + forecast + RT SMP를 읽어, BANK를 만들고, 노이즈 주입한
실측에 build_features를 입혀 학습 데이터를 만들고, build_pipeline으로
운영 모델 한 벌을 학습한 뒤 joblib으로 저장한다.

CLI에서는 `examples/run_smp_forecast.py train` 으로 호출한다.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from net_load_forecaster.db_manager import JejuEnergyDB

from .config import (
    CORE_COLS, DB_PATH, SMP_MODEL_PATH,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
)
from .db_extension import get_realtime_smp
from .features import build_features, assert_parity
from .pipeline import build_bank, build_pipeline, inject_noise


def _load_historical_features(db: JejuEnergyDB) -> pd.DataFrame:
    """학습창+검증창 전체를 historical에서 읽어 학습 모드 피처로 변환."""
    hist = db.get_historical(
        start_date=str(TRAIN_START),
        end_date=str(VAL_END + pd.Timedelta('2h')),  # nl_lead_2 NaN 최소화
    )
    if hist.empty:
        raise RuntimeError(
            f"historical_data가 비었다 (구간 {TRAIN_START} ~ {VAL_END}). "
            f"Stage 1 fetch_data 로 먼저 채워야 한다."
        )
    hist.index = pd.to_datetime(hist.index)
    return hist


def _load_forecast_val_features(db: JejuEnergyDB) -> pd.DataFrame:
    """BANK 구성용으로 검증창 forecast를 읽어 서빙 모드 피처로 변환."""
    fc = db.get_forecast(
        start_date=str(VAL_START),
        end_date=str(VAL_END),
    )
    if fc.empty:
        raise RuntimeError(
            f"forecast_data가 비었다 (검증창 {VAL_START} ~ {VAL_END}). "
            f"BANK 구성을 위해 Stage 1 fetch_data 로 forecast를 채워야 한다."
        )
    fc.index = pd.to_datetime(fc.index)
    return fc


def _attach_target(features: pd.DataFrame, db: JejuEnergyDB) -> pd.DataFrame:
    """RT SMP를 features 인덱스에 join한다."""
    rt = get_realtime_smp(
        db,
        start=str(features.index.min()),
        end=str(features.index.max()),
    )
    if rt.empty:
        raise RuntimeError(
            "realtime_smp 테이블이 비었다. smp_forecaster.data_pipeline."
            "ingest_rt_smp 로 먼저 인제스트해야 한다."
        )
    return features.join(rt['smp_rt'], how='inner')


def train(
    db_path: str | Path | None = None,
    save_path: str | Path | None = None,
    seed_override: int | None = None,
) -> dict:
    """SMP 운영 모델을 학습하고 디스크에 저장한다.

    Args:
        db_path: SQLite 경로. None이면 config.DB_PATH.
        save_path: 모델 저장 경로. None이면 config.SMP_MODEL_PATH.
        seed_override: 노이즈 주입 시드 (디버그용).

    Returns:
        저장 결과 요약 dict.
    """
    db_path = Path(db_path or DB_PATH)
    save_path = Path(save_path or SMP_MODEL_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] DB 로드 / Loading DB: {db_path}")
    db = JejuEnergyDB(str(db_path))
    try:
        hist_raw = _load_historical_features(db)
        fc_val_raw = _load_forecast_val_features(db)

        print(f"[2/5] BANK 구성 / Building noise bank "
              f"(검증창 {VAL_START.date()} ~ {VAL_END.date()})")
        actual_val_feats = build_features(
            hist_raw.loc[VAL_START:VAL_END], mode='train',
        )
        # forecast 피처에서 net_load 계산용 컬럼이 빠지지 않도록 raw에 합치기 전 처리
        fc_val_feats = build_features(fc_val_raw, mode='serve')
        bank = build_bank(actual_val_feats, fc_val_feats)
        print(f"  BANK 시각별 표본수 (solar/wind): "
              f"평균 {sum(len(b['s']) for b in bank.values()) / 24:.0f} / "
              f"{sum(len(b['w']) for b in bank.values()) / 24:.0f}")

        print(f"[3/5] 노이즈 주입 + 피처 빌드 / Noise injection + features")
        utils = hist_raw[['Solar_Utilization', 'Wind_Utilization']]
        seed = seed_override if seed_override is not None else None
        noised = (inject_noise(utils, bank, seed=seed)
                  if seed is not None else inject_noise(utils, bank))
        hist_noised = hist_raw.copy()
        hist_noised['Solar_Utilization'] = noised['Solar_Utilization']
        hist_noised['Wind_Utilization']  = noised['Wind_Utilization']
        train_feats = build_features(hist_noised, mode='train')

        # parity 검사: 학습본 컬럼과 서빙본 컬럼이 같아야 한다.
        assert_parity(train_feats, fc_val_feats)

        train_feats = _attach_target(train_feats, db)
        train_feats = train_feats.dropna(subset=CORE_COLS + ['smp_rt'])
        train_df = train_feats.loc[TRAIN_START:TRAIN_END]
        print(f"  학습 행수 {len(train_df):,} | "
              f"음수<=0 {(train_df['smp_rt'] <= 0).sum()}")

        print(f"[4/5] 모델 학습 / Training pipeline (피처 {len(CORE_COLS)}개)")
        model = build_pipeline(train_df, CORE_COLS)

        print(f"[5/5] 저장 / Saving → {save_path}")
        artifact = {
            'floor_clf': model.floor_clf,
            'neg_clf':   model.neg_clf,
            'reg':       model.reg,
            'feature_cols': model.feature_cols,
            'floor_val': model.floor_val,
            'deep_neg':  model.deep_neg,
            'bank': bank,
            'trained_at': datetime.now().isoformat(),
            'training_window': (str(TRAIN_START), str(TRAIN_END)),
            'val_window': (str(VAL_START), str(VAL_END)),
        }
        joblib.dump(artifact, save_path)
    finally:
        db.close()

    return {
        'save_path': str(save_path),
        'train_rows': int(len(train_df)),
        'feature_cols': model.feature_cols,
        'floor_pr_auc': model.floor_clf.oof_pr_auc,
        'neg_pr_auc':   model.neg_clf.oof_pr_auc,
    }
