"""특정 날짜의 24시간 SMP 예측 / 24h SMP prediction for a target date.

Stage 1의 forecast_data 테이블에서 그 날짜의 예보 피처를 읽고, build_features
(serve 모드)로 입력을 만든 뒤 combine로 최종 예측을 생성한다.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from net_load_forecaster.db_manager import JejuEnergyDB

from .combine import combine
from .config import DB_PATH
from .features import build_features
from .loader import load_model


def predict_smp(
    target_date: str | datetime,
    db_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """target_date의 시간별 SMP 예측을 24행 DataFrame으로 반환.

    Args:
        target_date: 'YYYY-MM-DD' 또는 datetime.
        db_path: SQLite 경로. None이면 config.DB_PATH.
        model_path: 모델 .pkl 경로. None이면 config.SMP_MODEL_PATH.

    Returns:
        index = target_date의 24시간 timestamp.
        columns: smp_pred, neg_proba, danger.
    """
    target_str = (target_date.strftime('%Y-%m-%d')
                  if isinstance(target_date, datetime) else str(target_date))
    target_day = pd.Timestamp(target_str)
    next_day = target_day + pd.Timedelta(days=1)

    model, _meta = load_model(model_path)

    db = JejuEnergyDB(str(db_path or DB_PATH))
    try:
        # nl_lead_2 계산을 위해 +2h 까지 같이 읽는다.
        fc = db.get_forecast(
            start_date=target_str,
            end_date=str(next_day + pd.Timedelta('2h')),
        )
        if fc.empty:
            raise RuntimeError(
                f"forecast_data에 {target_str} 데이터가 없다. "
                f"먼저 net_load_forecaster.fetch_data(kind='forecast')로 채워야 한다."
            )
        fc.index = pd.to_datetime(fc.index)
    finally:
        db.close()

    features = build_features(fc, mode='serve')
    day_features = features.loc[target_day:next_day - pd.Timedelta('1h')]
    if day_features.empty:
        raise RuntimeError(
            f"build_features 후 {target_str} 구간이 비었다 (forecast 결측?)."
        )

    result = combine(model, day_features)
    return pd.DataFrame({
        'smp_pred': result.yhat,
        'neg_proba': result.neg_proba,
        'danger':   result.danger.astype(int),
    }, index=day_features.index)
