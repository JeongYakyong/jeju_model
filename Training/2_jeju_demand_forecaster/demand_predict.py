"""제주 D+1 24시간 수요 예측 — 운영 추론 CLI (demand_predict.py).

================================================================================
이 스크립트는 무엇을 하나
================================================================================
어제까지의 실측 수요(history) + 내일(D+1) 24시간의 기상예보(weather) 두
CSV 를 받아, 학습된 LightGBM 모델로 D+1 24시간 수요를 예측해 CSV 1개로 저장한다.
이전 버전의 TimesFM CSV 입력은 이 스크립트 안에서 PatchTST 가 직접 만든다.

학습은 `training/patchtst_lgbm_train.py` 가 담당하고, 이 스크립트는 학습 산출물만
읽어 추론만 수행한다:
  - models/lgbm_pipeline.pkl       (pickle 된 LightGBM booster)
  - models/pipeline_config.json    (피처 스키마 + best_iteration)
  - models/patchtst_demand.pth     (PatchTST 가중치)
  - models/patchtst_demand_meta.pkl(PatchTST HP)

================================================================================
입력 파일 스키마 (2개)
================================================================================
1) --history  : 실측 수요 시계열
   필수 컬럼  : timestamp, real_demand
   시간 범위  : D+1 직전 최소 seq_len(=672 시간 = 28일) 포함해야 함
   허용 형태  : 더 긴 history 도 OK (필요 구간만 잘라 씀)
   주의       : 그 28일 안에 real_demand NaN 있으면 에러 → 미리 보간 필요

2) --weather  : D+1 의 24시간 기상 예보
   필수 컬럼  : timestamp, temp_c, humidity, solar_rad, wind_spd, day_type
   행 수      : 정확히 24행 (D+1 00:00 ~ 23:00, 1시간 간격)
   day_type   : 'weekday' | 'weekend' | 'holiday' 중 하나

================================================================================
출력 CSV
================================================================================
--out 경로에 timestamp, est_demand_new 두 컬럼, 24행 저장.

================================================================================
실행 예시
================================================================================
# (A) CLI 로 실행 — CSV 로 저장
python demand_predict.py \\
    --history data/history_demand.csv \\
    --weather data/forecast_weather_d1.csv \\
    --out     data/pred_d1.csv

# (B) 파이썬 모듈로 import — DataFrame 만 받기
from demand_predict import predict_24h
df = predict_24h(
    history_path='data/history_demand.csv',
    weather_path='data/forecast_weather_d1.csv',
)
# df : 24행, columns = ['timestamp', 'est_demand_new']

================================================================================
자주 발생하는 에러
================================================================================
[weather] D+1 24행이어야 함. ... → weather CSV 가 24행이 아니거나 1시간 간격 아님
[history] PatchTST 입력 윈도우 ... → history 가 D+1 직전 28일을 안 덮음
[history] ... real_demand NaN     → 보간 후 재시도
[day_type] 학습 시점에 없던 카테고리 → 'weekday'/'weekend'/'holiday' 외 값
모델 파일 없음 / config 누락       → training/patchtst_lgbm_train.py 재학습 필요

================================================================================
내부 동작 (참고)
================================================================================
- PatchTST 가 D+1 24h 예측을 생성 → patchtst_target 컬럼으로 LGBM 에 들어감
- 피처 스키마는 pipeline_config.json 에서 읽어옴 → 학습 노트북 변경에 자동 대응
- 범주형 day_type 의 카테고리 순서도 config 에 저장된 그대로 복원
- 사이클 피처 (hour_sin/cos, dow_sin/cos) 는 timestamp 에서 직접 계산
- lag_24h, roll_mean_24h 는 D+1 0시 → 23시 순서로 한 행씩 (iterative) 계산
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

from patchtst_predict import predict_d1 as patchtst_predict_d1


# ---------------------------------------------------------------------------
# 설정 / 입력 로딩
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """pipeline_config.json 로드. 필수 키 검증."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    required = ['feature_cols', 'categorical_cols', 'categorical_categories', 'best_iteration']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f'config 에 누락된 키: {missing}. 재학습 필요.')
    return cfg


def load_booster_pkl(model_path: str) -> lgb.Booster:
    """pickle 된 LightGBM booster 로드."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'모델 파일 없음: {model_path}')
    with open(model_path, 'rb') as f:
        booster = pickle.load(f)
    if not isinstance(booster, lgb.Booster):
        raise ValueError(f'모델 파일이 lgb.Booster 가 아님: {model_path}')
    return booster


def load_csv(path: str, name: str) -> pd.DataFrame:
    """CSV 로드 + timestamp 파싱. 파일 없으면 친절한 에러."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'[{name}] 경로 없음: {path}')
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# D+1 24h 입력 검증
# ---------------------------------------------------------------------------

def resolve_target_date(weather_df: pd.DataFrame,
                        cli_target: str | None,
                        verbose: bool = False) -> pd.Timestamp:
    """--target-date 가 주어지면 그걸 쓰고, 없으면 weather CSV 의 첫 날짜로 추정."""
    if cli_target:
        d = pd.Timestamp(cli_target).normalize()
    else:
        d = weather_df['timestamp'].iloc[0].normalize()
        if verbose:
            print(f'[info] --target-date 미지정 → weather CSV 첫 행에서 추정: {d.date()}')
    # weather 가 이 날짜를 가지고 있는지 가벼운 체크
    if not (weather_df['timestamp'].dt.normalize() == d).any():
        raise ValueError(f'[weather] CSV 에 target_date={d.date()} 행이 없음.')
    return d


def slice_24h(df: pd.DataFrame, target_date: pd.Timestamp, name: str) -> pd.DataFrame:
    """D+1 24행 (00:00 ~ 23:00) 정확히 만족하는지 검사 후 슬라이스."""
    start = target_date
    end   = target_date + pd.Timedelta(hours=23)
    sub = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].copy()
    if len(sub) != 24:
        raise ValueError(
            f'[{name}] D+1 24행이어야 함. {start.date()} 00:00 ~ 23:00 → 발견 {len(sub)}행')
    expected = pd.date_range(start, end, freq='h')
    if not (sub['timestamp'].values == expected.values).all():
        raise ValueError(f'[{name}] 시간이 1시간 간격으로 24행 연속이어야 함.')
    return sub.reset_index(drop=True)


def build_history_series(history_df: pd.DataFrame,
                         target_date: pd.Timestamp,
                         seq_len: int) -> pd.Series:
    """history 가 D+1 직전 seq_len 시간 (PatchTST 입력) + 48h (lag/roll) 를 덮는지 확인.

    PatchTST 는 D 23:00 까지의 seq_len(=672) 시간 데이터가 필요하다.
    lag/roll 은 그 안에 자연스레 포함되니 seq_len 만 만족하면 충분.

    반환: timestamp 인덱스의 real_demand Series.
    """
    if 'real_demand' not in history_df.columns:
        raise ValueError('[history] real_demand 컬럼 필요.')

    last_needed  = target_date - pd.Timedelta(hours=1)              # D 23:00
    first_needed = last_needed - pd.Timedelta(hours=seq_len - 1)    # D-27 00:00

    in_range = history_df[(history_df['timestamp'] >= first_needed) &
                          (history_df['timestamp'] <= last_needed)]
    if len(in_range) < seq_len:
        raise ValueError(
            f'[history] PatchTST 입력 윈도우 ({first_needed} ~ {last_needed}, '
            f'{seq_len}시간) 가 부족. 있는 행 수 = {len(in_range)}.')

    if in_range['real_demand'].isna().any():
        n_na = int(in_range['real_demand'].isna().sum())
        raise ValueError(f'[history] PatchTST 입력 윈도우에 real_demand NaN {n_na}개. 보간 필요.')

    series = history_df.set_index('timestamp')['real_demand'].sort_index()
    return series


# ---------------------------------------------------------------------------
# 사이클 피처 / day_type
# ---------------------------------------------------------------------------

def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour_sin/cos, dow_sin/cos 를 timestamp 에서 직접 계산."""
    h   = df['timestamp'].dt.hour
    dow = df['timestamp'].dt.dayofweek
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * h   / 24)
    df['hour_cos'] = np.cos(2 * np.pi * h   / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow / 7)
    return df


def coerce_categoricals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """범주형 컬럼을 학습 시점과 동일한 카테고리 순서로 변환."""
    df = df.copy()
    for col, cats in cfg['categorical_categories'].items():
        if col not in df.columns:
            raise ValueError(f'[weather] 범주형 컬럼 {col} 누락.')
        unknown = set(df[col].dropna().unique()) - set(cats)
        if unknown:
            raise ValueError(f'[{col}] 학습 시점에 없던 카테고리 {unknown}. '
                             f'학습된 카테고리: {cats}')
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


# ---------------------------------------------------------------------------
# Iterative 예측 (D+1 0시 → 23시)
# ---------------------------------------------------------------------------

def predict_iterative(booster: lgb.Booster,
                      feature_cols: list,
                      base_df: pd.DataFrame,
                      history_series: pd.Series,
                      target_date: pd.Timestamp,
                      best_iter: int) -> np.ndarray:
    """D+1 의 각 시각에 대해 lag/roll 을 갱신하며 한 행씩 예측."""
    series = history_series.copy()
    preds = np.zeros(24, dtype=float)

    for h in range(24):
        ts_h     = target_date + pd.Timedelta(hours=h)
        ts_lag24 = ts_h - pd.Timedelta(hours=24)

        # 1) lag_24h — 24시간 전 시점의 값
        if ts_lag24 not in series.index:
            raise ValueError(f'lag_24h 계산용 history 누락: {ts_lag24}')
        lag_24h = float(series.loc[ts_lag24])

        # 2) roll_mean_24h — [t-25h, t-1h] 평균 (학습 시 shift(1).rolling(24) 와 일치)
        window_start = ts_h - pd.Timedelta(hours=25)
        window_end   = ts_h - pd.Timedelta(hours=1)
        win = series.loc[(series.index >= window_start) & (series.index <= window_end)]
        if len(win) < 24:
            raise ValueError(
                f'roll_mean_24h 윈도우 부족 (t={ts_h}, 필요 24, 발견 {len(win)})')
        roll_mean_24h = float(win.mean())

        # 3) 한 행짜리 입력 만들기
        row = base_df.iloc[[h]].copy()
        row['lag_24h']       = lag_24h
        row['roll_mean_24h'] = roll_mean_24h

        X_row = row[feature_cols]
        yhat  = float(booster.predict(X_row, num_iteration=best_iter)[0])
        preds[h] = yhat

        # 4) 다음 시각의 roll_mean_24h 가 자기 예측을 보도록 series 에 append
        series.loc[ts_h] = yhat

    return preds


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def predict_24h(history_path: str,
                weather_path: str,
                out_path: str | None = None,
                model_path: str | None = None,
                config_path: str | None = None,
                target_date: str | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """파일 경로 → 24행 예측 DataFrame.

    Streamlit / 노트북 / 다른 파이썬 코드에서 import 해서 쓰는 메인 진입점.
    반환: columns=['timestamp', 'est_demand_new'], 24행.

    out_path 가 주어지면 같은 내용을 CSV 로도 저장.
    model_path / config_path 미지정시 스크립트 옆 models/ 폴더의 기본 파일 사용.
    """
    # === 1. 설정 / 모델 ===
    model_path  = model_path  or _default('models/lgbm_pipeline.pkl')
    config_path = config_path or _default('models/pipeline_config.json')

    cfg = load_config(config_path)
    booster = load_booster_pkl(model_path)

    feature_cols = cfg['feature_cols']
    best_iter    = cfg['best_iteration']
    if verbose:
        print(f'[모델] {model_path}  (best_iter={best_iter}, 피처={len(feature_cols)})')

    # === 2. 입력 로드 ===
    history_df = load_csv(history_path, 'history')
    weather_df = load_csv(weather_path, 'weather')

    # === 3. target_date + history 검증 ===
    tgt = resolve_target_date(weather_df, target_date, verbose=verbose)
    weather_24 = slice_24h(weather_df, tgt, 'weather')

    # PatchTST 의 seq_len 만큼 history 가 충분한지 확인 + 시리즈 추출
    # (seq_len 은 모델 메타에 박혀 있지만 매번 로드하지 않게 patchtst_predict 가 알아서 함)
    # 여기선 lag/roll 계산에 필요한 28일치를 보장하면 PatchTST 입력은 자동 만족.
    PATCHTST_SEQ_LEN = 672
    history_series = build_history_series(history_df, tgt, seq_len=PATCHTST_SEQ_LEN)

    # === 4. PatchTST 로 D+1 patchtst_target 24시간 생성 ===
    if verbose:
        print(f'[patchtst] D+1={tgt.date()} 24h 추론')
    patchtst_24 = patchtst_predict_d1(history_series, target_date=tgt)   # shape (24,)

    # === 5. base_df 조립 (weather + patchtst + 사이클) ===
    base = weather_24.copy()
    base['patchtst_target'] = patchtst_24
    base = add_cycle_features(base)
    base = coerce_categoricals(base, cfg)

    # 학습에 쓴 모든 피처가 있는지 (lag/roll 제외) 확인
    need = [c for c in feature_cols if c not in ('lag_24h', 'roll_mean_24h')]
    missing = [c for c in need if c not in base.columns]
    if missing:
        raise ValueError(f'D+1 입력에 누락된 피처: {missing}')

    # === 6. Iterative 예측 ===
    preds = predict_iterative(
        booster=booster,
        feature_cols=feature_cols,
        base_df=base,
        history_series=history_series,
        target_date=tgt,
        best_iter=best_iter,
    )

    out = pd.DataFrame({
        'timestamp':      base['timestamp'].values,
        'est_demand_new': preds.round(3),
    })

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
        out.to_csv(out_path, index=False)
        if verbose:
            print(f'[저장] {out_path}  ({len(out)}행)')
    if verbose:
        print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def _default(rel: str) -> str:
    """스크립트 위치 기준 기본 경로 — 사용자가 인자 안 줘도 동작하도록."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, rel)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description='제주 D+1 24h 수요 예측 (LGBM + PatchTST)')
    p.add_argument('--history',     required=True,
                   help='timestamp,real_demand (D+1 직전 ≥28일)')
    p.add_argument('--weather',     required=True,
                   help='timestamp + weather + day_type (D+1 24행)')
    p.add_argument('--out',         required=True,
                   help='출력 CSV 경로 (timestamp,est_demand_new)')
    p.add_argument('--model',       default=_default('models/lgbm_pipeline.pkl'))
    p.add_argument('--config',      default=_default('models/pipeline_config.json'))
    p.add_argument('--target-date', default=None,
                   help='D+1 날짜 YYYY-MM-DD (생략시 weather CSV 첫 행에서 추정)')
    args = p.parse_args(argv)

    try:
        predict_24h(
            history_path=args.history,
            weather_path=args.weather,
            out_path=args.out,
            model_path=args.model,
            config_path=args.config,
            target_date=args.target_date,
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
