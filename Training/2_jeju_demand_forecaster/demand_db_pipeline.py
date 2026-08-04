"""제주 D+1 수요 예측 — DB 전용 파이프라인 (CSV 탈피).

================================================================================
무엇을 하나
================================================================================
input_data_jeju.db 한 곳에서 모두 읽고 쓴다.
  - 입력  : historical (실측 real_demand_jeju + 3지점 기상),
            forecast   (D+1 예보 3지점 기상)
  - 신호  : patchtst_signal  (jeju_patchtst_target)  ← CSV 대신 DB 테이블
  - 출력  : forecast.jeju_est_demand_new  ← D+1 24h 수요 예측 UPSERT

기상은 모두 "제주 3지점 공간평균" 으로 집약한다 (학습과 동일).
  temp_c   = mean(temp[_c]_{w,e,s})
  humidity = mean(humidity_{w,e,s})  /  forecast 는 reh_{w,e,s}
  wind_spd = mean(wind_spd_{w,e,s})  /  forecast 는 wind_spd_10m_{w,e,s}
  solar_rad= mean(solar_rad_{w,s})   /  forecast 는 radiation_{w,s}   (east 일사 없음)

================================================================================
공개 API
================================================================================
    migrate_signal_from_csv()         # patchtst_features.csv → DB patchtst_signal
    refresh_signal(start, end)        # PatchTST 로 신호 재생성/연장 → DB UPSERT
    predict_demand_to_db(date)        # D+1 24h 예측 → forecast.jeju_est_demand_new
"""
from __future__ import annotations

import os
import sqlite3
import numpy as np
import pandas as pd

from patchtst_predict import predict_d1
from demand_predict import (
    load_config, load_booster_pkl,
    add_cycle_features, coerce_categoricals, predict_iterative,
)

# =============================================================================
# 0. 경로 / 상수
# =============================================================================
HERE     = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.normpath(os.path.join(
    HERE, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
MODELS   = os.path.join(HERE, 'models')
PKL      = os.path.join(MODELS, 'lgbm_pipeline.pkl')
CFG      = os.path.join(MODELS, 'pipeline_config.json')
CSV_SIGNAL = os.path.join(HERE, 'patchtst_features.csv')

SIGNAL_TABLE = 'patchtst_signal'
SIGNAL_COL   = 'jeju_patchtst_target'
OUT_COL      = 'jeju_est_demand_new'
PATCHTST_SEQ_LEN = 672

# 4기상 ← 지점 평균 (학습 historical / 추론 forecast)
HIST_WEATHER = {
    'temp_c':    ['temp_c_west', 'temp_c_east', 'temp_c_south'],
    'humidity':  ['humidity_west', 'humidity_east', 'humidity_south'],
    'wind_spd':  ['wind_spd_west', 'wind_spd_east', 'wind_spd_south'],
    'solar_rad': ['solar_rad_west', 'solar_rad_south'],
}
FORE_WEATHER = {
    'temp_c':    ['temp_west', 'temp_east', 'temp_south'],
    'humidity':  ['reh_west', 'reh_east', 'reh_south'],
    'wind_spd':  ['wind_spd_10m_west', 'wind_spd_10m_east', 'wind_spd_10m_south'],
    'solar_rad': ['radiation_west', 'radiation_south'],
}


def _conn():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# 1. 수요 시계열 (historical) — 0 은 결측 보고 시간보간
# =============================================================================
def load_demand_series() -> pd.Series:
    with _conn() as con:
        d = pd.read_sql('SELECT timestamp, real_demand_jeju FROM historical',
                        con, parse_dates=['timestamp'])
    d = d.sort_values('timestamp')
    d.loc[d['real_demand_jeju'] == 0, 'real_demand_jeju'] = np.nan
    s = d.set_index('timestamp')['real_demand_jeju'].interpolate(method='time')
    return s


# =============================================================================
# 2. patchtst 신호 테이블 (CSV → DB / 재생성)
# =============================================================================
def _ensure_signal_table(con):
    con.execute(f'CREATE TABLE IF NOT EXISTS "{SIGNAL_TABLE}" '
                f'("timestamp" TEXT PRIMARY KEY, "{SIGNAL_COL}" REAL)')


def _upsert_signal(con, df: pd.DataFrame):
    """df: columns=[timestamp(str), value]."""
    _ensure_signal_table(con)
    rows = [(t, float(v)) for t, v in zip(df['timestamp'], df[SIGNAL_COL])]
    con.executemany(
        f'INSERT INTO "{SIGNAL_TABLE}" ("timestamp","{SIGNAL_COL}") VALUES (?,?) '
        f'ON CONFLICT("timestamp") DO UPDATE SET "{SIGNAL_COL}"=excluded."{SIGNAL_COL}"',
        rows)


def migrate_signal_from_csv(csv_path: str | None = None) -> int:
    """기존 patchtst_features.csv 를 DB patchtst_signal 테이블로 이전."""
    csv_path = csv_path or CSV_SIGNAL
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'patchtst_target': SIGNAL_COL})
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[['timestamp', SIGNAL_COL]].dropna()
    with _conn() as con:
        _upsert_signal(con, df)
        con.commit()
        n = con.execute(f'SELECT COUNT(*) FROM "{SIGNAL_TABLE}"').fetchone()[0]
    return int(n)


def read_signal() -> pd.DataFrame:
    with _conn() as con:
        _ensure_signal_table(con)
        df = pd.read_sql(f'SELECT timestamp, "{SIGNAL_COL}" FROM "{SIGNAL_TABLE}"',
                         con, parse_dates=['timestamp'])
    return df.rename(columns={SIGNAL_COL: 'patchtst_target'}).sort_values('timestamp')


def refresh_signal(start: str, end: str) -> int:
    """[start, end] 의 각 D+1 에 대해 PatchTST 신호를 재생성해 DB UPSERT.

    start/end 는 D+1(예측 대상일) 기준 'YYYY-MM-DD'. 기존 가중치 그대로 사용.
    """
    s = load_demand_series()
    dates = pd.date_range(start, end, freq='D')
    out = []
    for t in dates:
        y = predict_d1(s, t)
        idx = pd.date_range(t, periods=24, freq='h')
        out.append(pd.DataFrame({
            'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
            SIGNAL_COL: y,
        }))
    allrows = pd.concat(out, ignore_index=True)
    with _conn() as con:
        _upsert_signal(con, allrows)
        con.commit()
    return len(allrows)


# =============================================================================
# 3. forecast 테이블 → D+1 24h 4기상 평균 (+ day_type)
# =============================================================================
def load_forecast_weather(date: str) -> pd.DataFrame:
    tgt = pd.Timestamp(date).normalize()
    start = tgt.strftime('%Y-%m-%d %H:%M:%S')
    end   = (tgt + pd.Timedelta(hours=23)).strftime('%Y-%m-%d %H:%M:%S')
    need = sorted({c for cols in FORE_WEATHER.values() for c in cols} | {'day_type'})
    sel = ', '.join(f'"{c}"' for c in ['timestamp'] + need)
    with _conn() as con:
        df = pd.read_sql(
            f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
            con, params=(start, end), parse_dates=['timestamp'])
    if len(df) != 24:
        raise ValueError(f'[forecast] {date} 24행이어야 함 — 발견 {len(df)}행 '
                         f'(forecast 테이블 보유 범위 확인)')
    w = pd.DataFrame({'timestamp': df['timestamp'], 'day_type': df['day_type']})
    for feat, cols in FORE_WEATHER.items():
        # 일부 지점 예보 미발행(None→object) 대비 숫자 강제변환 후 평균
        num = df[cols].apply(pd.to_numeric, errors='coerce')
        w[feat] = num.mean(axis=1)   # 한쪽 NaN 이면 보유 지점 평균으로 폴백
    return w


# =============================================================================
# 4. D+1 24h 예측 → forecast.jeju_est_demand_new UPSERT
# =============================================================================
def predict_demand_to_db(date: str, write: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
    """forecast/historical 직접 읽어 D+1 24h 수요 예측. write 시 DB 저장."""
    tgt = pd.Timestamp(date).normalize()

    cfg = load_config(CFG)
    booster = load_booster_pkl(PKL)
    feature_cols = cfg['feature_cols']
    best_iter    = cfg['best_iteration']

    # 1) 수요 시계열 + PatchTST 입력 윈도우 검증
    series = load_demand_series()
    last_needed  = tgt - pd.Timedelta(hours=1)
    first_needed = last_needed - pd.Timedelta(hours=PATCHTST_SEQ_LEN - 1)
    win = series.loc[first_needed:last_needed]
    if len(win) < PATCHTST_SEQ_LEN or win.isna().any():
        raise ValueError(f'[history] PatchTST 윈도우({first_needed}~{last_needed}) '
                         f'부족/NaN — 보유 {len(win)}행')

    # 2) D+1 예보 4기상 평균
    weather_24 = load_forecast_weather(date)

    # 3) PatchTST 신호
    patchtst_24 = predict_d1(series, target_date=tgt)

    # 4) base 조립 → iterative 예측 (demand_predict 로직 재사용)
    base = weather_24.copy()
    base['patchtst_target'] = patchtst_24
    base = add_cycle_features(base)
    base = coerce_categoricals(base, cfg)

    preds = predict_iterative(
        booster=booster, feature_cols=feature_cols, base_df=base,
        history_series=series, target_date=tgt, best_iter=best_iter)

    out = pd.DataFrame({
        'timestamp': base['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        OUT_COL: preds.round(3),
    })

    if write:
        with _conn() as con:
            cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
            if OUT_COL not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{OUT_COL}" REAL')
            con.executemany(
                f'INSERT INTO forecast ("timestamp","{OUT_COL}") VALUES (?,?) '
                f'ON CONFLICT("timestamp") DO UPDATE SET "{OUT_COL}"=excluded."{OUT_COL}"',
                list(zip(out['timestamp'], out[OUT_COL].astype(float))))
            con.commit()
        if verbose:
            print(f'[DB] forecast.{OUT_COL} ← {date} 24행 UPSERT')

    if verbose:
        print(out.to_string(index=False))
    return out


# =============================================================================
# 4b. 전 구간 백필 — 모델/시계열 1회 로드 후 반복
# =============================================================================
def backfill_demand_to_db(start: str, end: str,
                          verbose: bool = True) -> pd.DataFrame:
    """[start, end] 의 forecast 보유 날짜 전부에 jeju_est_demand_new 채움.

    예측 가능 조건: 그 날짜의 forecast 24행 존재 + historical 수요가
    D+1 직전 672h 윈도우를 NaN 없이 덮음. 불가한 날짜는 건너뛰고 사유를 모은다.
    """
    cfg = load_config(CFG)
    booster = load_booster_pkl(PKL)
    feature_cols, best_iter = cfg['feature_cols'], cfg['best_iteration']
    series = load_demand_series()

    # forecast 에 24행 있는 날짜만 후보로
    with _conn() as con:
        fdates = pd.read_sql(
            "SELECT substr(timestamp,1,10) d, COUNT(*) n FROM forecast "
            "WHERE timestamp BETWEEN ? AND ? GROUP BY d HAVING n=24 ORDER BY d",
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'))['d'].tolist()

    done, skipped, allrows = [], [], []
    for date in fdates:
        tgt = pd.Timestamp(date).normalize()
        last_needed = tgt - pd.Timedelta(hours=1)
        first_needed = last_needed - pd.Timedelta(hours=PATCHTST_SEQ_LEN - 1)
        win = series.loc[first_needed:last_needed]
        if len(win) < PATCHTST_SEQ_LEN or win.isna().any():
            skipped.append((date, 'history 윈도우 부족/NaN')); continue
        try:
            base = load_forecast_weather(date)
            base['patchtst_target'] = predict_d1(series, target_date=tgt)
            base = coerce_categoricals(add_cycle_features(base), cfg)
            preds = predict_iterative(booster, feature_cols, base, series, tgt, best_iter)
            allrows += list(zip(base['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                                np.round(preds, 3).astype(float)))
            done.append(date)
        except Exception as e:
            skipped.append((date, str(e)[:60]))

    with _conn() as con:
        cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
        if OUT_COL not in cols:
            con.execute(f'ALTER TABLE forecast ADD COLUMN "{OUT_COL}" REAL')
        con.executemany(
            f'INSERT INTO forecast ("timestamp","{OUT_COL}") VALUES (?,?) '
            f'ON CONFLICT("timestamp") DO UPDATE SET "{OUT_COL}"=excluded."{OUT_COL}"',
            allrows)
        con.commit()

    if verbose:
        print(f'[backfill] 완료 {len(done)}일 / 건너뜀 {len(skipped)}일 '
              f'/ {len(allrows)}행 UPSERT')
        if done:
            print(f'  범위: {done[0]} ~ {done[-1]}')
        for d, why in skipped[:8]:
            print(f'  skip {d}: {why}')
    return pd.DataFrame(skipped, columns=['date', 'reason'])


# =============================================================================
# 5. CLI
# =============================================================================
if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 수요 D+1 예측 (DB 전용)')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('migrate-signal')
    rs = sub.add_parser('refresh-signal'); rs.add_argument('start'); rs.add_argument('end')
    pp = sub.add_parser('predict'); pp.add_argument('date')
    pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    a = p.parse_args()

    if a.cmd == 'migrate-signal':
        print('patchtst_signal rows:', migrate_signal_from_csv())
    elif a.cmd == 'refresh-signal':
        print('refreshed rows:', refresh_signal(a.start, a.end))
    elif a.cmd == 'predict':
        predict_demand_to_db(a.date, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_demand_to_db(a.start, a.end)
