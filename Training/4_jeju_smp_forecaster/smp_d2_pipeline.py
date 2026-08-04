"""4단계 보너스 — D+2(뒤24h) SMP 서빙 파이프라인 (DB 전용).

추가작업.md §3 Step2~3. D+2 예측 DA 가격선 위에 D+1 음수검지기를 그대로 오버레이.

무엇을 하나 (전부 D+2 전용 신규 컬럼, D+1 A안/Phase2 산출물 불변)
  ① 가격선 : train_smp_d2_da.smp_d2_da.pkl 로 D+2 DA 예측 → est_smp_jeju_d2
  ② 음수경보: models_weight/smp_binary.pkl(D+1 분류기 그대로) 재사용.
             입력만 D+2 예측값으로 — 가격선 피처 smp_jeju_da = 위 예측 DA로 치환.
             → smp_neg_proba_d2, smp_danger_d2 (θ + 2h 지속)
  ③ 깊이   : models_weight/smp_depth_lookup.json(Phase2) 그대로 → smp_neg_depth_d2_p10/50/90

D+2 피처 가용성(누수 없음):
  lag24/lag168 = 실측 DA(t-24=D+1 발표값, t-168=D-5) / net_load·est_demand = forecast 예측
입력이 전부 예측값이라 경보 성능을 D+1과 분리 보고(§3).
"""
from __future__ import annotations

import os
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from train_smp_db import DB_PATH, FEATURES as CLF_FEATURES, load_forecast
from train_binary_smp import persist, NEG_THRESH
from train_smp_d2_da import FEATURES as DA_FEATURES, _predict_da
from smp_phase2_depth import lookup_depth

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models_weight')
DA_BUNDLE = os.path.join(MODELS, 'smp_d2_da.pkl')
CLF_BUNDLE = os.path.join(MODELS, 'smp_binary.pkl')
DEPTH_LOOKUP = os.path.join(MODELS, 'smp_depth_lookup.json')

OUT_PRICE = 'est_smp_jeju_d2'
OUT_PROBA = 'smp_neg_proba_d2'
OUT_DANGER = 'smp_danger_d2'         # 균형 운영점(θ_lo): recall 우선
OUT_DANGER_HI = 'smp_danger_d2_hi'   # 고확신 운영점(θ_hi): precision 우선
OUT_DEPTH = ['smp_neg_depth_d2_p10', 'smp_neg_depth_d2_p50', 'smp_neg_depth_d2_p90']
OUT_COLS = [OUT_PRICE, OUT_PROBA, OUT_DANGER, OUT_DANGER_HI, *OUT_DEPTH]

# D+2 전용 이중 운영점(θ 탐색 2026-06-05, 서빙구간). D+1 분류기 proba를 D+2 입력으로 재추론한
# 확률띠(≤0.275)에서 측정 — lo=균형(recall0.86/prec0.37), hi=고확신(recall0.41/prec0.46).
D2_THETA = {'lo': 0.250, 'hi': 0.258}


def _conn():
    return sqlite3.connect(DB_PATH)


def _da_lags(end=None):
    """실측 DA 시간연속 시계열 → lag24/lag168 (D+2 앵커·주간교정). 누수 없음.

    end(서빙 타깃 끝 날짜)를 주면 인덱스를 그 날까지 늘려 미래 D+2 타깃의 lag도 만든다.
    lag 값은 과거 발표 DA만 참조(lag24=하루 전·lag168=일주일 전)하므로 누수 없음 —
    인덱스를 안 늘리면 발표 DA 끝(오늘) 이후 타깃의 lag가 통째로 NaN이 돼 그 base의 D+2가 빈다.
    """
    with _conn() as con:
        s = pd.read_sql('SELECT timestamp, smp_jeju_da FROM historical ORDER BY timestamp',
                        con, parse_dates=['timestamp']).set_index('timestamp')['smp_jeju_da']
    hi = s.index.max()
    if end is not None:
        hi = max(hi, pd.Timestamp(f'{end} 23:00'))
    full = pd.date_range(s.index.min(), hi, freq='h')
    s = s.reindex(full)
    return pd.DataFrame({'lag24': s.shift(24), 'lag168': s.shift(168)})


def _build_serving(start: str, end: str):
    """forecast 예측입력 + 실측 DA lag로 D+2 서빙 프레임 구성(평가용 rt 포함).
    Δ(전일대비) 피처는 전체 연속 프레임에서 계산 후 구간 절단 → 단일일 서빙도 정합."""
    fc = load_forecast(with_target=True)                  # net_load·est_demand·leads·hour·month·smp_jeju_da·rt (전구간)
    lags = _da_lags(end)
    d = fc.join(lags, how='left').sort_index()
    # Step1 잔차회귀 변화량 피처 (forecast 예측 net_load·est_demand의 전일대비 변화)
    d['d_net_load'] = d['net_load'] - d['net_load'].shift(24)
    d['d_est_demand'] = d['est_demand'] - d['est_demand'].shift(24)
    d['dow'] = d.index.dayofweek
    return d[(d.index >= start) & (d.index <= f'{end} 23:00')]


def run(start: str, end: str, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    da_b = pickle.load(open(DA_BUNDLE, 'rb'))
    clf_b = pickle.load(open(CLF_BUNDLE, 'rb'))
    clf = clf_b['clf']
    th_lo, th_hi = D2_THETA['lo'], D2_THETA['hi']   # D+2 전용 이중 운영점
    min_run = int(clf_b.get('min_run', 2))
    with open(DEPTH_LOOKUP, encoding='utf-8') as f:
        depth_tbl = json.load(f)

    d = _build_serving(start, end).dropna(subset=DA_FEATURES)

    # ① D+2 DA 가격선 (잔차회귀)
    d[OUT_PRICE] = np.round(_predict_da(da_b['model'], d), 2)

    # ② 음수검지기 오버레이 — D+1 분류기 그대로, 가격선 피처만 D+2 예측 DA로 치환
    clf_in = d[CLF_FEATURES].copy()
    clf_in['smp_jeju_da'] = d[OUT_PRICE].values
    proba = clf.predict_proba(clf_in[CLF_FEATURES])[:, 1]
    d[OUT_PROBA] = np.round(proba, 4)
    d[OUT_DANGER] = persist(pd.Series(proba >= th_lo, index=d.index), min_run=min_run).astype(int).values
    d[OUT_DANGER_HI] = persist(pd.Series(proba >= th_hi, index=d.index), min_run=min_run).astype(int).values

    # ③ 깊이 overlay (Phase2 룩업 재사용, est_solar_utilization 키)
    su = pd.to_numeric(_solar_util(start, end), errors='coerce').reindex(d.index)
    p10 = p50 = p90 = None
    rows = [lookup_depth(depth_tbl, ts.hour, su.get(ts, np.nan)) for ts in d.index]
    d[OUT_DEPTH[0]] = [r[0] for r in rows]
    d[OUT_DEPTH[1]] = [r[1] for r in rows]
    d[OUT_DEPTH[2]] = [r[2] for r in rows]

    if write:
        _upsert(d)

    if verbose:
        _report(d)
    return d


def predict_d2_to_db(date: str, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """단일일(date)의 D+2 24h 서빙 — D+1 predict_*_to_db(date)와 대칭 API."""
    return run(date, date, write=write, verbose=verbose)


def _solar_util(start, end):
    with _conn() as con:
        s = pd.read_sql(
            'SELECT timestamp, est_solar_utilization_jeju FROM forecast '
            'WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'),
            parse_dates=['timestamp']).set_index('timestamp')['est_solar_utilization_jeju']
    return s


def _upsert(d: pd.DataFrame):
    out = pd.DataFrame({'timestamp': d.index.strftime('%Y-%m-%d %H:%M:%S')})
    for c in OUT_COLS:
        out[c] = d[c].values
    with _conn() as con:
        cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
        for c in OUT_COLS:
            if c not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
        setclause = ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS)
        placeholders = ','.join(['?'] * (len(OUT_COLS) + 1))
        con.executemany(
            f'INSERT INTO forecast ("timestamp",{",".join(chr(34)+c+chr(34) for c in OUT_COLS)}) '
            f'VALUES ({placeholders}) ON CONFLICT("timestamp") DO UPDATE SET {setclause}',
            list(out.itertuples(index=False, name=None)))
        con.commit()
    print(f'[DB] forecast ← {OUT_COLS} {len(out)}행 UPSERT ({out.timestamp.iloc[0]} ~ {out.timestamp.iloc[-1]})')


def _report(d: pd.DataFrame):
    print(f'\n═══ D+2 서빙 검증 (예측입력 기준, n={len(d)}) ═══')
    # ① 가격선 MAE (예측 DA vs 실측 DA)
    base = mean_absolute_error(d['smp_jeju_da'], d['lag24'])
    new = mean_absolute_error(d['smp_jeju_da'], d[OUT_PRICE])
    print(f'  ① DA 가격선 MAE: baseline(lag24)={base:.2f}  new={new:.2f}  Δ={new-base:+.2f}')
    # ② 음수경보 성능 (vs 실측 rt<5) — 이중 운영점, D+1과 분리 보고
    yneg = (d['smp_jeju_rt'] < NEG_THRESH).values
    n_neg = int(yneg.sum())
    print(f'  ② D+2 음수경보 (rt<{NEG_THRESH}={n_neg}, 이중 운영점):')
    for lbl, col, th in [('균형 lo', OUT_DANGER, D2_THETA['lo']),
                         ('고확신 hi', OUT_DANGER_HI, D2_THETA['hi'])]:
        alarm = d[col].values.astype(bool)
        tp = int((alarm & yneg).sum()); fp = int((alarm & ~yneg).sum())
        rec = tp / n_neg if n_neg else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        print(f'     [{lbl} θ={th}] 경보 {int(alarm.sum()):>3}h  recall={rec:.3f}  '
              f'precision={prec:.3f}  헛경보={fp}')
    print(f'     ※ 입력이 전부 예측값 → D+1 실측입력 경보와 직접 비교 불가, 별도 지표.')
    # 가격선 불변·overlay 확인
    print(f'  ③ est_smp_jeju_d2 는 신규컬럼 — D+1 est_smp_jeju 불변(별도 컬럼).')


if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 D+2 SMP 서빙 (DB 전용)')
    p.add_argument('start'); p.add_argument('end')
    p.add_argument('--no-write', action='store_true')
    a = p.parse_args()
    run(a.start, a.end, write=not a.no_write)
