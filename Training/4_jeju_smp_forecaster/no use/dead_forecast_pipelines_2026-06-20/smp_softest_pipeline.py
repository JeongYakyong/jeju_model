"""4단계 SMP D-1 위험구간 — 위험 레이어 서빙 파이프라인 (DB 전용).

쇼케이스(smp_risk_profile)를 DB만으로 재현할 수 있도록 forecast 테이블에 위험 레이어를 UPSERT.

입력 : forecast (timestamp, est_smp_jeju=DA, smp_neg_proba_jeju=raw proba)
보정 : models_weight/smp_calibrator.pkl (isotonic + d_cond=E[rt|rt<5])
출력 컬럼:
  smp_neg_proba_cal_jeju  보정 음수확률 P_cal (0~1)  → "음수 위험 N%"
  smp_rt_soft_est         위험조정 기대선 = (1-P_cal)·DA + P_cal·d_cond  (RT 점예측 아님)
  smp_danger_day_jeju     주간 경보구간 0/1 (θ=0.25 + 2h지속, 주간[8-16])
  smp_danger_night_jeju   야간 경보구간 0/1 (주간 밖)

가격선(est_smp_jeju=DA)·이진경보(smp_danger_jeju)는 안 건드림 — 별도 컬럼이다.
"""
from __future__ import annotations

import os
import sqlite3
import numpy as np
import pandas as pd

from train_smp_db import DB_PATH
from train_binary_smp import persist
from smp_calibrate import load_calibrator

THETA_ALARM = 0.25
DAY = range(8, 17)
PCAL = 'smp_neg_proba_cal_jeju'
SOFT = 'smp_rt_soft_est'
DDAY = 'smp_danger_day_jeju'
DNIGHT = 'smp_danger_night_jeju'
OUT_COLS = [PCAL, SOFT, DDAY, DNIGHT]


def _conn():
    return sqlite3.connect(DB_PATH)


def _runs(mask):
    f = np.asarray(mask, bool); out = []; i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def _frame(d: pd.DataFrame, iso, d_cond) -> pd.DataFrame:
    """하루치 forecast(index=timestamp) → 위험 레이어 4컬럼."""
    da = pd.to_numeric(d['est_smp_jeju'], errors='coerce').values
    proba = pd.to_numeric(d['smp_neg_proba_jeju'], errors='coerce').fillna(0).values
    pcal = iso.predict(proba)
    soft = (1 - pcal) * da + pcal * d_cond
    hours = np.array([t.hour for t in d.index])
    alarm = persist(pd.Series(proba >= THETA_ALARM, index=d.index)).values
    day = np.zeros(len(d), int); night = np.zeros(len(d), int)
    for (s, e) in _runs(alarm):
        is_day = np.mean([h in DAY for h in hours[s:e + 1]]) >= 0.5
        (day if is_day else night)[s:e + 1] = 1
    return pd.DataFrame({
        'timestamp': d.index.strftime('%Y-%m-%d %H:%M:%S'),
        PCAL: np.round(pcal, 4), SOFT: np.round(soft, 2),
        DDAY: day, DNIGHT: night,
    })


def _upsert(out: pd.DataFrame):
    with _conn() as con:
        cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
        for c in OUT_COLS:
            if c not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
        con.executemany(
            f'INSERT INTO forecast ("timestamp","{PCAL}","{SOFT}","{DDAY}","{DNIGHT}") '
            f'VALUES (?,?,?,?,?) ON CONFLICT("timestamp") DO UPDATE SET '
            + ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS),
            list(out.itertuples(index=False, name=None)))
        con.commit()


def _load_range(start, end):
    with _conn() as con:
        return pd.read_sql(
            'SELECT timestamp, est_smp_jeju, smp_neg_proba_jeju FROM forecast '
            'WHERE smp_neg_proba_jeju IS NOT NULL AND timestamp BETWEEN ? AND ? '
            'ORDER BY timestamp',
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'),
            parse_dates=['timestamp']).set_index('timestamp')


def predict_softest_to_db(date: str, write=True, verbose=True) -> pd.DataFrame:
    iso, d_cond = load_calibrator()
    d = _load_range(date, date)
    if d.empty:
        raise ValueError(f'[forecast] {date} proba 없음')
    out = _frame(d, iso, d_cond)
    if write:
        _upsert(out)
        if verbose:
            print(f'[DB] forecast ← {date} {OUT_COLS} {len(out)}행 UPSERT '
                  f'(주간{int(out[DDAY].sum())}h/야간{int(out[DNIGHT].sum())}h)')
    if verbose:
        print(out.to_string(index=False))
    return out


def backfill_softest_to_db(start: str, end: str, verbose=True) -> pd.DataFrame:
    iso, d_cond = load_calibrator()
    d = _load_range(start, end)
    if d.empty:
        print('[backfill] 대상 없음'); return pd.DataFrame()
    rows = [_frame(g, iso, d_cond) for _, g in d.groupby(d.index.date)]
    out = pd.concat(rows, ignore_index=True)
    _upsert(out)
    if verbose:
        print(f'[backfill] {len(out)}행 UPSERT  범위 {out["timestamp"].iloc[0]} ~ {out["timestamp"].iloc[-1]}  '
              f'(주간경보 {int(out[DDAY].sum())}h / 야간경보 {int(out[DNIGHT].sum())}h)')
    return out


if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 SMP D-1 위험 레이어 서빙 (DB 전용)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('date'); pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    a = p.parse_args()
    if a.cmd == 'predict':
        predict_softest_to_db(a.date, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_softest_to_db(a.start, a.end)
