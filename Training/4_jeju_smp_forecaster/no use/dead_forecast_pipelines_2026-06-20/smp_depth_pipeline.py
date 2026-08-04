"""4단계 SMP Phase 2-B — 음수 깊이 overlay 서빙 파이프라인 (DB 전용).

무엇을 하나
  - 입력 : forecast (timestamp, est_solar_utilization_jeju)
  - 룩업 : models_weight/smp_depth_lookup.json  (smp_phase2_depth.py 산출)
  - 출력 : forecast 테이블에 3컬럼 UPSERT
      smp_neg_depth_p10 / _p50 / _p90  = "음수 발생 조건부" 예상 깊이[원]

대전제: 가격선(est_smp_jeju=DA)·경보(smp_danger_jeju)는 **안 건드린다**. 이 값은 별도 컬럼이며,
표시층에서 smp_danger_jeju=1 인 구간에만 overlay로 보여준다(전 구간 점예측 아님).

smp_db_pipeline.py(P3 서빙)의 UPSERT 패턴을 그대로 따른다. P3 코드는 import하지 않음(독립).

공개 API:
    predict_depth_to_db(date)         # 해당일 24h 깊이 overlay → forecast UPSERT
    backfill_depth_to_db(start, end)  # 구간 일괄
"""
from __future__ import annotations

import os
import json
import sqlite3
import numpy as np
import pandas as pd

from train_smp_db import DB_PATH
from smp_phase2_depth import lookup_depth, LOOKUP

OUT_COLS = ['smp_neg_depth_p10', 'smp_neg_depth_p50', 'smp_neg_depth_p90']


def _conn():
    return sqlite3.connect(DB_PATH)


def _load_table():
    with open(LOOKUP, encoding='utf-8') as f:
        return json.load(f)


def _frame(d: pd.DataFrame, table) -> pd.DataFrame:
    """forecast 행(timestamp index, est_solar_utilization_jeju) → 깊이 3분위."""
    rows = []
    for ts, su in d['est_solar_utilization_jeju'].items():
        p10, p50, p90, _ = lookup_depth(table, ts.hour, su)
        rows.append((ts.strftime('%Y-%m-%d %H:%M:%S'), p10, p50, p90))
    return pd.DataFrame(rows, columns=['timestamp', *OUT_COLS])


def _upsert(out: pd.DataFrame):
    with _conn() as con:
        cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
        for c in OUT_COLS:
            if c not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
        con.executemany(
            f'INSERT INTO forecast ("timestamp","{OUT_COLS[0]}","{OUT_COLS[1]}","{OUT_COLS[2]}") '
            f'VALUES (?,?,?,?) ON CONFLICT("timestamp") DO UPDATE SET '
            + ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS),
            list(out.itertuples(index=False, name=None)))
        con.commit()


def _load_day(date: str) -> pd.DataFrame:
    tgt = pd.Timestamp(date).normalize()
    with _conn() as con:
        d = pd.read_sql(
            'SELECT timestamp, est_solar_utilization_jeju FROM forecast '
            'WHERE est_solar_utilization_jeju IS NOT NULL AND timestamp BETWEEN ? AND ? '
            'ORDER BY timestamp',
            con, params=(f'{date} 00:00:00', f'{date} 23:00:00'),
            parse_dates=['timestamp']).set_index('timestamp')
    return d


def predict_depth_to_db(date: str, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    table = _load_table()
    d = _load_day(date)
    if d.empty:
        raise ValueError(f'[forecast] {date} est_solar_utilization 없음')
    d['est_solar_utilization_jeju'] = pd.to_numeric(d['est_solar_utilization_jeju'], errors='coerce')
    out = _frame(d, table)
    if write:
        _upsert(out)
        if verbose:
            print(f'[DB] forecast ← {date} {OUT_COLS} {len(out)}행 UPSERT')
    if verbose:
        print(out.to_string(index=False))
    return out


def backfill_depth_to_db(start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    table = _load_table()
    with _conn() as con:
        d = pd.read_sql(
            'SELECT timestamp, est_solar_utilization_jeju FROM forecast '
            'WHERE est_solar_utilization_jeju IS NOT NULL AND timestamp BETWEEN ? AND ? '
            'ORDER BY timestamp',
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'),
            parse_dates=['timestamp']).set_index('timestamp')
    if d.empty:
        print('[backfill] 대상 없음'); return pd.DataFrame()
    d['est_solar_utilization_jeju'] = pd.to_numeric(d['est_solar_utilization_jeju'], errors='coerce')
    out = _frame(d, table)
    _upsert(out)
    if verbose:
        print(f'[backfill] {len(out)}행 UPSERT  범위 {out["timestamp"].iloc[0]} ~ {out["timestamp"].iloc[-1]}')
    return out


if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 SMP 음수 깊이 overlay 서빙 (DB 전용)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('date'); pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    a = p.parse_args()
    if a.cmd == 'predict':
        predict_depth_to_db(a.date, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_depth_to_db(a.start, a.end)
