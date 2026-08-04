"""4단계 제주 SMP — DB 전용 서빙 파이프라인 (A안: DA 가격선 + 음수경보).

================================================================================
무엇을 하나
================================================================================
input_data_jeju.db 한 곳에서 읽고 쓴다.
  - 입력 : forecast (예보피처 = est_net_load_jeju·est_*_utilization_jeju·
           radiation_south·wind_spd_10m_west·smp_jeju_da)
  - 모델 : models_weight/smp_binary.pkl (이진 음수분류기 + θ + 2시간 지속규칙)
  - 출력 : forecast 테이블에 3컬럼 UPSERT
      est_smp_jeju        = smp_jeju_da (예측 가격선; A안은 DA를 그대로 씀)
      smp_neg_proba_jeju  = P(음수)  (0~1)
      smp_danger_jeju     = 음수경보 0/1 (P>=θ + 연속 MIN_RUN시간 지속)

피처빌더는 학습과 동일 파이프(train_smp_db.load_forecast)를 재사용 → train/serve parity 보장.

================================================================================
공개 API
================================================================================
    predict_smp_to_db(date)            # 해당일 24h 경보 → forecast UPSERT
    backfill_smp_to_db(start, end)     # 구간 일괄
"""
from __future__ import annotations

import os
import pickle
import sqlite3

import numpy as np
import pandas as pd

from train_smp_db import DB_PATH, FEATURES, load_forecast
from train_binary_smp import persist

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, 'models_weight', 'smp_binary.pkl')

OUT_PRICE = 'est_smp_jeju'
OUT_PROBA = 'smp_neg_proba_jeju'
OUT_DANGER = 'smp_danger_jeju'
OUT_COLS = [OUT_PRICE, OUT_PROBA, OUT_DANGER]


def _conn():
    return sqlite3.connect(DB_PATH)


def _load_model(which='p25'):
    b = pickle.load(open(BUNDLE, 'rb'))
    th = b.get('theta')
    if th is None:
        raise ValueError('번들에 θ가 없음 — train_binary_smp 후 운영점 저장 필요')
    if isinstance(th, dict):           # 이중 운영점 — 기본 p25
        th = th.get(which, next(iter(th.values())))
    return b['clf'], float(th), int(b.get('min_run', 2))


def _predict_frame(d: pd.DataFrame, clf, theta, min_run) -> pd.DataFrame:
    proba = clf.predict_proba(d[FEATURES])[:, 1]
    raw = pd.Series(proba >= theta, index=d.index)
    danger = persist(raw, min_run=min_run)
    return pd.DataFrame({
        'timestamp': d.index.strftime('%Y-%m-%d %H:%M:%S'),
        OUT_PRICE: d['smp_jeju_da'].round(2).values,
        OUT_PROBA: np.round(proba, 4),
        OUT_DANGER: danger.astype(int).values,
    })


def _upsert(out: pd.DataFrame):
    with _conn() as con:
        cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
        for c in OUT_COLS:
            if c not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
        con.executemany(
            f'INSERT INTO forecast ("timestamp","{OUT_PRICE}","{OUT_PROBA}","{OUT_DANGER}") '
            f'VALUES (?,?,?,?) ON CONFLICT("timestamp") DO UPDATE SET '
            f'"{OUT_PRICE}"=excluded."{OUT_PRICE}", "{OUT_PROBA}"=excluded."{OUT_PROBA}", '
            f'"{OUT_DANGER}"=excluded."{OUT_DANGER}"',
            list(zip(out['timestamp'], out[OUT_PRICE].astype(float),
                     out[OUT_PROBA].astype(float), out[OUT_DANGER].astype(int))))
        con.commit()


def predict_smp_to_db(date: str, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """date(YYYY-MM-DD)의 24h 음수경보를 forecast 테이블에 UPSERT."""
    clf, theta, min_run = _load_model()
    tgt = pd.Timestamp(date).normalize()
    fc = load_forecast(with_target=False)          # 미래날짜=rt 없음 → 타깃 join 생략
    d = fc[(fc.index >= tgt) & (fc.index <= tgt + pd.Timedelta(hours=23))]
    if len(d) != 24:
        raise ValueError(f'[forecast] {date} 24행 필요 — 발견 {len(d)}행')
    if d[FEATURES].isna().any().any():
        raise ValueError(f'[forecast] {date} 피처 결측 — 예보 보유 확인')

    out = _predict_frame(d, clf, theta, min_run)
    if write:
        _upsert(out)
        if verbose:
            print(f'[DB] forecast ← {date} {OUT_COLS} 24행 UPSERT (θ={theta}, '
                  f'경보 {int(out[OUT_DANGER].sum())}h)')
    if verbose:
        print(out.to_string(index=False))
    return out


def backfill_smp_to_db(start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """[start,end] forecast 보유 날짜 전부 경보 채움."""
    clf, theta, min_run = _load_model()
    fc = load_forecast(with_target=False)
    with _conn() as con:
        dates = pd.read_sql(
            "SELECT substr(timestamp,1,10) d, COUNT(*) n FROM forecast "
            "WHERE est_net_load_jeju IS NOT NULL AND timestamp BETWEEN ? AND ? "
            "GROUP BY d HAVING n=24 ORDER BY d",
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'))['d'].tolist()

    done, skipped, rows = [], [], []
    for date in dates:
        tgt = pd.Timestamp(date).normalize()
        d = fc[(fc.index >= tgt) & (fc.index <= tgt + pd.Timedelta(hours=23))]
        if len(d) != 24 or d[FEATURES].isna().any().any():
            skipped.append(date); continue
        rows.append(_predict_frame(d, clf, theta, min_run)); done.append(date)
    if rows:
        allout = pd.concat(rows, ignore_index=True)
        _upsert(allout)
    if verbose:
        n_alarm = int(pd.concat(rows)[OUT_DANGER].sum()) if rows else 0
        print(f'[backfill] {len(done)}일 / 스킵 {len(skipped)}일 / 경보 {n_alarm}h '
              f'(θ={theta})')
        if done:
            print(f'  범위: {done[0]} ~ {done[-1]}')
    return pd.DataFrame({'date': done})


if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 SMP 음수경보 서빙 (DB 전용)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('date'); pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    a = p.parse_args()
    if a.cmd == 'predict':
        predict_smp_to_db(a.date, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_smp_to_db(a.start, a.end)
