"""4단계 SMP — 데이터/피처 공유 모듈 (학습·서빙 공통 로더).

input_data_jeju.db 에서 학습창(historical 실측)·서빙창(forecast 예보) 피처를
같은 규칙으로 만든다. train_binary_smp.py(학습)·smp_db_pipeline.py(서빙)가 import.

train/serve 매핑:
  net_load   : train real_demand_jeju-real_renew_gen_jeju / serve est_net_load_jeju
  est_demand : train real_demand_jeju                     / serve jeju_est_demand_new (1단계 출력)
  (캘린더 is_midday/is_neg_season/hour/month는 train/serve 동일)
"""
from __future__ import annotations

import os
import sys
import sqlite3

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# 폴더 분리(2026-06-05): 학습/EDA 스크립트는 training/ 에 둔다. 서빙 파이프라인이
# training/ 의 serve-util(persist·lookup_depth·load_calibrator·_predict_da)을 import하므로
# 공통 로더인 이 모듈이 import될 때 training/ 를 path에 올려 둔다(모든 서빙이 이 모듈을 먼저 import).
_TRAINING = os.path.join(HERE, 'training')
if os.path.isdir(_TRAINING) and _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
DB_PATH = os.path.normpath(os.path.join(
    HERE, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))

# 피처셋(사용자 개정 2026-06-05): 기상/신재생 4피처 제거, est_demand(+lead) 추가.
# is_midday/is_neg_season은 hour/month와 중복(중요도 최하위)으로 판명 → 제거.
FEATURES = ['smp_jeju_da', 'net_load', 'nl_lead_1', 'nl_lead_2',
            'est_demand', 'est_demand_lead_1', 'est_demand_lead_2', 'hour', 'month']
# 실시간 SMP 컬럼 — **회귀 타깃이 아니다.**  음수경보 라벨(rt<NEG_THRESH)을 만들고 검증
# 채점에만 쓴다.  가격선은 항상 DA(=smp_jeju_da)를 예측한다: D+1 은 발표된 DA 그대로(A안),
# D+2 는 DA 잔차회귀(train_smp_d2_da, TARGET='smp_jeju_da').  rt 점예측은 잠긴 실패경로
# (trial_error.md) — 다시 시도하지 않는다.
RT_COL = 'smp_jeju_rt'

# 음수경보 이벤트 임계: rt < NEG_THRESH (5 = 사실상 0에 가까운 무가치 SMP ≈ 음수).
NEG_THRESH = 5

TRAIN_START = '2024-03-01'
TRAIN_END   = '2025-12-31 23:00'
VAL_START   = '2026-01-01'
TEST_START  = '2025-12-13'   # forecast est_net_load 시작

SEASON = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄',
          6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을', 11: '가을'}


def _conn():
    return sqlite3.connect(DB_PATH)


def _add_features(df):
    """net_load·est_demand 컬럼이 채워진 df에 lead·캘린더·플래그 피처 부착."""
    df = df.sort_index()
    df['nl_lead_1'] = df['net_load'].shift(-1)
    df['nl_lead_2'] = df['net_load'].shift(-2)
    df['est_demand_lead_1'] = df['est_demand'].shift(-1)
    df['est_demand_lead_2'] = df['est_demand'].shift(-2)
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    return df


def load_historical():
    """학습/검증용 — 실측 피처 + 타깃(smp_jeju_rt). rt 보유 구간만.
    est_demand = real_demand_jeju (서빙의 jeju_est_demand_new와 train/serve parity)."""
    cols = ['timestamp', 'smp_jeju_da', 'real_demand_jeju', 'real_renew_gen_jeju',
            'smp_jeju_rt']
    with _conn() as con:
        df = pd.read_sql(f'SELECT {",".join(cols)} FROM historical '
                         "WHERE smp_jeju_rt IS NOT NULL ORDER BY timestamp",
                         con, parse_dates=['timestamp']).set_index('timestamp')
    df['net_load'] = df['real_demand_jeju'] - df['real_renew_gen_jeju']
    df['est_demand'] = df['real_demand_jeju']
    return _add_features(df)


def load_forecast(with_target=True):
    """서빙/평가용 — 예보 피처. with_target=False면 rt join 생략(미래날짜 서빙용).
    net_load=est_net_load_jeju(3단계), est_demand=jeju_est_demand_new(1단계)."""
    fcols = ['timestamp', 'smp_jeju_da', 'est_net_load_jeju', 'jeju_est_demand_new']
    with _conn() as con:
        f = pd.read_sql(f'SELECT {",".join(fcols)} FROM forecast '
                        "WHERE est_net_load_jeju IS NOT NULL ORDER BY timestamp",
                        con, parse_dates=['timestamp']).set_index('timestamp')
        t = pd.read_sql('SELECT timestamp, smp_jeju_rt FROM historical '
                        'WHERE smp_jeju_rt IS NOT NULL',
                        con, parse_dates=['timestamp']).set_index('timestamp')
    f = f.rename(columns={'est_net_load_jeju': 'net_load',
                          'jeju_est_demand_new': 'est_demand'})
    for c in ['smp_jeju_da', 'net_load', 'est_demand']:
        f[c] = pd.to_numeric(f[c], errors='coerce')
    f[f.columns] = f[f.columns].interpolate(limit_direction='both').ffill().bfill()
    f = _add_features(f)
    return f.join(t, how='inner') if with_target else f
