"""4단계 보너스 — D+2(뒤24h) DA SMP 가격선 회귀 (잔차회귀, A안 확장).

추가작업.md §3 Step1. D-2 운영의 뒤24h(모레)는 DA가 미발표 → DA를 예측해 앵커 생성.
타깃 = smp_jeju_da, 앵커 = lag24(=D+1 DA, 예측시점 발표됨), 모델 = LGBM 잔차회귀.

피처셋(Feature Gate 합의 2026-06-05, D+1 음수경보와 철학이 완전히 다름):
  앵커/시간구조 : lag24, lag168, hour, month, dow
  변화량(핵심)  : d_net_load, d_est_demand   (어제↔오늘 차이 = 잔차 신호)
  레벨(맥락)    : net_load, est_demand
  ※ solar_util 제외(사용자). 신재생 정보는 est_demand−net_load로 암묵 보존.

train/serve parity:
  net_load   : train real_demand_jeju-real_renew_gen_jeju / serve est_net_load_jeju
  est_demand : train real_demand_jeju                     / serve jeju_est_demand_new

평가: baseline(lag24 persist) MAE vs new MAE — 둘 다 실제 smp_jeju_da 기준.
원본 DB·D+1 A안/Phase 2 코드 미변경. 신규 파일.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(
    HERE, '..', '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
MODELS = os.path.normpath(os.path.join(HERE, '..', 'models_weight'))
BUNDLE = os.path.join(MODELS, 'smp_d2_da.pkl')

TARGET = 'smp_jeju_da'
ANCHOR = 'lag24'
FEATURES = ['lag24', 'lag168', 'net_load', 'd_net_load',
            'est_demand', 'd_est_demand', 'hour', 'month', 'dow']

# 데이터 컷오프(결측 회피, 사용자 지시) / 학습창 2020~ 전구간(DA는 레짐 무관)
DATA_START = '2020-01-01'
DATA_END   = '2026-05-28 23:00'
# 시계열 split (test에 봄 floor 계절 포함 = 어려운·공정한 검증)
TRAIN_END  = '2025-08-31 23:00'
VAL_START  = '2025-09-01'
VAL_END    = '2025-12-31 23:00'
TEST_START = '2026-01-01'


def _conn():
    import sqlite3
    return sqlite3.connect(DB_PATH)


def _build(df):
    """net_load·est_demand 채워진 시간연속 df에 잔차회귀 피처 부착."""
    df = df.sort_index()
    df['lag24'] = df[TARGET].shift(24)
    df['lag168'] = df[TARGET].shift(168)
    df['d_net_load'] = df['net_load'] - df['net_load'].shift(24)
    df['d_est_demand'] = df['est_demand'] - df['est_demand'].shift(24)
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['dow'] = df.index.dayofweek
    return df


def load_historical():
    """학습/검증 — 실측 피처 + 타깃(smp_jeju_da). 시간연속 reindex로 shift 정합."""
    cols = ['timestamp', 'smp_jeju_da', 'real_demand_jeju', 'real_renew_gen_jeju']
    with _conn() as con:
        df = pd.read_sql(f'SELECT {",".join(cols)} FROM historical ORDER BY timestamp',
                         con, parse_dates=['timestamp']).set_index('timestamp')
    df = df[(df.index >= DATA_START) & (df.index <= DATA_END)]
    # 결측 행이 있어도 shift(24/168)가 어긋나지 않도록 완전한 시간격자로 reindex
    full = pd.date_range(df.index.min(), df.index.max(), freq='h')
    df = df.reindex(full)
    df['net_load'] = df['real_demand_jeju'] - df['real_renew_gen_jeju']
    df['est_demand'] = df['real_demand_jeju']
    return _build(df)


def _predict_da(model, X):
    """잔차회귀: pred_DA = lag24 + 모델(잔차)."""
    return X['lag24'].values + model.predict(X[FEATURES])


def main():
    df = load_historical()
    df = df.dropna(subset=FEATURES + [TARGET])

    tr = df[df.index <= TRAIN_END]
    va = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    te = df[df.index >= TEST_START]
    print(f'TRAIN n={len(tr)} ({tr.index.min().date()}~{tr.index.max().date()})  '
          f'VAL n={len(va)}  TEST n={len(te)}\n')

    # 잔차 타깃 = DA - lag24
    ytr = (tr[TARGET] - tr['lag24']).values
    yva = (va[TARGET] - va['lag24']).values

    model = lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=2000, learning_rate=0.03,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=40, random_state=42, verbose=-1)
    model.fit(tr[FEATURES], ytr,
              eval_set=[(va[FEATURES], yva)], eval_metric='l1',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f'best_iter={model.best_iteration_}\n')

    print('═══ baseline(lag24 persist) vs new(잔차회귀) — 실제 smp_jeju_da 기준 MAE ═══')
    print(f'    {"split":>6} {"n":>6} {"baseline":>9} {"new":>8} {"Δ":>7}')
    for tag, d in [('TRAIN', tr), ('VAL', va), ('TEST', te)]:
        base = mean_absolute_error(d[TARGET], d['lag24'])
        new = mean_absolute_error(d[TARGET], _predict_da(model, d))
        print(f'    {tag:>6} {len(d):>6} {base:>9.2f} {new:>8.2f} {new-base:>+7.2f}')

    # 봄 floor 계절(2~5,11월) 분리 — 모델이 실제 싸우는 구간
    print('\n── TEST 계절 분리 (neg-season=2,3,4,5,11월) ──')
    for lbl, mask in [('neg-season', te.index.month.isin([2, 3, 4, 5, 11])),
                      ('rest', ~te.index.month.isin([2, 3, 4, 5, 11]))]:
        d = te[mask]
        if len(d):
            base = mean_absolute_error(d[TARGET], d['lag24'])
            new = mean_absolute_error(d[TARGET], _predict_da(model, d))
            print(f'    {lbl:>10} n={len(d):>5}  baseline={base:.2f}  new={new:.2f}  Δ={new-base:+.2f}')

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n── 피처 중요도 ──'); print(imp.to_string())

    os.makedirs(MODELS, exist_ok=True)
    with open(BUNDLE, 'wb') as fh:
        pickle.dump({'model': model, 'features': FEATURES, 'anchor': ANCHOR,
                     'target': TARGET, 'data_end': DATA_END}, fh)
    print(f'\n[saved] {BUNDLE}')


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
