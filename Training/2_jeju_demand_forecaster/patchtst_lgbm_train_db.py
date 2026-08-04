"""LGBM 재학습 (DB 직접 로드 + 3지점 기상).

patchtst_lgbm_train.py 의 DB 버전.
================================================================================
바뀐 점
================================================================================
1. 입력을 features_train.csv / features_test.csv 에서 → input_data_jeju.db
   (historical 테이블) 직접 추출로 변경. (임시 CSV 는 data/_features_from_db.csv 로 덤프)
2. 기상 피처는 원본과 동일한 4개 (temp_c/humidity/solar_rad/wind_spd) 이지만,
   1지점 값이 아니라 제주 3지점 "공간평균" 으로 계산한다.
     temp_c / humidity / wind_spd = mean(west, east, south)
     solar_rad                    = mean(west, south)   ← east 는 일사 없음(풍력지점)
3. patchtst_target 신호는 기존 patchtst_features.csv 재사용 (univariate → DB 무관).

산출 (원본과 동일 — demand_predict.py 가 그대로 읽음)
================================================================================
- models/lgbm_pipeline.pkl
- models/pipeline_config.json

DB → 모델 4기상 매핑 (학습 historical / 추론 forecast, 양쪽 다 3지점 평균)
================================================================================
  모델 피처     학습 historical 평균            추론 forecast 평균
  temp_c       mean(temp_c_{w,e,s})           mean(temp_{w,e,s})
  humidity     mean(humidity_{w,e,s})         mean(reh_{w,e,s})
  wind_spd     mean(wind_spd_{w,e,s})         mean(wind_spd_10m_{w,e,s})
  solar_rad    mean(solar_rad_{w,s})          mean(radiation_{w,s})
"""
import os
import sys
import json
import pickle
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# =============================================================================
# 0. 경로
# =============================================================================
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, 'data')
MODELS_DIR = os.path.join(HERE, 'models')
DB_PATH    = os.path.normpath(os.path.join(
    HERE, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))

PATCHTST_PATH = os.path.join(HERE, 'patchtst_features.csv')   # 기존 신호 재사용
DUMP_CSV      = os.path.join(DATA_DIR, '_features_from_db.csv')  # 임시 학습용 덤프

OUT_PKL  = os.path.join(MODELS_DIR, 'lgbm_pipeline.pkl')
OUT_JSON = os.path.join(MODELS_DIR, 'pipeline_config.json')

# =============================================================================
# 1. DB 추출 (historical) → 임시 CSV 덤프
# =============================================================================
print('=== 1. DB 추출 ===')
# 3지점 원자료를 끌어와 "제주 공간평균" 4개 기상으로 집약한다.
# (제주 전력수요는 지역 단일 물리량 → 지점별 분리보다 평균이 자연스럽고 과적합 위험 낮음)
STATION_COLS = [
    'temp_c_west', 'temp_c_east', 'temp_c_south',
    'humidity_west', 'humidity_east', 'humidity_south',
    'wind_spd_west', 'wind_spd_east', 'wind_spd_south',
    'solar_rad_west', 'solar_rad_south',          # east 일사 없음 → west,south 평균
]
WEATHER_COLS = ['temp_c', 'humidity', 'solar_rad', 'wind_spd']  # 모델이 쓰는 4개
PULL_COLS = ['timestamp', 'real_demand_jeju', 'jeju_est_demand_da',
             'day_type'] + STATION_COLS

con = sqlite3.connect(DB_PATH)
sel = ', '.join(f'"{c}"' for c in PULL_COLS)
raw = pd.read_sql(f'SELECT {sel} FROM historical', con, parse_dates=['timestamp'])
con.close()
raw = raw.sort_values('timestamp').reset_index(drop=True)

# 표준 이름으로 정리: real_demand (타깃), est_demand (KPX baseline, 평가용)
raw = raw.rename(columns={'real_demand_jeju': 'real_demand',
                          'jeju_est_demand_da': 'est_demand'})

# 3지점(일사는 2지점) 공간평균 → 4개 기상 피처
raw['temp_c']    = raw[['temp_c_west', 'temp_c_east', 'temp_c_south']].mean(axis=1)
raw['humidity']  = raw[['humidity_west', 'humidity_east', 'humidity_south']].mean(axis=1)
raw['wind_spd']  = raw[['wind_spd_west', 'wind_spd_east', 'wind_spd_south']].mean(axis=1)
raw['solar_rad'] = raw[['solar_rad_west', 'solar_rad_south']].mean(axis=1)
raw = raw.drop(columns=STATION_COLS)

# real_demand 의 0 은 결측으로 보고 시간보간 (제주 수요 0 불가 — dropna 금지 규칙)
n_zero = int((raw['real_demand'] == 0).sum())
raw.loc[raw['real_demand'] == 0, 'real_demand'] = np.nan
raw['real_demand'] = (raw.set_index('timestamp')['real_demand']
                         .interpolate(method='time').values)
print(f'  DB rows  : {raw.shape}   {raw.timestamp.min()} ~ {raw.timestamp.max()}')
print(f'  real_demand 0→보간 처리 : {n_zero}개')

os.makedirs(DATA_DIR, exist_ok=True)
raw.to_csv(DUMP_CSV, index=False)
print(f'  임시 덤프 : {DUMP_CSV}')

# =============================================================================
# 2. patchtst 신호 병합 + lag/roll 재생성
# =============================================================================
print('\n=== 2. patchtst 병합 + lag/roll ===')
# patchtst 신호는 DB 테이블(patchtst_signal)에서 읽는다 (CSV 탈피).
con = sqlite3.connect(DB_PATH)
ptst_df = pd.read_sql('SELECT timestamp, jeju_patchtst_target AS patchtst_target '
                      'FROM patchtst_signal', con, parse_dates=['timestamp'])
con.close()
print(f'  patchtst : {ptst_df.shape}   {ptst_df.timestamp.min()} ~ {ptst_df.timestamp.max()}')

full_df = raw.merge(ptst_df, on='timestamp', how='left')

s = full_df['real_demand']
full_df['lag_24h']       = s.shift(24)
full_df['roll_mean_24h'] = s.shift(1).rolling(window=24, min_periods=24).mean()
print('  lag_24h, roll_mean_24h 생성 완료')

# =============================================================================
# 3. 사이클 피처 + 분할 마스크
# =============================================================================
h   = full_df['timestamp'].dt.hour
dow = full_df['timestamp'].dt.dayofweek
full_df['hour_sin'] = np.sin(2 * np.pi * h   / 24)
full_df['hour_cos'] = np.cos(2 * np.pi * h   / 24)
full_df['dow_sin']  = np.sin(2 * np.pi * dow / 7)
full_df['dow_cos']  = np.cos(2 * np.pi * dow / 7)

ts = full_df['timestamp']
mask_train = ts <= '2025-02-28 23:00'
mask_val   = (ts >= '2025-03-01') & (ts <= '2026-03-21 23:00')
mask_test  = (ts >= '2026-03-22') & (ts <= '2026-05-31 23:00')

# =============================================================================
# 4. 피처 정의 (19개 = 3 prior + 11 weather + 4 cycle + 1 cat)
# =============================================================================
TARGET = 'real_demand'
FEATURE_COLS = (
    ['patchtst_target', 'lag_24h', 'roll_mean_24h']
    + WEATHER_COLS
    + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'day_type']
)
CATEGORICAL_COLS = ['day_type']

print(f'\n=== 4. 피처 ({len(FEATURE_COLS)}개) ===')
for i, c in enumerate(FEATURE_COLS, 1):
    print(f'  {i:>2d}. {c}')

for c in CATEGORICAL_COLS:
    full_df[c] = full_df[c].astype('category')

X = full_df[FEATURE_COLS]
y = full_df[TARGET]
X_train, y_train = X[mask_train], y[mask_train]
X_val,   y_val   = X[mask_val],   y[mask_val]
X_test,  y_test  = X[mask_test],  y[mask_test]
print(f'\n  Train : {X_train.shape},  Val : {X_val.shape},  Test : {X_test.shape}')
print(f'  patchtst_target NaN  Train={int(X_train.patchtst_target.isna().sum())}'
      f'  Val={int(X_val.patchtst_target.isna().sum())}'
      f'  Test={int(X_test.patchtst_target.isna().sum())}')

# =============================================================================
# 5. LightGBM 학습 (원본과 동일 params)
# =============================================================================
print('\n=== 5. LightGBM 학습 ===')
params = dict(
    objective='regression_l1', metric='mae',
    learning_rate=0.024, num_leaves=244, min_data_in_leaf=76,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=0.1, verbosity=-1, random_state=42,
)
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
dval   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=CATEGORICAL_COLS,
                     reference=dtrain)
model = lgb.train(
    params, dtrain, num_boost_round=3000,
    valid_sets=[dval], valid_names=['val'],
    callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)],
)
best_iter = int(model.best_iteration)
print(f'\n  best_iteration : {best_iter}')
print(f'  val best MAE   : {model.best_score["val"]["l1"]:.3f}')

# =============================================================================
# 6. 평가
# =============================================================================
def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true > 0)
    yt, yp = y_true[m], y_pred[m]
    return dict(n=int(len(yt)),
                mae=float(np.mean(np.abs(yt - yp))),
                rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
                mape=float(np.mean(np.abs((yt - yp) / yt)) * 100))

def pr(label, sc):
    print(f'  {label:<20s} N={sc["n"]:>5d}  MAE={sc["mae"]:7.3f}  '
          f'RMSE={sc["rmse"]:7.3f}  MAPE={sc["mape"]:5.3f}%')

pred_val  = model.predict(X_val,  num_iteration=best_iter)
pred_test = model.predict(X_test, num_iteration=best_iter)
print('\n=== 6. 평가 ===')
print('--- Val (실측 기상) ---')
pr('KPX baseline',    score(y_val,  full_df.loc[mask_val,  'est_demand'].values))
pr('PatchTST only',   score(y_val,  full_df.loc[mask_val,  'patchtst_target'].values))
pr('LGBM + PatchTST', score(y_val,  pred_val))
print('--- Test (예보 기상) ---')
pr('KPX baseline',    score(y_test, full_df.loc[mask_test, 'est_demand'].values))
pr('PatchTST only',   score(y_test, full_df.loc[mask_test, 'patchtst_target'].values))
pr('LGBM + PatchTST', score(y_test, pred_test))

# =============================================================================
# 7. 피처 중요도
# =============================================================================
imp = pd.DataFrame({
    'feature': model.feature_name(),
    'gain':    model.feature_importance(importance_type='gain'),
    'split':   model.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)
print('\n=== 7. 피처 중요도 (gain 내림차순) ===')
print(imp.to_string(index=False))

# =============================================================================
# 8. 저장
# =============================================================================
os.makedirs(MODELS_DIR, exist_ok=True)
trimmed = model.model_to_string(num_iteration=best_iter)
with open(OUT_PKL, 'wb') as f:
    pickle.dump(lgb.Booster(model_str=trimmed), f)

cfg = {
    'feature_cols':           FEATURE_COLS,
    'categorical_cols':       CATEGORICAL_COLS,
    'categorical_categories': {c: list(full_df[c].cat.categories) for c in CATEGORICAL_COLS},
    'target':                 TARGET,
    'best_iteration':         best_iter,
    'val_best_mae':           float(model.best_score['val']['l1']),
    'lgbm_version':           lgb.__version__,
    'trained_at':             datetime.now().isoformat(timespec='seconds'),
    'train_range':            ['2020-01-01', '2025-02-28'],
    'val_range':              ['2025-03-01', '2026-03-21'],
    'signal_source':          'patchtst',
    'data_source':            'input_data_jeju.db/historical (3-station weather)',
    'weather_cols':           WEATHER_COLS,
    'params':                 params,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print('\n=== 8. 저장 ===')
print(f'  {OUT_PKL}  ({os.path.getsize(OUT_PKL) // 1024} KB)')
print(f'  {OUT_JSON} ({os.path.getsize(OUT_JSON)} B)')
print('완료.')
