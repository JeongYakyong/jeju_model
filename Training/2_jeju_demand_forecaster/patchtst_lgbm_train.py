"""Step 4: LGBM 재학습 (timesfm_target → patchtst_target).

용도
====
train_lean.py 의 변형. 차이점은 단 하나 — TimesFM 신호 대신
PatchTST 백필 결과 (patchtst_features.csv) 를 사용한다.
다른 모든 부분 (12 lean 피처, Stage 3 best params, 학습 절차) 은 동일.

산출
====
- models/lgbm_pipeline.pkl      (pickle 된 LightGBM booster — demand_predict.py 가 읽음)
- models/pipeline_config.json   (피처 스키마 + best_iteration)

사용법
======
    cd demand_forecast/training
    python patchtst_lgbm_train.py
"""
import os
import sys
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

# Windows 콘솔이 cp949 라 한글/em-dash 같은 유니코드 출력에서 깨지는 걸 막음
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# =============================================================================
# 0. 경로
# =============================================================================
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, 'data')
MODELS_DIR = os.path.normpath(os.path.join(HERE, '..', 'models'))

TRAIN_PATH    = os.path.join(DATA_DIR, 'features_train.csv')
TEST_PATH     = os.path.join(DATA_DIR, 'features_test.csv')
PATCHTST_PATH = os.path.join(DATA_DIR, 'patchtst_features.csv')

OUT_PKL  = os.path.join(MODELS_DIR, 'lgbm_pipeline.pkl')
OUT_JSON = os.path.join(MODELS_DIR, 'pipeline_config.json')


# =============================================================================
# 1. 데이터 로딩 + 병합
# =============================================================================
print('=== 1. 데이터 로딩 ===')
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=['timestamp'])
ptst_df  = pd.read_csv(PATCHTST_PATH, parse_dates=['timestamp'])

print(f'  train    : {train_df.shape}   {train_df.timestamp.min()} ~ {train_df.timestamp.max()}')
print(f'  test     : {test_df.shape}    {test_df.timestamp.min()} ~ {test_df.timestamp.max()}')
print(f'  patchtst : {ptst_df.shape}    {ptst_df.timestamp.min()} ~ {ptst_df.timestamp.max()}')

full_df = (pd.concat([train_df, test_df], axis=0, ignore_index=True)
             .sort_values('timestamp').reset_index(drop=True)
             .merge(ptst_df, on='timestamp', how='left'))


# =============================================================================
# 2. lag / rolling 재생성 (필요한 2개만)
# =============================================================================
print('\n=== 2. lag/roll 재계산 ===')
lag_roll_cols = ['lag_24h', 'lag_48h', 'lag_72h', 'lag_168h', 'lag_336h',
                 'roll_mean_24h', 'roll_std_24h', 'roll_max_24h',
                 'roll_min_24h', 'roll_mean_168h']
full_df = full_df.drop(columns=[c for c in lag_roll_cols if c in full_df.columns])

s = full_df['real_demand']
full_df['lag_24h']       = s.shift(24)
full_df['roll_mean_24h'] = s.shift(1).rolling(window=24, min_periods=24).mean()
print('  lag_24h, roll_mean_24h 생성 완료 (앞 24h warmup NaN → LGBM native 처리)')


# =============================================================================
# 3. 분할 마스크
# =============================================================================
ts = full_df['timestamp']
mask_train = ts <= '2025-02-28 23:00'
mask_val   = (ts >= '2025-03-01') & (ts <= '2026-03-21 23:00')
mask_test  = (ts >= '2026-03-22') & (ts <= '2026-05-22 23:00')


# =============================================================================
# 4. 12 lean 피처 정의 — patchtst_target 이 timesfm_target 자리에
# =============================================================================
TARGET = 'real_demand'
FEATURE_COLS = [
    # 시계열 prior + 신호
    'patchtst_target', 'lag_24h', 'roll_mean_24h',
    # 기상 raw
    'temp_c', 'humidity', 'solar_rad', 'wind_spd',
    # 사이클 시간 / 요일
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # 범주
    'day_type',
]
CATEGORICAL_COLS = ['day_type']

print(f'\n=== 4. 피처 ({len(FEATURE_COLS)}개) ===')
for i, c in enumerate(FEATURE_COLS, 1):
    print(f'  {i:>2d}. {c}')

# day_type 카테고리화
for c in CATEGORICAL_COLS:
    full_df[c] = full_df[c].astype('category')

X = full_df[FEATURE_COLS]
y = full_df[TARGET]
X_train, y_train = X[mask_train], y[mask_train]
X_val,   y_val   = X[mask_val],   y[mask_val]
X_test,  y_test  = X[mask_test],  y[mask_test]
print(f'\n  Train : {X_train.shape},  Val : {X_val.shape},  Test : {X_test.shape}')

# patchtst_target NaN 점검 (앞 28일 + 그 외)
n_nan_train_p = int(X_train['patchtst_target'].isna().sum())
n_nan_val_p   = int(X_val['patchtst_target'].isna().sum())
n_nan_test_p  = int(X_test['patchtst_target'].isna().sum())
print(f'  patchtst_target NaN  Train={n_nan_train_p}  Val={n_nan_val_p}  Test={n_nan_test_p}')
print('  (앞 28일 warmup NaN 정상 — LGBM native 처리)')


# =============================================================================
# 5. LightGBM 학습 (Stage 3 best params 고정 — train_lean.py 와 동일)
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
    params, dtrain,
    num_boost_round=3000,
    valid_sets=[dval], valid_names=['val'],
    callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)],
)
best_iter = int(model.best_iteration)
print(f'\n  best_iteration : {best_iter}')
print(f'  val best MAE   : {model.best_score["val"]["l1"]:.3f}')


# =============================================================================
# 6. 평가 (Val + Test — KPX / PatchTST single / LGBM)
# =============================================================================
def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true > 0)
    yt, yp = y_true[m], y_pred[m]
    return dict(n=int(len(yt)),
                mae  = float(np.mean(np.abs(yt - yp))),
                rmse = float(np.sqrt(np.mean((yt - yp) ** 2))),
                mape = float(np.mean(np.abs((yt - yp) / yt)) * 100))

def pr(label, s):
    print(f'  {label:<20s} N={s["n"]:>5d}  MAE={s["mae"]:7.3f}  '
          f'RMSE={s["rmse"]:7.3f}  MAPE={s["mape"]:5.3f}%')

pred_val  = model.predict(X_val,  num_iteration=best_iter)
pred_test = model.predict(X_test, num_iteration=best_iter)

print('\n=== 6. 평가 ===')
print('--- Val (실측 기상) ---')
pr('KPX baseline',     score(y_val, full_df.loc[mask_val,  'est_demand'].values))
pr('PatchTST only',    score(y_val, full_df.loc[mask_val,  'patchtst_target'].values))
pr('LGBM + PatchTST',  score(y_val, pred_val))
print('--- Test (예보 기상) ---')
pr('KPX baseline',     score(y_test, full_df.loc[mask_test, 'est_demand'].values))
pr('PatchTST only',    score(y_test, full_df.loc[mask_test, 'patchtst_target'].values))
pr('LGBM + PatchTST',  score(y_test, pred_test))


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
# 8. 운영 산출물 저장 — demand_predict.py 가 직접 읽는 두 파일
# =============================================================================
os.makedirs(MODELS_DIR, exist_ok=True)

# (a) booster pickle — lgbm_pipeline.pkl
# best_iteration 까지만 사용하도록 트리 잘라낸 booster 를 저장
trimmed = model.model_to_string(num_iteration=best_iter)
booster_to_save = lgb.Booster(model_str=trimmed)
with open(OUT_PKL, 'wb') as f:
    pickle.dump(booster_to_save, f)

# (b) pipeline_config.json
categorical_categories = {
    c: list(full_df[c].cat.categories) for c in CATEGORICAL_COLS
}
cfg = {
    'feature_cols':            FEATURE_COLS,
    'categorical_cols':        CATEGORICAL_COLS,
    'categorical_categories':  categorical_categories,
    'target':                  TARGET,
    'best_iteration':          best_iter,
    'val_best_mae':            float(model.best_score['val']['l1']),
    'lgbm_version':            lgb.__version__,
    'trained_at':              datetime.now().isoformat(timespec='seconds'),
    'train_range':             ['2020-01-01', '2025-02-28'],
    'val_range':               ['2025-03-01', '2026-03-21'],
    'signal_source':           'patchtst',   # timesfm 아닌 patchtst 신호임을 명시
    'params':                  params,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f'\n=== 8. 저장 ===')
print(f'  {OUT_PKL}  ({os.path.getsize(OUT_PKL) // 1024} KB)')
print(f'  {OUT_JSON} ({os.path.getsize(OUT_JSON)} B)')


# =============================================================================
# 9. CONFIG 변경점 안내
# =============================================================================
print('\n' + '=' * 78)
print('▶ demand_predict.py 는 위 두 파일을 자동으로 읽으니 코드 수정 불필요.')
print('  단, --timesfm 인자명을 그대로 두면 의미가 헷갈리므로 step 6 에서 정리 권장.')
print('=' * 78)
print('완료.')
