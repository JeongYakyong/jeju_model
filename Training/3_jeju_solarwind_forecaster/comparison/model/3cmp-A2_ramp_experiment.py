"""3cmp-A2 — 기상 급변 피처(ramp/vol) 탐색 실험 (사용자 제안 2026-06-08).

질문: 일사·풍속의 '급변'(시간차분 ramp, 변동성 vol) 피처가 이용률 예측을 개선하나?
특히 흐린날(broken cloud) 과대예측·풍력 gust 구간.
base(현 3cmp-A 피처) vs base+ramp/vol 을 test 2026 util MAE·흐린날 bias로 비교.
※ 탐색 단계(§0.6) — 최종 채택은 사용자 확정 후.
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
sp = importlib.util.spec_from_file_location('m3a', os.path.join(HERE, '3cmp-A_lgbm_solarwind.py'))
m3a = importlib.util.module_from_spec(sp); sp.loader.exec_module(m3a)

CSV = os.path.normpath(os.path.join(HERE, '..', '..', 'training', 'solarwind_raw_jeju.csv'))
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'

raw = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
raw['year'] = raw.index.year
_, clim = m3a.build_features(raw[raw.year <= 2024])
feat, _ = m3a.build_features(raw, clim=clim)

# ── 급변 피처 후보 (각 날짜 윈도우 내 지역 계산 → horizon-agnostic 유지) ──
def add_ramp(df):
    for st in ['west', 'south']:
        df[f'solar_rad_ramp_{st}'] = df[f'solar_rad_{st}'].diff().fillna(0)          # 시간차분(부호)
        df[f'solar_rad_vol_{st}']  = df[f'solar_rad_{st}'].rolling(3, center=True, min_periods=1).std().fillna(0)  # 변동성
    for st in ['west', 'east']:
        df[f'wind_spd_ramp_{st}'] = df[f'wind_spd_{st}'].diff().fillna(0)
        df[f'wind_spd_vol_{st}']  = df[f'wind_spd_{st}'].rolling(3, center=True, min_periods=1).std().fillna(0)
    return df
feat = add_ramp(feat)
feat['split'] = np.where(feat.year <= 2024, 'train', np.where(feat.year == 2025, 'val', 'test'))

SOLAR_RAMP = ['solar_rad_ramp_west', 'solar_rad_vol_west', 'solar_rad_ramp_south', 'solar_rad_vol_south']
WIND_RAMP  = ['wind_spd_ramp_west', 'wind_spd_vol_west', 'wind_spd_ramp_east', 'wind_spd_vol_east']

params = dict(objective='regression_l1', n_estimators=1200, learning_rate=0.03, num_leaves=63,
              min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

def run(feats, target):
    tr = feat[feat.split == 'train']; va = feat[feat.split == 'val']; te = feat[feat.split == 'test'].copy()
    m = lgb.LGBMRegressor(**params)
    m.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], callbacks=[lgb.early_stopping(60, verbose=False)])
    te['p'] = np.clip(m.predict(te[feats]), 0, 1)
    return te, m

def solar_report(tag, te):
    day = te[(te.index.hour >= 8) & (te.index.hour <= 17)].copy()
    dc = day.groupby(day.index.date)['total_cloud_west'].transform('mean')
    day['reg'] = np.where(dc >= 0.7, 'cloudy', np.where(dc <= 0.3, 'sunny', 'mixed'))
    e = day['p'] - day[SU]
    out = {'set': tag, 'ALL_MAE': round(e.abs().mean(), 4), 'ALL_bias': round(e.mean(), 4)}
    for r in ['sunny', 'cloudy']:
        s = day[day.reg == r]; es = s['p'] - s[SU]
        out[f'{r}_MAE'] = round(es.abs().mean(), 4); out[f'{r}_bias'] = round(es.mean(), 4)
    return out

print('=' * 70); print('SOLAR  base vs base+ramp/vol (test 2026, 낮 8-17h)')
te_b, mb = run(m3a.SOLAR_FINAL, SU)
te_r, mr = run(m3a.SOLAR_FINAL + SOLAR_RAMP, SU)
sr = pd.DataFrame([solar_report('base', te_b), solar_report('base+ramp', te_r)])
print(sr.to_string(index=False))
imp = pd.Series(mr.booster_.feature_importance('gain'), index=m3a.SOLAR_FINAL + SOLAR_RAMP)
print('\n  +ramp 모델 내 ramp 피처 gain%:')
print((imp / imp.sum() * 100).reindex(SOLAR_RAMP).round(2).to_string())

print('\n' + '=' * 70); print('WIND  base vs base+ramp/vol (test 2026, 전시간)')
def wind_report(tag, te):
    e = te['p'] - te[WU]
    return {'set': tag, 'MAE': round(e.abs().mean(), 4), 'bias': round(e.mean(), 4)}
te_wb, _ = run(m3a.WIND_FINAL, WU)
te_wr, mwr = run(m3a.WIND_FINAL + WIND_RAMP, WU)
wr = pd.DataFrame([wind_report('base', te_wb), wind_report('base+ramp', te_wr)])
print(wr.to_string(index=False))
impw = pd.Series(mwr.booster_.feature_importance('gain'), index=m3a.WIND_FINAL + WIND_RAMP)
print('\n  +ramp 모델 내 ramp 피처 gain%:')
print((impw / impw.sum() * 100).reindex(WIND_RAMP).round(2).to_string())
