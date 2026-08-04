"""3cmp-C — forecast 기상(실서빙 D+1) PatchTST vs LGBM 비교.

3cmp-B(실측기상)는 '모델 자체' 편향을 봤다. 여기서는 사용자가 실제 관찰한
'흐린날 과대예측'의 무대 = forecast 예보 기상으로 D+1을 비교한다.
- 두 모델 모두 forecast 기상 입력(공정). regime(맑음/흐림)은 ★실측 구름으로 분류.
- forecast 기상 가용구간(2025-12-13~2026-06-01)만, lead=D+1(아카이브 없음).
산출: tab/3cmp-C_forecast_d1.csv, fig/3cmp-C_forecast_regime.png
"""
import os, sys, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CMP  = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(CMP, '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import solarwind_db_pipeline as sw
import importlib.util
_spec = importlib.util.spec_from_file_location('m3a', os.path.join(HERE, '3cmp-A_lgbm_solarwind.py'))
m3a = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m3a)
import lightgbm as lgb
for _f in ['Malgun Gothic', 'Gulim']:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f; break
plt.rcParams['axes.unicode_minus'] = False

DB  = os.path.normpath(os.path.join(ROOT, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
CSV = os.path.join(ROOT, 'training', 'solarwind_raw_jeju.csv')
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'

# clearsky 평년(train) 재산출
raw = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
_, clim = m3a.build_features(raw[raw.index.year <= 2024])
m_solar = lgb.Booster(model_file=os.path.join(HERE, 'lgbm_solar_util.txt'))
m_wind  = lgb.Booster(model_file=os.path.join(HERE, 'lgbm_wind_util.txt'))

# forecast 기상 → 캐노니컬 컬럼(LGBM build_features 입력 형태)
con = sqlite3.connect(DB)
fore_cols = {'radiation_west': 'solar_rad_west', 'radiation_south': 'solar_rad_south',
             'total_cloud_west': 'total_cloud_west', 'total_cloud_south': 'total_cloud_south',
             'midlow_cloud_west': 'midlow_cloud_west', 'midlow_cloud_south': 'midlow_cloud_south',
             'rainfall_west': 'rainfall_west', 'rainfall_south': 'rainfall_south',
             'wind_spd_10m_west': 'wind_spd_west', 'wind_spd_10m_east': 'wind_spd_east',
             'wind_spd_10m_south': 'wind_spd_south',
             'wd_sin_10m_west': 'wd_sin_west', 'wd_cos_10m_west': 'wd_cos_west',
             'wd_sin_10m_east': 'wd_sin_east', 'wd_cos_10m_east': 'wd_cos_east',
             'wd_sin_10m_south': 'wd_sin_south', 'wd_cos_10m_south': 'wd_cos_south'}
sel = ', '.join(f'"{c}"' for c in ['timestamp'] + list(fore_cols))
fdf = pd.read_sql(f'SELECT {sel} FROM forecast ORDER BY timestamp', con,
                  parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
fdf = fdf.rename(columns=fore_cols)
# 실측(타깃 util + 실측구름 regime + capacity + gen + demand)
act = pd.read_sql("SELECT timestamp, real_solar_utilization_jeju, real_wind_utilization_jeju, "
                  "real_solar_capacity_jeju, real_wind_capacity_jeju, real_solar_gen_jeju, "
                  "real_wind_gen_jeju, real_demand_jeju, total_cloud_west AS act_cloud "
                  "FROM historical", con, parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
con.close()

feat, _ = m3a.build_features(fdf, clim=clim)
feat['lg_su'] = np.clip(m_solar.predict(feat[m3a.SOLAR_FINAL]), 0, 1)
feat['lg_wu'] = np.clip(m_wind.predict(feat[m3a.WIND_FINAL]), 0, 1)

# PatchTST D+1: 파이프라인으로 forecast 보유 날짜별 추론
fdates = pd.read_sql_table if False else None
con = sqlite3.connect(DB)
days = pd.read_sql("SELECT substr(timestamp,1,10) d, COUNT(*) n FROM forecast "
                   "WHERE radiation_west IS NOT NULL GROUP BY d HAVING n=24 ORDER BY d", con)['d'].tolist()
con.close()
pt_rows = []
for d in days:
    try:
        out = sw.predict_solarwind_to_db(d, write=False, verbose=False)
        out['timestamp'] = pd.to_datetime(out['timestamp'])
        pt_rows.append(out[['timestamp', 'est_solar_utilization_jeju', 'est_wind_utilization_jeju']])
    except Exception:
        continue
PT = pd.concat(pt_rows).set_index('timestamp')

# 결합
R = feat[['lg_su', 'lg_wu']].join(PT.rename(columns={'est_solar_utilization_jeju': 'pt_su',
                                                     'est_wind_utilization_jeju': 'pt_wu'}), how='inner')
R = R.join(act, how='inner').dropna(subset=[SU, WU, 'act_cloud'])
R['hour'] = R.index.hour
print('forecast D+1 비교 표본', len(R), '시간 /', R.index.normalize().nunique(), '일')

# regime: 실측 낮구름 일평균
day = R[(R.hour >= 8) & (R.hour <= 17)].copy()
dc = day.groupby(day.index.date)['act_cloud'].transform('mean')
day['regime'] = np.where(dc >= 0.7, 'cloudy', np.where(dc <= 0.3, 'sunny', 'mixed'))

rows = []
for r in ['sunny', 'mixed', 'cloudy', 'ALL']:
    s = day if r == 'ALL' else day[day.regime == r]
    row = {'regime': r, 'n': len(s)}
    for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]:
        e = s[f'{tag}_su'] - s[SU]
        row[f'{name}_bias'] = round(e.mean(), 4); row[f'{name}_MAE'] = round(e.abs().mean(), 4)
    rows.append(row)
solar = pd.DataFrame(rows)
print('\n[★ forecast D+1 SOLAR 이용률 regime별 (낮 8-17h)]'); print(solar.to_string(index=False))

ew_pt = R['pt_wu'] - R[WU]; ew_lg = R['lg_wu'] - R[WU]
wind = pd.DataFrame([{'model': 'PatchTST', 'bias': round(ew_pt.mean(), 4), 'MAE': round(ew_pt.abs().mean(), 4)},
                     {'model': 'LGBM', 'bias': round(ew_lg.mean(), 4), 'MAE': round(ew_lg.abs().mean(), 4)}])
print('\n[forecast D+1 WIND 이용률 (전시간)]'); print(wind.to_string(index=False))

# net_load (demand 고정)
for tag in ['pt', 'lg']:
    R[f'{tag}_nl'] = R['real_demand_jeju'] - R[f'{tag}_su']*R['real_solar_capacity_jeju'] - R[f'{tag}_wu']*R['real_wind_capacity_jeju']
R['act_nl'] = R['real_demand_jeju'] - R['real_solar_gen_jeju'] - R['real_wind_gen_jeju']
NLm = R['act_nl'].mean()
nl = pd.DataFrame([{'model': name,
                    'MAE': round(np.abs(R[f'{tag}_nl']-R['act_nl']).mean(), 1),
                    'nMAE%': round(np.abs(R[f'{tag}_nl']-R['act_nl']).mean()/NLm*100, 2)}
                   for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]])
print('\n[forecast D+1 net_load]'); print(nl.to_string(index=False))

solar.to_csv(os.path.join(CMP, 'tab', '3cmp-C_forecast_d1_solar.csv'), index=False)
nl.to_csv(os.path.join(CMP, 'tab', '3cmp-C_forecast_d1_netload.csv'), index=False)

# 그림: regime별 bias 막대 (perfect vs forecast 대비용)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
x = np.arange(3); w = 0.35; sub = solar[solar.regime != 'ALL']
ax[0].bar(x - w/2, sub['PatchTST_bias'], w, label='PatchTST', color='crimson')
ax[0].bar(x + w/2, sub['LGBM_bias'], w, label='LGBM', color='navy')
ax[0].axhline(0, color='k', lw=.7); ax[0].set_xticks(x); ax[0].set_xticklabels(sub['regime'])
ax[0].set_title('forecast D+1 solar bias by regime (낮)'); ax[0].set_ylabel('bias(util)'); ax[0].legend()
ax[1].bar(x - w/2, sub['PatchTST_MAE'], w, label='PatchTST', color='crimson')
ax[1].bar(x + w/2, sub['LGBM_MAE'], w, label='LGBM', color='navy')
ax[1].set_xticks(x); ax[1].set_xticklabels(sub['regime']); ax[1].set_title('forecast D+1 solar MAE by regime')
ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(CMP, 'fig', '3cmp-C_forecast_regime.png')); plt.close(fig)
print('\n완료 → tab/3cmp-C_*.csv, fig/3cmp-C_forecast_regime.png')
