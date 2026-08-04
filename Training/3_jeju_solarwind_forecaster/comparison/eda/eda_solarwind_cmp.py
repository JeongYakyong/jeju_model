"""3단계 비교용 G-9 EDA — solar/wind 이용률 시계열·기상관계·흐린날·분포안정성.

목적: PatchTST vs LGBM 비교(흐린날 과대예측 중점) 착수 전, 모델링 선행 게이트(G-9).
- 타깃(이용률)의 시계열 구조: 일/계절 주기, 추세, 안정성
- 입력(기상) ↔ 타깃 관계: solar_rad/cloud/damping/wind_spd
- 흐린날 정의와 흐린날에서의 이용률 분포(과대예측 표적 구간)
- 용량 표류 → 이용률 정규화 타당성 + 연도별 분포 안정(covariate shift)
- train/test 분포 겹침

산출: comparison/fig/3cmp-0_*.png, comparison/tab/3cmp-0_*.csv, 콘솔 요약.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CMP  = os.path.normpath(os.path.join(HERE, '..'))
CSV  = os.path.normpath(os.path.join(CMP, '..', 'training', 'solarwind_raw_jeju.csv'))
FIG  = os.path.join(CMP, 'fig')
TAB  = os.path.join(CMP, 'tab')
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
for _f in ['Malgun Gothic', 'Gulim', 'NanumGothic']:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f; break
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.apply(pd.to_numeric, errors='coerce') if False else df
df['year']  = df.index.year
df['month'] = df.index.month
df['hour']  = df.index.hour
df['doy']   = df.index.dayofyear
SU = 'real_solar_utilization_jeju'
WU = 'real_wind_utilization_jeju'

# 학습창 정의 (제주 솔라윈드 학습과 동일 골격: ~2024 train / 2025 val / 2026 test)
df['split'] = np.where(df['year'] <= 2024, 'train',
              np.where(df['year'] == 2025, 'val', 'test'))

print('=' * 70)
print('범위', df.index.min(), '~', df.index.max(), '| rows', len(df))
print(df['split'].value_counts())

# ---------------------------------------------------------------- 1. 시계열 구조
# 일중 곡선(시간별 평균 이용률) + 월별
hourly = df.groupby('hour')[[SU, WU]].mean()
monthly = df.groupby('month')[[SU, WU]].mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
hourly.plot(ax=ax[0], marker='o'); ax[0].set_title('시간별 평균 이용률'); ax[0].set_xlabel('hour')
monthly.plot(ax=ax[1], marker='o'); ax[1].set_title('월별 평균 이용률'); ax[1].set_xlabel('month')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3cmp-0_diurnal_seasonal.png')); plt.close(fig)

# ---------------------------------------------------------------- 2. 용량 표류 vs 이용률 안정
cap = df.groupby('year')[['real_solar_capacity_jeju', 'real_wind_capacity_jeju']].median()
# 낮시간(태양광 발전 구간)만 이용률 평균 — 연도별
day = df[(df.hour >= 9) & (df.hour <= 16)]
util_year = pd.DataFrame({
    'solar_util_day(9-16h)': day.groupby('year')[SU].mean(),
    'wind_util_allhour': df.groupby('year')[WU].mean(),
})
drift = cap.join(util_year)
drift.to_csv(os.path.join(TAB, '3cmp-0_capacity_util_drift.csv'))
print('\n[용량·이용률 연도별]'); print(drift.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cap.plot(ax=ax[0], marker='o'); ax[0].set_title('연도별 설비용량(MW) — 표류 큼')
util_year.plot(ax=ax[1], marker='o'); ax[1].set_title('연도별 평균 이용률 — 정규화 후 안정성')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3cmp-0_capacity_drift.png')); plt.close(fig)

# ---------------------------------------------------------------- 3. 기상 ↔ 이용률 관계
# solar: 낮시간만, 일사/구름/damping
sol = df[(df.hour >= 8) & (df.hour <= 17)].copy()
sol_feats = ['solar_rad_west', 'solar_rad_south', 'total_cloud_west', 'total_cloud_south',
             'midlow_cloud_west', 'midlow_cloud_south', 'rainfall_west', 'rainfall_south']
corr_s = sol[sol_feats + [SU]].corr()[SU].drop(SU).sort_values()
wind_feats = ['wind_spd_west', 'wind_spd_east', 'wind_spd_south']
corr_w = df[wind_feats + [WU]].corr()[WU].drop(WU).sort_values()
print('\n[solar 낮시간 피처 ↔ 이용률 상관]'); print(corr_s.round(3))
print('\n[wind 피처 ↔ 이용률 상관]'); print(corr_w.round(3))
pd.concat([corr_s.rename('corr_solar_util'), corr_w.rename('corr_wind_util')], axis=1)\
  .to_csv(os.path.join(TAB, '3cmp-0_weather_corr.csv'))

# solar_rad vs util 산점 + cloud별 색
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sc = ax[0].scatter(sol['solar_rad_west'], sol[SU], c=sol['total_cloud_west'],
                   s=4, cmap='viridis_r', alpha=0.4)
ax[0].set_xlabel('solar_rad_west'); ax[0].set_ylabel('solar_util'); ax[0].set_title('일사 vs 이용률 (색=총운량)')
plt.colorbar(sc, ax=ax[0], label='total_cloud')
ax[1].scatter(df['wind_spd_west'], df[WU], s=3, alpha=0.2)
ax[1].set_xlabel('wind_spd_west'); ax[1].set_ylabel('wind_util'); ax[1].set_title('풍속 vs 이용률')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3cmp-0_weather_scatter.png')); plt.close(fig)

# ---------------------------------------------------------------- 4. 흐린날 정의 + 흐린날 이용률
# clear-sky 대용: (월,시) 일사 상위 분위 = 맑은 기준. solar_deficit = 1 - rad/clearsky
clim = sol.groupby([sol.month, sol.hour])['solar_rad_west'].transform(lambda x: x.quantile(0.90))
sol['clearsky_rad'] = clim
sol['rad_ratio'] = (sol['solar_rad_west'] / sol['clearsky_rad'].replace(0, np.nan)).clip(0, 1.5)
# 일(day) 단위 맑음/흐림: 낮시간 평균 총운량
day_cloud = sol.groupby(sol.index.date)['total_cloud_west'].mean()
sunny_days = set(day_cloud[day_cloud <= 0.3].index)
cloudy_days = set(day_cloud[day_cloud >= 0.7].index)
sol['day'] = sol.index.date
sol['regime'] = np.where(sol['day'].isin(cloudy_days), 'cloudy',
                np.where(sol['day'].isin(sunny_days), 'sunny', 'mixed'))
reg = sol.groupby('regime')[SU].agg(['mean', 'std', 'count'])
reg.to_csv(os.path.join(TAB, '3cmp-0_regime_util.csv'))
print('\n[낮시간 맑음/흐림별 solar 이용률]'); print(reg.round(3))
print(f"  맑은날 {len(sunny_days)}일 / 흐린날 {len(cloudy_days)}일 / 전체일 {len(day_cloud)}")

# 맑음/흐림 일중 곡선
fig, ax = plt.subplots(figsize=(7, 4))
for r, c in [('sunny', 'orange'), ('mixed', 'gray'), ('cloudy', 'navy')]:
    sol[sol.regime == r].groupby('hour')[SU].mean().plot(ax=ax, marker='o', label=r, color=c)
ax.legend(); ax.set_title('맑음/흐림별 시간별 solar 이용률'); ax.set_ylabel('util')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3cmp-0_regime_diurnal.png')); plt.close(fig)

# ---------------------------------------------------------------- 5. train/test 분포 겹침
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for sp, c in [('train', 'C0'), ('val', 'C1'), ('test', 'C2')]:
    d = sol[sol.split == sp]
    ax[0].hist(d[SU], bins=40, density=True, alpha=0.4, label=sp, color=c)
    ax[1].hist(d['solar_rad_west'], bins=40, density=True, alpha=0.4, label=sp, color=c)
    dw = df[df.split == sp]
    ax[2].hist(dw[WU], bins=40, density=True, alpha=0.4, label=sp, color=c)
ax[0].set_title('solar_util 분포(낮)'); ax[1].set_title('solar_rad 분포(낮)'); ax[2].set_title('wind_util 분포')
for a in ax: a.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3cmp-0_split_overlap.png')); plt.close(fig)

# 분포 요약표
ov = []
for name, frame, col in [('solar_util', sol, SU), ('solar_rad', sol, 'solar_rad_west'),
                          ('wind_util', df, WU), ('wind_spd', df, 'wind_spd_west')]:
    row = {'var': name}
    for sp in ['train', 'val', 'test']:
        s = frame[frame.split == sp][col]
        row[f'{sp}_mean'] = round(s.mean(), 3); row[f'{sp}_p90'] = round(s.quantile(.9), 3)
    ov.append(row)
ov = pd.DataFrame(ov); ov.to_csv(os.path.join(TAB, '3cmp-0_split_summary.csv'), index=False)
print('\n[split별 분포 요약]'); print(ov.to_string(index=False))

# 자기상관(이용률) — 장지평에서 lag 유효성 점검
def acf(x, lags):
    x = x - x.mean(); return [1.0] + [np.corrcoef(x[:-l], x[l:])[0, 1] for l in lags]
lags = [1, 24, 48, 72, 168]
su_acf = acf(df[SU].fillna(0).values, lags)
wu_acf = acf(df[WU].fillna(0).values, lags)
print('\n[이용률 자기상관] lag', [0] + lags)
print('  solar', [round(v, 3) for v in su_acf])
print('  wind ', [round(v, 3) for v in wu_acf])

print('\nEDA 완료 → fig/3cmp-0_*.png, tab/3cmp-0_*.csv')
