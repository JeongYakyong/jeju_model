"""3cmp-2 — forecast 전용 변수(cape/hpbl/gust/cinn/tcog) 상관분석 (후처리 후보 탐색).

배경(사용자 2026-06-08): 이 변수들은 historical에 없어 모델 입력 불가. 단,
solar/wind/rain 및 우리 모델 '잔차'와 상관이 깊으면 후처리 보정에 쓸 수 있다.
forecast 가용구간(2025-12-13~2026-06)만, lead=D+1.

분석:
  A. 특이변수(원시+파생) ↔ 실측 solar_util(낮)/wind_util/rainfall  (Pearson·Spearman)
  B. 특이변수 ↔ LGBM 잔차(실측−예측)  ← 후처리 실효성의 핵심
  파생: gustiness=gust−wind_spd, gust/wind, cape·tcog(대류운 proxy) 등
산출: tab/3cmp-2_corr_*.csv, fig/3cmp-2_*.png
"""
import os, sys, sqlite3
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); CMP = os.path.normpath(os.path.join(HERE, '..'))
MODEL = os.path.join(CMP, 'model'); sys.path.insert(0, MODEL)
import importlib.util
sp = importlib.util.spec_from_file_location('m3a', os.path.join(MODEL, '3cmp-A_lgbm_solarwind.py'))
m3a = importlib.util.module_from_spec(sp); sp.loader.exec_module(m3a)
for _f in ['Malgun Gothic', 'Gulim']:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f; break
plt.rcParams['axes.unicode_minus'] = False
DB = os.path.normpath(os.path.join(CMP, '..', '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
CSV = os.path.normpath(os.path.join(CMP, '..', 'training', 'solarwind_raw_jeju.csv'))
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'
SPECIAL = ['cape', 'hpbl', 'gust', 'cinn', 'tcog', 'tcoh']

# clim + LGBM 예측(forecast 기상)
raw = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
_, clim = m3a.build_features(raw[raw.index.year <= 2024])
m_solar = lgb.Booster(model_file=os.path.join(MODEL, 'lgbm_solar_util.txt'))
m_wind  = lgb.Booster(model_file=os.path.join(MODEL, 'lgbm_wind_util.txt'))

con = sqlite3.connect(DB)
fcols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
# forecast 표준기상(예측용) → 캐노니컬
fmap = {'radiation_west': 'solar_rad_west', 'radiation_south': 'solar_rad_south',
        'total_cloud_west': 'total_cloud_west', 'total_cloud_south': 'total_cloud_south',
        'midlow_cloud_west': 'midlow_cloud_west', 'midlow_cloud_south': 'midlow_cloud_south',
        'rainfall_west': 'rainfall_west', 'rainfall_south': 'rainfall_south',
        'wind_spd_10m_west': 'wind_spd_west', 'wind_spd_10m_east': 'wind_spd_east',
        'wd_sin_10m_west': 'wd_sin_west', 'wd_cos_10m_west': 'wd_cos_west'}
spcols = [f'{v}_{st}' for v in SPECIAL for st in ['west', 'east', 'south'] if f'{v}_{st}' in fcols]
allcols = list(fmap) + spcols
sel = ', '.join(f'"{c}"' for c in ['timestamp'] + allcols)
F = pd.read_sql(f'SELECT {sel} FROM forecast ORDER BY timestamp', con, parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
print('forecast special 커버리지(원시):')
for v in SPECIAL:
    c = f'{v}_west'
    if c in F: print(f'  {c}: n={F[c].notna().sum()}  median={F[c].median():.3f}  >=9000비중 {(F[c]>=9000).mean()*100:.1f}%')

# ── sentinel(9999) 처리 ──  cape/cinn은 9999=대류불안정 없음(안정). tcoh=상수0 → 제거.
SENTINEL = 9000
for v in ['cape', 'cinn']:
    for st in ['west', 'east', 'south']:
        c = f'{v}_{st}'
        if c in F:
            F[f'{c}_present'] = (F[c] < SENTINEL).astype(float)   # 1=대류 존재(값 유효)
            F.loc[F[c] >= SENTINEL, c] = np.nan                   # 9999 → NaN(실값만 분석)
SPECIAL = ['cape', 'hpbl', 'gust', 'cinn', 'tcog']   # tcoh(상수0) 제외
print('  → cape/cinn 9999 마스킹 후 실값비율: cape_west %.1f%%, cinn_west %.1f%%' % (
    F['cape_west'].notna().mean()*100, F['cinn_west'].notna().mean()*100))
act = pd.read_sql(f"SELECT timestamp, {SU} su, {WU} wu, real_solar_gen_jeju sg, real_wind_gen_jeju wg, "
                  "rainfall_west rw, rainfall_south rs FROM historical", con,
                  parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
con.close()

# LGBM 예측(forecast 표준기상)
feat, _ = m3a.build_features(F.rename(columns=fmap), clim=clim)
feat['pred_su'] = np.clip(m_solar.predict(feat[m3a.SOLAR_FINAL]), 0, 1)
feat['pred_wu'] = np.clip(m_wind.predict(feat[m3a.WIND_FINAL]), 0, 1)

R = feat.join(act, how='inner')
# 특이변수 공간평균(solar=west+south, wind=west+east) + west 단독
for v in SPECIAL:
    cols_ws = [f'{v}_{s}' for s in ['west', 'south'] if f'{v}_{s}' in R]
    cols_we = [f'{v}_{s}' for s in ['west', 'east'] if f'{v}_{s}' in R]
    if cols_ws: R[f'{v}_solarmean'] = R[cols_ws].mean(axis=1)
    if cols_we: R[f'{v}_windmean'] = R[cols_we].mean(axis=1)
# 파생: gustiness
for s in ['west', 'east']:
    if f'gust_{s}' in R: R[f'gustiness_{s}'] = R[f'gust_{s}'] - R[f'wind_spd_{s}']
R['gustiness_windmean'] = R[[c for c in ['gustiness_west', 'gustiness_east'] if c in R]].mean(axis=1)
R['rain'] = R[['rw', 'rs']].mean(axis=1)
R['resid_su'] = R['su'] - R['pred_su']   # +면 우리가 과소, −면 과대
R['resid_wu'] = R['wu'] - R['pred_wu']
R['hour'] = R.index.hour
day = R[(R.hour >= 8) & (R.hour <= 17)].copy()

def corr2(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 30: return (np.nan, np.nan, int(m.sum()))
    pe = np.corrcoef(x[m], y[m])[0, 1]; sp_ = spearmanr(x[m], y[m]).correlation
    return (round(pe, 3), round(sp_, 3), int(m.sum()))

# ── A. 특이변수 ↔ 실측 ──
rowsA = []
solar_vars = [f'{v}_solarmean' for v in SPECIAL] + ['cape_west', 'tcog_west', 'tcoh_west', 'cinn_west', 'hpbl_west']
for var in solar_vars:
    if var not in day: continue
    pe, sp_, n = corr2(day[var], day['su'])
    per, sper, _ = corr2(day[var], day['resid_su'])
    rowsA.append(dict(var=var, vs='solar_util(낮)', pearson=pe, spearman=sp_,
                      resid_pear=per, resid_spear=sper, n=n))
wind_vars = [f'{v}_windmean' for v in SPECIAL] + ['gust_west', 'gust_east', 'gustiness_west', 'gustiness_windmean', 'hpbl_west']
for var in wind_vars:
    if var not in R: continue
    pe, sp_, n = corr2(R[var], R['wu'])
    per, sper, _ = corr2(R[var], R['resid_wu'])
    rowsA.append(dict(var=var, vs='wind_util', pearson=pe, spearman=sp_,
                      resid_pear=per, resid_spear=sper, n=n))
# rain
for var in [f'{v}_solarmean' for v in SPECIAL]:
    if var not in R: continue
    pe, sp_, n = corr2(R[var], R['rain'])
    rowsA.append(dict(var=var, vs='rainfall', pearson=pe, spearman=sp_, resid_pear=np.nan, resid_spear=np.nan, n=n))
A = pd.DataFrame(rowsA)
A.to_csv(os.path.join(CMP, 'tab', '3cmp-2_corr_special.csv'), index=False)
pd.set_option('display.width', 200)
print('\n[특이변수 ↔ 실측 util/rain + 우리모델 잔차]  (resid: +면 과소예측, −면 과대예측)')
print(A.to_string(index=False))

# ── 그룹별 평균 잔차 (후처리 규칙 해석용: 플래그/임계 켜졌을 때 우리가 얼마나 틀리나) ──
print('\n[그룹별 평균 잔차 — 후처리 규칙 후보]  (resid 음수=우리가 과대예측 → 깎을 여지)')
grp = []
def grprow(label, mask, resid_col, frame):
    s = frame[mask]; b = frame[~mask]
    if len(s) < 30 or len(b) < 30: return
    grp.append(dict(rule=label, n_on=len(s), resid_on=round(frame.loc[mask, resid_col].mean(), 4),
                    resid_off=round(frame.loc[~mask, resid_col].mean(), 4),
                    diff=round(frame.loc[mask, resid_col].mean() - frame.loc[~mask, resid_col].mean(), 4)))
grprow('SOLAR: cape_present(대류있음)', day['cape_west_present'] == 1, 'resid_su', day)
grprow('SOLAR: tcog_west>0(대류운)',    day['tcog_west'] > 0,           'resid_su', day)
grprow('SOLAR: gust_solarmean>10(강풍)', day['gust_solarmean'] > 10,    'resid_su', day)
grprow('WIND: tcog_windmean>0(대류운)',  R['tcog_windmean'] > 0,         'resid_wu', R)
grprow('WIND: cape_west_present(대류)',  R['cape_west_present'] == 1,    'resid_wu', R)
grprow('WIND: hpbl_west>800(깊은혼합)',  R['hpbl_west'] > 800,           'resid_wu', R)
G = pd.DataFrame(grp); G.to_csv(os.path.join(CMP, 'tab', '3cmp-2_group_resid.csv'), index=False)
print(G.to_string(index=False))

# ── 그림: 핵심 4개 산점 ──
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
def sc(a, x, y, xl, yl, t):
    m = R[x].notna() & R[y].notna()
    a.scatter(R.loc[m, x], R.loc[m, y], s=4, alpha=0.2); a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(t); a.grid(True)
sc(ax[0,0], 'cape_solarmean', 'resid_su' if 'resid_su' in R else 'su', 'cape(solar mean)', 'solar resid(실측−예측)', 'cape vs solar 잔차')
# daytime만
md = day['cape_solarmean'].notna()
ax[0,0].clear(); ax[0,0].scatter(day.loc[md,'cape_solarmean'], day.loc[md,'resid_su'], s=4, alpha=0.2)
ax[0,0].axhline(0,color='r',lw=.7); ax[0,0].set_title('cape vs solar 잔차(낮)'); ax[0,0].set_xlabel('cape'); ax[0,0].set_ylabel('resid_su'); ax[0,0].grid(True)
mt = day['tcog_west'].notna()
ax[0,1].scatter(day.loc[mt,'tcog_west'], day.loc[mt,'resid_su'], s=4, alpha=0.2)
ax[0,1].axhline(0,color='r',lw=.7); ax[0,1].set_title('tcog vs solar 잔차(낮)'); ax[0,1].set_xlabel('tcog_west'); ax[0,1].set_ylabel('resid_su'); ax[0,1].grid(True)
mg = R['gustiness_windmean'].notna()
ax[1,0].scatter(R.loc[mg,'gustiness_windmean'], R.loc[mg,'wu'], s=4, alpha=0.2)
ax[1,0].set_title('gustiness(gust−wind) vs wind_util'); ax[1,0].set_xlabel('gustiness'); ax[1,0].set_ylabel('wind_util'); ax[1,0].grid(True)
ax[1,1].scatter(R.loc[mg,'gustiness_windmean'], R.loc[mg,'resid_wu'], s=4, alpha=0.2)
ax[1,1].axhline(0,color='r',lw=.7); ax[1,1].set_title('gustiness vs wind 잔차'); ax[1,1].set_xlabel('gustiness'); ax[1,1].set_ylabel('resid_wu'); ax[1,1].grid(True)
fig.tight_layout(); fig.savefig(os.path.join(CMP, 'fig', '3cmp-2_special_scatter.png')); plt.close(fig)
print('\n저장: tab/3cmp-2_corr_special.csv, fig/3cmp-2_special_scatter.png')
