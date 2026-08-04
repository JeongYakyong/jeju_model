"""3cmp-G — direct 지평별 PatchTST(신규 학습) vs LGBM vs recursive PatchTST 비교.

신규 가중치: solarwind_patchTST_pkl/best_patchtst_{solar,wind}_model_D{2..6}.pth
direct 모델은 학습 offset((n-1)*24h)이 origin↔target 간격을 정확히 메운다 →
발행일 origin(=D0 23:00)의 과거는 전 지평 동일, 모델·대상일 기상만 교체(재귀 아님, 누수 없음).

3cmp-B 산출(tab/3cmp-B_raw_predictions.parquet: recursive PatchTST + LGBM + 실측 + cap/demand)을
불러와 같은 (D0,ts)에 direct 예측을 추가 → 지평별 direct/recursive/LGBM 3자 비교.
실측기상(perfect) 기준. 목적: PatchTST/LGBM 경계를 지평별로 실측 결정.
산출: tab/3cmp-G_*.csv, fig/3cmp-G_*.png
"""
import os, sys
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
CMP = os.path.normpath(os.path.join(HERE, '..')); ROOT = os.path.normpath(os.path.join(CMP, '..'))
sys.path.insert(0, ROOT)
import solarwind_db_pipeline as sw
for _f in ['Malgun Gothic', 'Gulim']:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f; break
plt.rcParams['axes.unicode_minus'] = False
PKL = os.path.join(ROOT, 'solarwind_patchTST_pkl')
CSV = os.path.join(ROOT, 'training', 'solarwind_raw_jeju.csv')
HORIZONS = [1, 2, 3, 4, 5, 6]

# ── 자산: D+1(기존) + direct D2~D6 ──
solar1, wind1, sc_solar, sc_wind, md, device = sw.load_assets()
ff_solar, ff_wind = md['future_features_solar'], md['future_features_wind']
SEQ_S, SEQ_W, PL = md['SEQ_LEN_SOLAR'], md['SEQ_LEN_WIND'], md['PRED_LEN']

def _load(kind, n):
    hp = sw.SOLAR_HP if kind == 'solar' else sw.WIND_HP
    nf = len(md['features_solar']) if kind == 'solar' else len(md['features_wind'])
    seq = SEQ_S if kind == 'solar' else SEQ_W
    m = sw.PatchTST_Weather_Model(num_features=nf, seq_len=seq, pred_len=PL, **hp).to(device)
    m.load_state_dict(torch.load(os.path.join(PKL, f'best_patchtst_{kind}_model_D{n}.pth'), map_location=device))
    m.eval(); return m
solar_m = {1: solar1, **{n: _load('solar', n) for n in [2, 3, 4, 5, 6]}}
wind_m  = {1: wind1,  **{n: _load('wind',  n) for n in [2, 3, 4, 5, 6]}}

# ── 캐노니컬 스케일 행렬(3cmp-B와 동일 구성) ──
raw = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
M = pd.DataFrame(index=raw.index)
M['Hour_sin'] = np.sin(2*np.pi*M.index.hour/24); M['Hour_cos'] = np.cos(2*np.pi*M.index.hour/24)
M['Year_sin'] = np.sin(2*np.pi*M.index.dayofyear/365); M['Year_cos'] = np.cos(2*np.pi*M.index.dayofyear/365)
for st in ['west', 'south']:
    for c in [f'solar_rad_{st}', f'total_cloud_{st}', f'midlow_cloud_{st}']:
        M[c] = raw[c]
    daily = raw.groupby(raw.index.date)[f'rainfall_{st}'].transform(lambda x: x.between_time('06:00', '20:00').sum())
    M[f'solar_damping_{st}'] = np.exp(-0.163*daily.clip(upper=10))
for st in ['west', 'east']:
    sp = raw[f'wind_spd_{st}']; cond = [sp < 15, (sp >= 15)&(sp < 20), (sp >= 20)&(sp < 25), sp >= 25]
    M[f'wind_zone_{st}'] = np.select(cond, [0.,1.,.5,0.], default=0.); M[f'wind_spd_{st}'] = sp.clip(upper=20.)
M['wd_sin'] = raw['wd_sin_west']; M['wd_cos'] = raw['wd_cos_west']
M = M.interpolate(limit=3).ffill().bfill()
Ssolar = pd.DataFrame(sc_solar.transform(M[ff_solar]), index=M.index, columns=ff_solar)
Swind  = pd.DataFrame(sc_wind.transform(M[ff_wind]), index=M.index, columns=ff_wind)
act_su, act_wu = raw['real_solar_utilization_jeju'], raw['real_wind_utilization_jeju']

@torch.no_grad()
def _direct(model, Smat, ff, seq, target_day, past_end, py_series):
    d = pd.Timestamp(target_day)
    past_idx = pd.date_range(past_end - pd.Timedelta(hours=seq-1), past_end, freq='h')
    fut_idx = pd.date_range(d, periods=PL, freq='h')
    pn = Smat.reindex(past_idx)[ff].values; fn = Smat.reindex(fut_idx)[ff].values
    py = py_series.reindex(past_idx).values.reshape(-1, 1)
    if np.isnan(pn).any() or np.isnan(fn).any() or np.isnan(py).any():
        return None
    b = {'past_numeric': torch.FloatTensor(pn).unsqueeze(0), 'past_y': torch.FloatTensor(py).unsqueeze(0),
         'future_numeric': torch.FloatTensor(fn).unsqueeze(0)}
    return np.clip(model(b, device=device).squeeze(0).cpu().numpy(), 0, 1)

# ── 3cmp-B 결과 로드(recursive+lgbm+실측) + direct 추가 ──
B = pd.read_parquet(os.path.join(CMP, 'tab', '3cmp-B_raw_predictions.parquet'))
B['ts'] = pd.to_datetime(B['ts'])
pd_su = {}; pd_wu = {}
for D0, g in B.groupby('D0'):
    past_end = pd.Timestamp(D0) + pd.Timedelta(hours=23)   # 발행 origin 마지막 known
    for h in HORIZONS:
        d = pd.Timestamp(D0) + pd.Timedelta(days=h)
        ps = _direct(solar_m[h], Ssolar, ff_solar, SEQ_S, d, past_end, act_su)
        pw = _direct(wind_m[h], Swind, ff_wind, SEQ_W, d, past_end, act_wu)
        if ps is None or pw is None: continue
        fidx = pd.date_range(d, periods=PL, freq='h')
        for t, a, b2 in zip(fidx, ps, pw): pd_su[(D0, t)] = a; pd_wu[(D0, t)] = b2
B['dr_su'] = [pd_su.get((r.D0, r.ts), np.nan) for r in B.itertuples()]
B['dr_wu'] = [pd_wu.get((r.D0, r.ts), np.nan) for r in B.itertuples()]
B = B.dropna(subset=['dr_su', 'dr_wu'])
B['hour'] = B['ts'].dt.hour
print('direct 비교 표본', len(B), '| 발행일', B['D0'].nunique())

# net_load (direct)
for tag in ['dr']:
    B[f'{tag}_nl'] = B['demand'] - B[f'{tag}_su']*B['solar_cap'] - B[f'{tag}_wu']*B['wind_cap']
NLm = B['act_nl'].mean()

# ── 표: 지평별 solar(낮)/wind util + net_load — direct vs recursive vs LGBM ──
day = B[(B.hour >= 8) & (B.hour <= 17)].copy()
rows = []
for h in HORIZONS:
    s = day[day.h == h]; r = {'h': h, 'n_day': len(s)}
    for tag, nm in [('dr', 'direct'), ('pt', 'recursive'), ('lg', 'LGBM')]:
        e = s[f'{tag}_su'] - s['act_su']; r[f'solar_{nm}_MAE'] = round(e.abs().mean(), 4); r[f'solar_{nm}_bias'] = round(e.mean(), 4)
    rows.append(r)
solar_h = pd.DataFrame(rows); solar_h.to_csv(os.path.join(CMP, 'tab', '3cmp-G_solar_by_horizon.csv'), index=False)
print('\n[SOLAR 이용률 지평별 (낮)] direct vs recursive vs LGBM'); print(solar_h.to_string(index=False))

rows = []
for h in HORIZONS:
    s = B[B.h == h]; r = {'h': h, 'n': len(s)}
    for tag, nm in [('dr', 'direct'), ('pt', 'recursive'), ('lg', 'LGBM')]:
        e = s[f'{tag}_wu'] - s['act_wu']; r[f'wind_{nm}_MAE'] = round(e.abs().mean(), 4); r[f'wind_{nm}_bias'] = round(e.mean(), 4)
    rows.append(r)
wind_h = pd.DataFrame(rows); wind_h.to_csv(os.path.join(CMP, 'tab', '3cmp-G_wind_by_horizon.csv'), index=False)
print('\n[WIND 이용률 지평별]'); print(wind_h.to_string(index=False))

rows = []
for h in HORIZONS:
    s = B[B.h == h]; r = {'h': h, 'n': len(s)}
    for tag, nm in [('dr', 'direct'), ('pt', 'recursive'), ('lg', 'LGBM')]:
        mae = np.abs(s[f'{tag}_nl'] - s['act_nl']).mean(); r[f'{nm}_nMAE%'] = round(mae/NLm*100, 2)
    rows.append(r)
nl_h = pd.DataFrame(rows); nl_h.to_csv(os.path.join(CMP, 'tab', '3cmp-G_netload_by_horizon.csv'), index=False)
print(f'\n[net_load 지평별 nMAE%] (평균 net_load {NLm:.0f}MW)'); print(nl_h.to_string(index=False))

# ── 그림 ──
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ax[0].plot(solar_h.h, solar_h['solar_direct_MAE'], 'o-', label='PatchTST-direct', color='green')
ax[0].plot(solar_h.h, solar_h['solar_recursive_MAE'], '^--', label='PatchTST-recursive', color='crimson')
ax[0].plot(solar_h.h, solar_h['solar_LGBM_MAE'], 's-', label='LGBM', color='navy')
ax[0].set_title('solar util MAE by horizon (낮)'); ax[0].set_xlabel('D+h'); ax[0].legend()
ax[1].plot(wind_h.h, wind_h['wind_direct_MAE'], 'o-', label='PatchTST-direct', color='green')
ax[1].plot(wind_h.h, wind_h['wind_recursive_MAE'], '^--', label='PatchTST-recursive', color='crimson')
ax[1].plot(wind_h.h, wind_h['wind_LGBM_MAE'], 's-', label='LGBM', color='navy')
ax[1].set_title('wind util MAE by horizon'); ax[1].set_xlabel('D+h'); ax[1].legend()
ax[2].plot(nl_h.h, nl_h['direct_nMAE%'], 'o-', label='PatchTST-direct', color='green')
ax[2].plot(nl_h.h, nl_h['recursive_nMAE%'], '^--', label='PatchTST-recursive', color='crimson')
ax[2].plot(nl_h.h, nl_h['LGBM_nMAE%'], 's-', label='LGBM', color='navy')
ax[2].set_title('net_load nMAE% by horizon'); ax[2].set_xlabel('D+h'); ax[2].legend()
fig.tight_layout(); fig.savefig(os.path.join(CMP, 'fig', '3cmp-G_direct_compare.png')); plt.close(fig)
print('\n완료 → tab/3cmp-G_*.csv, fig/3cmp-G_direct_compare.png')
