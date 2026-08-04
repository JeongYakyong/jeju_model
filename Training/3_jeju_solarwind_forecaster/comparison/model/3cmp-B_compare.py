"""3cmp-B — PatchTST(재귀 롤링) vs LGBM(순수기상 direct) 비교 하니스.

설계(사용자 확정):
  - 평가 레벨: 이용률 + net_load 둘 다
  - 기상 입력: 실측기상(perfect) 우선 — 모델 자체의 흐린날 편향을 격리
  - 장지평: PatchTST는 고정 origin에서 예측 이용률을 past_y로 이어붙이는 재귀 롤링(D+1~D+6),
            LGBM은 순수기상 horizon-무관(각 대상일 기상으로 직접)
  - net_load = 실측 demand − solar_gen − wind_gen (demand 고정 → 오차는 순수 신재생)

test = 2026. 발행일 D0마다 D+1..D+6 블록(각 24h, 00:00~23:00) 생성.
산출: tab/3cmp-B_util_by_horizon.csv, _solar_regime_by_horizon.csv, _netload_by_horizon.csv
      fig/3cmp-B_*.png
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CMP  = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(CMP, '..'))      # 3. jeju_solarwind_forecaster
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import solarwind_db_pipeline as sw            # PatchTST 모델 클래스·자산 로더
import importlib.util
_spec = importlib.util.spec_from_file_location('m3a', os.path.join(HERE, '3cmp-A_lgbm_solarwind.py'))
m3a = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m3a)
import lightgbm as lgb

for _f in ['Malgun Gothic', 'Gulim']:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f; break
plt.rcParams['axes.unicode_minus'] = False

CSV = os.path.join(ROOT, 'training', 'solarwind_raw_jeju.csv')
DB  = os.path.normpath(os.path.join(ROOT, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'
HORIZONS = [1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------------- 데이터·피처 준비
raw = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index()
raw = raw.apply(pd.to_numeric, errors='coerce')
# real_demand 병합 (net_load용)
import sqlite3
con = sqlite3.connect(DB)
dem = pd.read_sql("SELECT timestamp, real_demand_jeju FROM historical", con,
                  parse_dates=['timestamp']).set_index('timestamp')
con.close()
raw['real_demand'] = pd.to_numeric(dem['real_demand_jeju'], errors='coerce').reindex(raw.index)

# LGBM 피처(전체구간, clearsky 평년은 train으로)
tr = raw[raw.index.year <= 2024]
_, clim = m3a.build_features(tr)
feat_all, _ = m3a.build_features(raw, clim=clim)

m_solar = lgb.Booster(model_file=os.path.join(HERE, 'lgbm_solar_util.txt'))
m_wind  = lgb.Booster(model_file=os.path.join(HERE, 'lgbm_wind_util.txt'))
feat_all['lgbm_su'] = np.clip(m_solar.predict(feat_all[m3a.SOLAR_FINAL]), 0, 1)
feat_all['lgbm_wu'] = np.clip(m_wind.predict(feat_all[m3a.WIND_FINAL]), 0, 1)

# ---------------------------------------------------------------- PatchTST 자산·입력행렬
solar_m, wind_m, sc_solar, sc_wind, md, device = sw.load_assets()
ff_solar = md['future_features_solar']; ff_wind = md['future_features_wind']
SEQ_S, SEQ_W, PL = md['SEQ_LEN_SOLAR'], md['SEQ_LEN_WIND'], md['PRED_LEN']

# PatchTST 캐노니컬 피처 행렬 (학습 파이프라인과 동일 구성)
M = pd.DataFrame(index=raw.index)
M['Hour_sin'] = np.sin(2*np.pi*M.index.hour/24); M['Hour_cos'] = np.cos(2*np.pi*M.index.hour/24)
M['Year_sin'] = np.sin(2*np.pi*M.index.dayofyear/365); M['Year_cos'] = np.cos(2*np.pi*M.index.dayofyear/365)
for st in ['west', 'south']:
    for c in [f'solar_rad_{st}', f'total_cloud_{st}', f'midlow_cloud_{st}']:
        M[c] = raw[c]
    daily = raw.groupby(raw.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    M[f'solar_damping_{st}'] = np.exp(-0.163*daily.clip(upper=10))
for st in ['west', 'east']:
    sp = raw[f'wind_spd_{st}']
    cond = [sp < 15, (sp >= 15)&(sp < 20), (sp >= 20)&(sp < 25), sp >= 25]
    M[f'wind_zone_{st}'] = np.select(cond, [0.,1.,.5,0.], default=0.)
    M[f'wind_spd_{st}'] = sp.clip(upper=20.0)
M['wd_sin'] = raw['wd_sin_west']; M['wd_cos'] = raw['wd_cos_west']
M = M.interpolate(limit=3).ffill().bfill()
# 스케일 적용(학습과 동일: future_features 컬럼만 transform)
Ssolar = pd.DataFrame(sc_solar.transform(M[ff_solar]), index=M.index, columns=ff_solar)
Swind  = pd.DataFrame(sc_wind.transform(M[ff_wind]),  index=M.index, columns=ff_wind)
act_su = raw[SU]; act_wu = raw[WU]


@torch.no_grad()
def _pt_block(model, Smat, ff, seq, target_day, past_y_series):
    """target_day(00:00)의 24h PatchTST 예측. past_y_series: 과거 util(롤링 반영)."""
    d0 = pd.Timestamp(target_day)
    past_idx = pd.date_range(d0 - pd.Timedelta(hours=seq), d0 - pd.Timedelta(hours=1), freq='h')
    fut_idx  = pd.date_range(d0, periods=PL, freq='h')
    pn = Smat.reindex(past_idx)[ff].values
    fn = Smat.reindex(fut_idx)[ff].values
    py = past_y_series.reindex(past_idx).values.reshape(-1, 1)
    if np.isnan(pn).any() or np.isnan(fn).any() or np.isnan(py).any():
        return None, fut_idx
    batch = {'past_numeric': torch.FloatTensor(pn).unsqueeze(0),
             'past_y': torch.FloatTensor(py).unsqueeze(0),
             'future_numeric': torch.FloatTensor(fn).unsqueeze(0)}
    pred = model(batch, device=device).squeeze(0).cpu().numpy()
    return np.clip(pred, 0, 1), fut_idx


# ---------------------------------------------------------------- 롤링 비교 루프
test_days = pd.date_range('2026-01-01', '2026-05-25', freq='D')   # 발행일 D0 (D+6 데이터 존재 범위)
recs = []
for D0 in test_days:
    # 롤링 상태: 예측 util로 past_y override (발행 이후 시각만)
    roll_su = act_su.copy(); roll_wu = act_wu.copy()
    issue_cut = D0 + pd.Timedelta(hours=23)     # D0 23:00까지 실측 known
    for k in HORIZONS:
        d = D0 + pd.Timedelta(days=k)
        # roll_su/roll_wu: 실측이되 직전 블록들은 이미 예측으로 override됨(재귀 롤링)
        ps, fidx = _pt_block(solar_m, Ssolar, ff_solar, SEQ_S, d, roll_su)
        pw, _    = _pt_block(wind_m,  Swind,  ff_wind,  SEQ_W, d, roll_wu)
        if ps is None or pw is None:
            continue
        # 롤링: 이 블록 예측을 past_y에 기록(다음 k에서 사용)
        roll_su.loc[fidx] = ps; roll_wu.loc[fidx] = pw
        sub = feat_all.reindex(fidx)
        df = pd.DataFrame({
            'ts': fidx, 'D0': D0, 'h': k,
            'act_su': act_su.reindex(fidx).values, 'act_wu': act_wu.reindex(fidx).values,
            'pt_su': ps, 'pt_wu': pw,
            'lg_su': sub['lgbm_su'].values, 'lg_wu': sub['lgbm_wu'].values,
            'solar_cap': raw['real_solar_capacity_jeju'].reindex(fidx).values,
            'wind_cap': raw['real_wind_capacity_jeju'].reindex(fidx).values,
            'real_solar_gen': raw['real_solar_gen_jeju'].reindex(fidx).values,
            'real_wind_gen': raw['real_wind_gen_jeju'].reindex(fidx).values,
            'demand': raw['real_demand'].reindex(fidx).values,
            'total_cloud_west': raw['total_cloud_west'].reindex(fidx).values,
        })
        recs.append(df)

R = pd.concat(recs, ignore_index=True)
R['hour'] = R['ts'].dt.hour
R = R.dropna(subset=['act_su', 'act_wu', 'demand'])
print('비교 표본', len(R), '행, 발행일', R['D0'].nunique(), '일')

# net_load 계산 (demand 고정)
for tag in ['pt', 'lg']:
    R[f'{tag}_sg'] = R[f'{tag}_su'] * R['solar_cap']
    R[f'{tag}_wg'] = R[f'{tag}_wu'] * R['wind_cap']
    R[f'{tag}_nl'] = R['demand'] - R[f'{tag}_sg'] - R[f'{tag}_wg']
R['act_nl'] = R['demand'] - R['real_solar_gen'] - R['real_wind_gen']

# ---------------------------------------------------------------- 표 1: 지평별 이용률
def by_h(col_pred, col_true, frame=R):
    out = []
    for h in HORIZONS:
        s = frame[frame.h == h]
        e = s[col_pred] - s[col_true]
        out.append(dict(h=h, MAE=round(e.abs().mean(), 4), bias=round(e.mean(), 4), n=len(s)))
    return pd.DataFrame(out)

day = R[(R.hour >= 8) & (R.hour <= 17)].copy()
tbl = []
for h in HORIZONS:
    s = day[day.h == h]
    row = {'h': h}
    for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]:
        e = s[f'{tag}_su'] - s['act_su']
        row[f'{name}_MAE'] = round(e.abs().mean(), 4); row[f'{name}_bias'] = round(e.mean(), 4)
    tbl.append(row)
solar_h = pd.DataFrame(tbl); solar_h.to_csv(os.path.join(CMP, 'tab', '3cmp-B_solar_util_by_horizon.csv'), index=False)
print('\n[SOLAR 이용률 지평별 (낮 8-17h)]'); print(solar_h.to_string(index=False))

tbl = []
for h in HORIZONS:
    s = R[R.h == h]; row = {'h': h}
    for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]:
        e = s[f'{tag}_wu'] - s['act_wu']
        row[f'{name}_MAE'] = round(e.abs().mean(), 4); row[f'{name}_bias'] = round(e.mean(), 4)
    tbl.append(row)
wind_h = pd.DataFrame(tbl); wind_h.to_csv(os.path.join(CMP, 'tab', '3cmp-B_wind_util_by_horizon.csv'), index=False)
print('\n[WIND 이용률 지평별 (전시간)]'); print(wind_h.to_string(index=False))

# ---------------------------------------------------------------- 표 2: 흐린날 regime × 지평 (핵심)
dcloud = day.groupby([day.D0, day.h, day.ts.dt.date])['total_cloud_west'].transform('mean')
day['regime'] = np.where(dcloud >= 0.7, 'cloudy', np.where(dcloud <= 0.3, 'sunny', 'mixed'))
rows = []
for r in ['sunny', 'cloudy']:
    for h in HORIZONS:
        s = day[(day.regime == r) & (day.h == h)]
        if not len(s): continue
        row = {'regime': r, 'h': h, 'n': len(s)}
        for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]:
            e = s[f'{tag}_su'] - s['act_su']
            row[f'{name}_bias'] = round(e.mean(), 4); row[f'{name}_MAE'] = round(e.abs().mean(), 4)
        rows.append(row)
reg_h = pd.DataFrame(rows); reg_h.to_csv(os.path.join(CMP, 'tab', '3cmp-B_solar_regime_by_horizon.csv'), index=False)
print('\n[★ SOLAR 흐린날/맑은날 × 지평 bias (낮)]'); print(reg_h.to_string(index=False))

# ---------------------------------------------------------------- 표 3: net_load 지평별
# net_load는 정오에 0 근처라 MAPE 폭발 → nMAE%(=MAE/평균net_load)로 보고 + |nl|>50 한정 MAPE 참고
NL_MEAN = R['act_nl'].mean()
def cmape(p, a):
    m = np.abs(a) > 50
    return float((np.abs(p[m] - a[m]) / np.abs(a[m])).mean() * 100)
tbl = []
for h in HORIZONS:
    s = R[R.h == h]; row = {'h': h, 'n': len(s)}
    for tag, name in [('pt', 'PatchTST'), ('lg', 'LGBM')]:
        mae = np.abs(s[f'{tag}_nl'] - s['act_nl']).mean()
        row[f'{name}_MAE'] = round(mae, 1)
        row[f'{name}_nMAE%'] = round(mae / NL_MEAN * 100, 2)
        row[f'{name}_MAPE%'] = round(cmape(s[f'{tag}_nl'].values, s['act_nl'].values), 2)
    tbl.append(row)
nl_h = pd.DataFrame(tbl); nl_h.to_csv(os.path.join(CMP, 'tab', '3cmp-B_netload_by_horizon.csv'), index=False)
print('\n[net_load 지평별 (전시간, demand 고정)]'); print(nl_h.to_string(index=False))

# ---------------------------------------------------------------- 그림
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ax[0].plot(solar_h.h, solar_h['PatchTST_bias'], 'o-', label='PatchTST', color='crimson')
ax[0].plot(solar_h.h, solar_h['LGBM_bias'], 's-', label='LGBM', color='navy')
ax[0].axhline(0, color='k', lw=.7); ax[0].set_title('solar 이용률 bias by horizon (낮)')
ax[0].set_xlabel('D+h'); ax[0].legend()
cl = reg_h[reg_h.regime == 'cloudy']
ax[1].plot(cl.h, cl['PatchTST_bias'], 'o-', label='PatchTST', color='crimson')
ax[1].plot(cl.h, cl['LGBM_bias'], 's-', label='LGBM', color='navy')
ax[1].axhline(0, color='k', lw=.7); ax[1].set_title('★ 흐린날 solar bias by horizon')
ax[1].set_xlabel('D+h'); ax[1].legend()
ax[2].plot(nl_h.h, nl_h['PatchTST_nMAE%'], 'o-', label='PatchTST', color='crimson')
ax[2].plot(nl_h.h, nl_h['LGBM_nMAE%'], 's-', label='LGBM', color='navy')
ax[2].set_title('net_load nMAE% by horizon'); ax[2].set_xlabel('D+h'); ax[2].legend()
fig.tight_layout(); fig.savefig(os.path.join(CMP, 'fig', '3cmp-B_horizon_compare.png')); plt.close(fig)

R.to_parquet(os.path.join(CMP, 'tab', '3cmp-B_raw_predictions.parquet'))
print('\n비교 완료 → tab/3cmp-B_*.csv, fig/3cmp-B_horizon_compare.png')
