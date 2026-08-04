"""3cmp-3 — tcog 후처리(대류일 보정) 가볍게 적합·검증.

3cmp-2: tcog>0(대류운)에서 solar 과대·wind 과소예측(forecast 전용 신호, 모델입력 불가).
배포 모델(solar=PatchTST, wind=LGBM)의 forecast 잔차에 tcog 보정을 적합:
  corrected = pred + beta * tcog   (no-intercept → tcog=0이면 보정 0)
  solar beta<0(깎음), wind beta>0(올림) 기대.
평가: 랜덤 5-fold(계절 골고루, 사용자 지시). fold train으로 beta 적합 → test 적용 →
      MAE before/after (전체 + tcog>0 부분집합).
산출: tab/3cmp-3_tcog_postproc.csv, model/tcog_postproc.json(beta)
"""
import os, sys, json, sqlite3
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
HERE = os.path.dirname(os.path.abspath(__file__)); CMP = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(CMP, '..')); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import solarwind_db_pipeline as sw
import importlib.util
spx = importlib.util.spec_from_file_location('m3a', os.path.join(HERE, '3cmp-A_lgbm_solarwind.py'))
m3a = importlib.util.module_from_spec(spx); spx.loader.exec_module(m3a)
DB = sw.DB_PATH
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'

# clim + LGBM wind
raw = pd.read_csv(os.path.join(ROOT, 'training', 'solarwind_raw_jeju.csv'), parse_dates=['timestamp']).set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
_, clim = m3a.build_features(raw[raw.index.year <= 2024])
m_wind = lgb.Booster(model_file=os.path.join(ROOT, 'lgbm_models', 'lgbm_wind_util.txt'))

con = sqlite3.connect(DB)
# forecast wind 기상 + tcog + 표준(없으면 skip)
fmap = {'wind_spd_10m_west': 'wind_spd_west', 'wind_spd_10m_east': 'wind_spd_east',
        'wd_sin_10m_west': 'wd_sin_west', 'wd_cos_10m_west': 'wd_cos_west'}
spc = ['tcog_west', 'tcog_east', 'tcog_south']
sel = ', '.join(f'"{c}"' for c in ['timestamp'] + list(fmap) + spc)
F = pd.read_sql(f'SELECT {sel} FROM forecast ORDER BY timestamp', con, parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
act = pd.read_sql(f'SELECT timestamp, {SU} su, {WU} wu FROM historical', con, parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce')
# 표준 solar 기상(PatchTST 출력은 pipeline로 따로) — tcog만 여기서
days = pd.read_sql("SELECT substr(timestamp,1,10) d, COUNT(*) n FROM forecast WHERE radiation_west IS NOT NULL GROUP BY d HAVING n=24 ORDER BY d", con)['d'].tolist()
con.close()

# wind LGBM 예측(forecast 기상)
W = F.rename(columns=fmap)
d = pd.DataFrame(index=W.index)
d['hour_sin'] = np.sin(2*np.pi*d.index.hour/24); d['hour_cos'] = np.cos(2*np.pi*d.index.hour/24)
d['year_sin'] = np.sin(2*np.pi*d.index.dayofyear/365); d['year_cos'] = np.cos(2*np.pi*d.index.dayofyear/365)
d['wd_sin_west'] = W['wd_sin_west']; d['wd_cos_west'] = W['wd_cos_west']
for st in ['west', 'east']:
    sp = W[f'wind_spd_{st}']; cond = [sp < 15, (sp >= 15)&(sp < 20), (sp >= 20)&(sp < 25), sp >= 25]
    d[f'wind_zone_{st}'] = np.select(cond, [0., 1., .5, 0.], default=0.); d[f'wind_spd_{st}'] = sp.clip(upper=20.)
meta = json.load(open(os.path.join(ROOT, 'lgbm_models', 'feat_meta.json'), encoding='utf-8'))
d['lg_wu'] = np.clip(m_wind.predict(d[meta['WIND_FINAL']]), 0, 1)

# solar PatchTST 예측(forecast, D+1) — 파이프라인 일별
pt = []
for dd in days:
    try:
        o = sw.predict_solarwind_to_db(dd, write=False, verbose=False)
        o['timestamp'] = pd.to_datetime(o['timestamp']); pt.append(o[['timestamp', 'est_solar_utilization_jeju']])
    except Exception:
        continue
PT = pd.concat(pt).set_index('timestamp')

R = pd.DataFrame(index=F.index)
# 지점 선택(잔차적합 비교, 3cmp-3b): solar=south(+10.1%) / wind=east(+10.7%). west는 모델 주피처라 잉여.
R['tcog_solar'] = F['tcog_south']
R['tcog_wind'] = F['tcog_east']
R['pt_su'] = PT['est_solar_utilization_jeju']; R['lg_wu'] = d['lg_wu']
R = R.join(act).dropna(subset=['su', 'wu', 'pt_su', 'lg_wu', 'tcog_solar'])
R['hour'] = R.index.hour

def beta_fit(tcog, resid):
    m = tcog > 0
    if m.sum() < 10: return 0.0
    return float(np.sum(tcog[m]*resid[m]) / np.sum(tcog[m]**2))

def kfold_eval(frame, pred_col, true_col, tcog_col, daytime=False):
    fr = frame[(frame.hour >= 8) & (frame.hour <= 17)] if daytime else frame
    X = fr[[tcog_col]].values; tcog = fr[tcog_col].values
    resid = (fr[true_col] - fr[pred_col]).values; pred = fr[pred_col].values; true = fr[true_col].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_b = []; mae_a = []; mae_b_c = []; mae_a_c = []; betas = []
    for tr, te in kf.split(fr):
        b = beta_fit(tcog[tr], resid[tr]); betas.append(b)
        corr = np.clip(pred[te] + b*tcog[te], 0, 1)
        eb = np.abs(pred[te]-true[te]); ea = np.abs(corr-true[te])
        mae_b.append(eb.mean()); mae_a.append(ea.mean())
        cm = tcog[te] > 0
        if cm.sum(): mae_b_c.append(eb[cm].mean()); mae_a_c.append(ea[cm].mean())
    return (np.mean(mae_b), np.mean(mae_a), np.mean(mae_b_c), np.mean(mae_a_c), np.mean(betas),
            int((tcog > 0).sum()), len(fr))

rows = []
sb = kfold_eval(R, 'pt_su', 'su', 'tcog_solar', daytime=True)
rows.append(dict(channel='SOLAR(PatchTST,낮)', MAE_before=round(sb[0],4), MAE_after=round(sb[1],4),
                 MAE_before_tcog=round(sb[2],4), MAE_after_tcog=round(sb[3],4), beta=round(sb[4],4), n_tcog=sb[5], n=sb[6]))
wb = kfold_eval(R, 'lg_wu', 'wu', 'tcog_wind', daytime=False)
rows.append(dict(channel='WIND(LGBM,전시간)', MAE_before=round(wb[0],4), MAE_after=round(wb[1],4),
                 MAE_before_tcog=round(wb[2],4), MAE_after_tcog=round(wb[3],4), beta=round(wb[4],4), n_tcog=wb[5], n=wb[6]))
T = pd.DataFrame(rows)
print('[tcog 후처리 5-fold 랜덤 평가] corrected = pred + beta*tcog (tcog>0만 보정)')
print(T.to_string(index=False))
T.to_csv(os.path.join(CMP, 'tab', '3cmp-3_tcog_postproc.csv'), index=False)
json.dump({'solar_beta': round(sb[4], 5), 'wind_beta': round(wb[4], 5),
           'solar_tcog': 'south', 'wind_tcog': 'east',
           'note': 'corrected_util = clip(pred + beta*tcog_station, 0,1); solar=tcog_south(낮시간), wind=tcog_east(전시간). 잔차적합으로 지점 선택(west는 모델 주피처라 잉여).'},
          open(os.path.join(HERE, 'tcog_postproc.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nbeta 저장 → model/tcog_postproc.json')
