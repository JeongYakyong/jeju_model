# -*- coding: utf-8 -*-
"""배포된 PatchTST+LGBM(D+1) vs 신규 직접 다지평 LGBM(D+1) 동일 구간 비교.

동일 test 구간(2026-03-22~05-31)에서 각 target_date(D+1)의 24h 예측을:
  (1) 배포 모델  : demand_predict.predict_iterative + patchtst_predict_d1 (기존 운영 파이프라인 재현)
  (2) 신규 모델  : lgbm_jeju_demand_direct.txt 직접 다지평(h=1..24)
  (3) PatchTST 단독 신호
  (4) KPX 하루전 jeju_est_demand_da
모두 **실측 기상(완전기상)** 입력으로 공정 비교. MAPE(24h 블록) 집계.
"""
import os, sys, sqlite3, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from demand_predict import (load_config, load_booster_pkl, add_cycle_features,
                            coerce_categoricals, predict_iterative, _default)
from patchtst_predict import predict_d1 as patchtst_predict_d1
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.normpath(os.path.join(HERE, '..', '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
STATIONS = ['west','east','south']; SOLAR_ST=['west','south']
WX=['temp_c','humidity','solar_rad','wind_spd']

# ---- 데이터 로드 + 제주 공간평균 + reindex/보간 (2-A 와 동일) ----
pull=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type']
for st in STATIONS:
    for w in ['temp_c','humidity','wind_spd']: pull.append(f'{w}_{st}')
for st in SOLAR_ST: pull.append(f'solar_rad_{st}')
con=sqlite3.connect(DB)
raw=pd.read_sql(f"SELECT {', '.join(pull)} FROM historical", con, parse_dates=['timestamp'])
ptst=pd.read_sql('SELECT timestamp, jeju_patchtst_target FROM patchtst_signal', con, parse_dates=['timestamp'])
con.close()
raw=raw.sort_values('timestamp').reset_index(drop=True)
raw['temp_c']=raw[[f'temp_c_{s}' for s in STATIONS]].mean(axis=1)
raw['humidity']=raw[[f'humidity_{s}' for s in STATIONS]].mean(axis=1)
raw['wind_spd']=raw[[f'wind_spd_{s}' for s in STATIONS]].mean(axis=1)
raw['solar_rad']=raw[[f'solar_rad_{s}' for s in SOLAR_ST]].mean(axis=1)
raw=raw.merge(ptst,on='timestamp',how='left')
idx=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
d=raw.set_index('timestamp').reindex(idx); d.index.name='timestamp'
d.loc[d.real_demand_jeju==0,'real_demand_jeju']=np.nan
d['real_demand_jeju']=d['real_demand_jeju'].interpolate(method='time')
for w in WX: d[w]=d[w].interpolate(method='time')
d['day_type']=d['day_type'].ffill().bfill()

# ---- 모델 로드 ----
cfg=load_config(_default('models/pipeline_config.json'))
booster=load_booster_pkl(_default('models/lgbm_pipeline.pkl'))
feat_dep=cfg['feature_cols']; best_dep=cfg['best_iteration']
direct=lgb.Booster(model_file=os.path.join(HERE,'models','lgbm_jeju_demand_direct.txt'))
FEAT=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
hist_series=d['real_demand_jeju'].copy()
dem=d.real_demand_jeju.values.astype(float)
rec24_full=pd.Series(dem).rolling(24,min_periods=24).mean().values
rec168_full=pd.Series(dem).rolling(168,min_periods=168).mean().values
pos={ts:i for i,ts in enumerate(d.index)}

dates=pd.date_range('2026-03-22','2026-05-31',freq='D')
rows=[]
for tgt in dates:
    end=tgt+pd.Timedelta(hours=23)
    if end not in pos: continue
    # 실측 타깃
    y=d.loc[tgt:end,'real_demand_jeju'].values
    if np.isnan(y).any() or len(y)!=24: continue
    # (1) 배포 모델
    try:
        ptst24=patchtst_predict_d1(hist_series, target_date=tgt)
        base=d.loc[tgt:end,WX].reset_index().rename(columns={'index':'timestamp'})
        base['day_type']=d.loc[tgt:end,'day_type'].values
        base['patchtst_target']=ptst24
        base=add_cycle_features(base); base=coerce_categoricals(base,cfg)
        dep=predict_iterative(booster,feat_dep,base,hist_series,tgt,best_dep)
    except Exception as e:
        print('dep fail',tgt.date(),e); continue
    # (2) 신규 직접
    o=pos[tgt]-1  # origin = D 23:00
    H=np.arange(1,25); tg=o+H
    samp=pd.DataFrame({
        'h':H,'lag168':dem[tg-168],
        'rec24':rec24_full[o],'rec168':rec168_full[o],
        'temp_c':d['temp_c'].values[tg],'humidity':d['humidity'].values[tg],
        'solar_rad':d['solar_rad'].values[tg],'wind_spd':d['wind_spd'].values[tg],
    })
    hr=d.index.hour.values[tg]; dw=d.index.dayofweek.values[tg]; mo=d.index.month.values[tg]
    samp['hour_sin']=np.sin(2*np.pi*hr/24); samp['hour_cos']=np.cos(2*np.pi*hr/24)
    samp['dow_sin']=np.sin(2*np.pi*dw/7); samp['dow_cos']=np.cos(2*np.pi*dw/7)
    samp['month_sin']=np.sin(2*np.pi*mo/12); samp['month_cos']=np.cos(2*np.pi*mo/12)
    samp['day_type']=pd.Categorical(d['day_type'].values[tg], categories=cfg['categorical_categories']['day_type'])
    new=direct.predict(samp[FEAT])
    # (3)(4) 신호
    ptst_sig=d['jeju_patchtst_target'].values[tg]
    kpx=d['jeju_est_demand_da'].values[tg]
    for k in range(24):
        rows.append(dict(ts=tgt+pd.Timedelta(hours=k), y=y[k], dep=dep[k], new=new[k],
                         ptst=ptst_sig[k], kpx=kpx[k]))

R=pd.DataFrame(rows)
def mape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); m=(a>0)&np.isfinite(a)&np.isfinite(p)
    return float(np.mean(np.abs(a[m]-p[m])/a[m])*100)
print(f'\n=== D+1 비교 (실측 기상, {R.ts.min()} ~ {R.ts.max()}, {len(R)}h / {len(R)//24}일) ===')
res=pd.DataFrame([
    dict(model='배포 PatchTST+LGBM (iterative)', mape=round(mape(R.y,R.dep),3)),
    dict(model='신규 LGBM 직접 다지평',          mape=round(mape(R.y,R.new),3)),
    dict(model='PatchTST 단독',                  mape=round(mape(R.y,R.ptst),3)),
    dict(model='KPX 하루전',                     mape=round(mape(R.y,R.kpx),3)),
]).sort_values('mape')
print(res.to_string(index=False))
TAB=os.path.join(HERE,'tab'); os.makedirs(TAB,exist_ok=True)
res.to_csv(os.path.join(TAB,'2-A_deployed_compare.csv'),index=False)
R.to_csv(os.path.join(TAB,'2-A_deployed_compare_hourly.csv'),index=False)
print('saved tab/2-A_deployed_compare.csv')
