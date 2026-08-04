# -*- coding: utf-8 -*-
"""낮시간 surge 과소예측 공략 실험: 비대칭손실(분위수/낮가중) + 흐린날 특화피처.

사용자 결정:
- BTM = cap×이용률 추정은 만들지 않음(임의추정 금지). cap_btmppa_mw(raw)만 유지.
- 비대칭 손실 + 흐린날 특화 피처(순수 기상): 평년대비 일사결손비율 + 일사램프.
평가기준(사용자):
- D+1~D+3 완전기상 전시간 MAE ≤ KPX MAE × 1.05 (5%까지 악화 허용)
- 낮시간(08~16h) MAPE 가 흐림·맑음 모두 KPX보다 의미있게 우위.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd, lightgbm as lgb
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.normpath(os.path.join(HERE,'..','..','1. data_fetcher_and_db','data','input_data_jeju.db'))
CAP=os.path.normpath(os.path.join(HERE,'..','data','jeju_ppa_btm_capacity_mw.csv'))
ST=['west','east','south']; SOLAR=['west','south']
WX=['temp_c','humidity','solar_rad','wind_spd']
CLOUD=['total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south']
pull=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_utilization_jeju']
for s in ST:
    for w in ['temp_c','humidity','wind_spd']: pull.append(f'{w}_{s}')
for s in SOLAR: pull.append(f'solar_rad_{s}')
pull+=CLOUD
con=sqlite3.connect(DB)
raw=pd.read_sql(f"SELECT {','.join(pull)} FROM historical",con,parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
con.close()
raw['temp_c']=raw[[f'temp_c_{s}' for s in ST]].mean(axis=1)
raw['humidity']=raw[[f'humidity_{s}' for s in ST]].mean(axis=1)
raw['wind_spd']=raw[[f'wind_spd_{s}' for s in ST]].mean(axis=1)
raw['solar_rad']=raw[[f'solar_rad_{s}' for s in SOLAR]].mean(axis=1)
raw=raw[['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_utilization_jeju']+WX+CLOUD]
idx=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
d=raw.set_index('timestamp').reindex(idx); d.index.name='timestamp'
d.loc[d.real_demand_jeju==0,'real_demand_jeju']=np.nan
d['real_demand_jeju']=d['real_demand_jeju'].interpolate(method='time')
for w in WX+CLOUD: d[w]=d[w].interpolate(method='time')
d['day_type']=d['day_type'].ffill().bfill(); d['real_solar_utilization_jeju']=d['real_solar_utilization_jeju'].interpolate(method='time')
cap=pd.read_csv(CAP); d['year']=d.index.year; d['mo']=d.index.month
d=d.reset_index().merge(cap,left_on=['year','mo'],right_on=['year','month'],how='left').set_index('timestamp')

dem=d.real_demand_jeju.values.astype(float); base=d.jeju_est_demand_da.values.astype(float); util=d.real_solar_utilization_jeju.values
hour=d.index.hour.values; dow=d.index.dayofweek.values; month=d.index.month.values
# 흐린날 특화 피처(순수 기상, BTM 추정 아님): 평년(월,시) 대비 일사결손비율 + 일사램프
clim_solar=pd.Series(d.solar_rad.values).groupby([month,hour]).transform('mean').values
with np.errstate(divide='ignore',invalid='ignore'):
    solar_deficit=np.where(clim_solar>5, np.clip(1-d.solar_rad.values/clim_solar,-0.5,1.5), 0.0)  # 흐릴수록↑
solar_ramp=np.abs(np.diff(d.solar_rad.values,prepend=np.nan))
arr={c:d[c].values for c in WX+CLOUD}; arr['cap_btmppa_mw']=d.cap_btmppa_mw.values
arr['solar_deficit']=solar_deficit; arr['solar_ramp']=solar_ramp
dtype_arr=d.day_type.values.astype(object); N=len(d)
rec24=pd.Series(dem).rolling(24,min_periods=24).mean().values
rec168=pd.Series(dem).rolling(168,min_periods=168).mean().values
H=np.arange(1,169)
origins=np.where((hour==23)&(np.arange(N)>=167)&(np.arange(N)<=N-1-168))[0]
tgt=(origins[:,None]+H[None,:]).ravel(); hh=np.broadcast_to(H,(len(origins),168)).ravel()
def col(a): return a[tgt]
s=pd.DataFrame({'y':col(dem),'h':hh.astype(int),'lag168':dem[tgt-168],
 'rec24':np.repeat(rec24[origins],168),'rec168':np.repeat(rec168[origins],168),
 'temp_c':col(arr['temp_c']),'humidity':col(arr['humidity']),'solar_rad':col(arr['solar_rad']),'wind_spd':col(arr['wind_spd']),
 'cap_btmppa_mw':col(arr['cap_btmppa_mw']),
 'hour':col(hour),'dow':col(dow),'month':col(month),'day_type':col(dtype_arr),
 'base':col(base),'util':col(util),'tts':d.index.values[tgt]})
for c in CLOUD: s[c]=np.where(s.h<=48,col(arr[c]),np.nan)
# 흐린날 특화 피처도 h<=48만(예보 신뢰)
s['solar_deficit']=np.where(s.h<=48,col(arr['solar_deficit']),np.nan)
s['solar_ramp']=np.where(s.h<=48,col(arr['solar_ramp']),np.nan)
s['hour_sin']=np.sin(2*np.pi*s.hour/24); s['hour_cos']=np.cos(2*np.pi*s.hour/24)
s['dow_sin']=np.sin(2*np.pi*s.dow/7); s['dow_cos']=np.cos(2*np.pi*s.dow/7)
s['month_sin']=np.sin(2*np.pi*s.month/12); s['month_cos']=np.cos(2*np.pi*s.month/12)
s=s[s.y.notna()&s.lag168.notna()].reset_index(drop=True)
s['day_type']=s['day_type'].astype('category')
s['is_day']=((s.hour>=8)&(s.hour<=16)).astype(int)
tts=pd.to_datetime(s.tts)
tr=s[tts<='2025-02-28 23:00']; va=s[(tts>='2025-03-01')&(tts<='2026-03-21 23:00')]; te=s[(tts>='2026-03-22')&(tts<='2026-05-31 23:00')].copy()

BASE=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south','cap_btmppa_mw',
      'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
CLOUDY=['solar_deficit','solar_ramp']
CAT=['day_type']
HP=dict(num_leaves=244,min_data_in_leaf=76,feature_fraction=0.9,bagging_fraction=0.8,
        bagging_freq=5,lambda_l2=0.1,verbosity=-1,random_state=42,learning_rate=0.024)
def mape(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=(a>0)&np.isfinite(a)&np.isfinite(p); return float(np.mean(np.abs(a[m]-p[m])/a[m])*100)
def mae(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=(a>0)&np.isfinite(a)&np.isfinite(p); return float(np.mean(np.abs(a[m]-p[m])))
te['q']=np.nan
dmask=(te.hour>=8)&(te.hour<=16)
ted=te[dmask].copy(); ted['uq']=pd.qcut(ted.util,4,labels=False,duplicates='drop')
def run(feats,name,objective='regression_l1',alpha=0.5,day_w=1.0):
    p=dict(HP)
    if objective=='quantile': p.update(objective='quantile',alpha=alpha,metric='quantile')
    else: p.update(objective='regression_l1',metric='mae')
    w=np.where(tr.is_day.values==1, day_w, 1.0)
    wv=np.where(va.is_day.values==1, day_w, 1.0)
    dtr=lgb.Dataset(tr[feats],tr.y,weight=w,categorical_feature=[c for c in CAT if c in feats])
    dva=lgb.Dataset(va[feats],va.y,weight=wv,categorical_feature=[c for c in CAT if c in feats],reference=dtr)
    m=lgb.train(p,dtr,num_boost_round=4000,valid_sets=[dva],valid_names=['val'],
                callbacks=[lgb.early_stopping(150)])
    pr=m.predict(te[feats],num_iteration=m.best_iteration)
    te['pred']=pr; ted['pred']=m.predict(ted[feats],num_iteration=m.best_iteration)
    da=((te.h-1)//24+1)
    out={'name':name,'best':m.best_iteration}
    # 제약: D+1~3 전시간 MAE 비율
    for dn in [1,2,3]:
        g=te[da==dn]; out[f'MAE_D{dn}/KPX']=round(mae(g.y,g.pred)/mae(g.y,g.base),3)
    # 낮시간 MAPE
    out['낮MAPE']=round(mape(ted.y,ted.pred),2)
    cl=ted[ted.uq==0]; cr=ted[ted.uq==3]
    out['낮흐림MAPE']=round(mape(cl.y,cl.pred),2); out['낮흐림편향']=round((cl.pred-cl.y).mean(),1)
    out['낮맑음MAPE']=round(mape(cr.y,cr.pred),2)
    out['전체MAPE']=round(mape(te.y,te.pred),2)
    return out
# KPX 기준
kpx={'name':'KPX(기준)'}
da=((te.h-1)//24+1)
for dn in [1,2,3]: kpx[f'MAE_D{dn}/KPX']=1.0
kpx['낮MAPE']=round(mape(ted.y,ted.base),2)
cl=ted[ted.uq==0]; cr=ted[ted.uq==3]
kpx['낮흐림MAPE']=round(mape(cl.y,cl.base),2); kpx['낮흐림편향']=round((cl.base-cl.y).mean(),1); kpx['낮맑음MAPE']=round(mape(cr.y,cr.base),2)
kpx['전체MAPE']=round(mape(te.y,te.base),2); kpx['best']=0
rows=[kpx]
rows.append(run(BASE,'V0 baseline(L1)'))
rows.append(run(BASE,'V1 L1+낮가중3',day_w=3.0))
rows.append(run(BASE,'V2 quantile.55',objective='quantile',alpha=0.55))
rows.append(run(BASE,'V3 quantile.60',objective='quantile',alpha=0.60))
rows.append(run(BASE+CLOUDY,'V4 L1+흐린날피처'))
rows.append(run(BASE+CLOUDY,'V5 q.58+낮가중2+흐린날',objective='quantile',alpha=0.58,day_w=2.0))
rows.append(run(BASE+CLOUDY,'V6 q.60+흐린날',objective='quantile',alpha=0.60))
R=pd.DataFrame(rows)
cols=['name','best','MAE_D1/KPX','MAE_D2/KPX','MAE_D3/KPX','낮MAPE','낮흐림MAPE','낮흐림편향','낮맑음MAPE','전체MAPE']
print(R[cols].to_string(index=False))
TAB=os.path.join(HERE,'tab'); os.makedirs(TAB,exist_ok=True)
R[cols].to_csv(os.path.join(TAB,'2-A_exp_daytime_asym.csv'),index=False)
print('\n기준: MAE_Dn/KPX <= 1.05 (전시간), 낮흐림/낮맑음 MAPE < KPX 그리고 의미있게 낮아야.')
print('KPX 낮흐림 %.2f / 낮맑음 %.2f'%(kpx['낮흐림MAPE'],kpx['낮맑음MAPE']))
