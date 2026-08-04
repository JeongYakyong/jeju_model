# -*- coding: utf-8 -*-
"""2-A 피처 A/B 실험: 구름(h<=48 마스킹) + BTM/PPA 용량.

baseline(2-A 15피처) 대비:
  + 구름@h<=48 : total_cloud, midlow_cloud (D+1/D+2 지평만, 그 외 NaN)
  + 용량       : cap_btmppa_mw (전 지평, 월별 알려진 값)
  + 둘다
  + 구름@전지평 (참고용)
완전기상(실측) 평가. 모델 저장 안 함(실험). 채택은 사용자 Decision Gate.
"""
import os, sqlite3
import numpy as np, pandas as pd, lightgbm as lgb
HERE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.normpath(os.path.join(HERE,'..','..','1. data_fetcher_and_db','data','input_data_jeju.db'))
CAP=os.path.normpath(os.path.join(HERE,'..','data','jeju_ppa_btm_capacity_mw.csv'))
STATIONS=['west','east','south']; SOLAR_ST=['west','south']
WX=['temp_c','humidity','solar_rad','wind_spd']

pull=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type']
for st in STATIONS:
    for w in ['temp_c','humidity','wind_spd','total_cloud','midlow_cloud']: pull.append(f'{w}_{st}')
for st in SOLAR_ST: pull.append(f'solar_rad_{st}')
con=sqlite3.connect(DB)
raw=pd.read_sql(f"SELECT {', '.join(pull)} FROM historical", con, parse_dates=['timestamp'])
con.close()
raw=raw.sort_values('timestamp').reset_index(drop=True)
raw['temp_c']=raw[[f'temp_c_{s}' for s in STATIONS]].mean(axis=1)
raw['humidity']=raw[[f'humidity_{s}' for s in STATIONS]].mean(axis=1)
raw['wind_spd']=raw[[f'wind_spd_{s}' for s in STATIONS]].mean(axis=1)
raw['solar_rad']=raw[[f'solar_rad_{s}' for s in SOLAR_ST]].mean(axis=1)
raw['total_cloud']=raw[[f'total_cloud_{s}' for s in STATIONS]].mean(axis=1)
raw['midlow_cloud']=raw[[f'midlow_cloud_{s}' for s in STATIONS]].mean(axis=1)
keep=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type']+WX+['total_cloud','midlow_cloud']
raw=raw[keep]
idx=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
d=raw.set_index('timestamp').reindex(idx); d.index.name='timestamp'
d.loc[d.real_demand_jeju==0,'real_demand_jeju']=np.nan
d['real_demand_jeju']=d['real_demand_jeju'].interpolate(method='time')
for w in WX+['total_cloud','midlow_cloud']: d[w]=d[w].interpolate(method='time')
d['day_type']=d['day_type'].ffill().bfill()
# 용량 병합
cap=pd.read_csv(CAP)
d['year']=d.index.year; d['month']=d.index.month
d=d.reset_index().merge(cap,on=['year','month'],how='left').set_index('timestamp')

dem=d.real_demand_jeju.values.astype(float)
base=d.jeju_est_demand_da.values.astype(float)
arr={c:d[c].values for c in WX+['total_cloud','midlow_cloud','cap_btmppa_mw']}
hour=d.index.hour.values; dow=d.index.dayofweek.values; month=d.index.month.values
dtype_arr=d.day_type.values.astype(object)
N=len(d)
rec24=pd.Series(dem).rolling(24,min_periods=24).mean().values
rec168=pd.Series(dem).rolling(168,min_periods=168).mean().values
H=np.arange(1,169)
origins=np.where((hour==23)&(np.arange(N)>=167)&(np.arange(N)<=N-1-168))[0]
P=origins
tgt=(P[:,None]+H[None,:]).ravel(); hh=np.broadcast_to(H,(len(P),168)).ravel()
def col(a): return a[tgt]
s=pd.DataFrame({
 'y':col(dem),'h':hh.astype(np.int16),'lag168':dem[tgt-168],
 'rec24':np.repeat(rec24[P],168),'rec168':np.repeat(rec168[P],168),
 'temp_c':col(arr['temp_c']),'humidity':col(arr['humidity']),'solar_rad':col(arr['solar_rad']),'wind_spd':col(arr['wind_spd']),
 'total_cloud':col(arr['total_cloud']),'midlow_cloud':col(arr['midlow_cloud']),
 'cap_btmppa_mw':col(arr['cap_btmppa_mw']),
 'hour':col(hour),'dow':col(dow),'month':col(month),'day_type':col(dtype_arr),'base':col(base),
 'tts':d.index.values[tgt]})
s['hour_sin']=np.sin(2*np.pi*s.hour/24); s['hour_cos']=np.cos(2*np.pi*s.hour/24)
s['dow_sin']=np.sin(2*np.pi*s.dow/7); s['dow_cos']=np.cos(2*np.pi*s.dow/7)
s['month_sin']=np.sin(2*np.pi*s.month/12); s['month_cos']=np.cos(2*np.pi*s.month/12)
# 구름 h<=48 마스킹 버전
s['total_cloud_48']=np.where(s.h<=48, s.total_cloud, np.nan)
s['midlow_cloud_48']=np.where(s.h<=48, s.midlow_cloud, np.nan)
s=s[s.y.notna()&s.lag168.notna()].reset_index(drop=True)
s['day_type']=s['day_type'].astype('category')
tts=pd.to_datetime(s.tts)
trm=s[tts<='2025-02-28 23:00']; vam=s[(tts>='2025-03-01')&(tts<='2026-03-21 23:00')]; tem=s[(tts>='2026-03-22')&(tts<='2026-05-31 23:00')]

BASE=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
CAT=['day_type']
params=dict(objective='regression_l1',metric='mae',learning_rate=0.024,num_leaves=244,
            min_data_in_leaf=76,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=5,
            lambda_l2=0.1,verbosity=-1,random_state=42)
def mape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); m=(a>0)&np.isfinite(a)&np.isfinite(p)
    return float(np.mean(np.abs(a[m]-p[m])/a[m])*100)
def run(feats,name):
    dtr=lgb.Dataset(trm[feats],trm.y,categorical_feature=[c for c in CAT if c in feats])
    dva=lgb.Dataset(vam[feats],vam.y,categorical_feature=[c for c in CAT if c in feats],reference=dtr)
    m=lgb.train(params,dtr,num_boost_round=4000,valid_sets=[dva],valid_names=['val'],
                callbacks=[lgb.early_stopping(150)])
    te=tem.copy(); te['pred']=m.predict(te[feats],num_iteration=m.best_iteration)
    te['da']=((te.h-1)//24+1)
    row={'model':name,'best':m.best_iteration}
    for dn in [1,2,3,7]:
        g=te[te.da==dn]; row[f'D+{dn}']=round(mape(g.y,g.pred),3)
    row['전체']=round(mape(te.y,te.pred),3)
    return row, m

variants=[
 (BASE,'baseline(2-A)'),
 (BASE+['total_cloud_48','midlow_cloud_48'],'+구름@h<=48'),
 (BASE+['cap_btmppa_mw'],'+BTM용량'),
 (BASE+['total_cloud_48','midlow_cloud_48','cap_btmppa_mw'],'+구름@h<=48+용량'),
 (BASE+['total_cloud','midlow_cloud'],'+구름@전지평(참고)'),
]
rows=[]; models={}
for feats,name in variants:
    r,m=run(feats,name); rows.append(r); models[name]=m
    print(f"{name:24s} best={r['best']:4d}  D+1={r['D+1']}  D+2={r['D+2']}  D+3={r['D+3']}  D+7={r['D+7']}  전체={r['전체']}")
res=pd.DataFrame(rows)
TAB=os.path.join(HERE,'tab'); os.makedirs(TAB,exist_ok=True)
res.to_csv(os.path.join(TAB,'2-A_exp_cloud_cap.csv'),index=False)
print('\nKPX 하루전 비교:', 'D+1', round(mape(tem[tem.h<=24].y,tem[tem.h<=24].base),3),
      '전체', round(mape(tem.y,tem.base),3))
# 채택 후보 모델 중요도(구름@h<=48+용량)
mm=models['+구름@h<=48+용량']
imp=pd.DataFrame({'f':mm.feature_name(),'gain':mm.feature_importance('gain')}).sort_values('gain',ascending=False)
imp['pct']=(imp.gain/imp.gain.sum()*100).round(1)
print('\n[+구름@h<=48+용량] 중요도 top:'); print(imp.head(10).to_string(index=False))
print('saved tab/2-A_exp_cloud_cap.csv')
