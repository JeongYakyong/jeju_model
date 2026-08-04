# -*- coding: utf-8 -*-
"""현재 2-A 모델의 낮시간(08~16h) 정확도를 KPX 대비 따로 평가.
+ 흐림/맑음(이용률 하위/상위 25%)별 낮시간 정확도 + 일사 급변(|Δsolar| 상위) 정확도.
완전기상 기준. 채택 모델(lgbm_jeju_demand_direct.txt) 그대로 로드.
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
keep=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_utilization_jeju']+WX+CLOUD
raw=raw[keep]
idx=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
d=raw.set_index('timestamp').reindex(idx); d.index.name='timestamp'
d.loc[d.real_demand_jeju==0,'real_demand_jeju']=np.nan
d['real_demand_jeju']=d['real_demand_jeju'].interpolate(method='time')
for w in WX+CLOUD: d[w]=d[w].interpolate(method='time')
d['day_type']=d['day_type'].ffill().bfill()
d['real_solar_utilization_jeju']=d['real_solar_utilization_jeju'].interpolate(method='time')
cap=pd.read_csv(CAP); d['year']=d.index.year; d['mo']=d.index.month
d=d.reset_index().merge(cap,left_on=['year','mo'],right_on=['year','month'],how='left').set_index('timestamp')

dem=d.real_demand_jeju.values.astype(float); base=d.jeju_est_demand_da.values.astype(float)
util=d.real_solar_utilization_jeju.values
arr={c:d[c].values for c in WX+CLOUD+['cap_btmppa_mw']}
hour=d.index.hour.values; dow=d.index.dayofweek.values; month=d.index.month.values
dtype_arr=d.day_type.values.astype(object); N=len(d)
rec24=pd.Series(dem).rolling(24,min_periods=24).mean().values
rec168=pd.Series(dem).rolling(168,min_periods=168).mean().values
dsolar=np.abs(np.diff(arr['solar_rad'],prepend=np.nan))   # |Δ일사| 급변 지표
H=np.arange(1,169)
origins=np.where((hour==23)&(np.arange(N)>=167)&(np.arange(N)<=N-1-168))[0]
tgt=(origins[:,None]+H[None,:]).ravel(); hh=np.broadcast_to(H,(len(origins),168)).ravel()
def col(a): return a[tgt]
s=pd.DataFrame({'y':col(dem),'h':hh.astype(int),'lag168':dem[tgt-168],
 'rec24':np.repeat(rec24[origins],168),'rec168':np.repeat(rec168[origins],168),
 'temp_c':col(arr['temp_c']),'humidity':col(arr['humidity']),'solar_rad':col(arr['solar_rad']),'wind_spd':col(arr['wind_spd']),
 'cap_btmppa_mw':col(arr['cap_btmppa_mw']),
 'hour':col(hour),'dow':col(dow),'month':col(month),'day_type':col(dtype_arr),
 'base':col(base),'util':col(util),'dsolar':col(dsolar),'tts':d.index.values[tgt]})
for c in CLOUD: s[c]=np.where(s.h<=48,col(arr[c]),np.nan)
s['hour_sin']=np.sin(2*np.pi*s.hour/24); s['hour_cos']=np.cos(2*np.pi*s.hour/24)
s['dow_sin']=np.sin(2*np.pi*s.dow/7); s['dow_cos']=np.cos(2*np.pi*s.dow/7)
s['month_sin']=np.sin(2*np.pi*s.month/12); s['month_cos']=np.cos(2*np.pi*s.month/12)
s=s[s.y.notna()&s.lag168.notna()].reset_index(drop=True)
s['day_type']=s['day_type'].astype('category')
tts=pd.to_datetime(s.tts); te=s[(tts>='2026-03-22')&(tts<='2026-05-31 23:00')].copy()
FEAT=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south','cap_btmppa_mw',
      'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
m=lgb.Booster(model_file=os.path.join(HERE,'models','lgbm_jeju_demand_direct.txt'))
te['pred']=m.predict(te[FEAT])
te['da']=((te.h-1)//24+1)
def mape(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);msk=(a>0)&np.isfinite(a)&np.isfinite(p)
    return float(np.mean(np.abs(a[msk]-p[msk])/a[msk])*100)
def mae(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);msk=(a>0)&np.isfinite(a)&np.isfinite(p)
    return float(np.mean(np.abs(a[msk]-p[msk])))
day=te[(te.hour>=8)&(te.hour<=16)]
print('=== 전체시간 vs 낮시간(08~16h) — 모델 vs KPX (test, 완전기상) ===')
print('%-8s %12s %12s %12s %12s'%('지평','모델_전시간','KPX_전시간','모델_낮','KPX_낮'))
for dn in [1,2,3,7]:
    g=te[te.da==dn]; gd=day[day.da==dn]
    print('D+%-6d %12.3f %12.3f %12.3f %12.3f'%(dn,mape(g.y,g.pred),mape(g.y,g.base),mape(gd.y,gd.pred),mape(gd.y,gd.base)))
print('%-8s %12.3f %12.3f %12.3f %12.3f'%('전체',mape(te.y,te.pred),mape(te.y,te.base),mape(day.y,day.pred),mape(day.y,day.base)))
print('\n=== 낮시간 흐림/맑음(이용률 4분위) — 모델 vs KPX (MAPE / 평균오차편향) ===')
day=day.copy(); day['q']=pd.qcut(day.util,4,labels=False,duplicates='drop')
for q,lab in [(0,'흐림(하위25%)'),(3,'맑음(상위25%)')]:
    gg=day[day.q==q]
    bias_m=(gg.pred-gg.y).mean(); bias_k=(gg.base-gg.y).mean()
    print('%-12s 모델 MAPE %.2f%% (편향 %+.1fMW) | KPX MAPE %.2f%% (편향 %+.1fMW, −=과소예측)'
          %(lab,mape(gg.y,gg.pred),bias_m,mape(gg.y,gg.base),bias_k))
print('\n=== 낮시간 일사급변(|Δsolar| 상위25%) — 모델 vs KPX ===')
dd=day[day.dsolar>=day.dsolar.quantile(0.75)]
print('급변시각 n=%d  모델 MAPE %.2f%% | KPX MAPE %.2f%%'%(len(dd),mape(dd.y,dd.pred),mape(dd.y,dd.base)))
