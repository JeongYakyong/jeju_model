# -*- coding: utf-8 -*-
"""V5(채택) 모델을 실측기상 vs forecast기상 둘 다로 test 평가 (실서빙 정확도 비교).

V5 = quantile(alpha=0.58) + 낮시간 가중2 + 흐린날피처(solar_deficit, solar_ramp).
학습은 실측기상(historical)으로, 평가는 (a)실측기상 (b)forecast기상 두 가지.
forecast 매핑: temp←temp / humidity←reh / wind←wind_spd_10m / solar←radiation / cloud←cloud (west·south).
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
con=sqlite3.connect(DB)
pull=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_utilization_jeju']
for s in ST:
    for w in ['temp_c','humidity','wind_spd']: pull.append(f'{w}_{s}')
for s in SOLAR: pull.append(f'solar_rad_{s}')
pull+=CLOUD
raw=pd.read_sql(f"SELECT {','.join(pull)} FROM historical",con,parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
# forecast 기상
fc=pd.read_sql("SELECT timestamp, temp_west,temp_south, reh_west,reh_south, wind_spd_10m_west,wind_spd_10m_south, radiation_west,radiation_south, total_cloud_west,total_cloud_south, midlow_cloud_west,midlow_cloud_south FROM forecast",con,parse_dates=['timestamp'])
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
# forecast 기상을 동일 인덱스로 정렬
F=pd.DataFrame(index=idx)
F['temp_c']=fc.set_index('timestamp')[['temp_west','temp_south']].mean(axis=1).reindex(idx)
F['humidity']=fc.set_index('timestamp')[['reh_west','reh_south']].mean(axis=1).reindex(idx)
F['wind_spd']=fc.set_index('timestamp')[['wind_spd_10m_west','wind_spd_10m_south']].mean(axis=1).reindex(idx)
F['solar_rad']=fc.set_index('timestamp')[['radiation_west','radiation_south']].mean(axis=1).reindex(idx)
for c in CLOUD: F[c]=fc.set_index('timestamp')[c].reindex(idx)

hour=d.index.hour.values; dow=d.index.dayofweek.values; month=d.index.month.values; N=len(d)
dem=d.real_demand_jeju.values.astype(float); base=d.jeju_est_demand_da.values.astype(float); util=d.real_solar_utilization_jeju.values
clim_solar=pd.Series(d.solar_rad.values).groupby([month,hour]).transform('mean').values
def deficit(solar):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(clim_solar>5, np.clip(1-solar/clim_solar,-0.5,1.5),0.0)
def ramp(solar): return np.abs(np.diff(solar,prepend=np.nan))
rec24=pd.Series(dem).rolling(24,min_periods=24).mean().values; rec168=pd.Series(dem).rolling(168,min_periods=168).mean().values
dtype_arr=d.day_type.values.astype(object)
H=np.arange(1,169); origins=np.where((hour==23)&(np.arange(N)>=167)&(np.arange(N)<=N-1-168))[0]
tgt=(origins[:,None]+H[None,:]).ravel(); hh=np.broadcast_to(H,(len(origins),168)).ravel()
def col(a): return a[tgt]
def build(weather_src):
    # weather_src: dict of arrays (temp_c,humidity,solar_rad,wind_spd, CLOUD..) + deficit/ramp from its solar
    df=pd.DataFrame({'y':col(dem),'h':hh.astype(int),'lag168':dem[tgt-168],
        'rec24':np.repeat(rec24[origins],168),'rec168':np.repeat(rec168[origins],168),
        'temp_c':col(weather_src['temp_c']),'humidity':col(weather_src['humidity']),
        'solar_rad':col(weather_src['solar_rad']),'wind_spd':col(weather_src['wind_spd']),
        'cap_btmppa_mw':col(d.cap_btmppa_mw.values),
        'hour':col(hour),'dow':col(dow),'month':col(month),'day_type':col(dtype_arr),
        'base':col(base),'util':col(util),'tts':d.index.values[tgt]})
    for c in CLOUD: df[c]=np.where(df.h<=48,col(weather_src[c]),np.nan)
    df['solar_deficit']=np.where(df.h<=48,col(deficit(weather_src['solar_rad'])),np.nan)
    df['solar_ramp']=np.where(df.h<=48,col(ramp(weather_src['solar_rad'])),np.nan)
    df['hour_sin']=np.sin(2*np.pi*df.hour/24); df['hour_cos']=np.cos(2*np.pi*df.hour/24)
    df['dow_sin']=np.sin(2*np.pi*df.dow/7); df['dow_cos']=np.cos(2*np.pi*df.dow/7)
    df['month_sin']=np.sin(2*np.pi*df.month/12); df['month_cos']=np.cos(2*np.pi*df.month/12)
    return df
W_act={k:d[k].values for k in WX+CLOUD}
W_fc ={k:F[k].values for k in WX+CLOUD}
s_act=build(W_act); s_fc=build(W_fc)
# 학습은 실측기상으로
m_act=s_act.copy(); tts=pd.to_datetime(m_act.tts)
keep=m_act.y.notna()&m_act.lag168.notna()
m_act=m_act[keep].reset_index(drop=True); tts=tts[keep].reset_index(drop=True)
m_act['day_type']=m_act['day_type'].astype('category')
s_fc=s_fc[keep.values].reset_index(drop=True); s_fc['day_type']=s_fc['day_type'].astype('category')
m_act['is_day']=((m_act.hour>=8)&(m_act.hour<=16)).astype(int)
tr=m_act[tts<='2025-02-28 23:00']; va=m_act[(tts>='2025-03-01')&(tts<='2026-03-21 23:00')]
te_mask=((tts>='2026-03-22')&(tts<='2026-05-31 23:00')).values
FEAT=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south','cap_btmppa_mw',
      'solar_deficit','solar_ramp','hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
CAT=['day_type']
params=dict(objective='quantile',alpha=0.58,metric='quantile',num_leaves=244,min_data_in_leaf=76,
            feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=5,lambda_l2=0.1,verbosity=-1,
            random_state=42,learning_rate=0.024)
w=np.where(tr.is_day.values==1,2.0,1.0); wv=np.where(va.is_day.values==1,2.0,1.0)
dtr=lgb.Dataset(tr[FEAT],tr.y,weight=w,categorical_feature=CAT)
dva=lgb.Dataset(va[FEAT],va.y,weight=wv,categorical_feature=CAT,reference=dtr)
model=lgb.train(params,dtr,num_boost_round=4000,valid_sets=[dva],valid_names=['val'],callbacks=[lgb.early_stopping(150)])
bi=model.best_iteration
def mape(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);msk=(a>0)&np.isfinite(a)&np.isfinite(p);return float(np.mean(np.abs(a[msk]-p[msk])/a[msk])*100)
def mae(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);msk=(a>0)&np.isfinite(a)&np.isfinite(p);return float(np.mean(np.abs(a[msk]-p[msk])))
te_act=m_act[te_mask].copy(); te_fc=s_fc[te_mask].copy()
te_act['pred']=model.predict(te_act[FEAT],num_iteration=bi)
te_fc['pred']=model.predict(te_fc[FEAT],num_iteration=bi)
da=((te_act.h-1)//24+1).values
print('V5 best_iter',bi,'  test n',len(te_act))
print('\n=== D+1~3 전시간 MAE ÷ KPX (제약 <=1.05) ===')
for dn in [1,2,3]:
    g=te_act[da==dn]; gf=te_fc[((te_fc.h-1)//24+1)==dn]
    print(' D+%d  실측 %.3f  forecast %.3f'%(dn, mae(g.y,g.pred)/mae(g.y,g.base), mae(gf.y,gf.pred)/mae(gf.y,gf.base)))
def daytable(te):
    dd=te[(te.hour>=8)&(te.hour<=16)].copy(); dd['uq']=pd.qcut(dd.util,4,labels=False,duplicates='drop')
    cl=dd[dd.uq==0]; cr=dd[dd.uq==3]
    return mape(dd.y,dd.pred),mape(cl.y,cl.pred),(cl.pred-cl.y).mean(),mape(cr.y,cr.pred),mape(te.y,te.pred)
da1,cl1,b1,cr1,all1=daytable(te_act); da2,cl2,b2,cr2,all2=daytable(te_fc)
ddk=te_act[(te_act.hour>=8)&(te_act.hour<=16)].copy(); ddk['uq']=pd.qcut(ddk.util,4,labels=False,duplicates='drop')
kday=mape(ddk.y,ddk.base); kcl=mape(ddk[ddk.uq==0].y,ddk[ddk.uq==0].base); kcr=mape(ddk[ddk.uq==3].y,ddk[ddk.uq==3].base)
print('\n=== 낮시간(08~16h) MAPE 비교 ===')
print('%-16s %8s %8s %8s %8s'%('','낮전체','낮흐림','낮맑음','전체'))
print('%-16s %8.2f %8.2f %8.2f %8.2f'%('KPX', kday,kcl,kcr, mape(te_act.y,te_act.base)))
print('%-16s %8.2f %8.2f %8.2f %8.2f'%('V5 실측기상', da1,cl1,cr1, all1))
print('%-16s %8.2f %8.2f %8.2f %8.2f'%('V5 forecast기상', da2,cl2,cr2, all2))
print('\n낮흐림 편향(MW): V5실측 %+.1f / V5forecast %+.1f (KPX 기준 흐림 과소예측)'%(b1,b2))
TAB=os.path.join(HERE,'tab'); os.makedirs(TAB,exist_ok=True)
pd.DataFrame([
 dict(src='KPX',낮전체=round(kday,2),낮흐림=round(kcl,2),낮맑음=round(kcr,2),전체=round(mape(te_act.y,te_act.base),2)),
 dict(src='V5_actual',낮전체=round(da1,2),낮흐림=round(cl1,2),낮맑음=round(cr1,2),전체=round(all1,2)),
 dict(src='V5_forecast',낮전체=round(da2,2),낮흐림=round(cl2,2),낮맑음=round(cr2,2),전체=round(all2,2)),
]).to_csv(os.path.join(TAB,'2-A_v5_actual_vs_forecast.csv'),index=False)
print('saved tab/2-A_v5_actual_vs_forecast.csv')
