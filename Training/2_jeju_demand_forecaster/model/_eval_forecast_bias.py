# -*- coding: utf-8 -*-
"""forecast 기상 bias 점검 + quantile-mapping 보정 후 V5 재평가.

LGBM은 단조변환 불변이지만 학습 split 임계값이 historical 단위라, forecast 분포가
다르면 정확도가 깨진다 → forecast를 historical 주변분포로 정렬(QM).
QM 적합 구간 = test 이전 겹침(2025-12-13~2026-03-21), test(2026-03-22~)에 적용(누수방지).
solar는 시간별 QM, 그외(temp/hum/wind/cloud)는 전역 QM.
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
# forecast 원본 정렬
F=pd.DataFrame(index=idx)
F['temp_c']=fc.set_index('timestamp')[['temp_west','temp_south']].mean(axis=1).reindex(idx)
F['humidity']=fc.set_index('timestamp')[['reh_west','reh_south']].mean(axis=1).reindex(idx)
F['wind_spd']=fc.set_index('timestamp')[['wind_spd_10m_west','wind_spd_10m_south']].mean(axis=1).reindex(idx)
F['solar_rad']=fc.set_index('timestamp')[['radiation_west','radiation_south']].mean(axis=1).reindex(idx)
for c in CLOUD: F[c]=fc.set_index('timestamp')[c].reindex(idx)

# ---- QM 적합: 겹침-train(2025-12-13~2026-03-21) ----
hour=d.index.hour.values
fit_mask=(idx>='2025-12-13')&(idx<='2026-03-21 23:00')
QS=np.linspace(0,1,201)
def qm_global(fvar,hvar):
    fok=F[fvar].values[fit_mask]; hok=d[hvar].values[fit_mask]
    fok=fok[np.isfinite(fok)]; hok=hok[np.isfinite(hok)]
    fq=np.quantile(fok,QS); hq=np.quantile(hok,QS)
    return lambda x: np.interp(x,fq,hq)
def qm_hourly(fvar,hvar):
    maps={}
    for hh in range(24):
        mm=fit_mask&(hour==hh)
        fok=F[fvar].values[mm]; hok=d[hvar].values[mm]
        fok=fok[np.isfinite(fok)]; hok=hok[np.isfinite(hok)]
        if len(fok)>20 and len(hok)>20: maps[hh]=(np.quantile(fok,QS),np.quantile(hok,QS))
    def ap(x,hr):
        out=np.array(x,float).copy()
        for hh,(fq,hq) in maps.items():
            m=hr==hh; out[m]=np.interp(out[m],fq,hq)
        return out
    return ap
# 보정 forecast 생성 (전 구간)
Fc=F.copy()
print('=== 기상 bias 점검 + QM 보정 (겹침-train 적합) ===')
print('%-13s %8s %8s %8s %8s'%('var','hist','fcst_raw','fcst_QM','corr'))
hr_all=hour
for v in WX+CLOUD:
    if v=='solar_rad':
        ap=qm_hourly('solar_rad','solar_rad'); Fc[v]=ap(F[v].values,hr_all)
    else:
        ap=qm_global(v,v); Fc[v]=ap(F[v].values)
    tm=(idx>='2026-03-22')&(idx<='2026-05-31 23:00')  # test 구간 분포 비교
    print('%-13s %8.3f %8.3f %8.3f %8.3f'%(v, d[v].values[tm].mean(), F[v].values[tm].mean(), Fc[v].values[tm].mean(),
          pd.Series(F[v].values[tm]).corr(pd.Series(d[v].values[tm]))))

# ---- 샘플 구성 + V5 학습(실측) + 3종 평가 ----
month=d.index.month.values; dow=d.index.dayofweek.values; N=len(d)
dem=d.real_demand_jeju.values.astype(float); base=d.jeju_est_demand_da.values.astype(float); util=d.real_solar_utilization_jeju.values
clim_solar=pd.Series(d.solar_rad.values).groupby([month,hour]).transform('mean').values
def deficit(solar):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(clim_solar>5,np.clip(1-solar/clim_solar,-0.5,1.5),0.0)
def ramp(solar): return np.abs(np.diff(solar,prepend=np.nan))
rec24=pd.Series(dem).rolling(24,min_periods=24).mean().values; rec168=pd.Series(dem).rolling(168,min_periods=168).mean().values
dtype_arr=d.day_type.values.astype(object)
H=np.arange(1,169); origins=np.where((hour==23)&(np.arange(N)>=167)&(np.arange(N)<=N-1-168))[0]
tgt=(origins[:,None]+H[None,:]).ravel(); hh=np.broadcast_to(H,(len(origins),168)).ravel()
def col(a): return a[tgt]
def build(src):
    df=pd.DataFrame({'y':col(dem),'h':hh.astype(int),'lag168':dem[tgt-168],
        'rec24':np.repeat(rec24[origins],168),'rec168':np.repeat(rec168[origins],168),
        'temp_c':col(src['temp_c']),'humidity':col(src['humidity']),'solar_rad':col(src['solar_rad']),'wind_spd':col(src['wind_spd']),
        'cap_btmppa_mw':col(d.cap_btmppa_mw.values),'hour':col(hour),'dow':col(dow),'month':col(month),
        'day_type':col(dtype_arr),'base':col(base),'util':col(util),'tts':d.index.values[tgt]})
    for c in CLOUD: df[c]=np.where(df.h<=48,col(src[c]),np.nan)
    df['solar_deficit']=np.where(df.h<=48,col(deficit(src['solar_rad'])),np.nan)
    df['solar_ramp']=np.where(df.h<=48,col(ramp(src['solar_rad'])),np.nan)
    df['hour_sin']=np.sin(2*np.pi*df.hour/24); df['hour_cos']=np.cos(2*np.pi*df.hour/24)
    df['dow_sin']=np.sin(2*np.pi*df.dow/7); df['dow_cos']=np.cos(2*np.pi*df.dow/7)
    df['month_sin']=np.sin(2*np.pi*df.month/12); df['month_cos']=np.cos(2*np.pi*df.month/12)
    return df
s_act=build({k:d[k].values for k in WX+CLOUD})
s_fc =build({k:F[k].values for k in WX+CLOUD})
s_qm =build({k:Fc[k].values for k in WX+CLOUD})
# QM 비-구름만(temp/hum/wind/solar 보정, 구름은 raw 유지)
src_nc={k:Fc[k].values for k in WX};
for c in CLOUD: src_nc[c]=F[c].values
s_qm2=build(src_nc)
keep=(s_act.y.notna()&s_act.lag168.notna()).values
for df in (s_act,s_fc,s_qm,s_qm2):
    df.drop(df.index[~keep],inplace=True); df.reset_index(drop=True,inplace=True); df['day_type']=df['day_type'].astype('category')
tts=pd.to_datetime(s_act.tts)
s_act['is_day']=((s_act.hour>=8)&(s_act.hour<=16)).astype(int)
tr=s_act[tts<='2025-02-28 23:00']; va=s_act[(tts>='2025-03-01')&(tts<='2026-03-21 23:00')]
te_mask=((tts>='2026-03-22')&(tts<='2026-05-31 23:00')).values
FEAT=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south','cap_btmppa_mw',
      'solar_deficit','solar_ramp','hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
params=dict(objective='quantile',alpha=0.58,metric='quantile',num_leaves=244,min_data_in_leaf=76,
            feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=5,lambda_l2=0.1,verbosity=-1,random_state=42,learning_rate=0.024)
w=np.where(tr.is_day.values==1,2.0,1.0); wv=np.where(va.is_day.values==1,2.0,1.0)
dtr=lgb.Dataset(tr[FEAT],tr.y,weight=w,categorical_feature=['day_type'])
dva=lgb.Dataset(va[FEAT],va.y,weight=wv,categorical_feature=['day_type'],reference=dtr)
model=lgb.train(params,dtr,num_boost_round=4000,valid_sets=[dva],valid_names=['val'],callbacks=[lgb.early_stopping(150)])
bi=model.best_iteration
def mape(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=(a>0)&np.isfinite(a)&np.isfinite(p);return float(np.mean(np.abs(a[m]-p[m])/a[m])*100)
def daytab(df):
    te=df[te_mask].copy(); te['pred']=model.predict(te[FEAT],num_iteration=bi)
    dd=te[(te.hour>=8)&(te.hour<=16)].copy(); dd['uq']=pd.qcut(dd.util,4,labels=False,duplicates='drop')
    cl=dd[dd.uq==0]; cr=dd[dd.uq==3]
    return mape(dd.y,dd.pred),mape(cl.y,cl.pred),(cl.pred-cl.y).mean(),mape(cr.y,cr.pred),mape(te.y,te.pred)
tek=s_act[te_mask].copy(); ddk=tek[(tek.hour>=8)&(tek.hour<=16)].copy(); ddk['uq']=pd.qcut(ddk.util,4,labels=False,duplicates='drop')
print('\n=== V5 낮시간(08~16h) MAPE: actual vs forecast-raw vs forecast-QM ===  best_iter',bi)
print('%-18s %8s %8s %8s %8s'%('','낮전체','낮흐림','낮맑음','전체'))
print('%-18s %8.2f %8.2f %8.2f %8.2f'%('KPX',mape(ddk.y,ddk.base),mape(ddk[ddk.uq==0].y,ddk[ddk.uq==0].base),mape(ddk[ddk.uq==3].y,ddk[ddk.uq==3].base),mape(tek.y,tek.base)))
for nm,df in [('V5 actual',s_act),('V5 forecast-raw',s_fc),('V5 forecast-QM(전체)',s_qm),('V5 QM(구름제외)',s_qm2)]:
    a,b,bias,c,al=daytab(df); print('%-20s %8.2f %8.2f %8.2f %8.2f   (흐림편향 %+.1fMW)'%(nm,a,b,c,al,bias))
