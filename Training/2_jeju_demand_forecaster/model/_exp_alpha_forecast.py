# -*- coding: utf-8 -*-
"""서빙(forecast) 기준으로 비대칭 alpha 재튜닝 (+QM on/off).

V5의 alpha=0.58은 실측기상 기준. 실서빙은 forecast라 흐린날 과소예측이 더 크다 →
forecast 조건에서 alpha를 다시 찾는다. 누수방지: alpha 선택은 forecast-VAL
(2025-12-13~2026-03-21, forecast 존재구간)로, 최종 보고는 forecast-TEST(2026-03-22~05-31).
모델은 실측기상으로 학습(forecast 이력이 짧아 학습 불가) — train/serve 미스매치를 alpha/QM로 흡수.
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
raw['temp_c']=raw[[f'temp_c_{s}' for s in ST]].mean(axis=1); raw['humidity']=raw[[f'humidity_{s}' for s in ST]].mean(axis=1)
raw['wind_spd']=raw[[f'wind_spd_{s}' for s in ST]].mean(axis=1); raw['solar_rad']=raw[[f'solar_rad_{s}' for s in SOLAR]].mean(axis=1)
raw=raw[['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_utilization_jeju']+WX+CLOUD]
idx=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
d=raw.set_index('timestamp').reindex(idx); d.index.name='timestamp'
d.loc[d.real_demand_jeju==0,'real_demand_jeju']=np.nan; d['real_demand_jeju']=d['real_demand_jeju'].interpolate(method='time')
for w in WX+CLOUD: d[w]=d[w].interpolate(method='time')
d['day_type']=d['day_type'].ffill().bfill(); d['real_solar_utilization_jeju']=d['real_solar_utilization_jeju'].interpolate(method='time')
cap=pd.read_csv(CAP); d['year']=d.index.year; d['mo']=d.index.month
d=d.reset_index().merge(cap,left_on=['year','mo'],right_on=['year','month'],how='left').set_index('timestamp')
F=pd.DataFrame(index=idx)
F['temp_c']=fc.set_index('timestamp')[['temp_west','temp_south']].mean(axis=1).reindex(idx)
F['humidity']=fc.set_index('timestamp')[['reh_west','reh_south']].mean(axis=1).reindex(idx)
F['wind_spd']=fc.set_index('timestamp')[['wind_spd_10m_west','wind_spd_10m_south']].mean(axis=1).reindex(idx)
F['solar_rad']=fc.set_index('timestamp')[['radiation_west','radiation_south']].mean(axis=1).reindex(idx)
for c in CLOUD: F[c]=fc.set_index('timestamp')[c].reindex(idx)
hour=d.index.hour.values; month=d.index.month.values; dow=d.index.dayofweek.values; N=len(d)
# QM 적합: forecast-val 구간(2025-12-13~2026-03-21)
fit_mask=(idx>='2025-12-13')&(idx<='2026-03-21 23:00'); QS=np.linspace(0,1,201)
Fc=F.copy()
for v in WX+CLOUD:
    if v=='solar_rad':
        for hh2 in range(24):
            mm=fit_mask&(hour==hh2); fok=F[v].values[mm]; hok=d[v].values[mm]
            fok=fok[np.isfinite(fok)]; hok=hok[np.isfinite(hok)]
            if len(fok)>20:
                fq=np.quantile(fok,QS); hq=np.quantile(hok,QS); sel=(hour==hh2)
                Fc.loc[sel,v]=np.interp(F[v].values[sel],fq,hq)
    else:
        fok=F[v].values[fit_mask]; hok=d[v].values[fit_mask]; fok=fok[np.isfinite(fok)]; hok=hok[np.isfinite(hok)]
        fq=np.quantile(fok,QS); hq=np.quantile(hok,QS); Fc[v]=np.interp(F[v].values,fq,hq)

dem=d.real_demand_jeju.values.astype(float); base=d.jeju_est_demand_da.values.astype(float); util=d.real_solar_utilization_jeju.values
clim_solar=pd.Series(d.solar_rad.values).groupby([month,hour]).transform('mean').values
def deficit(solar):
    with np.errstate(divide='ignore',invalid='ignore'): return np.where(clim_solar>5,np.clip(1-solar/clim_solar,-0.5,1.5),0.0)
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
s_act=build({k:d[k].values for k in WX+CLOUD}); s_fc=build({k:F[k].values for k in WX+CLOUD}); s_qm=build({k:Fc[k].values for k in WX+CLOUD})
keep=(s_act.y.notna()&s_act.lag168.notna()).values
for df in (s_act,s_fc,s_qm): df.drop(df.index[~keep],inplace=True); df.reset_index(drop=True,inplace=True); df['day_type']=df['day_type'].astype('category')
tts=pd.to_datetime(s_act.tts); s_act['is_day']=((s_act.hour>=8)&(s_act.hour<=16)).astype(int)
tr=s_act[tts<='2025-02-28 23:00']; va=s_act[(tts>='2025-03-01')&(tts<='2026-03-21 23:00')]
valfc_mask=((tts>='2025-12-13')&(tts<='2026-03-21 23:00')).values   # forecast-VAL(튜닝)
test_mask=((tts>='2026-03-22')&(tts<='2026-05-31 23:00')).values    # forecast-TEST(보고)
FEAT=['h','lag168','rec24','rec168','temp_c','humidity','solar_rad','wind_spd',
      'total_cloud_west','total_cloud_south','midlow_cloud_west','midlow_cloud_south','cap_btmppa_mw',
      'solar_deficit','solar_ramp','hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','day_type']
def mape(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=(a>0)&np.isfinite(a)&np.isfinite(p);return float(np.mean(np.abs(a[m]-p[m])/a[m])*100)
def mae(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=(a>0)&np.isfinite(a)&np.isfinite(p);return float(np.mean(np.abs(a[m]-p[m])))
def daymetrics(df,pred,mask):
    t=df[mask].copy(); t['pred']=pred[mask]
    dd=t[(t.hour>=8)&(t.hour<=16)].copy(); dd['uq']=pd.qcut(dd.util,4,labels=False,duplicates='drop')
    cl=dd[dd.uq==0]; cr=dd[dd.uq==3]
    da=((t.h-1)//24+1).values
    maer=[mae(t[da==n].y,t[da==n].pred)/mae(t[da==n].y,t[da==n].base) for n in [1,2,3]]
    return dict(흐림=mape(cl.y,cl.pred),맑음=mape(cr.y,cr.pred),낮=mape(dd.y,dd.pred),전체=mape(t.y,t.pred),maeD=max(maer))
def kpxday(df,mask):
    t=df[mask]; dd=t[(t.hour>=8)&(t.hour<=16)].copy(); dd['uq']=pd.qcut(dd.util,4,labels=False,duplicates='drop')
    return mape(dd[dd.uq==0].y,dd[dd.uq==0].base),mape(dd[dd.uq==3].y,dd[dd.uq==3].base)
kv_c,kv_r=kpxday(s_act,valfc_mask); kt_c,kt_r=kpxday(s_act,test_mask)
print('KPX  VAL 흐림 %.2f 맑음 %.2f | TEST 흐림 %.2f 맑음 %.2f'%(kv_c,kv_r,kt_c,kt_r))
print('\n%-22s | VAL흐림 VAL맑음 | TEST흐림 TEST맑음 TEST낮 TEST전체 MAE/KPX'%'(alpha,dayw,src)')
HP=dict(num_leaves=244,min_data_in_leaf=76,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=5,lambda_l2=0.1,verbosity=-1,random_state=42,learning_rate=0.024)
rows=[]
for alpha in [0.55,0.60,0.65,0.70,0.75]:
  for dayw in [2.0]:
    p=dict(HP,objective='quantile',alpha=alpha,metric='quantile')
    w=np.where(tr.is_day.values==1,dayw,1.0); wv=np.where(va.is_day.values==1,dayw,1.0)
    dtr=lgb.Dataset(tr[FEAT],tr.y,weight=w,categorical_feature=['day_type'])
    dva=lgb.Dataset(va[FEAT],va.y,weight=wv,categorical_feature=['day_type'],reference=dtr)
    m=lgb.train(p,dtr,num_boost_round=4000,valid_sets=[dva],valid_names=['val'],callbacks=[lgb.early_stopping(150)])
    for src,sdf in [('raw',s_fc),('QM',s_qm)]:
        pred=m.predict(sdf[FEAT],num_iteration=m.best_iteration)
        v=daymetrics(sdf,pred,valfc_mask); t=daymetrics(sdf,pred,test_mask)
        rows.append((alpha,dayw,src,v,t))
        flagc='✓' if t['흐림']<kt_c else 'x'; flagr='✓' if t['맑음']<kt_r else 'x'
        print('a=%.2f w=%g %-4s | %6.2f %6.2f | %6.2f%s %6.2f%s %6.2f %6.2f  %.2f'
              %(alpha,dayw,src,v['흐림'],v['맑음'],t['흐림'],flagc,t['맑음'],flagr,t['낮'],t['전체'],t['maeD']))
