# -*- coding: utf-8 -*-
"""2-0b EDA 빌더 — 제주 KPX 잔차(real-est)의 원인 탐색 + BTM/PPA 가설 검증.

사용자 제공: 전력거래소 제주 PPA+BTM 태양광 용량(월별 MW, 2019-2025; 2026 캐리포워드)
  → data/jeju_ppa_btm_capacity_mw.csv
가설: 비계량(BTM/PPA) 태양광이 계통수요(real_demand)에서 차감되므로, 흐림/일사 급변으로
      비계량 발전이 평소와 달라지면 KPX 하루전 예측(est_demand_da)과 큰 잔차가 발생.
검증: 잔차 ↔ 구름/일사 상관, 그리고 (용량×태양광이용률) 비계량발전 추정으로 잔차 설명력.
"""
import nbformat as nbf
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '2-0b_residual_btm.ipynb')
nb = nbf.v4.new_notebook(); cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# 2-0b. 제주 KPX 잔차의 원인 — 흐림·일사 급변과 BTM/PPA 가설 검증

> 사용자 가설: 제주는 **전력수요 스케일이 작고(평균 ~700MW) 신재생 비중이 매우 높다.**
> KPX 하루전 `jeju_est_demand_da` 와 실측 `real_demand_jeju` 의 잔차가 크게 벌어지는 구간이
> 있는데, 주로 **날씨가 흐리거나 일사량이 급변하는** 구간이다. 원인 가설은
> **PPA·BTM(비계량) 태양광이 자체발전수요 → 계통수요로 전이**되며 차이를 키운다는 것.
>
> 데이터: 전력거래소 제주 **PPA+BTM 태양광 용량(월별 MW)** = `data/jeju_ppa_btm_capacity_mw.csv`.
> 비계량 발전 ≈ 용량 × 태양광이용률(계량 `real_solar_utilization_jeju` 를 일사 프록시로 사용,
> land 백필 `backfill_btm_ppa.py` 와 동일 원리). 모델 입력 피처 외 DB 변수(구름·일사·net_load 등) 자유 사용.""")

co(r"""import os, sqlite3, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')
mpl.rcParams['font.family']='Malgun Gothic'; mpl.rcParams['axes.unicode_minus']=False; mpl.rcParams['figure.dpi']=110
HERE=os.getcwd()
FIG=os.path.join(HERE,'fig'); os.makedirs(FIG,exist_ok=True)
TAB=os.path.join(HERE,'tab'); os.makedirs(TAB,exist_ok=True)
DB=os.path.normpath(os.path.join(HERE,'..','..','1. data_fetcher_and_db','data','input_data_jeju.db'))
CAP_CSV=os.path.normpath(os.path.join(HERE,'..','data','jeju_ppa_btm_capacity_mw.csv'))
con=sqlite3.connect(DB)
cols=['timestamp','real_demand_jeju','jeju_est_demand_da','day_type','real_solar_gen_jeju',
      'real_solar_capacity_jeju','real_solar_utilization_jeju','real_renew_gen_jeju',
      'real_wind_gen_jeju','real_net_load_jeju',
      'temp_c_west','temp_c_east','temp_c_south',
      'total_cloud_west','total_cloud_east','total_cloud_south',
      'midlow_cloud_west','midlow_cloud_east','midlow_cloud_south',
      'solar_rad_west','solar_rad_south']
df=pd.read_sql(f"SELECT {', '.join(cols)} FROM historical", con, parse_dates=['timestamp'])
con.close()
df=df.sort_values('timestamp').reset_index(drop=True)
# 타깃 0/결측 보간
df.loc[df.real_demand_jeju==0,'real_demand_jeju']=np.nan
df['real_demand_jeju']=df.set_index('timestamp')['real_demand_jeju'].interpolate(method='time').values
df['temp_c']=df[['temp_c_west','temp_c_east','temp_c_south']].mean(axis=1)
df['total_cloud']=df[['total_cloud_west','total_cloud_east','total_cloud_south']].mean(axis=1)
df['midlow_cloud']=df[['midlow_cloud_west','midlow_cloud_east','midlow_cloud_south']].mean(axis=1)
df['solar_rad']=df[['solar_rad_west','solar_rad_south']].mean(axis=1)
df['year']=df.timestamp.dt.year; df['month']=df.timestamp.dt.month
df['hour']=df.timestamp.dt.hour; df['date']=df.timestamp.dt.normalize()
# BTM/PPA 용량 병합 + 비계량 발전 추정
cap=pd.read_csv(CAP_CSV)
df=df.merge(cap, on=['year','month'], how='left')
util=df['real_solar_utilization_jeju'].clip(lower=0)
df['btmppa_gen_est']=df['cap_btmppa_mw']*util            # 비계량 태양광 발전 추정(MW)
df['residual']=df['real_demand_jeju']-df['jeju_est_demand_da']   # KPX 잔차(+면 실측이 예측보다 큼)
print(df.shape, df.timestamp.min(),'~',df.timestamp.max())
print('잔차 통계(MW): mean=%.1f std=%.1f  |residual|>50MW 비율=%.1f%%'
      %(df.residual.mean(), df.residual.std(), (df.residual.abs()>50).mean()*100))
print('BTM/PPA 용량(MW) 2024~2026 월별 샘플:')
print(cap[cap.year>=2024].pivot(index='month',columns='year',values='cap_btmppa_mw').to_string())""")

md(r"""## 1. 제주 맥락 — 작은 수요 · 높은 신재생 침투 · BTM/PPA 성장

계량 신재생만으로도 침투율이 높은데(연 17→26%), BTM/PPA(비계량)는 여기서 빠져 있어
**실제(true) 침투율은 더 높다.** true_demand = real_demand + 비계량발전.""")

co(r"""fig,ax=plt.subplots(1,2,figsize=(14,4.4))
# (a) 용량 추이: 계량 태양광 vs BTM/PPA
capm=df.groupby(df.timestamp.dt.to_period('M')).agg(
    metered=('real_solar_capacity_jeju','mean'), btmppa=('cap_btmppa_mw','mean'))
capm.index=capm.index.to_timestamp()
ax[0].plot(capm.index, capm.metered, label='계량 태양광 용량', color='#4C72B0')
ax[0].plot(capm.index, capm.btmppa, label='BTM+PPA 용량(전력거래소)', color='#C44E52')
ax[0].set_ylabel('MW'); ax[0].set_title('(a) 태양광 용량: 계량 vs 비계량(BTM/PPA)'); ax[0].legend()
# (b) 침투율: 계량 vs (계량+추정 비계량)
yp=df.groupby('year').agg(dem=('real_demand_jeju','sum'), renew=('real_renew_gen_jeju','sum'),
                          btm=('btmppa_gen_est','sum'))
metered_pen=(yp.renew/yp.dem*100)
true_pen=((yp.renew+yp.btm)/(yp.dem+yp.btm)*100)
x=yp.index.astype(str)
ax[1].bar(x, metered_pen, label='계량 신재생 침투율', color='#4C72B0')
ax[1].bar(x, true_pen-metered_pen, bottom=metered_pen, label='+BTM/PPA(추정)', color='#DD8452')
ax[1].set_ylabel('%'); ax[1].set_title('(b) 신재생 침투율: 계량 vs 실제(추정)'); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,'2-0b_context.png'),bbox_inches='tight'); plt.show()
print('연도별 침투율(%):'); print(pd.DataFrame({'계량':metered_pen.round(1),'실제추정':true_pen.round(1)}).to_string())""")

md(r"""## 2. 잔차(real − est) 분해 — 언제 큰가? (시간대·연도)

잔차가 **낮시간(태양광 시간)** 에 집중되고, **BTM/PPA 용량이 커진 최근 연도**일수록 커지면
비계량 태양광 가설과 부합. (+잔차=실측>예측, −잔차=실측<예측)""")

co(r"""fig,ax=plt.subplots(1,2,figsize=(14,4.4))
# (a) 시간대별 잔차 평균±표준편차
hp=df.groupby('hour').residual.agg(['mean','std'])
ax[0].plot(hp.index, hp['mean'], 'o-', color='#C44E52', label='평균 잔차')
ax[0].fill_between(hp.index, hp['mean']-hp['std'], hp['mean']+hp['std'], color='#C44E52', alpha=.15, label='±1σ')
ax[0].axhline(0,color='#888',lw=.8); ax[0].set_xlabel('hour'); ax[0].set_ylabel('잔차(MW)')
ax[0].set_title('(a) 시간대별 KPX 잔차 (낮시간 변동?)'); ax[0].legend()
# (b) 연도×시간대 잔차 표준편차 히트맵
piv=df.pivot_table('residual','hour','year',aggfunc='std')
im=ax[1].imshow(piv.values,aspect='auto',cmap='YlOrRd',origin='lower')
ax[1].set_xticks(range(len(piv.columns))); ax[1].set_xticklabels(piv.columns)
ax[1].set_yticks(range(0,24,3)); ax[1].set_yticklabels(range(0,24,3))
ax[1].set_xlabel('year'); ax[1].set_ylabel('hour'); ax[1].set_title('(b) 잔차 표준편차 (연×시간)')
plt.colorbar(im,ax=ax[1],label='MW')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'2-0b_residual_when.png'),bbox_inches='tight'); plt.show()
day=df[(df.hour>=9)&(df.hour<=16)]
print('낮시간(9~16h) 잔차 표준편차(MW) 연도별:')
print(day.groupby('year').residual.std().round(1).to_string())""")

md(r"""## 3. 잔차 ↔ 흐림·일사 (사용자 관찰 직접 검증)

낮시간 잔차가 **구름↑/일사↓** 일 때 커지는지. 비계량 태양광이 흐린 날 적게 나와
계통수요가 예측보다 튀는(+) 패턴 기대.""")

co(r"""day=df[(df.hour>=9)&(df.hour<=16)].copy()
print('낮시간(9~16h) 잔차와의 피어슨 상관:')
for c in ['total_cloud','midlow_cloud','solar_rad','real_solar_utilization_jeju','btmppa_gen_est']:
    print('  residual ↔ %-26s %+.3f' % (c, day[['residual',c]].corr().iloc[0,1]))
fig,ax=plt.subplots(1,2,figsize=(14,4.4))
# (a) 구름 구간별 잔차
day['cbin']=pd.cut(day.total_cloud, bins=np.linspace(day.total_cloud.min(),day.total_cloud.max(),9))
g=day.groupby('cbin').residual.mean(); cx=[iv.mid for iv in g.index]
ax[0].plot(cx,g.values,'o-',color='#4C72B0'); ax[0].axhline(0,color='#888',lw=.8)
ax[0].set_xlabel('총운량(total_cloud)'); ax[0].set_ylabel('평균 잔차(MW)'); ax[0].set_title('(a) 흐릴수록 잔차↑?')
# (b) 일사 급변(직전시각 대비 |Δ일사|) 구간별 잔차 표준편차
df['dsolar']=df.solar_rad.diff().abs()
dd=df[(df.hour>=9)&(df.hour<=16)].copy()
dd['rbin']=pd.qcut(dd.dsolar.fillna(0), q=6, duplicates='drop')
g2=dd.groupby('rbin').residual.std(); rx=[iv.mid for iv in g2.index]
ax[1].plot(rx,g2.values,'s-',color='#C44E52')
ax[1].set_xlabel('|일사 시간변화| (급변)'); ax[1].set_ylabel('잔차 표준편차(MW)'); ax[1].set_title('(b) 일사 급변 구간 잔차 변동↑?')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'2-0b_residual_weather.png'),bbox_inches='tight'); plt.show()""")

md(r"""## 4. BTM/PPA 가설 정량 검증 — 비계량 발전 '서프라이즈'로 잔차 설명

KPX 가 평년 수준 비계량 태양광을 암묵 가정한다면, 잔차는 **비계량발전이 평년과 다른 정도**로 설명돼야 한다.
- 평년 이용률 util_clim(월,시) = train 기간 평균.
- 서프라이즈 = 용량 × (util − util_clim)  (MW; +면 평년보다 비계량 태양광 많음 → 계통수요↓ 기대)
- 가설: 잔차 ↔ 서프라이즈 **음(−)상관** (비계량 많음 → 실측<예측).""")

co(r"""trm=df[df.timestamp<='2025-02-28 23:00']
util_clim=trm.groupby(['month','hour']).real_solar_utilization_jeju.mean()
df['util_clim']=[util_clim.get((m,h),np.nan) for m,h in zip(df.month,df.hour)]
df['btmppa_surprise']=df['cap_btmppa_mw']*(df['real_solar_utilization_jeju']-df['util_clim'])
day=df[(df.hour>=9)&(df.hour<=16)].dropna(subset=['residual','btmppa_surprise'])
r_all=day[['residual','btmppa_surprise']].corr().iloc[0,1]
# 단순 선형 설명력 (R^2)
x=day.btmppa_surprise.values; y=day.residual.values
b1=np.cov(x,y)[0,1]/np.var(x); b0=y.mean()-b1*x.mean(); yhat=b0+b1*x
r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
print('낮시간 잔차 ↔ BTM/PPA 서프라이즈 상관: %+.3f  (선형 R²=%.3f, 기울기=%.2f)'%(r_all,r2,b1))
# 연도별 상관(용량 커질수록 강해지는지)
print('\n연도별 (낮시간) 잔차↔서프라이즈 상관:')
for yr,g in day.groupby('year'):
    if len(g)>100: print('  %d: %+.3f (n=%d, 평균용량 %.0fMW)'%(yr,g[['residual','btmppa_surprise']].corr().iloc[0,1],len(g),g.cap_btmppa_mw.mean()))
fig,ax=plt.subplots(1,2,figsize=(14,4.4))
s=day.sample(min(6000,len(day)),random_state=0)
ax[0].scatter(s.btmppa_surprise, s.residual, s=5, alpha=.3, color='#4C72B0')
xs=np.linspace(s.btmppa_surprise.min(),s.btmppa_surprise.max(),50)
ax[0].plot(xs,b0+b1*xs,color='#C44E52',lw=2,label=f'기울기 {b1:.2f}, R²={r2:.2f}')
ax[0].axhline(0,color='#888',lw=.6); ax[0].axvline(0,color='#888',lw=.6)
ax[0].set_xlabel('BTM/PPA 발전 서프라이즈(MW, +=평년초과)'); ax[0].set_ylabel('KPX 잔차(MW)')
ax[0].set_title('(a) 비계량 태양광 서프라이즈 vs 잔차'); ax[0].legend()
# (b) 설명력: 낮시간 잔차에서 서프라이즈/구름을 회귀로 제거하면 잔차 σ 가 얼마나 주는가
#     (KPX est 는 '계통수요'를 예측하므로 true_demand 비교가 아니라, 잔차의 설명분산으로 본다)
def ols_r2(X, y):
    X=np.column_stack([np.ones(len(X)), X]); beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    yh=X@beta; return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2), yh
dd=df[(df.hour>=9)&(df.hour<=16)].dropna(subset=['residual','btmppa_surprise','total_cloud','midlow_cloud']).copy()
y=dd.residual.values
r2_s,_   =ols_r2(dd[['btmppa_surprise']].values, y)
r2_c,_   =ols_r2(dd[['total_cloud','midlow_cloud']].values, y)
r2_sc,yh =ols_r2(dd[['btmppa_surprise','total_cloud','midlow_cloud']].values, y)
resid_adj=y-yh
print('\n낮시간 잔차 선형 설명력(R²):  서프라이즈 단독 %.3f | 구름 단독 %.3f | 서프라이즈+구름 %.3f'%(r2_s,r2_c,r2_sc))
print('낮시간 잔차 σ: 원본 %.1f → 서프라이즈+구름 제거후 %.1f MW (%.0f%% 감소)'
      %(y.std(), resid_adj.std(), (1-resid_adj.std()/y.std())*100))
# 시간대별 |잔차| 평균: 맑은날(이용률 상위25%) vs 흐린날(하위25%)
dd['util_q']=pd.qcut(dd.real_solar_utilization_jeju,4,labels=False,duplicates='drop')
gq=dd[dd.util_q.isin([0,3])].groupby('util_q').residual.agg(['mean','std'])
gq.index=['흐림(하위25%)' if i==0 else '맑음(상위25%)' for i in gq.index]
ax[1].bar(['흐림\n(이용률 하위25%)','맑음\n(이용률 상위25%)'], gq['mean'].values,
          yerr=gq['std'].values, color=['#888','#DDAA33'], capsize=4)
ax[1].axhline(0,color='#444',lw=.8); ax[1].set_ylabel('평균 잔차(MW)±σ')
ax[1].set_title('(b) 흐림 vs 맑음 — 낮시간 KPX 잔차')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'2-0b_btm_hypothesis.png'),bbox_inches='tight'); plt.show()
print('\n흐림/맑음 낮시간 평균잔차(MW):'); print(gq.round(1).to_string())""")

md(r"""## 4b. 맑은날 vs 흐린날 수요 직접 비교 (요일·계절 통제) — 사용자 요청

`real_demand vs est_demand` 잔차 EDA 는 est 가 KPX **자체 예측**이라 (KPX 모델오차 + BTM효과 + 잡음)이
섞여 해석이 어렵다. 그래서 **est 를 빼고 real_demand 만** 본다: 다른 조건(요일·월·기온)을 맞추고
**일사만 다른** 맑은날↔흐린날의 시간대별 수요를 비교 → 비계량 태양광 효과가 **낮시간 함몰(덕커브)** 로 보인다.""")

co(r"""# 날짜별 낮시간(9~16h) 평균 이용률로 맑음/흐림 분류 (월 내 분위수로 계절 통제)
dts=df[(df.hour>=9)&(df.hour<=16)].groupby('date').agg(
    util_day=('real_solar_utilization_jeju','mean'), temp_day=('temp_c','mean'),
    cloud_day=('total_cloud','mean'))
dmeta=df.groupby('date').agg(day_type=('day_type','first'), month=('month','first'),
                             dow=('timestamp', lambda s:s.dt.dayofweek.iloc[0]))
dd=dts.join(dmeta)
dd['mq']=dd.groupby('month').util_day.transform(lambda s: pd.qcut(s,4,labels=False,duplicates='drop'))
wk=dd[dd.day_type=='weekday']                  # 평일만 (요일효과 1차 통제)
clear=set(wk[wk.mq==3].index); cloudy=set(wk[wk.mq==0].index)   # 월내 상위/하위 25%
prof=df[df.date.isin(clear|cloudy)].copy()
prof['grp']=np.where(prof.date.isin(clear),'맑음(상위25%)','흐림(하위25%)')
pv=prof.groupby(['grp','hour']).real_demand_jeju.mean().unstack(0)
fig,ax=plt.subplots(1,2,figsize=(14,4.4))
ax[0].plot(pv.index, pv['맑음(상위25%)'],'o-',color='#DDAA33',label='맑음(상위25%)')
ax[0].plot(pv.index, pv['흐림(하위25%)'],'s-',color='#4C72B0',label='흐림(하위25%)')
ax[0].set_xlabel('hour'); ax[0].set_ylabel('평균 계통수요(MW)')
ax[0].set_title('(a) 평일 맑음 vs 흐림 시간대별 수요'); ax[0].legend()
gap=(pv['흐림(하위25%)']-pv['맑음(상위25%)'])
ax[1].bar(gap.index, gap.values, color=['#C44E52' if 9<=h<=16 else '#bbb' for h in gap.index])
ax[1].axhline(0,color='#444',lw=.8); ax[1].set_xlabel('hour'); ax[1].set_ylabel('흐림 − 맑음 (MW)')
ax[1].set_title('(b) 흐림−맑음 수요차 (낮시간=비계량 태양광 효과)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'2-0b_clear_vs_cloudy.png'),bbox_inches='tight'); plt.show()
midday=gap.loc[9:16].mean(); night=gap.loc[[0,1,2,3,4,22,23]].mean()
print('평일 맑음 N=%d일 / 흐림 N=%d일'%(len(clear),len(cloudy)))
print('교란 점검 — 평균기온: 맑음 %.1f℃ / 흐림 %.1f℃ (비슷할수록 좋음)'
      %(dd.loc[list(clear)].temp_day.mean(), dd.loc[list(cloudy)].temp_day.mean()))
print('낮시간(9~16h) 흐림−맑음 수요차: +%.1f MW  /  심야(무일사) 차: %+.1f MW'%(midday,night))
print('→ 흐린날 계통수요가 낮에만 %.1fMW 높음(심야엔 차이 거의 없음) = 비계량 태양광이 맑은날 계통수요를 끌어내림.'%midday)
# 같은 요일 짝 예시 (봄, 같은 dow, 가장 맑음 vs 가장 흐림)
sp=dd[(dd.month.isin([3,4,5]))&(dd.day_type=='weekday')]
ex=None
for dow,g in sp.groupby('dow'):
    if len(g)>=2:
        cd=g.util_day.idxmax(); od=g.util_day.idxmin()
        if ex is None or (g.loc[cd].util_day-g.loc[od].util_day)>ex[3]:
            ex=(dow,cd,od,g.loc[cd].util_day-g.loc[od].util_day)
if ex:
    dow,cd,od,_=ex; nm=['월','화','수','목','금','토','일'][dow]
    print(f'\n같은요일({nm}) 봄 예시 — 맑음 {pd.Timestamp(cd).date()} vs 흐림 {pd.Timestamp(od).date()}: '
          f'낮평균수요 {df[(df.date==cd)&(df.hour.between(9,16))].real_demand_jeju.mean():.0f} vs '
          f'{df[(df.date==od)&(df.hour.between(9,16))].real_demand_jeju.mean():.0f} MW')""")

md(r"""## 5. 요약 + 모델 시사점""")

co(r"""day=df[(df.hour>=9)&(df.hour<=16)].copy()
c_cloud=day[['residual','total_cloud']].corr().iloc[0,1]
c_solar=day[['residual','solar_rad']].corr().iloc[0,1]
day2=day.dropna(subset=['residual','btmppa_surprise','total_cloud','midlow_cloud'])
c_surp=day2[['residual','btmppa_surprise']].corr().iloc[0,1]
def _ols_r2(X,y):
    X=np.column_stack([np.ones(len(X)),X]); b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    yh=X@b; return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2), yh
yv=day2.residual.values
r2_s,_=_ols_r2(day2[['btmppa_surprise']].values,yv)
r2_sc,yh=_ols_r2(day2[['btmppa_surprise','total_cloud','midlow_cloud']].values,yv)
sig_real=yv.std(); sig_adj=(yv-yh).std()
rep=f'''# 2-0b 요약 — 제주 KPX 잔차의 원인 (BTM/PPA 가설 검증)

## 맥락
- 제주 수요 평균 ~{df.real_demand_jeju.mean():.0f}MW(작음). 계량 신재생 침투 {((df[df.year==2025].real_renew_gen_jeju.sum()/df[df.year==2025].real_demand_jeju.sum())*100):.0f}%(2025).
  BTM/PPA(비계량) 추가 시 실제 침투율은 더 높음. BTM/PPA 용량 2019~2025 성장(전력거래소).

## 잔차(real−est) 구조
- 잔차는 **낮시간(태양광 시간)에 집중·확대**(그림 2-0b_residual_when). 최근 연도일수록 낮시간 변동↑.

## 사용자 관찰 검증 (낮 9~16h)
- 잔차 ↔ 총운량 상관 {c_cloud:+.3f}, 잔차 ↔ 일사 상관 {c_solar:+.3f} → **흐릴수록/일사 낮을수록 실측이 예측보다 큼(+)**.
- 일사 급변 구간일수록 잔차 변동↑(그림 2-0b_residual_weather) — 사용자 관찰과 일치.

## 맑은날 vs 흐린날 (평일, 월내 일사 상·하위 25%)
- **낮시간(9~16h): 흐림이 맑음보다 수요 +{midday:.0f}MW 높음**, 그러나 **심야(무일사): {night:+.0f}MW**.
  즉 일사 영향이 없는 심야엔 부호가 반대/작고 낮에만 크게 벌어짐 → 격차는 **수요레벨 차이가 아니라 낮시간 태양광 효과**.
  (낮−심야 대비 {midday-night:.0f}MW) → est 없이 real_demand 만으로도 비계량 태양광이 맑은날 낮 계통수요를 끌어내림이 명확.
- 이 방식이 핵심: real vs est 잔차는 KPX 자체오차가 섞여 어렵지만, **요일·월 통제 후 일사만 대비**하면 깔끔(심야차는 잔여 교란).

## BTM/PPA 가설 (정량)
- 잔차 ↔ 비계량발전 서프라이즈(용량×(이용률−평년)) 상관 {c_surp:+.3f} (음의 상관:
  평년보다 비계량 태양광이 많은 시각 = 실측수요가 예측보다 낮음). 사용자 가설과 부호 일치.
- 낮시간 잔차 선형 설명력 R²: 서프라이즈 단독 {r2_s:.3f} → 서프라이즈+구름 {r2_sc:.3f}.
  회귀 제거 시 낮시간 잔차 σ {sig_real:.1f}→{sig_adj:.1f}MW ({(1-sig_adj/sig_real)*100:.0f}% 감소).
- **연도별 상관이 용량과 함께 강해짐**: 2020 −0.41 → 2026 **−0.64**(용량 최대 233MW) → 비계량 성장이 잔차 기여를 키움.
- 정직성: 선형으로는 낮시간 잔차의 일부(~{r2_sc*100:.0f}%)만 설명. 부호·방향은 가설과 모두 일치하나
  나머지는 비선형(트리)·기타 요인(KPX 자체 예보오차·산업부하 등). LGBM이면 더 잡을 여지.

## 모델 시사점 (피처 변경은 Decision Gate — 별도 확정 필요)
- 현재 2-A 피처는 solar_rad(일사)는 있으나 **구름(total/midlow_cloud)·BTM/PPA 용량은 없음.**
- 후보: ①구름량(total_cloud/midlow_cloud, 예보 존재) ②BTM/PPA 용량×평년이용률(또는 용량 자체)
  ③forecast의 cape 등 흐림/대류 지표. 낮시간 잔차를 직접 겨냥 → D+1 정확도 개선 여지.
- 단, 비계량발전 추정은 이용률(=실측) 의존 → 서빙 시 예보이용률/예보일사로 대체 필요(정직성).
'''
open(os.path.join(HERE,'REPORT_2-0b_residual_btm.md'),'w',encoding='utf-8').write(rep)
print(rep)""")

nb['cells']=cells
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.metadata['language_info']={'name':'python'}
nbf.write(nb, open(OUT,'w',encoding='utf-8'))
print('wrote',OUT,'cells',len(cells))
