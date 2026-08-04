# -*- coding: utf-8 -*-
"""제주 지평 백테스트 진단 — horizon_backtest_jeju.parquet → 표·그림.

산출: tab/diag_by_horizon_jeju.csv, tab/diag_by_season_jeju.csv,
      fig/diag_by_horizon_jeju.png, fig/diag_season_solar_jeju.png
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, 'horizon_backtest_jeju.parquet')
TAB = os.path.join(HERE, 'tab'); FIG = os.path.join(HERE, 'fig')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

for cand in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = cand; break
plt.rcParams['axes.unicode_minus'] = False

SEASON = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄', 6: '여름'}
DAY = (9, 15)   # 태양광 낮(사용자 우선순위)


def _mape(a, p):
    m = (a > 0)
    return float(np.mean(np.abs(a[m] - p[m]) / a[m]) * 100) if m.any() else np.nan


def _nmae(a, p):
    m = a.notna() & p.notna()
    return float(np.mean(np.abs(a[m] - p[m]))) if m.any() else np.nan


def by_horizon(df):
    out = []
    for (mode, n), g in df.groupby(['mode', 'horizon']):
        dd = g.dropna(subset=['real_demand']); dd = dd[dd.real_demand > 0]
        day = dd[(dd.hour >= DAY[0]) & (dd.hour <= DAY[1])]
        sd = g[(g.hour >= DAY[0]) & (g.hour <= DAY[1])]
        nn = g.dropna(subset=['real_net_load'])
        nl_mae = _nmae(nn.real_net_load, nn.est_net_load)
        out.append(dict(
            mode=mode, horizon=n,
            demand_mape=_mape(dd.real_demand.values, dd.est_demand.values),
            demand_mape_day=_mape(day.real_demand.values, day.est_demand.values),
            demand_bias=float(np.mean((dd.est_demand - dd.real_demand) / dd.real_demand) * 100),
            solar_nmae_day=_nmae(sd.real_solar_util, sd.est_solar_util),
            wind_nmae=_nmae(g.real_wind_util, g.est_wind_util),
            net_load_mae=nl_mae,
            net_load_nmae=(nl_mae / float(dd.real_demand.mean()) * 100) if len(dd) else np.nan,
            n=len(g)))
    return pd.DataFrame(out)


def by_season(df):
    d = df.copy(); d['season'] = d.timestamp.dt.month.map(SEASON)
    d['daypart'] = np.where((d.hour >= DAY[0]) & (d.hour <= DAY[1]), '낮', '밤')
    out = []
    for (mode, season, dp), g in d.groupby(['mode', 'season', 'daypart']):
        dd = g.dropna(subset=['real_demand']); dd = dd[dd.real_demand > 0]
        out.append(dict(mode=mode, season=season, daypart=dp,
                        demand_mape=_mape(dd.real_demand.values, dd.est_demand.values),
                        solar_nmae=_nmae(g.real_solar_util, g.est_solar_util),
                        wind_nmae=_nmae(g.real_wind_util, g.est_wind_util),
                        n=len(g)))
    return pd.DataFrame(out)


def fig_horizon(bh):
    f = bh[bh['mode'] == 'forecast'].sort_values('horizon')
    o = bh[bh['mode'] == 'oracle'].sort_values('horizon')
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = [('demand_mape', '수요 MAPE (%)'), ('solar_nmae_day', '태양광 이용률 nMAE (낮 09-15h)'),
             ('net_load_nmae', 'net_load nMAE (%)')]
    for a, (col, title) in zip(ax, specs):
        a.plot(f.horizon, f[col], 'o-', color='#c0392b', label='실예보')
        a.plot(o.horizon, o[col], 's--', color='#2471a3', label='ORACLE(입력완벽)')
        a.fill_between(f.horizon, o[col].values, f[col].values, color='#e74c3c', alpha=0.10)
        a.set_title(title, fontsize=10); a.set_xlabel('지평 (D+)'); a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle('제주 지평 진단 — 실예보 vs ORACLE (격차 = 예보 품질의 대가)', fontsize=12)
    fig.tight_layout()
    p = os.path.join(FIG, 'diag_by_horizon_jeju.png'); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_season(bs):
    f = bs[(bs['mode'] == 'forecast') & (bs['daypart'] == '낮')]
    order = ['겨울', '봄', '여름']
    f = f.set_index('season').reindex(order)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(order, f['solar_nmae'], color='#e67e22'); ax[0].set_title('태양광 이용률 nMAE (낮·실예보)', fontsize=10)
    ax[1].bar(order, f['demand_mape'], color='#16a085'); ax[1].set_title('수요 MAPE (낮·실예보)', fontsize=10)
    for a in ax: a.grid(alpha=0.3, axis='y')
    fig.suptitle('제주 계절×낮 분해 (실예보)', fontsize=12); fig.tight_layout()
    p = os.path.join(FIG, 'diag_season_jeju.png'); fig.savefig(p, dpi=130); plt.close(fig)
    return p


if __name__ == '__main__':
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    df = pd.read_parquet(PARQUET)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    bh = by_horizon(df); bs = by_season(df)
    bh.to_csv(os.path.join(TAB, 'diag_by_horizon_jeju.csv'), index=False, encoding='utf-8-sig')
    bs.to_csv(os.path.join(TAB, 'diag_by_season_jeju.csv'), index=False, encoding='utf-8-sig')
    p1 = fig_horizon(bh); p2 = fig_season(bs)
    print('=== 지평별 (forecast vs oracle) ===')
    print(bh.round(2).to_string(index=False))
    print('\n=== 계절×낮밤 (forecast) ===')
    print(bs[bs['mode'] == 'forecast'].round(3).to_string(index=False))
    print('\nsaved:', p1, p2)
