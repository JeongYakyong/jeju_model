"""4단계 SMP Phase 1 — RT-DA 잔차 분해 (예측 불가 입증).

목적(smp_step4_instruction §3): RT 자체가 아니라 **잔차 = smp_jeju_rt - smp_jeju_da**를
분해해, 잔차 분산 중 (트렌드/계절)·(기상으로 설명)·(설명 안 되는 순수 노이즈) 비율을 보여
"가격선은 DA가 잡고, 남은 잔차의 대부분은 예측 불가한 시장노이즈"임을 정량 입증한다.

산출물:
  - fig_stl_residual.png        : STL 분해 그림(잔차 trend/seasonal/remainder)
  - 콘솔: 분산 분해 표 + 음수/비음수 구간 기상 R² + Decision Gate 측정값(N·런길이·neg_num)

원본 DB·A안 코드(train_binary_smp / smp_db_pipeline / train_smp_db)는 **건드리지 않음**.
이 스크립트는 train_smp_db.load_historical()만 재사용한다(읽기 전용).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # 루트(공통 로더 train_smp_db) 접근
from train_smp_db import DB_PATH
FIG = os.path.join(HERE, 'fig_stl_residual.png')

WINDOW_START = '2024-06-01'          # 레짐 경계(시범사업) 이후만
# 잔차에서 DA(가격레벨)는 이미 제거됨 → 기상/물리 연속피처만으로 회귀(R²).
WEATHER_FEATS = ['net_load', 'nl_lead_1', 'nl_lead_2',
                 'solar_util', 'wind_util', 'wind_spd_west', 'solar_rad_south']

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_window():
    """학습창 잔차 + 기상피처 + neg_num (DB 직접조회, 읽기 전용).
    A안 load_historical()은 구 컬럼 smp_rt_neg_flag(현재 smp_rt_neg_num로 교체됨)를 참조해
    현재 깨져 있어 의존하지 않는다. RT NULL 시점은 WHERE로 제외(보간 없음)."""
    cols = ['timestamp', 'smp_jeju_da', 'real_demand_jeju', 'real_renew_gen_jeju',
            'real_solar_utilization_jeju', 'real_wind_utilization_jeju',
            'wind_spd_west', 'solar_rad_south', 'smp_jeju_rt', 'smp_rt_neg_num']
    with sqlite3.connect(DB_PATH) as con:
        h = pd.read_sql(f'SELECT {",".join(cols)} FROM historical '
                        'WHERE smp_jeju_rt IS NOT NULL ORDER BY timestamp',
                        con, parse_dates=['timestamp']).set_index('timestamp')
    h['net_load'] = h['real_demand_jeju'] - h['real_renew_gen_jeju']
    h = h.rename(columns={'real_solar_utilization_jeju': 'solar_util',
                          'real_wind_utilization_jeju': 'wind_util'})
    h = h.sort_index()
    h['nl_lead_1'] = h['net_load'].shift(-1)
    h['nl_lead_2'] = h['net_load'].shift(-2)
    h['hour'] = h.index.hour
    h['month'] = h.index.month
    h = h[h.index >= WINDOW_START].copy()
    h['residual'] = h['smp_jeju_rt'] - h['smp_jeju_da']
    return h


def stl_decompose(resid: pd.Series):
    """STL(period=24). STL은 결손 불가 → 구조적 index 결손(24h, 0.14%)만 그림용 선형보간.
    보간된 행은 분산분해 수치에서 제외(원본 잔차만 사용)."""
    full = pd.date_range(resid.index.min(), resid.index.max(), freq='h')
    s = resid.reindex(full)
    gap_mask = s.isna()
    n_gap = int(gap_mask.sum())
    s_filled = s.interpolate(method='linear', limit_direction='both')
    stl = STL(s_filled, period=24, robust=True).fit()
    comp = pd.DataFrame({'trend': stl.trend, 'seasonal': stl.seasonal,
                         'remainder': stl.resid}, index=full)
    comp['gap'] = gap_mask.values
    return comp, n_gap


def save_fig(resid: pd.Series, comp: pd.DataFrame):
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(comp.index, resid.reindex(comp.index), lw=0.4, color='#333')
    ax[0].set_ylabel('잔차\n(RT-DA)')
    ax[0].set_title('RT-DA 잔차의 STL 분해 (학습창 2024-06~, period=24)')
    ax[1].plot(comp.index, comp['trend'], lw=0.8, color='#1f77b4'); ax[1].set_ylabel('trend')
    ax[2].plot(comp.index, comp['seasonal'], lw=0.4, color='#2ca02c'); ax[2].set_ylabel('seasonal\n(일주기)')
    ax[3].plot(comp.index, comp['remainder'], lw=0.4, color='#d62728'); ax[3].set_ylabel('remainder\n(잔여)')
    ax[3].set_xlabel('시각')
    for a in ax:
        a.grid(alpha=0.3); a.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    plt.close(fig)


def variance_decomposition(h: pd.DataFrame, comp: pd.DataFrame):
    """잔차 총분산 = 트렌드 + 계절 + (기상설명) + (순수노이즈)로 분해.
    - 트렌드/계절: STL 성분 분산 비율(원본 행만; 보간행 제외)
    - 기상설명: STL remainder를 기상 연속피처로 선형회귀한 R² × (remainder 분산비)
    - 순수노이즈 = remainder 분산비 × (1 − R²_weather)
    """
    # 원본(보간 아님) 행만 사용
    real = comp[~comp['gap']].copy()
    resid_real = (h['smp_jeju_rt'] - h['smp_jeju_da']).reindex(real.index)
    V = float(resid_real.var())

    sh_trend = float(real['trend'].var()) / V
    sh_seasonal = float(real['seasonal'].var()) / V
    sh_remainder = float(real['remainder'].var()) / V

    # remainder를 기상피처로 회귀 (피처 결측 행 제외)
    feat = h[WEATHER_FEATS].reindex(real.index)
    rem = real['remainder']
    ok = feat.notna().all(axis=1) & rem.notna()
    lr = LinearRegression().fit(feat[ok], rem[ok])
    r2_w = r2_score(rem[ok], lr.predict(feat[ok]))
    r2_w = max(0.0, r2_w)

    sh_weather = sh_remainder * r2_w
    sh_noise = sh_remainder * (1 - r2_w)

    return {
        'total_var': V, 'n_used': int(len(real)),
        'trend': sh_trend, 'seasonal': sh_seasonal, 'remainder': sh_remainder,
        'r2_weather_on_remainder': r2_w,
        'weather_explained': sh_weather, 'pure_noise': sh_noise,
    }


def weather_r2_by_sign(h: pd.DataFrame):
    """음수구간 vs 비음수구간에서 잔차를 기상피처로 회귀한 R² 비교.
    음수구간 R²가 유의미하게 높으면 = '음수만 설명가능' 정당성."""
    resid = h['smp_jeju_rt'] - h['smp_jeju_da']
    out = {}
    for name, mask in [('전체', pd.Series(True, index=h.index)),
                       ('음수(rt<0)', h['smp_jeju_rt'] < 0),
                       ('비음수(rt>=0)', h['smp_jeju_rt'] >= 0)]:
        sub = h[mask]
        feat = sub[WEATHER_FEATS]
        y = resid[mask]
        ok = feat.notna().all(axis=1) & y.notna()
        if ok.sum() < 20:
            out[name] = (int(ok.sum()), float('nan')); continue
        lr = LinearRegression().fit(feat[ok], y[ok])
        out[name] = (int(ok.sum()), float(r2_score(y[ok], lr.predict(feat[ok]))))
    return out


def run_lengths(neg_flag: pd.Series):
    """연속 음수(rt<0) run-length 분포. neg_flag는 시간순 정렬된 bool."""
    f = neg_flag.values.astype(bool)
    runs = []
    i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return np.array(runs)


def main():
    h = load_window()
    resid = h['smp_jeju_rt'] - h['smp_jeju_da']

    print('═══ Phase 1 — RT−DA 잔차 분해 (학습창 2024-06~) ═══')
    print(f'행수={len(h)}  잔차 mean={resid.mean():.2f}  std={resid.std():.2f}  '
          f'min={resid.min():.1f}  max={resid.max():.1f}\n')

    comp, n_gap = stl_decompose(resid)
    save_fig(resid, comp)
    print(f'[fig] {FIG} 저장  (STL period=24, 구조결손 {n_gap}h는 그림용 보간·수치 제외)\n')

    vd = variance_decomposition(h, comp)
    print('── 분산 분해 표 (잔차 총분산 = 100%) ──')
    print(f'  총분산 V = {vd["total_var"]:.1f}  (사용 행 {vd["n_used"]})')
    print(f'  (1) 트렌드        : {vd["trend"]*100:6.1f}%')
    print(f'  (2) 계절(일주기)  : {vd["seasonal"]*100:6.1f}%')
    print(f'  (3) 기상설명      : {vd["weather_explained"]*100:6.1f}%   '
          f'(remainder R²_weather={vd["r2_weather_on_remainder"]:.3f})')
    print(f'  (4) 순수 노이즈   : {vd["pure_noise"]*100:6.1f}%   ★예측 불가 시장노이즈★')
    print(f'      [참고] remainder 분산비 = {vd["remainder"]*100:.1f}% '
          f'(= (3)+(4))\n')

    print('── 음수/비음수 구간 기상 R² (음수만 설명가능 확인) ──')
    for name, (n, r2) in weather_r2_by_sign(h).items():
        print(f'  {name:<14} n={n:>6}  R²(기상→잔차)={r2:.3f}')
    print()

    # ── Decision Gate 측정값 ──
    print('═══ Decision Gate 측정값 (Phase 1 → Phase 2) ═══')
    neg = h['smp_jeju_rt'] < 0
    N = int(neg.sum())
    print(f'  1) 음수 샘플 수 N (rt<0, 학습창) = {N}   '
          f'→ {"N≥300 : P2-A(LGBM quantile) 후보" if N >= 300 else "N<300 : P2-B(경험적 분포)"}')

    runs = run_lengths(neg.sort_index())
    if len(runs):
        print(f'  2a) 연속 음수 run-length: 사건수={len(runs)}  '
              f'평균={runs.mean():.2f}h  중앙={np.median(runs):.0f}h  max={runs.max()}h')
        vc = pd.Series(runs).value_counts().sort_index()
        print('      런길이 분포(시간:건수): ' + ', '.join(f'{k}h:{v}' for k, v in vc.items()))

    nn = h.loc[neg, 'smp_rt_neg_num'].dropna()
    if len(nn):
        print(f'  2b) 음수시간의 smp_rt_neg_num(15분 심도 0~4) 분포:')
        vc = nn.astype(int).value_counts().sort_index()
        for k, v in vc.items():
            print(f'        neg_num={k}: {v}건 ({v/len(nn)*100:.1f}%)')

    # 음수 깊이(magnitude) 요약 — P2 타깃 감
    depth = h.loc[neg, 'smp_jeju_rt']
    print(f'\n  [음수 깊이 참고] rt<0 magnitude: mean={depth.mean():.1f}  '
          f'P10={depth.quantile(0.1):.1f}  P50={depth.median():.1f}  '
          f'P90={depth.quantile(0.9):.1f}  min={depth.min():.1f}')


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
