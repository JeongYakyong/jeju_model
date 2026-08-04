"""4단계 SMP — 음수 위험 프로파일 시각화 (단순화판).

요소 5개만:
  - DA 가격선(메인, solid bold)        : est_smp_jeju — 정상 시 예상 SMP, 앵커
  - 위험조정 기대선(서브, dotted)       : (1-P_cal)·DA + P_cal·d  (smp_calibrate)
  - RT 참고선(서브2, 실제)              : smp_jeju_rt — 실제 실시간 SMP(이중모드 확인용)
  - 주간 경보 zone(깊이 미정)           : 주간[8-16] 음수경보 구간
  - 야간 경보 zone(깊이 미정)           : 주간 밖 음수경보 구간(비물리 시간, 별도색)

경보 = proba ≥ θ(=0.25) + 2h 지속. 깊이밴드/등급/고확신밴드는 제외(구간만).
평가는 recall/precision만. 읽기 전용. 모델·DB 미변경.
"""
from __future__ import annotations

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 루트(공통 로더) 접근
from train_smp_db import DB_PATH
from train_binary_smp import persist
from smp_calibrate import load_calibrator

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

THETA_ALARM = 0.25                 # 경보 운영점
DAY = range(8, 17)                  # 음수 물리 가능 시간(주간). 밖은 '야간 경보'
DAY_C, NIGHT_C = '#e74c3c', '#5d6d7e'


def load_day(date):
    with sqlite3.connect(DB_PATH) as con:
        f = pd.read_sql(
            "SELECT timestamp, est_smp_jeju da, smp_neg_proba_jeju proba FROM forecast "
            f"WHERE substr(timestamp,1,10)='{date}' ORDER BY timestamp",
            con, parse_dates=['timestamp']).set_index('timestamp')
        r = pd.read_sql(
            "SELECT timestamp, smp_jeju_rt rt FROM historical "
            f"WHERE substr(timestamp,1,10)='{date}' ORDER BY timestamp",
            con, parse_dates=['timestamp']).set_index('timestamp')
    df = f.join(r, how='left')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def runs(mask):
    f = np.asarray(mask, bool); out = []; i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def plot_day(date, save=True):
    df = load_day(date)
    if df.empty:
        raise ValueError(f'{date} forecast 없음')
    iso, d_cond = load_calibrator()
    n = len(df); x = np.arange(n)
    da = df['da'].values
    rt = df['rt'].values
    proba = df['proba'].fillna(0).values
    pcal = iso.predict(proba)
    risk = (1 - pcal) * da + pcal * d_cond
    hours = np.array([t.hour for t in df.index])
    alarm = persist(pd.Series(proba >= THETA_ALARM, index=df.index)).values

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.axhline(0, color='gray', lw=1, ls=':')

    summary = []
    for (s, e) in runs(alarm):
        dur = e - s + 1
        is_day = np.mean([h in DAY for h in hours[s:e + 1]]) >= 0.5
        c = DAY_C if is_day else NIGHT_C
        ax.axvspan(s - .5, e + .5, color=c, alpha=0.16, zorder=0)
        ax.annotate(f'{"주간" if is_day else "야간"} 경보 {dur}h',
                    xy=((s + e) / 2, 6), ha='center', va='bottom', fontsize=8.5,
                    color=c, weight='bold')
        summary.append(('주간' if is_day else '야간', df.index[s].hour, df.index[e].hour, dur))

    # 선 3개
    ax.plot(x, da, color='#1b2631', lw=2.8, marker='o', ms=3.5, zorder=5,
            label='DA 가격선 (메인)')
    ax.plot(x, risk, color='#c0392b', lw=1.8, ls=(0, (1, 1.2)), marker='.', ms=4, zorder=6,
            label='위험조정 기대선 (서브)')
    if np.isfinite(rt).any():
        ax.plot(x, rt, color='#2980b9', lw=1.3, ls='--', marker='x', ms=4, alpha=0.9,
                zorder=4, label='RT 참고선 (실제)')

    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax.set_xlim(-.5, n - .5); ax.set_xlabel('시각 (h)'); ax.set_ylabel('SMP (원/kWh)')
    vals = np.concatenate([da, risk, rt[np.isfinite(rt)]]) if np.isfinite(rt).any() else np.concatenate([da, risk])
    ax.set_ylim(min(np.nanmin(vals), 0) - 12, max(np.nanmax(vals), 0) + 16)
    ax.set_title(f'제주 SMP 음수 위험 프로파일 — {date}')
    ax.grid(alpha=0.25)

    handles = [
        Patch(facecolor=DAY_C, alpha=0.3, label='주간 경보 (깊이 미정)'),
        Patch(facecolor=NIGHT_C, alpha=0.3, label='야간 경보 (깊이 미정)'),
        Line2D([], [], color='#c0392b', lw=1.8, ls=':', marker='.', label='위험조정 기대선 (서브)'),
        Line2D([], [], color='#1b2631', lw=2.8, marker='o', label='DA 가격선 (메인)'),
        Line2D([], [], color='#2980b9', lw=1.3, ls='--', marker='x', label='RT 참고선 (실제)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.92, ncol=2)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fig_risk_profile_{date}.png')
    if save:
        fig.savefig(out, dpi=120); plt.close(fig)

    print(f'── {date} 음수 위험 요약 ──')
    for kind, h0, h1, dur in summary:
        print(f'  [{kind} 경보] {h0:02d}~{h1:02d}시 {dur}h')
    if not summary:
        print('  경보 없음(정상)')
    print(f'  위험조정 기대선 최저 {risk.min():.0f}원'
          + (f' · RT 실제 최저 {np.nanmin(rt):.0f}원' if np.isfinite(rt).any() else ' · RT 미보유'))
    print(f'  [fig] {out}')
    return summary


def main():
    import sys
    for d in (sys.argv[1:] or ['2026-03-19']):
        plot_day(d); print()


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
