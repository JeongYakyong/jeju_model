# -*- coding: utf-8 -*-
"""inspect_smp_examples.py — SMP 출력을 실제 사건에서 눈으로 검증.

validate_smp_horizon_jeju 와 같은 substrate(horizon_backtest_jeju.parquet, 봄 음수가격 포함)로
SMP 서빙을 돌린 뒤, 집계가 아니라 **구체적 사례**를 보여준다:
  ① 모델이 음수 경보를 옳게 켠 날(TP) 시간별 원값(est_smp·neg_proba·danger vs 실측 rt)
  ② 헛경보 사례(FP)
  ③ 봄철 한 구간 그림 = 실측 rt + 경보 시점 + 음수확률 (fig/smp_inspect_spring.png)
DB 미오염(tempdb 사용).
"""
from __future__ import annotations
import os, sys, sqlite3, tempfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 라벨(윈도우)
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_smp_horizon_jeju as V   # build_tempdb 재사용

NEG = 5


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    tmp = os.path.join(tempfile.gettempdir(), 'inspect_smp.db')
    print('[inspect] tempdb 구성(parquet→est_horizon_jeju)...')
    nbase = V.build_tempdb(tmp)
    runner = V._imp('serve_smp_horizon_jeju', os.path.join(V.SMP_DIR, 'serve_smp_horizon_jeju.py'))
    runner.DB = tmp
    if V.SMP_DIR not in sys.path:
        sys.path.insert(0, V.SMP_DIR)
    import smp_db_pipeline as P1, smp_d2_pipeline as P2, train_smp_db
    P2.load_forecast = lambda with_target=True: train_smp_db.load_forecast(with_target=False)
    bases = runner.list_bases()
    scratch = os.path.join(tempfile.gettempdir(), 'inspect_smp_scratch.db')
    print(f'[inspect] {len(bases)} base SMP 서빙...')
    for bi, base in enumerate(bases, 1):
        r = runner.serve_base(base, P1, P2, scratch)
        if not r.empty:
            runner.upsert_est(r, tmp)

    con = sqlite3.connect(tmp)
    df = pd.read_sql('SELECT * FROM est_smp_horizon_jeju', con, parse_dates=['timestamp'])
    h = pd.read_sql('SELECT timestamp, smp_jeju_rt, smp_jeju_da FROM historical '
                    'WHERE smp_jeju_rt IS NOT NULL', con, parse_dates=['timestamp'])
    con.close()
    d = df.merge(h, on='timestamp', how='inner')
    d['neg'] = d['smp_jeju_rt'] < NEG
    d['day'] = d['timestamp'].dt.date

    d1 = d[d.horizon_d == 1].copy()
    # 날짜별 음수h / 경보h / 적중h
    g = d1.groupby('day').agg(neg_h=('neg', 'sum'),
                              alarm_h=('smp_danger', 'sum'),
                              tp=('neg', lambda s: int(((s) & (d1.loc[s.index, 'smp_danger'] == 1)).sum())),
                              rt_min=('smp_jeju_rt', 'min')).reset_index()
    tp_days = g[(g.neg_h > 0) & (g.tp > 0)].sort_values('tp', ascending=False)
    fp_days = g[(g.neg_h == 0) & (g.alarm_h > 0)].sort_values('alarm_h', ascending=False)

    def show_day(day, title):
        s = d1[d1.day == day].sort_values('timestamp')
        print(f'\n=== {title}: {day} (D+1, rt 최저 {s.smp_jeju_rt.min():.0f}) ===')
        v = s[['timestamp', 'est_smp', 'smp_jeju_da', 'smp_jeju_rt', 'smp_neg_proba', 'smp_danger', 'neg']].copy()
        v['timestamp'] = v['timestamp'].dt.strftime('%H:%M')
        v.columns = ['시각', 'est_smp', 'DA실측', 'rt실측', 'neg_proba', '경보', '실제음수']
        for c in ['est_smp', 'DA실측', 'rt실측']:
            v[c] = v[c].round(1)
        v['neg_proba'] = v['neg_proba'].round(3)
        # 음수/경보 시간대만 강조해서 핵심 시간(08~16h) 출력
        print(v[v['시각'].str[:2].astype(int).between(8, 17)].to_string(index=False))

    print('\n' + '=' * 70)
    print(f'D+1 음수일 {int((g.neg_h>0).sum())}일 / 그중 경보 적중일 {len(tp_days)}일 / 헛경보일 {len(fp_days)}일')
    if len(tp_days):
        show_day(tp_days.iloc[0]['day'], '① 음수 경보 적중(TP) 사례')
    if len(tp_days) > 1:
        show_day(tp_days.iloc[1]['day'], '① 음수 경보 적중(TP) 사례 2')
    if len(fp_days):
        show_day(fp_days.iloc[0]['day'], '② 헛경보(FP) 사례')

    # D+2 가격선 정확도 집계 (예측 DA vs 실측 DA, + 나이브 baseline lag24=어제 같은시각 DA)
    d2 = d[d.horizon_d == 2].copy()
    if len(d2):
        con2 = sqlite3.connect(tmp)
        da_full = pd.read_sql('SELECT timestamp, smp_jeju_da FROM historical ORDER BY timestamp',
                              con2, parse_dates=['timestamp']).set_index('timestamp')['smp_jeju_da']
        con2.close()
        da_full = da_full.reindex(pd.date_range(da_full.index.min(), da_full.index.max(), freq='h'))
        lag24 = da_full.shift(24)
        a = d2.dropna(subset=['est_smp', 'smp_jeju_da']).copy()
        a['lag24'] = lag24.reindex(a['timestamp']).values
        a = a.dropna(subset=['lag24'])
        mae_pred = (a.est_smp - a.smp_jeju_da).abs().mean()
        mae_base = (a.lag24 - a.smp_jeju_da).abs().mean()
        bias = (a.est_smp - a.smp_jeju_da).mean()
        corr = a[['est_smp', 'smp_jeju_da']].corr().iloc[0, 1]
        # 음수경보(D+2) recall/precision
        yneg = (a.smp_jeju_rt < NEG).values
        alarm = a.smp_danger.fillna(0).astype(int).values.astype(bool)
        tp = int((alarm & yneg).sum()); fp = int((alarm & ~yneg).sum()); fn = int((~alarm & yneg).sum())
        rec = tp / (tp + fn) if (tp + fn) else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        print('\n' + '=' * 70)
        print(f'=== D+2 정확도 집계 (n={len(a)}h, 178 base) ===')
        print(f'  가격선(예측 DA) MAE = {mae_pred:.2f}원/kWh   (나이브 baseline lag24 = {mae_base:.2f}'
              f' → 개선 {mae_base-mae_pred:+.2f})')
        print(f'  bias = {bias:+.2f}원   상관(예측↔실측 DA) = {corr:.3f}')
        print(f'  음수경보  recall={rec:.3f}  precision={prec:.3f}  (TP {tp}/FP {fp}/FN {fn})')
        # 계절별 가격선 MAE
        a['mon'] = a['timestamp'].dt.month
        seas = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄', 6: '여름'}
        a['계절'] = a['mon'].map(seas)
        sm = a.groupby('계절').apply(lambda g: pd.Series({
            'n': len(g), 'MAE': (g.est_smp - g.smp_jeju_da).abs().mean(),
            'baseline': (g.lag24 - g.smp_jeju_da).abs().mean()})).round(2)
        print('\n  [계절별 가격선 MAE]'); print(sm.to_string())

    # D+2 가격선 예측 한 사례 (시각별)
    if len(d2):
        ex = d2[d2.day == d2.day.max()].sort_values('timestamp')
        mae = (ex.est_smp - ex.smp_jeju_da).abs().mean()
        print(f'\n=== ③ D+2 가격선 예측 vs 실측 DA: {ex.day.iloc[0]} (MAE {mae:.1f}원) ===')
        vv = ex[['timestamp', 'est_smp', 'smp_jeju_da']].copy()
        vv['timestamp'] = vv['timestamp'].dt.strftime('%H:%M')
        vv.columns = ['시각', 'est_smp(예측DA)', 'DA실측']
        vv['est_smp(예측DA)'] = vv['est_smp(예측DA)'].round(1); vv['DA실측'] = vv['DA실측'].round(1)
        print(vv[vv['시각'].str[:2].astype(int) % 3 == 0].to_string(index=False))

    # ④ 봄철 한 구간 그림
    if len(tp_days):
        center = pd.Timestamp(tp_days.iloc[0]['day'])
        win = d1[(d1.timestamp >= center - pd.Timedelta(days=2)) &
                 (d1.timestamp <= center + pd.Timedelta(days=3))].sort_values('timestamp')
        if len(win) > 24:
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
            ax[0].plot(win.timestamp, win.smp_jeju_rt, color='#0f172a', lw=1.3, label='rt 실측 SMP')
            ax[0].axhline(NEG, color='#dc2626', ls='--', lw=0.9, label=f'음수 임계({NEG})')
            al = win[win.smp_danger == 1]
            ax[0].scatter(al.timestamp, al.smp_jeju_rt, color='#dc2626', s=28, zorder=5, label='모델 경보')
            ng = win[win.neg]
            ax[0].scatter(ng.timestamp, ng.smp_jeju_rt, facecolors='none', edgecolors='#2563eb',
                          s=70, lw=1.3, zorder=4, label='실제 음수사건')
            ax[0].set_ylabel('SMP (원/kWh)'); ax[0].legend(loc='upper right', fontsize=8)
            ax[0].set_title(f'제주 SMP 4단계 — 음수 경보 검증 (D+1, {center.date()} 전후)')
            ax[1].fill_between(win.timestamp, 0, win.smp_neg_proba, color='#f59e0b', alpha=0.6)
            ax[1].axhline(0.25, color='#6b7280', ls=':', lw=0.9)
            ax[1].set_ylabel('음수확률'); ax[1].set_ylim(0, 1)
            fig.tight_layout()
            figdir = os.path.join(HERE, 'fig'); os.makedirs(figdir, exist_ok=True)
            out = os.path.join(figdir, 'smp_inspect_spring.png')
            fig.savefig(out, dpi=110); plt.close(fig)
            print(f'\n[그림] {out}')


if __name__ == '__main__':
    main()
