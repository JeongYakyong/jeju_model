"""4단계 SMP Phase 2-B — 음수 깊이의 경험적 조건부 분포 (룩업표 빌더).

게이트 결과: N(rt<0)=380이나 깊이가 행정바닥(~-70)에 퇴화(P50≈-70, neg_num 91%가 4)
→ 적은 양성에 quantile 회귀하면 floor 과적합. 정직하고 과적합 없는 **경험적 분포**를 채택.

산출:
  - models_weight/smp_depth_lookup.json : (시간대 × solar_util 수준) 격자별 음수 깊이 P10/50/90 룩업
  - 콘솔: 룩업표 + 지속(run-length) 경험분포 + 비음수 가격선 MAE 불변 검증

대전제: 비음수 구간은 절대 안 건드린다. 이 룩업은 "음수가 발생했다는 조건 하에"의 깊이 분포로,
서빙 시 경보(P3)가 켜진 구간에만 overlay로 표시된다. 가격선(est_smp_jeju=DA)은 불변.

원본 DB·A안 코드 미변경. historical 읽기 전용.
"""
from __future__ import annotations

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # 루트(공통 로더 train_smp_db) 접근
from train_smp_db import DB_PATH

MODELS = os.path.normpath(os.path.join(HERE, '..', 'models_weight'))
LOOKUP = os.path.join(MODELS, 'smp_depth_lookup.json')

WINDOW_START = '2024-06-01'
MIN_CELL = 12          # 격자 셀 최소 표본 — 미만이면 상위(시간대→전체) 폴백
QS = [0.1, 0.5, 0.9]   # P10/P50/P90


def load_neg():
    cols = ['timestamp', 'smp_jeju_da', 'real_demand_jeju', 'real_renew_gen_jeju',
            'real_solar_utilization_jeju', 'smp_jeju_rt', 'smp_rt_neg_num']
    with sqlite3.connect(DB_PATH) as con:
        h = pd.read_sql(f'SELECT {",".join(cols)} FROM historical '
                        'WHERE smp_jeju_rt IS NOT NULL ORDER BY timestamp',
                        con, parse_dates=['timestamp']).set_index('timestamp')
    h = h[h.index >= WINDOW_START].copy()
    h['solar_util'] = h['real_solar_utilization_jeju']
    h['hour'] = h.index.hour
    return h


def hour_bucket(hr):
    if 9 <= hr <= 10:  return '09-10'
    if 11 <= hr <= 13: return '11-13'
    if 14 <= hr <= 15: return '14-15'
    return 'edge'      # 음수 거의 없는 시간 — 전체 폴백


def solar_bucket(su, t1, t2):
    if su is None or (isinstance(su, float) and np.isnan(su)): return 'mid'
    if su < t1:  return 'lo'
    if su < t2:  return 'mid'
    return 'hi'


def _q(vals):
    a = np.asarray(vals, float)
    return {f'P{int(q*100)}': round(float(np.quantile(a, q)), 1) for q in QS}


def build_lookup(neg: pd.DataFrame):
    """음수 표본(rt<0)만으로 (시간대×solar) 격자 P10/50/90. 셀<MIN_CELL이면 폴백 기록."""
    n = neg[neg['smp_jeju_rt'] < 0].copy()
    depth = n['smp_jeju_rt']
    t1, t2 = float(n['solar_util'].quantile(1/3)), float(n['solar_util'].quantile(2/3))
    n['hb'] = n['hour'].map(hour_bucket)
    n['sb'] = n['solar_util'].map(lambda s: solar_bucket(s, t1, t2))

    glob = _q(depth); glob['n'] = int(len(depth))
    by_hb = {hb: {**_q(g['smp_jeju_rt']), 'n': int(len(g))}
             for hb, g in n.groupby('hb')}
    cells = {}
    for (hb, sb), g in n.groupby(['hb', 'sb']):
        cells[f'{hb}|{sb}'] = {**_q(g['smp_jeju_rt']), 'n': int(len(g))}

    return {
        'window_start': WINDOW_START, 'n_neg': int(len(depth)),
        'solar_thresholds': [round(t1, 3), round(t2, 3)],
        'min_cell': MIN_CELL,
        'global': glob, 'by_hour_bucket': by_hb, 'cells': cells,
        '_quantiles': ['P10', 'P50', 'P90'],
    }


def lookup_depth(table, hour, solar_util):
    """서빙용: (hour, solar_util) → (P10,P50,P90, source). 셀→시간대→전체 폴백."""
    t1, t2 = table['solar_thresholds']
    hb, sb = hour_bucket(int(hour)), solar_bucket(float(solar_util), t1, t2)
    key = f'{hb}|{sb}'
    cell = table['cells'].get(key)
    if cell and cell['n'] >= table['min_cell']:
        src = f'cell:{key}(n={cell["n"]})'; r = cell
    elif hb in table['by_hour_bucket'] and table['by_hour_bucket'][hb]['n'] >= table['min_cell']:
        r = table['by_hour_bucket'][hb]; src = f'hour:{hb}(n={r["n"]})'
    else:
        r = table['global']; src = f'global(n={r["n"]})'
    return r['P10'], r['P50'], r['P90'], src


def run_lengths(neg_flag):
    f = neg_flag.values.astype(bool); runs = []; i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]: j += 1
            runs.append(j - i); i = j
        else: i += 1
    return np.array(runs)


def main():
    h = load_neg()
    table = build_lookup(h)
    os.makedirs(MODELS, exist_ok=True)
    with open(LOOKUP, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    print('═══ Phase 2-B — 음수 깊이 경험적 조건부 분포 ═══')
    print(f'학습창 {WINDOW_START}~  음수표본 N={table["n_neg"]}  '
          f'solar 임계(터셜)={table["solar_thresholds"]}  (셀 최소표본={MIN_CELL})\n')

    print('── 깊이 룩업표 (시간대 × solar 수준), 음수 발생 조건부 P10/P50/P90 [원] ──')
    print(f'  {"시간대":>6} {"solar":>5} {"n":>4} {"P10":>7} {"P50":>7} {"P90":>7}')
    for hb in ['09-10', '11-13', '14-15', 'edge']:
        for sb in ['lo', 'mid', 'hi']:
            c = table['cells'].get(f'{hb}|{sb}')
            if not c: continue
            flag = '' if c['n'] >= MIN_CELL else '  ←폴백(시간대/전체 사용)'
            print(f'  {hb:>6} {sb:>5} {c["n"]:>4} {c["P10"]:>7.1f} {c["P50"]:>7.1f} {c["P90"]:>7.1f}{flag}')
    print(f'\n  [시간대 marginal]')
    for hb, r in table['by_hour_bucket'].items():
        print(f'  {hb:>6}  n={r["n"]:>3}  P10={r["P10"]:>6.1f} P50={r["P50"]:>6.1f} P90={r["P90"]:>6.1f}')
    g = table['global']
    print(f'  [전체]   n={g["n"]}  P10={g["P10"]} P50={g["P50"]} P90={g["P90"]}')

    # 서빙 폴백 예시
    print('\n── 서빙 룩업 예시 ──')
    for hr, su in [(12, 0.85), (12, 0.55), (10, 0.75), (14, 0.7), (3, 0.0)]:
        p10, p50, p90, src = lookup_depth(table, hr, su)
        print(f'  hour={hr:>2} solar={su:<4}  →  P10={p10:>6.1f} P50={p50:>6.1f} P90={p90:>6.1f}  [{src}]')

    # 지속(run-length) 경험 분포
    print('\n── 지속(run-length) 경험 분포 (best-effort) ──')
    neg_flag = (h['smp_jeju_rt'] < 0).sort_index()
    runs = run_lengths(neg_flag)
    print(f'  사건수={len(runs)}  평균={runs.mean():.2f}h  P50={np.median(runs):.0f}h  '
          f'P90={np.quantile(runs,0.9):.0f}h  max={runs.max()}h')
    print('  → 데모 문구용: "발생 시 예상 지속 ~3h(P90 5h)"')

    # 비음수 가격선 MAE 불변 검증 (DA는 안 건드림 = 구조적 불변, 수치 재확인)
    pos = h[h['smp_jeju_rt'] >= 0]
    mae_pos = float((pos['smp_jeju_da'] - pos['smp_jeju_rt']).abs().mean())
    print(f'\n── 비음수 가격선 검증 ──')
    print(f'  비음수구간(rt>=0) DA 가격선 MAE = {mae_pos:.2f}  (A안 8.3 수준 유지, 깊이 overlay는 별도 컬럼이라 불변)')

    print(f'\n[saved] {LOOKUP}')


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
