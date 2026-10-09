# -*- coding: utf-8 -*-
"""풍력 입력 풍속 분위수 매핑(QM) 보정표 적합 — NWP 예보 풍속 → 실측(ASOS) 분포, 시각대별.

배경: 풍력 LGBM 은 실측 풍속(historical.wind_spd_{st})으로 학습했는데 서빙 땐 수치예보
(forecast_horizon.wind_spd_10m_{st})를 먹는다. 두 분포가 어긋나 있어(특히 east 는 예보가 평균
+1.4 m/s 과대) 분위수 매핑으로 입력을 학습 분포에 되돌린다. 모델 재학습과 무관한 입력 변환이다.

시각대별(2026-10-09): 예보 편향이 하루 중 일정하지 않다 — west 는 밤·아침 +0.6, 낮 +0.2 m/s
(실측은 낮에 0.7 m/s 더 센데 예보는 이 일변화가 약하다). 하루 전체를 한 번에 맞추면 낮이 과하게
깎여 정오 과소예측이 된다. 5개 시각대로 나눠 각각 분위수 매핑한다.
검증(예보 입력, 2025-12-20~, 월 단위 교차검증): 정오(10~15시) 편향 -0.031 → -0.019, MAE 0.1033 → 0.1025.
지평별 분리는 효과가 없어 넣지 않는다(D+1~3 풀링).

산출: models/solarwind_lgbm/wind_qm.json  (서빙 serve_solarwind._apply_wind_qm 이 읽는다)
실행: python Training/3_jeju_solarwind_forecaster/training/fit_wind_qm.py
"""
from __future__ import annotations
import os, sys, json, sqlite3
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P

OUT = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'wind_qm.json')
STATIONS = ['west', 'east']
HORIZONS = (1, 2, 3)
HOUR_BLOCKS = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
NQ = 100   # 분위수 격자(0~1, 양 끝 포함)


def load_pairs():
    """예보-실측 짝. 같은 (대상시각, 지평)에 base 가 여럿이면 가장 최근 발표만 쓴다."""
    con = sqlite3.connect(P.DB_JEJU)
    fh = pd.read_sql("SELECT timestamp, base, horizon_d, wind_spd_10m_west, wind_spd_10m_east "
                     "FROM forecast_horizon WHERE horizon_d BETWEEN ? AND ?", con,
                     params=(min(HORIZONS), max(HORIZONS)), parse_dates=['timestamp'])
    hist = pd.read_sql("SELECT timestamp, wind_spd_west, wind_spd_east FROM historical",
                       con, parse_dates=['timestamp']).set_index('timestamp')
    con.close()
    fh = fh.sort_values('base').groupby(['timestamp', 'horizon_d']).tail(1)
    pairs = fh.merge(hist, left_on='timestamp', right_index=True, suffixes=('_fc', '_obs'))
    pairs = pairs.rename(columns={'wind_spd_10m_west': 'fc_west', 'wind_spd_10m_east': 'fc_east',
                                  'wind_spd_west': 'obs_west', 'wind_spd_east': 'obs_east'})
    wind_columns = ['fc_west', 'fc_east', 'obs_west', 'obs_east']
    pairs[wind_columns] = pairs[wind_columns].apply(pd.to_numeric, errors='coerce')
    pairs['hour'] = pairs['timestamp'].dt.hour
    return pairs


def main():
    pairs = load_pairs()
    qs = np.linspace(0, 1, NQ)
    stations, report = {}, {}
    for st in STATIONS:
        blocks, rep = [], []
        for hours in HOUR_BLOCKS:
            sel = pairs[pairs['hour'].isin(hours)][[f'fc_{st}', f'obs_{st}']].dropna()
            fc, obs = sel[f'fc_{st}'].values, sel[f'obs_{st}'].values
            fc_q, obs_q = np.quantile(fc, qs), np.quantile(obs, qs)
            mapped = np.clip(np.interp(fc, fc_q, obs_q), 0, None)
            blocks.append({'hours': hours, 'fc_q': fc_q.round(4).tolist(), 'obs_q': obs_q.round(4).tolist()})
            rep.append({'hours': f'{hours[0]}-{hours[-1]}', 'n': int(len(sel)),
                        'bias_before': round(float((fc - obs).mean()), 3),
                        'bias_after': round(float((mapped - obs).mean()), 3)})
        stations[st] = {'blocks': blocks}
        report[st] = rep
    payload = {'_doc': 'NWP 풍속→실측 분위수 매핑(풍력 입력 보정, 시각대별). 서빙 _apply_wind_qm 사용.',
               'horizons_fit': list(HORIZONS), 'n_quantiles': NQ,
               'fit_period': [str(pairs['timestamp'].min()), str(pairs['timestamp'].max())],
               'stations': stations, 'report': report}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('saved', OUT, '| 적합 기간', payload['fit_period'])
    for st in STATIONS:
        for r in report[st]:
            print(f"  [{st}] {r['hours']:>5}시 n={r['n']:>5}  편향 {r['bias_before']:+.2f} → {r['bias_after']:+.2f} m/s")


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    main()
