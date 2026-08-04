# -*- coding: utf-8 -*-
"""풍력 입력 풍속 분위수 매핑(QM) 보정표 적합 — NWP 예보 풍속 → 실측 분포.

배경(2026-06-23 진단): 풍력 LGBM 은 실측 풍속(historical.wind_spd_{st})으로 학습했는데
서빙 땐 수치예보(forecast_horizon.wind_spd_10m_{st})를 먹는다. 두 분포가 어긋나 출력이
이용률을 +7.5%p 과대예측한다. 특히 east 는 NWP 가 +1.4m/s(+45%) 과대(분산도 팽창),
west 는 평균은 맞지만 분산 압축(약풍 과대·강풍 과소). 이를 분위수 매핑으로 학습 분포에
되돌려 입력을 정합시킨다(모델 재학습 없음).

검증(전 기간 5겹 OOF, 단지평 D+1~3): nMAE 13.57→11.27%(-17%), bias +7.53→+1.39%p,
전 계절·전 지평 재현. 단 실측 강풍(≥12m/s, ~16% 시간)은 NWP 가 애초에 못 잡아 과소예측이
조금 깊어짐 — 보정으로 못 만드는 NWP 한계(가스 관점 보수적). 상세 REPORT_wind_qm.md.

산출: lgbm_models/wind_qm.json  (서빙 serve_solarwind_hybrid._apply_wind_qm 가 읽음)
실행: python fit_wind_qm.py [--horizons 1,2,3]
"""
from __future__ import annotations
import os, sys, json, sqlite3, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
DB   = os.path.join(ROOT, '1. data_fetcher_and_db', 'data', 'input_data_jeju.db')
OUT  = os.path.join(HERE, '..', 'lgbm_models', 'wind_qm.json')
STATIONS = ['west', 'east']
NQ = 199   # 분위수 격자(0.005~0.995)


def fit(horizons):
    con = sqlite3.connect(DB)
    fh = pd.read_sql("SELECT base, timestamp, wind_spd_10m_west, wind_spd_10m_east "
                     "FROM forecast_horizon", con, parse_dates=['timestamp'])
    hist = pd.read_sql("SELECT timestamp, wind_spd_west, wind_spd_east FROM historical",
                       con, parse_dates=['timestamp']).set_index('timestamp')
    con.close()
    fh['horizon'] = (fh['timestamp'].dt.normalize() - pd.to_datetime(fh['base']).dt.normalize()).dt.days
    fh = fh[fh['horizon'].isin(horizons)]
    qs = np.linspace(0.005, 0.995, NQ)
    stations = {}; report = {}
    for st in STATIONS:
        x = pd.to_numeric(fh[f'wind_spd_10m_{st}'], errors='coerce')
        y = fh['timestamp'].map(pd.to_numeric(hist[f'wind_spd_{st}'], errors='coerce'))
        m = x.notna() & y.notna()
        x, y = x[m].values, y[m].values
        fc_q = np.quantile(x, qs); obs_q = np.quantile(y, qs)
        stations[st] = {'q': qs.round(4).tolist(),
                        'fc_q': fc_q.round(4).tolist(), 'obs_q': obs_q.round(4).tolist()}
        # 적합 진단: 매핑 전후 평균/편향
        xc = np.clip(np.interp(x, fc_q, obs_q), 0, None)
        report[st] = {'n': int(m.sum()),
                      'fc_mean': float(x.mean()), 'obs_mean': float(y.mean()),
                      'bias_before': float((x - y).mean()), 'bias_after': float((xc - y).mean()),
                      'std_fc': float(x.std()), 'std_obs': float(y.std())}
    payload = {'_doc': 'NWP 풍속→실측 분위수 매핑(풍력 입력 보정). 서빙 _apply_wind_qm 사용.',
               'horizons_fit': horizons, 'n_quantiles': NQ, 'stations': stations, 'report': report}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('saved', os.path.normpath(OUT))
    for st in STATIONS:
        r = report[st]
        print(f"  [{st}] n={r['n']}  예보평균 {r['fc_mean']:.2f}→실측 {r['obs_mean']:.2f}  "
              f"편향 {r['bias_before']:+.2f}→{r['bias_after']:+.2f} m/s  "
              f"std {r['std_fc']:.2f}/{r['std_obs']:.2f}")


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', default='1,2,3', help='적합에 쓸 지평(기본 단지평 1,2,3)')
    a = ap.parse_args()
    fit([int(x) for x in a.horizons.split(',')])
