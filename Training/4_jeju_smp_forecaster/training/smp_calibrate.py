"""4단계 SMP — 음수확률 보정(Isotonic) + 위험조정 기대선.

raw LGBM proba는 미보정(0.26인데 실제 rt<5 빈도 ~0.5-0.65). Isotonic 회귀로 P_cal(진짜 빈도)로
보정하고, 그 위에 '위험조정 기대선'을 만든다:

    E[SMP] = (1 - P_cal)·DA  +  P_cal·D_COND       (D_COND = E[rt | rt<5] ≈ -53)

※ 이 선은 RT 점예측이 아니라 '위험의 무게중심'을 잇는 연속 시나리오선이다.
   실제 RT는 이중모드(정상 또는 바닥)이며, MAE류로 평가하지 않는다(평가=recall/precision).

산출: models_weight/smp_calibrator.pkl  (iso + d_cond). 학습은 train+val proba→(rt<5).
읽기 전용 학습(historical). 원본 DB 미변경.
"""
from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # 루트(공통 로더 train_smp_db) 접근
from train_smp_db import (load_historical, FEATURES, RT_COL, NEG_THRESH,
                          TRAIN_START, TRAIN_END, VAL_START)

MODELS = os.path.normpath(os.path.join(HERE, '..', 'models_weight'))
BUNDLE = os.path.join(MODELS, 'smp_binary.pkl')
CALIB = os.path.join(MODELS, 'smp_calibrator.pkl')


def fit_and_save():
    clf = pickle.load(open(BUNDLE, 'rb'))['clf']
    h = load_historical()
    tr = h[(h.index >= TRAIN_START) & (h.index <= TRAIN_END)].dropna(subset=FEATURES)
    va = h[h.index >= VAL_START].dropna(subset=FEATURES)
    d = pd.concat([tr, va])
    p = clf.predict_proba(d[FEATURES])[:, 1]
    y = (d[RT_COL] < NEG_THRESH).astype(int).values
    iso = IsotonicRegression(out_of_bounds='clip').fit(p, y)
    d_cond = float(d[d[RT_COL] < NEG_THRESH][RT_COL].mean())   # E[rt|rt<5]
    with open(CALIB, 'wb') as f:
        pickle.dump({'iso': iso, 'd_cond': d_cond, 'neg_thresh': NEG_THRESH}, f)
    print(f'[saved] {CALIB}  d_cond(E[rt|rt<5])={d_cond:.1f}')
    print('  보정 매핑(raw→P_cal):',
          {round(r, 3): round(float(iso.predict([r])[0]), 3)
           for r in [0.024, 0.20, 0.24, 0.25, 0.26, 0.268]})
    return iso, d_cond


def load_calibrator():
    b = pickle.load(open(CALIB, 'rb'))
    return b['iso'], b['d_cond']


def calibrate(proba, iso=None):
    if iso is None:
        iso, _ = load_calibrator()
    return iso.predict(np.asarray(proba, float))


def risk_line(proba, da, iso=None, d_cond=None):
    """위험조정 기대선 E = (1-P_cal)·DA + P_cal·D_COND."""
    if iso is None or d_cond is None:
        iso, d_cond = load_calibrator()
    pcal = iso.predict(np.asarray(proba, float))
    return (1 - pcal) * np.asarray(da, float) + pcal * d_cond, pcal


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    fit_and_save()
