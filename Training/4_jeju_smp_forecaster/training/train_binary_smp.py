"""4단계 SMP — 이진 음수경보 분류기 (A안 최종형).

가격선은 DA(smp_jeju_da) 그대로 쓰고, 이 분류기는 "음수 위험 경보"만 담당.

치명(fatal) 정의 = DA는 정상인데 실제 RT가 무가치  →  (smp_jeju_da >= NEG_THRESH) & (smp_jeju_rt < NEG_THRESH)
  발전사업자가 돈 받을 줄 알았는데 무는 경우 = 분류기가 잡아야 할 핵심 목표.

타깃 = (smp_jeju_rt < NEG_THRESH)   (NEG_THRESH=5: 0에 가까운 무가치 SMP ≈ 음수)
경보 = P(음수) >= θ  에 **2시간 지속 규칙**(연속 2시간 이상만 경보) 적용.

평가: ROC-AUC / PR-AUC + 임계 스윕(총recall / 치명recall / precision / 헛경보).
"""
from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # 루트(공통 로더 train_smp_db) 접근
from train_smp_db import (FEATURES, RT_COL, NEG_THRESH, TRAIN_START, TRAIN_END, VAL_START,
                          TEST_START, load_historical, load_forecast, SEASON)

MODELS = os.path.normpath(os.path.join(HERE, '..', 'models_weight'))
BUNDLE = os.path.join(MODELS, 'smp_binary.pkl')

MIN_RUN = 2     # 경보 지속 규칙: 연속 N시간 이상만 경보
# 이중 운영점(보수적, 사용자 확정 2026-06-05): precision 우선 두 단계.
#   p25: 균형~precision (TEST prec 0.38 / 총recall 0.86)
#   p26: 고확신 (TEST prec 0.67 / 총recall 0.49)  ← 헛경보 최소
THETA = {'p25': 0.25, 'p26': 0.26}


def persist(flag: pd.Series, min_run=MIN_RUN) -> pd.Series:
    """연속 min_run 시간 이상인 경보만 유지(단발 제거). 런길이 기반."""
    f = flag.values.astype(bool)
    out = np.zeros(len(f), bool)
    i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(out, index=flag.index)


def sweep(tag, y_rt, da, proba):
    """임계별 총recall/치명recall/precision/헛경보 (2시간 지속 적용)."""
    yneg = (y_rt < NEG_THRESH).values                  # 이벤트: rt<5
    fatal = ((da >= NEG_THRESH).values & yneg)         # 치명: DA 정상(≥5)인데 RT<5
    nneg, nfatal = int(yneg.sum()), int(fatal.sum())
    roc = roc_auc_score(yneg, proba); pr = average_precision_score(yneg, proba)
    print(f'  [{tag}] rt<{NEG_THRESH}={nneg}  치명(da≥{NEG_THRESH}&rt<{NEG_THRESH})={nfatal}  '
          f'ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}')
    print(f'    {"θ":>5} {"총recall":>8} {"치명recall":>9} {"precision":>9} {"헛경보":>6} {"경보h":>6}')
    for th in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]:
        alarm = persist(pd.Series(proba >= th, index=y_rt.index)).values
        tp = int((alarm & yneg).sum()); fp = int((alarm & ~yneg).sum())
        ftp = int((alarm & fatal).sum())
        rec = tp / nneg if nneg else float('nan')
        frec = ftp / nfatal if nfatal else float('nan')
        prec = tp / (tp + fp) if tp + fp else float('nan')
        print(f'    {th:>5} {rec:>8.3f} {frec:>9.3f} {prec:>9.3f} {fp:>6} {int(alarm.sum()):>6}')


def main():
    hist = load_historical()
    tr = hist[(hist.index >= TRAIN_START) & (hist.index <= TRAIN_END)].dropna(subset=FEATURES)
    va = hist[hist.index >= VAL_START].dropna(subset=FEATURES)
    te = load_forecast(); te = te[te.index >= TEST_START].dropna(subset=FEATURES)

    ytr = (tr[RT_COL] < NEG_THRESH).astype(int)
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    print(f'TRAIN n={len(tr)} rt<{NEG_THRESH}={pos}  VAL n={len(va)}  TEST n={len(te)}\n')

    clf = lgb.LGBMClassifier(
        objective='binary', n_estimators=2000, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
        scale_pos_weight=neg / max(pos, 1), random_state=42, verbose=-1)
    clf.fit(tr[FEATURES], ytr, eval_set=[(va[FEATURES], (va[RT_COL] < NEG_THRESH).astype(int))],
            eval_metric='average_precision', callbacks=[lgb.early_stopping(100, verbose=False)])

    pva = clf.predict_proba(va[FEATURES])[:, 1]
    pte = clf.predict_proba(te[FEATURES])[:, 1]
    print(f'best_iter={clf.best_iteration_}  (경보 지속규칙 MIN_RUN={MIN_RUN}시간)\n')

    print('═══ 임계 스윕 (2시간 지속 적용) ═══')
    sweep('VAL ', va[RT_COL], va['smp_jeju_da'], pva)
    print()
    sweep('TEST', te[RT_COL], te['smp_jeju_da'], pte)

    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n── 피처 중요도 ──'); print(imp.to_string())

    with open(BUNDLE, 'wb') as fh:
        pickle.dump({'clf': clf, 'features': FEATURES, 'min_run': MIN_RUN,
                     'theta': THETA, 'neg_thresh': NEG_THRESH}, fh)
    print(f'\n[saved] {BUNDLE}  (θ={THETA}, neg_thresh={NEG_THRESH})')


def _resave_theta():
    """재학습 없이 번들의 운영점(θ)만 갱신."""
    b = pickle.load(open(BUNDLE, 'rb'))
    b['theta'] = THETA; b['neg_thresh'] = NEG_THRESH
    pickle.dump(b, open(BUNDLE, 'wb'))
    print(f'[bundle] θ={THETA} 갱신')


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
