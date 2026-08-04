# -*- coding: utf-8 -*-
"""풍력 실험 1회 — east 풍향(wd_sin/cos_east) 추가 ROI 점검.

배경: 진단에서 풍력 ORACLE 바닥선이 모델 천장 근처(D+1 ~0.087). 현행 WIND_FINAL 은 풍향이
west 만 들어감. 현장: 풍력단지 west 55% / east 45% → east 풍향이 wake/지형을 더 잡을 여지.
사용자 지시: 1회 실험, 과적합 의심되면 현행 복귀. **생산 lgbm_wind_util.txt 는 덮지 않음**
(실험 산출은 lgbm_wind_util_exp_east.txt).

평가: test 2026 실측기상(=ORACLE급, horizon-무관) 계절별 nMAE + train/val/test 격차(과적합)
+ east 피처 중요도. 현행 base 와 같은 split/params/데이터로 비교(공정).
"""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
CMP_MODEL = os.path.join(ROOT, '3. jeju_solarwind_forecaster', 'comparison', 'model', '3cmp-A_lgbm_solarwind.py')
CSV = os.path.join(HERE, 'solarwind_raw_jeju.csv')
OUT_MODEL = os.path.join(ROOT, '3. jeju_solarwind_forecaster', 'lgbm_models', 'lgbm_wind_util_exp_east.txt')
WU = 'real_wind_utilization_jeju'
SEASON = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄',
          6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을', 11: '가을'}


def _load_3cmpA():
    spec = importlib.util.spec_from_file_location('cmpA', CMP_MODEL)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    cmpA = _load_3cmpA()
    BASE = list(cmpA.WIND_FINAL)
    EXP = BASE + ['wd_sin_east', 'wd_cos_east']

    df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    tr = df[df.index.year <= 2024]
    _, clim = cmpA.build_features(tr)
    feat, _ = cmpA.build_features(df, clim=clim)
    feat['split'] = np.where(feat.index.year <= 2024, 'train',
                    np.where(feat.index.year == 2025, 'val', 'test'))
    feat['season'] = feat.index.month.map(SEASON)

    params = dict(objective='regression_l1', n_estimators=1200, learning_rate=0.03,
                  num_leaves=63, min_child_samples=80, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

    def fit_eval(feats, tag, save=None):
        trm = feat[feat.split == 'train']; vam = feat[feat.split == 'val']
        m = lgb.LGBMRegressor(**params)
        m.fit(trm[feats], trm[WU], eval_set=[(vam[feats], vam[WU])],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        if save:
            m.booster_.save_model(save)
        out = {}
        for split in ['train', 'val', 'test']:
            s = feat[feat.split == split]
            p = np.clip(m.predict(s[feats]), 0, 1)
            out[split] = float(np.mean(np.abs(p - s[WU].values)))
        te = feat[feat.split == 'test'].copy()
        te['pred'] = np.clip(m.predict(te[feats]), 0, 1)
        seas = te.groupby('season').apply(
            lambda x: float(np.mean(np.abs(x.pred - x[WU]))), include_groups=False)
        imp = pd.Series(m.booster_.feature_importance('gain'), index=feats)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        return out, seas, imp, m.best_iteration_

    print('=' * 64)
    print('풍력 east 풍향 실험 — test 2026 실측기상(ORACLE급) nMAE')
    print('=' * 64)
    b_split, b_seas, b_imp, b_it = fit_eval(BASE, 'base')
    e_split, e_seas, e_imp, e_it = fit_eval(EXP, 'exp', save=OUT_MODEL)

    print(f"\n[현행 base]  best_iter={b_it}")
    print(f"  nMAE train {b_split['train']:.4f} / val {b_split['val']:.4f} / test {b_split['test']:.4f}"
          f"  (과적합 격차 test-train {b_split['test']-b_split['train']:+.4f})")
    print(f"[+east     ]  best_iter={e_it}")
    print(f"  nMAE train {e_split['train']:.4f} / val {e_split['val']:.4f} / test {e_split['test']:.4f}"
          f"  (과적합 격차 test-train {e_split['test']-e_split['train']:+.4f})")
    print(f"\n  test 전체 변화: {b_split['test']:.4f} → {e_split['test']:.4f} ({e_split['test']-b_split['test']:+.4f})")

    print('\n  계절별 test nMAE (base → +east):')
    for s in ['겨울', '봄', '여름', '가을']:
        if s in b_seas.index:
            print(f'    {s:5s}  {b_seas[s]:.4f} → {e_seas[s]:.4f}  ({e_seas[s]-b_seas[s]:+.4f})')

    print('\n  +east 모델 중요도(gain) — east 풍향 위치:')
    print(e_imp.round(3).to_string())
    print(f'\n  east 풍향 합산 중요도: {e_imp.get("wd_sin_east",0)+e_imp.get("wd_cos_east",0):.3f}')
    print('\n저장(실험전용, 생산 미덮음):', OUT_MODEL if os.path.exists(OUT_MODEL) else '(미저장)')


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    main()
