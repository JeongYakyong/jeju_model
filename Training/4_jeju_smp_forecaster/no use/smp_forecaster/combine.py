"""분류기 + 회귀 합치기 (2단 임계, anchor 잔차, zoneA 통과).

model9 `combine_v(neg='two', anchor=True)` 한 경로만 가져왔다. 단일 임계,
위험띠 결합, raw 회귀 등 ablation 옵션은 모두 빠졌다.

결정 규칙(우선순위 순):
1. is_zoneA == 1 이면 → DA(smp_jeju) 그대로 (구조적 통과)
2. neg_proba ≥ TAU_HARD 이면 → 깊은 음수값
3. neg_proba ≥ TAU_SOFT 이면 → 0
4. floor_proba ≥ TAU_HI 이면 → 바닥값
5. TAU_LO ≤ floor_proba < TAU_HI 이면 → DA(smp_jeju) (바닥 애매 → DA 후퇴)
6. 그 외 → da_anchor + reg(잔차) 예측
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TAU_HARD, TAU_HI, TAU_LO, TAU_SOFT
from .pipeline import TrainedSmpModel


@dataclass
class CombineResult:
    yhat: np.ndarray            # 최종 24h SMP 예측값
    neg_proba: np.ndarray       # 음수 확률 (위험 알림용)
    danger: np.ndarray          # neg_proba >= TAU_SOFT (위험띠 마스크)


def combine(model: TrainedSmpModel, df: pd.DataFrame) -> CombineResult:
    """학습된 모델 + 피처 DataFrame → 최종 SMP 예측.

    Args:
        model: 학습된 TrainedSmpModel.
        df: build_features 출력. CORE_COLS + 'smp_jeju' + 'da_anchor' + 'is_zoneA'
            컬럼이 모두 있어야 한다.

    Returns:
        CombineResult.
    """
    feature_cols = model.feature_cols
    X = df[feature_cols]

    floor_proba = model.floor_clf.predict_proba(X)
    neg_proba   = model.neg_clf.predict_proba(X)

    da = df['smp_jeju'].values
    level = df['da_anchor'].values + model.reg.predict(X)

    # 기본 = 잔차회귀 예측
    yhat = level.copy()

    # 바닥 분류기 처리 (애매 → DA 후퇴, 확신 → 바닥값)
    floor_unsure = (floor_proba >= TAU_LO) & (floor_proba < TAU_HI)
    yhat = np.where(floor_unsure, da, yhat)
    yhat = np.where(floor_proba >= TAU_HI, model.floor_val, yhat)

    # 음수 분류기 처리 (애매 → 0, 확신 → 깊은 음수)
    yhat = np.where(neg_proba >= TAU_SOFT, 0.0, yhat)
    yhat = np.where(neg_proba >= TAU_HARD, model.deep_neg, yhat)

    # zoneA(net_load 상위) 구조적 DA 통과 — 위 모든 규칙을 덮어쓴다
    is_zone_a = df['is_zoneA'].values == 1
    yhat = np.where(is_zone_a, da, yhat)

    danger = neg_proba >= TAU_SOFT
    return CombineResult(yhat=yhat, neg_proba=neg_proba, danger=danger)
