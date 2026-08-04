"""학습 파이프라인 빌더 / Production training pipeline.

model9 운영 모델(핵심13@재선택)의 학습 절차:

1. **BANK** — 검증구간(2026-02 ~ 2026-05-13)에서 실측·예보 이용률 차이를
   시각별로 모아 둔 잔차 표본 은행.
2. **inject_noise** — 학습 직전에 실측 이용률에 시각별 BANK 잔차를 더해
   예보품질에 강건한 모델로 만든다.
3. **fit_calibrated** — 시간순 확장폴드 OOF → IsotonicRegression 보정 →
   전체 재학습 LGBM 분류기.
4. **build_pipeline** — 위 3개를 묶어 운영 모델 한 벌(분류기 2개 + 회귀)을
   학습한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from .config import CV_CUTS, RNG, TRAIN_END

UTIL_BOUNDS = {
    'Solar_Utilization': (0.0, 1.0),
    'Wind_Utilization':  (0.0, 1.5),
}


# ── 모델 팩토리 (model8/9 확정 하이퍼파라미터) ────────────────────────────
def make_classifier(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RNG, n_jobs=-1, verbose=-1,
    )


def make_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective='l2',
        n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
        random_state=RNG, n_jobs=-1, verbose=-1,
    )


# ── BANK + noise injection ────────────────────────────────────────────────
def build_bank(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict[int, dict[str, np.ndarray]]:
    """시각별 예보오차 표본 은행을 만든다.

    Args:
        actual:   실측 피처 (Solar_Utilization, Wind_Utilization 포함).
        forecast: 같은 인덱스의 예보 피처 (이미 컬럼명 통일된 build_features 결과).

    Returns:
        {hour: {'s': solar_residuals, 'w': wind_residuals}} — 길이는 시각별로 다름.
    """
    common = actual.index.intersection(forecast.index)
    if len(common) == 0:
        raise ValueError("BANK 구성에 필요한 실측·예보 겹침 구간이 없다")

    solar_resid = (forecast.loc[common, 'Solar_Utilization']
                   - actual.loc[common, 'Solar_Utilization']).values
    wind_resid  = (forecast.loc[common, 'Wind_Utilization']
                   - actual.loc[common, 'Wind_Utilization']).values
    hours = common.hour

    bank: dict[int, dict[str, np.ndarray]] = {}
    for h in range(24):
        mask = (hours == h)
        bank[h] = {
            's': solar_resid[mask],
            'w': wind_resid[mask],
        }
    return bank


def inject_noise(
    utilizations: pd.DataFrame,
    bank: dict[int, dict[str, np.ndarray]],
    seed: int = RNG,
) -> pd.DataFrame:
    """학습용 실측 이용률에 시각별 예보오차 잔차를 더한다.

    Args:
        utilizations: Solar_Utilization, Wind_Utilization 두 컬럼을 가진 DataFrame.
        bank: build_bank() 결과.
        seed: 재현용 시드.

    Returns:
        같은 모양의 DataFrame (값에 노이즈 추가, 물리 범위로 클립).
    """
    out = utilizations.copy()
    rng = np.random.RandomState(seed)
    hour_arr = out.index.hour.values

    for col, bank_key in [('Solar_Utilization', 's'), ('Wind_Utilization', 'w')]:
        lo, hi = UTIL_BOUNDS[col]
        noise = np.zeros(len(out))
        for h in range(24):
            mask = (hour_arr == h)
            pool = bank[h][bank_key]
            if mask.any() and len(pool):
                noise[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
        out[col] = np.clip(out[col].values + noise, lo, hi)
    return out


# ── 시간순 확장폴드 OOF + isotonic 보정 분류기 ────────────────────────────
@dataclass
class CalibratedClassifier:
    """학습된 LGBM 분류기 + isotonic 보정기 묶음."""
    clf: lgb.LGBMClassifier
    iso: IsotonicRegression
    feature_cols: list[str]
    oof_pr_auc: float

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(X[self.feature_cols])[:, 1]
        return self.iso.predict(raw)


def fit_calibrated(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_fn: Callable[[pd.DataFrame], pd.Series],
    label_name: str,
) -> CalibratedClassifier:
    """시간순 확장폴드 OOF → isotonic 보정 → 전체 재적합 LGBM 분류기.

    Args:
        train_df: 시간 인덱스 학습 DataFrame. feature_cols + 라벨 계산용 컬럼 포함.
        feature_cols: 모델 입력 컬럼.
        label_fn: train_df → 이진 라벨 Series. (e.g. lambda d: d['smp_rt'] < 5)
        label_name: 로그용 이름.

    Returns:
        보정된 분류기 + OOF PR-AUC 진단치.
    """
    segments = list(zip(CV_CUTS, CV_CUTS[1:] + [TRAIN_END + pd.Timedelta('1h')]))
    y = label_fn(train_df).astype(int)
    oof = pd.Series(np.nan, index=train_df.index)

    for seg_start, seg_end in segments:
        prev = train_df.loc[:seg_start - pd.Timedelta('1h')]
        fold = train_df.loc[seg_start:seg_end - pd.Timedelta('1h')]
        if len(prev) < 200 or len(fold) == 0:
            continue
        prev_y = label_fn(prev).astype(int)
        spw = (prev_y == 0).sum() / max(int((prev_y == 1).sum()), 1)
        fold_clf = make_classifier(spw).fit(prev[feature_cols], prev_y)
        oof.loc[fold.index] = fold_clf.predict_proba(fold[feature_cols])[:, 1]

    ok = oof.notna()
    iso = IsotonicRegression(out_of_bounds='clip').fit(oof[ok], y[ok])

    spw_full = (y == 0).sum() / max(int((y == 1).sum()), 1)
    final_clf = make_classifier(spw_full).fit(train_df[feature_cols], y)

    pr_auc = float(average_precision_score(y[ok], iso.predict(oof[ok])))
    print(f"  [{label_name}] OOF PR-AUC={pr_auc:.3f} "
          f"(양성 {int(y[ok].sum())}/{int(ok.sum())})")

    return CalibratedClassifier(
        clf=final_clf, iso=iso,
        feature_cols=list(feature_cols), oof_pr_auc=pr_auc,
    )


# ── 운영 모델 한 벌 학습 ──────────────────────────────────────────────────
@dataclass
class TrainedSmpModel:
    """학습된 SMP 운영 모델 = 분류기 2개 + 잔차 회귀 + 메타.

    floor_clf : smp_rt < 5  분류기 (바닥값)
    neg_clf   : smp_rt <= 0 분류기 (음수)
    reg       : (smp_rt - da_anchor) 잔차 회귀 (양수 가격대만 학습)
    """
    floor_clf: CalibratedClassifier
    neg_clf: CalibratedClassifier
    reg: lgb.LGBMRegressor
    feature_cols: list[str]
    floor_val: float       # 바닥 라벨이 켜졌을 때 대입할 값 (학습셋 중앙값)
    deep_neg: float        # 깊은 음수 라벨이 켜졌을 때 대입할 값


def build_pipeline(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> TrainedSmpModel:
    """학습 DataFrame과 피처 리스트로 운영 모델 한 벌을 학습한다.

    Args:
        train_df: TRAIN_START~TRAIN_END 구간 + smp_rt 타깃이 채워진 학습본.
                  feature_cols 와 'smp_rt', 'da_anchor' 컬럼이 모두 있어야 한다.
        feature_cols: 모델 입력 컬럼 (보통 CORE_COLS).

    Returns:
        TrainedSmpModel.
    """
    floor_val = float(train_df.loc[train_df['smp_rt'] < 5, 'smp_rt'].median())
    deep_neg  = float(train_df.loc[train_df['smp_rt'] <= 0, 'smp_rt'].median())
    print(f"  바닥값(중앙값)={floor_val:.2f} | 깊은음수값(중앙값)={deep_neg:.2f}")

    floor_clf = fit_calibrated(
        train_df, feature_cols,
        label_fn=lambda d: d['smp_rt'] < 5,
        label_name='바닥<5',
    )
    neg_clf = fit_calibrated(
        train_df, feature_cols,
        label_fn=lambda d: d['smp_rt'] <= 0,
        label_name='음수<=0',
    )

    # 잔차 회귀는 양수 가격대만(바닥/음수 분류기와 역할 분리)
    pos = train_df[train_df['smp_rt'] >= 5]
    reg = make_regressor().fit(
        pos[feature_cols],
        pos['smp_rt'] - pos['da_anchor'],
    )

    return TrainedSmpModel(
        floor_clf=floor_clf, neg_clf=neg_clf, reg=reg,
        feature_cols=list(feature_cols),
        floor_val=floor_val, deep_neg=deep_neg,
    )
