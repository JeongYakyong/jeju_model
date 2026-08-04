"""학습된 SMP 모델 로더 (메모이즈) / Memoized loader for the trained model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from .config import SMP_MODEL_PATH
from .pipeline import TrainedSmpModel


@lru_cache(maxsize=4)
def _load_artifact(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없다: {path}. smp_forecaster.train.train() 으로 "
            f"먼저 학습해야 한다."
        )
    return joblib.load(path)


def load_model(model_path: str | Path | None = None) -> tuple[TrainedSmpModel, dict]:
    """저장된 모델 아티팩트를 불러와 (TrainedSmpModel, raw artifact dict)로 반환.

    Args:
        model_path: 모델 .pkl 경로. None이면 config.SMP_MODEL_PATH.

    Returns:
        (model, artifact). artifact는 메타(bank, trained_at, training_window 등) 포함.
    """
    path = str(model_path or SMP_MODEL_PATH)
    artifact = _load_artifact(path)

    model = TrainedSmpModel(
        floor_clf=artifact['floor_clf'],
        neg_clf=artifact['neg_clf'],
        reg=artifact['reg'],
        feature_cols=artifact['feature_cols'],
        floor_val=artifact['floor_val'],
        deep_neg=artifact['deep_neg'],
    )
    return model, artifact
