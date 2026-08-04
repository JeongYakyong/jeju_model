"""smp_forecaster — Stage 2 of the Jeju forecasting pipeline.

운영 모델 = model9 핵심13@재선택 (피처 13개, TAU_SOFT=0.06, TAU_HARD=0.50).
음수(-)SMP 탐지 1순위 운영점.

Public API:
    train(...)              # DB → BANK → 학습 → joblib 저장
    predict_smp(date, ...)  # target_date의 24h SMP 예측 DataFrame
    load_model(...)         # 저장된 (model, artifact) 로드
    ingest_rt_smp(...)      # RT SMP CSV → DB (선택)

Stage 1을 라이브러리로 사용한다(`from net_load_forecaster import ...`).
"""
from __future__ import annotations

from .config import (
    CORE_COLS, SMP_MODEL_PATH, DB_PATH,
    TAU_SOFT, TAU_HARD, TAU_HI, TAU_LO, NL_CUTS,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)
from .data_pipeline import ingest_rt_smp, load_rt_smp_csv
from .db_extension import (
    ensure_realtime_smp_table, save_realtime_smp, get_realtime_smp,
)
from .features import build_features
from .loader import load_model
from .predict import predict_smp
from .train import train

__all__ = [
    # 핵심 API
    'train', 'predict_smp', 'load_model',
    'ingest_rt_smp',
    # 빌딩 블록
    'build_features', 'load_rt_smp_csv',
    'ensure_realtime_smp_table', 'save_realtime_smp', 'get_realtime_smp',
    # 상수
    'CORE_COLS',
    'TAU_SOFT', 'TAU_HARD', 'TAU_HI', 'TAU_LO', 'NL_CUTS',
    'TRAIN_START', 'TRAIN_END', 'VAL_START', 'VAL_END',
    'TEST_START', 'TEST_END',
    'SMP_MODEL_PATH', 'DB_PATH',
]
