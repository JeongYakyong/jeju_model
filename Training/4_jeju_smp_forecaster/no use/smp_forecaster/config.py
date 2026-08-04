"""smp_forecaster 설정 / Paths and constants.

운영 모델(model9 핵심13@재선택)의 모든 상수가 한곳에 모여 있다.
환경변수로 경로를 덮어쓸 수 있다 (예: `SMP_MODELS_DIR=/custom/path`).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from net_load_forecaster.config import (
    PROJECT_ROOT,
    KST,
    DB_PATH,
    MODELS_DIR as _STAGE1_MODELS_DIR,
)

# ── 경로 ──────────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent

# Stage 1과 같은 디렉터리를 기본값으로 (모델 아티팩트 한곳에 모음)
MODELS_DIR = Path(os.getenv('SMP_MODELS_DIR', _STAGE1_MODELS_DIR))
SMP_MODEL_PATH = MODELS_DIR / 'smp_model.pkl'

# RT SMP 정제본 CSV (production 입력) — clean_rt_smp.csv 가 정식 소스.
# realtime_smp_24-26.csv 는 그 *재료*일 뿐 직접 읽지 않는다.
RT_SMP_CSV = Path(os.getenv(
    'SMP_RT_CSV',
    PROJECT_ROOT.parent / 'AX_model2' / 'clean_rt_smp.csv',
))

# ── 운영 모델 = 핵심 13개 피처 (model9 Part D 결정) ───────────────────────
CORE_COLS = [
    # DA / 기준값
    'smp_jeju', 'smp_land', 'da_anchor',
    # 미래 net_load (Phase 6 신호)
    'nl_lead_1', 'nl_lead_2',
    # 태양광 (음수 핵심축)
    'Solar_Utilization', 'solar_rad',
    # 풍력 (출력제어 전 외생 독립축)
    'wind_spd_north', 'Wind_Utilization',
    # 달력 (음수 단일 최대축)
    'hour', 'month', 'spring_midday', 'is_market_pretest',
]

# ── 운영점 (model9 Part D-2 재선택값) ─────────────────────────────────────
# 음수 분류기 2단 임계 (핵심13 전용 재선택)
TAU_SOFT = 0.06   # 애매 → 점예측 0
TAU_HARD = 0.50   # 확신 → 깊은 음수

# 바닥 분류기 임계 (model8 부터 불변)
TAU_HI = 0.30
TAU_LO = 0.05

# 비대칭비용 가중 (참고: 학습엔 안 쓰고 평가용)
W_BAD = 5.0   # 치명 = 예측+ 실제-
W_OK  = 0.3   # 가벼움 = 예측- 실제+

# net_load 구간 컷 (A/B/C/D)
NL_CUTS = [181.3, 264.5, 383.6]

# ── 학습/검증/시험 시간창 ─────────────────────────────────────────────────
# TRAIN  : historical_data 기반 학습 입력 (실측)
# VAL    : historical_data 기반 BANK 구성용 (실측 vs 예보 잔차)
# TEST   : forecast_data 기반 out-of-sample 평가용 (예보)
#
# VAL/TEST 는 동일 시간창. 소스만 다르다.
# realtime_smp 타깃 커버리지는 2026-05-14 까지라, 05-15 이후는 features
# 만 있고 비교용 ground truth 가 없다.
TRAIN_START = pd.Timestamp('2024-06-01')
TRAIN_END   = pd.Timestamp('2026-01-31 23:00')
VAL_START   = pd.Timestamp('2026-02-01')
VAL_END     = pd.Timestamp('2026-05-23 23:00')
TEST_START  = VAL_START
TEST_END    = VAL_END

# 시간순 확장폴드 분할점 (model8/9 동일)
CV_CUTS = [pd.Timestamp(s) for s in [
    '2024-09-01', '2025-01-01', '2025-04-01',
    '2025-07-01', '2025-10-01', '2026-01-01',
]]

# ── 학습 시드 ─────────────────────────────────────────────────────────────
RNG = 42

__all__ = [
    'PACKAGE_DIR', 'PROJECT_ROOT', 'MODELS_DIR', 'SMP_MODEL_PATH',
    'DB_PATH', 'RT_SMP_CSV', 'KST',
    'CORE_COLS',
    'TAU_SOFT', 'TAU_HARD', 'TAU_HI', 'TAU_LO',
    'W_BAD', 'W_OK', 'NL_CUTS',
    'TRAIN_START', 'TRAIN_END', 'VAL_START', 'VAL_END',
    'TEST_START', 'TEST_END', 'CV_CUTS',
    'RNG',
]
