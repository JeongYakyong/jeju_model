"""핵심 13개 피처 빌더 / Feature builder for the production SMP model.

`build_features(src, mode)`만 외부에 노출한다.

- mode='train' : Stage 1의 historical_data 형식(실측 이용률). 이 함수가
  반환한 DataFrame은 학습 직전에 noise-injection을 입은 다음 사용된다.
- mode='serve' : Stage 1의 forecast_data 형식(예보 = est_* 접두사 + 결측치
  포함). 컬럼명을 통일하고 NaN을 양방향 보간한다.

두 모드 모두 결과의 컬럼 순서와 이름이 완전히 동일하다.
"""
from __future__ import annotations

from typing import Literal

import holidays
import numpy as np
import pandas as pd

from .config import CORE_COLS, NL_CUTS

# CORE_COLS(모델 입력) + combine_v가 구조적으로 쓰는 컬럼들
_EXTRA_FOR_COMBINE = ['is_zoneA', 'net_load', 'zone_code']
_OUTPUT_COLS = CORE_COLS + _EXTRA_FOR_COMBINE


def build_features(
    src: pd.DataFrame,
    mode: Literal['train', 'serve'],
) -> pd.DataFrame:
    """입력 DataFrame에서 핵심 피처 13개 + combine 보조 컬럼을 만든다.

    Args:
        src: 시간 인덱스 DataFrame. 'train'은 historical 컬럼,
             'serve'는 forecast 컬럼(est_ 접두사 포함).
        mode: 'train' | 'serve'.

    Returns:
        같은 인덱스에 _OUTPUT_COLS 만 담은 DataFrame.
    """
    if mode not in ('train', 'serve'):
        raise ValueError(f"mode must be 'train' or 'serve', got {mode!r}")

    df = src.copy()

    # 1) 예보 모드: 컬럼명 통일 + 수치 결측 보간 (model9와 동일)
    if mode == 'serve':
        df = df.rename(columns={
            'est_Solar_Utilization': 'Solar_Utilization',
            'est_Wind_Utilization':  'Wind_Utilization',
        })
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = (df[numeric_cols]
                            .interpolate(limit_direction='both')
                            .ffill().bfill())

    # 2) net_load = 수요 - (태양광 + 풍력)  (두 모드 동일 식)
    solar_mw = df['Solar_Utilization'] * df['Solar_Capacity_Est']
    wind_mw  = df['Wind_Utilization']  * df['Wind_Capacity_Est']
    df['net_load'] = df['est_demand'] - (solar_mw + wind_mw)

    # 3) 구간 코드 (A/B/C/D)
    nl = df['net_load']
    df['zone_code'] = np.select(
        [nl < NL_CUTS[0], nl < NL_CUTS[1], nl < NL_CUTS[2]],
        [0, 1, 2],
        default=3,
    )
    df['is_zoneA'] = (df['zone_code'] == 3).astype(int)

    # 4) 미래 net_load (Phase 6 핵심 신호) - 핵심13은 lead_1, lead_2만 사용
    df['nl_lead_1'] = nl.shift(-1)
    df['nl_lead_2'] = nl.shift(-2)

    # 5) DA 기준값 (Phase 3 보정식)
    df['da_anchor'] = 24.94 + 0.754 * df['smp_jeju']

    # 6) 달력 - timestamp에서 다시 계산
    years = range(df.index.year.min(), df.index.year.max() + 1)
    kr_holidays = holidays.KR(years=years)
    df['hour']  = df.index.hour
    df['month'] = df.index.month
    df['spring_midday'] = (
        df['month'].between(3, 5) & df['hour'].between(10, 13)
    ).astype(int)
    df['is_market_pretest'] = (df.index < pd.Timestamp('2024-06-01')).astype(int)

    # holidays는 핵심13엔 없지만, 향후 ablation/검증 호환을 위해 만들지는 않는다.
    _ = kr_holidays  # 의도적으로 사용하지 않음 (핵심13에 dow/holiday 없음)

    return df[_OUTPUT_COLS]


def assert_parity(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """학습본과 서빙본의 피처 컬럼/순서가 동일한지 확인."""
    if list(a.columns) != list(b.columns):
        raise AssertionError(
            f"피처 컬럼/순서 불일치: train={list(a.columns)} vs "
            f"serve={list(b.columns)}"
        )
