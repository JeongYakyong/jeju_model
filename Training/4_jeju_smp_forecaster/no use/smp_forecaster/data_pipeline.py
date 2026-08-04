"""RT SMP CSV → SQLite 인제스트 / RT SMP CSV to DB.

production 입력은 `clean_rt_smp.csv` (AX_model2/prep_rt_smp.py 가 만든
정제본). 컬럼 구조:

    timestamp, g1, g2, g3, g4,
    smp_rt_daily_max, smp_rt_daily_min, smp_rt_daily_wavg,
    smp_rt_hourly_mean,   # 4구간 평균 = 시간별 RT SMP 타깃
    smp_rt_neg_flag       # 4구간 최소가 < 0 (시간 내 음구간 포착)

이 두 컬럼만 뽑아 DB realtime_smp 테이블에 넣는다. 원본인
`realtime_smp_24-26.csv` 는 *재료* 라 직접 읽지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from net_load_forecaster.db_manager import JejuEnergyDB

from .config import RT_SMP_CSV
from .db_extension import save_realtime_smp


def load_rt_smp_csv(csv_path: str | Path | None = None) -> pd.DataFrame:
    """clean_rt_smp.csv → 시간별 (smp_rt, smp_rt_neg) DataFrame."""
    path = Path(csv_path or RT_SMP_CSV)
    df = pd.read_csv(path, encoding='cp949')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    neg_flag = df['smp_rt_neg_flag'].astype(str).str.upper().eq('TRUE')
    out = pd.DataFrame({
        'smp_rt': df['smp_rt_hourly_mean'].values,
        'smp_rt_neg': neg_flag.astype(int).values,
    }, index=df.index)
    return out


def ingest_rt_smp(
    db: JejuEnergyDB | None = None,
    csv_path: str | Path | None = None,
) -> int:
    """RT SMP CSV를 읽어 DB realtime_smp 테이블에 UPSERT한다.

    Args:
        db: 열린 JejuEnergyDB. None이면 기본 경로로 새로 연다(호출자가 닫지 않음).
        csv_path: RT SMP CSV 경로. None이면 config.RT_SMP_CSV.

    Returns:
        저장한 행 수.
    """
    own_db = db is None
    if own_db:
        db = JejuEnergyDB()
    try:
        df = load_rt_smp_csv(csv_path)
        return save_realtime_smp(db, df)
    finally:
        if own_db:
            db.close()
