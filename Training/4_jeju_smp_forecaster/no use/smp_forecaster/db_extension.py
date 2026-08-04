"""Stage 1 JejuEnergyDB에 RT SMP 테이블을 함수 형태로 얹는다.

Stage 1 코드는 건드리지 않는다. 모든 함수는 `JejuEnergyDB` 인스턴스를
받아 그 안의 `conn`(sqlite3.Connection)에서 동작한다.

테이블:
    realtime_smp(
        timestamp TEXT PRIMARY KEY,   -- 'YYYY-MM-DD HH:MM:SS'
        smp_rt    REAL,               -- 1~4구간 평균 (시간별 RT SMP)
        smp_rt_neg INTEGER            -- 1구간이라도 < 0 이면 1
    )
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from net_load_forecaster.db_manager import JejuEnergyDB


def ensure_realtime_smp_table(db: JejuEnergyDB) -> None:
    """realtime_smp 테이블이 없으면 생성한다."""
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS realtime_smp (
            timestamp  TEXT PRIMARY KEY,
            smp_rt     REAL,
            smp_rt_neg INTEGER,
            updated_at TEXT
        )
    """)
    db.conn.commit()


def save_realtime_smp(db: JejuEnergyDB, df: pd.DataFrame) -> int:
    """RT SMP 시간별 DataFrame을 UPSERT한다.

    Args:
        df: index가 datetime인 DataFrame. 컬럼: smp_rt, smp_rt_neg.

    Returns:
        저장(또는 갱신)한 행 수.
    """
    if df.empty:
        return 0

    ensure_realtime_smp_table(db)

    now = datetime.now().isoformat()
    rows = []
    for ts, row in df.iterrows():
        rows.append((
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            None if pd.isna(row['smp_rt']) else float(row['smp_rt']),
            int(bool(row['smp_rt_neg'])) if pd.notna(row.get('smp_rt_neg')) else None,
            now,
        ))

    db.conn.executemany("""
        INSERT INTO realtime_smp (timestamp, smp_rt, smp_rt_neg, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(timestamp) DO UPDATE SET
            smp_rt     = COALESCE(excluded.smp_rt,     realtime_smp.smp_rt),
            smp_rt_neg = COALESCE(excluded.smp_rt_neg, realtime_smp.smp_rt_neg),
            updated_at = excluded.updated_at
    """, rows)
    db.conn.commit()
    return len(rows)


def get_realtime_smp(
    db: JejuEnergyDB,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """RT SMP를 시간순 DataFrame으로 조회한다.

    Args:
        start, end: 'YYYY-MM-DD' 또는 'YYYY-MM-DD HH:MM:SS'. None이면 전체.

    Returns:
        timestamp(DatetimeIndex), smp_rt, smp_rt_neg. 비어있으면 빈 DataFrame.
    """
    ensure_realtime_smp_table(db)

    query = "SELECT timestamp, smp_rt, smp_rt_neg FROM realtime_smp"
    conditions = []
    if start:
        conditions.append(f"timestamp >= '{start}'")
    if end:
        conditions.append(f"timestamp <= '{end}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp"

    df = pd.read_sql(query, db.conn)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')
