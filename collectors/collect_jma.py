# -*- coding: utf-8 -*-
"""collect_jma.py -- JMA MSM 운량·일사 예보(Open-Meteo) -> 메인 DB forecast_jma.

왜 필요한가 (2026-10-02 사용자 확정)
----
태양광 PatchTST 를 "미래 입력 = 수치예보"로 재학습하면서 **운량은 JMA 만** 쓰기로 했다
(KIMG 전운량은 과대라 부적절).  학습은 Open-Meteo previous-runs 의 JMA D+1 예보로 했으므로
서빙도 같은 모델(jma_msm)·같은 좌표의 JMA 운량을 받아야 한다.

무엇을 받나
----
- 모델 jma_msm, 변수 cloud_cover(%) -> total_cloud_{west,south,east} (0~1)
- (2026-10-08) shortwave_radiation(W/m2) -> solar_rad_{west,south} (MJ/m2/h = W x 0.0036).
  태양광 재학습이 일사도 JMA 원본으로 배우기 때문이다 (KIMG 일사는 흐린 날 +19% 과대). east 는 일사계가 없다.
  학습 파일(jma_previous_runs_*.csv)의 W/m2 와 단위·시각 정렬이 같음을 확인(전일 00 UTC 실행, 최대차 8 W/m2).
- 좌표 = KMA 수집점(collect_forecast POINTS_JEJU_V2)과 동일.  학습 파일
  (data/refdata/meteo_data/jma_previous_runs_*.csv)과 값이 100% 일치함을 확인(2026-10-02).
- 실행: 기본은 그날의 **12 UTC 실행** (= 21 KST, KMA 12z base 와 같은 시각).
  ★12/00 UTC 실행만 78h 까지 있어 D+1~D+3 을 덮는다. 다른 실행(03·06·09·15...)은 39h 라 D+1 만.

시간 지연과 가교 (2026-10-06 사용자 결정)
----
Open-Meteo 에 실행이 올라오기까지 약 3.5시간 걸린다(03 UTC 실행 -> 06:33 UTC 확인).
12 UTC 실행은 00:30 KST 정규 실행보다 늦게 올라오는 날이 많다. 기다리지 않고:
  - 12 UTC 실행이 아직 없으면 아무것도 받지 않고 끝낸다 → 매시 복구(run_pipeline repair)가 다시 시도.
  - --bridge (cron 23:00 KST): 그날 **06 UTC 실행**(39h, D+1 을 덮음)을 받아 둔다. 12 UTC 가 늦는 동안
    serve_chain 이 D+1 만 임시로 예측하는 가교용이다.
  - 그날 12 UTC 실행을 저장하면 같은 날의 다른 실행(가교)은 지운다 — 정식 갱신 뒤에는 필요 없다.
실제로 어느 실행을 받았는지는 run_time_utc 컬럼에 남는다.

저장 (forecast_jma, 메인 DB)
----
(run_time_utc, timestamp) 가 키. timestamp 는 KST.  같은 실행을 다시 받으면 행을 교체한다.

사용 예
    python collectors/collect_jma.py                     # 오늘 12 UTC 실행 (아직 없으면 아무것도 안 받음)
    python collectors/collect_jma.py --bridge            # 오늘 06 UTC 실행 (D+1 가교용, cron 23:00 KST)
    python collectors/collect_jma.py --run 2026-10-01T12 # 지정 실행
    python collectors/collect_jma.py --backfill 30       # 과거 30일의 12 UTC 실행 (single-runs 는 2026-04-02~)
    python collectors/collect_jma.py --no-save           # 받기만 하고 저장 안 함

Weather data by Open-Meteo.com (CC BY 4.0)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # noqa: E402

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
META_URL = "https://api.open-meteo.com/data/jma_msm/static/meta.json"
MODEL = "jma_msm"
RUN_HOUR_UTC = 12
BRIDGE_RUN_HOUR_UTC = 6     # 가교 실행 (39h — D+1 만 덮는다)
TABLE = "forecast_jma"
MJ_PER_W = 0.0036            # W/m2 -> MJ/m2/h (학습 파일과 같은 환산)
RADIATION_STATIONS = ["west", "south"]
REQUEST_TIMEOUT = 60

# KMA 수집점과 같은 좌표 (collect_forecast POINTS_JEJU_V2)
POINTS = [
    {"suffix": "west",  "lat": 33.4427, "lon": 126.1713},   # 고산
    {"suffix": "east",  "lat": 33.3868, "lon": 126.8802},   # 성산
    {"suffix": "south", "lat": 33.3284, "lon": 126.8366},   # 남쪽 태양광 단지
]


def latest_available_run() -> datetime:
    """Open-Meteo 에 마지막으로 올라온 jma_msm 실행 시각 (UTC, tz 없음)."""
    meta = requests.get(META_URL, timeout=REQUEST_TIMEOUT).json()
    return datetime.fromtimestamp(meta["last_run_initialisation_time"], timezone.utc).replace(tzinfo=None)


def fetch_run(run_time: datetime) -> pd.DataFrame:
    """한 실행의 지점별 운량·일사. 반환: timestamp(KST), lead_hour, total_cloud_{지점}, solar_rad_{west,south}."""
    params = {
        "latitude": ",".join(str(p["lat"]) for p in POINTS),
        "longitude": ",".join(str(p["lon"]) for p in POINTS),
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": "cloud_cover,shortwave_radiation",
        "models": MODEL,
    }
    response = requests.get(SINGLE_RUNS_URL, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo {response.status_code}: {response.text[:200]}")
    locations = response.json()
    if isinstance(locations, dict):        # 지점이 하나면 리스트가 아니라 dict 로 온다
        locations = [locations]

    frame = None
    for point, location in zip(POINTS, locations):
        hourly = location["hourly"]
        times_utc = pd.to_datetime(hourly["time"])
        cloud = pd.to_numeric(pd.Series(hourly["cloud_cover"]), errors="coerce") / 100.0
        columns = {"time_utc": times_utc, f"total_cloud_{point['suffix']}": cloud.values}
        if point["suffix"] in RADIATION_STATIONS:
            radiation = pd.to_numeric(pd.Series(hourly["shortwave_radiation"]), errors="coerce") * MJ_PER_W
            columns[f"solar_rad_{point['suffix']}"] = radiation.values
        part = pd.DataFrame(columns)
        frame = part if frame is None else frame.merge(part, on="time_utc", how="outer")

    cloud_cols = [f"total_cloud_{p['suffix']}" for p in POINTS]
    radiation_cols = [f"solar_rad_{suffix}" for suffix in RADIATION_STATIONS]
    frame = frame.dropna(subset=cloud_cols, how="all")     # 예보 길이 밖은 전부 NaN 으로 온다
    frame["timestamp"] = frame["time_utc"] + timedelta(hours=9)
    frame["lead_hour"] = ((frame["time_utc"] - run_time) / timedelta(hours=1)).astype(int)
    frame["run_time_utc"] = run_time.strftime("%Y-%m-%d %H:%M:%S")
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame[["run_time_utc", "timestamp", "lead_hour"] + cloud_cols + radiation_cols]


def save(frame: pd.DataFrame) -> int:
    value_cols = [c for c in frame.columns if c.startswith(("total_cloud_", "solar_rad_"))]
    con = sqlite3.connect(P.DB_JEJU)
    try:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ("
            "run_time_utc TEXT NOT NULL, timestamp TEXT NOT NULL, lead_hour INTEGER, "
            + ", ".join(f"{c} REAL" for c in value_cols)
            + ", PRIMARY KEY (run_time_utc, timestamp))")
        existing = {row[1] for row in con.execute(f"PRAGMA table_info({TABLE})")}
        for column in value_cols:                      # 운량만 있던 옛 테이블에 일사 열 추가
            if column not in existing:
                con.execute(f"ALTER TABLE {TABLE} ADD COLUMN {column} REAL")
        before = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        columns = list(frame.columns)
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            frame.itertuples(index=False, name=None))
        con.commit()
        after = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    finally:
        con.close()
    return after - before


def delete_bridge_runs(run_time: datetime) -> int:
    """12 UTC 실행을 저장한 뒤, 같은 UTC 날짜의 다른 실행(06 UTC 가교 등)을 지운다."""
    day = run_time.strftime("%Y-%m-%d")
    keep = run_time.strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(P.DB_JEJU)
    try:
        deleted = con.execute(
            f"DELETE FROM {TABLE} WHERE substr(run_time_utc, 1, 10) = ? AND run_time_utc != ?",
            (day, keep)).rowcount
        con.commit()
    finally:
        con.close()
    return deleted


def target_run_for_today() -> datetime:
    """지금 시각 기준 '오늘 밤 12z base' 에 해당하는 12 UTC 실행.

    12z 파이프라인은 KST 자정 직후(00:20)에 돌고, 그 base 는 전날 21 KST = 전날 12 UTC 다.
    UTC 로는 '가장 최근에 지난 12:00' 이다.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    return run if now >= run else run - timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="JMA MSM 운량 예보 -> forecast_jma")
    parser.add_argument("--run", help="지정 실행 (UTC, 예: 2026-10-01T12)")
    parser.add_argument("--backfill", type=int, default=0, help="과거 N일의 12 UTC 실행")
    parser.add_argument("--bridge", action="store_true", help="오늘 06 UTC 실행 (D+1 가교용)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.run:
        runs = [datetime.strptime(args.run, "%Y-%m-%dT%H")]
    elif args.backfill:
        last = target_run_for_today()
        runs = [last - timedelta(days=d) for d in range(args.backfill, -1, -1)]
    elif args.bridge:
        runs = [target_run_for_today().replace(hour=BRIDGE_RUN_HOUR_UTC)]
    else:
        runs = [target_run_for_today()]

    latest = latest_available_run()
    failed = 0
    for run_time in runs:
        if run_time > latest:   # 아직 안 올라온 실행 — 다음 복구 때 다시 시도
            print(f"[미게시] {run_time:%Y-%m-%d %H}UTC 실행 아직 없음 (최신 {latest:%Y-%m-%d %H}UTC) — 건너뜀")
            continue
        try:
            frame = fetch_run(run_time)
        except Exception as error:   # 한 실행 실패가 백필 전체를 막지 않도록
            print(f"[실패] {run_time:%Y-%m-%d %H}UTC: {error}")
            failed += 1
            continue
        lead_max = frame["lead_hour"].max()
        message = f"{run_time:%Y-%m-%d %H}UTC 실행: {len(frame)}시각 (선행 {frame['lead_hour'].min()}~{lead_max}h)"
        if args.no_save:
            print(message + " [저장 안 함]")
        else:
            print(message + f" -> {TABLE} +{save(frame)}행 (교체 포함 시 0일 수 있음)")
            if run_time.hour == RUN_HOUR_UTC:
                deleted = delete_bridge_runs(run_time)
                if deleted:
                    print(f"  같은 날 가교 실행 {deleted}행 삭제 (12 UTC 정식 실행으로 대체)")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
