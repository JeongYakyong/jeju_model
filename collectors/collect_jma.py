# -*- coding: utf-8 -*-
"""collect_jma.py -- JMA MSM 운량 예보(Open-Meteo) -> 메인 DB forecast_jma.

왜 필요한가 (2026-10-02 사용자 확정)
----
태양광 PatchTST 를 "미래 입력 = 수치예보"로 재학습하면서 **운량은 JMA 만** 쓰기로 했다
(KIMG 전운량은 과대라 부적절).  학습은 Open-Meteo previous-runs 의 JMA D+1 예보로 했으므로
서빙도 같은 모델(jma_msm)·같은 좌표의 JMA 운량을 받아야 한다.

무엇을 받나
----
- 모델 jma_msm, 변수 cloud_cover(%) -> total_cloud_{west,south,east} (0~1)
- 좌표 = KMA 수집점(collect_forecast POINTS_JEJU_V2)과 동일.  학습 파일
  (data/refdata/meteo_data/jma_previous_runs_*.csv)과 값이 100% 일치함을 확인(2026-10-02).
- 실행: 기본은 그날의 **12 UTC 실행** (= 21 KST, KMA 12z base 와 같은 시각).
  ★12/00 UTC 실행만 78h 까지 있어 D+1~D+3 을 덮는다. 다른 실행(03·06·09·15...)은 39h 라 D+1 만.

시간 지연
----
Open-Meteo 에 실행이 올라오기까지 약 3.5시간 걸린다(03 UTC 실행 -> 06:33 UTC 확인).
12 UTC 실행은 ~15:30 UTC = **~00:30 KST** 로 추정 — 12z 파이프라인(00:20 KST)보다 늦을 수 있다.
그래서:
  --wait-minutes N : 12 UTC 실행이 아직이면 N분까지 5분 간격으로 기다린다.
  그래도 없으면 **가장 최근에 올라온 실행**으로 대체하고 로그에 남긴다(39h 라 D+1 만 덮을 수 있음).
실제로 어느 실행을 받았는지는 run_time_utc 컬럼에 남는다.

저장 (forecast_jma, 메인 DB)
----
(run_time_utc, timestamp) 가 키. timestamp 는 KST.  같은 실행을 다시 받으면 행을 교체한다.

사용 예
    python collectors/collect_jma.py                     # 오늘 12 UTC 실행 (없으면 최신 실행)
    python collectors/collect_jma.py --wait-minutes 60   # 12 UTC 실행을 최대 60분 기다림
    python collectors/collect_jma.py --run 2026-10-01T12 # 지정 실행
    python collectors/collect_jma.py --backfill 30       # 과거 30일의 12 UTC 실행 (single-runs 는 2026-04-02~)
    python collectors/collect_jma.py --no-save           # 받기만 하고 저장 안 함

Weather data by Open-Meteo.com (CC BY 4.0)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
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
TABLE = "forecast_jma"
WAIT_INTERVAL_MINUTES = 5
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
    """한 실행의 지점별 운량. 반환: timestamp(KST), lead_hour, total_cloud_{지점}."""
    params = {
        "latitude": ",".join(str(p["lat"]) for p in POINTS),
        "longitude": ",".join(str(p["lon"]) for p in POINTS),
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": "cloud_cover",
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
        part = pd.DataFrame({"time_utc": times_utc, f"total_cloud_{point['suffix']}": cloud.values})
        frame = part if frame is None else frame.merge(part, on="time_utc", how="outer")

    cloud_cols = [f"total_cloud_{p['suffix']}" for p in POINTS]
    frame = frame.dropna(subset=cloud_cols, how="all")     # 예보 길이 밖은 전부 NaN 으로 온다
    frame["timestamp"] = frame["time_utc"] + timedelta(hours=9)
    frame["lead_hour"] = ((frame["time_utc"] - run_time) / timedelta(hours=1)).astype(int)
    frame["run_time_utc"] = run_time.strftime("%Y-%m-%d %H:%M:%S")
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame[["run_time_utc", "timestamp", "lead_hour"] + cloud_cols]


def save(frame: pd.DataFrame) -> int:
    cloud_cols = [c for c in frame.columns if c.startswith("total_cloud_")]
    con = sqlite3.connect(P.DB_JEJU)
    try:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ("
            "run_time_utc TEXT NOT NULL, timestamp TEXT NOT NULL, lead_hour INTEGER, "
            + ", ".join(f"{c} REAL" for c in cloud_cols)
            + ", PRIMARY KEY (run_time_utc, timestamp))")
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


def target_run_for_today() -> datetime:
    """지금 시각 기준 '오늘 밤 12z base' 에 해당하는 12 UTC 실행.

    12z 파이프라인은 KST 자정 직후(00:20)에 돌고, 그 base 는 전날 21 KST = 전날 12 UTC 다.
    UTC 로는 '가장 최근에 지난 12:00' 이다.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    return run if now >= run else run - timedelta(days=1)


def resolve_run(target: datetime, wait_minutes: int) -> datetime:
    """target 실행이 올라왔으면 그대로, 아니면 기다렸다가 끝내 없으면 최신 실행."""
    deadline = time.time() + wait_minutes * 60
    while True:
        latest = latest_available_run()
        if latest >= target:
            return target
        if time.time() >= deadline:
            print(f"[대체] {target:%Y-%m-%d %H}UTC 실행이 아직 없음 -> 최신 실행 {latest:%Y-%m-%d %H}UTC 사용 "
                  f"(12/00 UTC 가 아니면 39h 라 D+1 만 덮는다)")
            return latest
        print(f"[대기] {target:%H}UTC 실행 미게시 (최신 {latest:%Y-%m-%d %H}UTC) — {WAIT_INTERVAL_MINUTES}분 후 재확인")
        time.sleep(WAIT_INTERVAL_MINUTES * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="JMA MSM 운량 예보 -> forecast_jma")
    parser.add_argument("--run", help="지정 실행 (UTC, 예: 2026-10-01T12)")
    parser.add_argument("--backfill", type=int, default=0, help="과거 N일의 12 UTC 실행")
    parser.add_argument("--wait-minutes", type=int, default=0, help="12 UTC 실행을 기다릴 최대 분")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.run:
        runs = [datetime.strptime(args.run, "%Y-%m-%dT%H")]
    elif args.backfill:
        last = target_run_for_today()
        runs = [last - timedelta(days=d) for d in range(args.backfill, -1, -1)]
    else:
        runs = [resolve_run(target_run_for_today(), args.wait_minutes)]

    failed = 0
    for run_time in runs:
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
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
