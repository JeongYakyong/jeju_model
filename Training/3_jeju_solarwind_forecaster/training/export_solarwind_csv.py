"""DB -> Colab 학습용 CSV 추출 (3지점 solar/wind PatchTST 재학습용).

`1. data_fetcher_and_db/data/input_data_jeju.db` 의 `historical` 테이블에서
태양광(west/south)·풍력(west/east/south) 재학습에 필요한 "원천(raw)" 컬럼만
뽑아 하나의 wide CSV 로 저장한다. 파생 피처(Hour_sin, solar_damping, wind_zone,
풍속 다항식 등)는 Colab 노트북 안에서 만든다 — 이 스크립트는 순수 추출만 담당.

설계 결정(2026-06-01 게이트):
  - 결합:   지점별 피처를 "별도 채널"로 concat (평균 X) -> 기존 cross-attention 재사용
  - solar:  west + south  (east 는 추론(forecast) 시점에 일사/구름이 없어 제외)
  - wind:   west + east + south  (wind_spd/wd 가 3지점 모두 100% 가용)
  - target: real_solar_utilization_jeju / real_wind_utilization_jeju (0~1, DB 기성)

사용법 (로컬, repo 루트 어디서든):
    python Training/3_jeju_solarwind_forecaster/training/export_solarwind_csv.py
    # -> 같은 폴더에 solarwind_raw_jeju.csv 생성 -> Colab 에 업로드

옵션:
    --db   PATH   입력 DB 경로 (기본: project_paths.DB_JEJU = data/input_data_jeju.db)
    --out  PATH   출력 CSV 경로 (기본: 같은 폴더 solarwind_raw_jeju.csv)
    --start / --end  YYYY-MM-DD 로 기간 제한 (기본: 전체)

2026-07-30: DB 경로를 옛 폴더 구조 하드코딩에서 project_paths(SSOT)로 바꿨다.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# 저장소 경로 SSOT — project_paths.py 가 있는 폴더가 루트다
_HERE = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in _HERE.parents if (p / "project_paths.py").exists()), _HERE)
sys.path.insert(0, str(REPO_ROOT))
import project_paths as P   # noqa: E402

DEFAULT_DB = Path(P.DB_JEJU)
DEFAULT_OUT = _HERE / "solarwind_raw_jeju.csv"

# ── 추출할 historical 컬럼 ────────────────────────────────────────────────
SOLAR_COLS = [
    # 일사 (west/south 만 존재)
    "solar_rad_west", "solar_rad_south",
    # 구름
    "total_cloud_west", "total_cloud_south",
    "midlow_cloud_west", "midlow_cloud_south",
    # 강수 (-> solar_damping 파생용)
    "rainfall_west", "rainfall_south",
]
WIND_COLS = [
    "wind_spd_west", "wind_spd_east", "wind_spd_south",
    "wd_sin_west", "wd_cos_west",
    "wd_sin_east", "wd_cos_east",
    "wd_sin_south", "wd_cos_south",
]
TARGET_COLS = [
    "real_solar_utilization_jeju",
    "real_wind_utilization_jeju",
]
# 참고/검증용 (학습엔 직접 안 쓰지만 MW 환산·플롯에 유용)
REF_COLS = [
    "real_solar_gen_jeju", "real_wind_gen_jeju",
    "real_solar_capacity_jeju", "real_wind_capacity_jeju",
    "day_type",
]

EXPORT_COLS = ["timestamp"] + SOLAR_COLS + WIND_COLS + TARGET_COLS + REF_COLS


def export(db_path: Path, out_path: Path, start: str | None, end: str | None) -> None:
    if not db_path.exists():
        sys.exit(f"[ERR] DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    try:
        avail = pd.read_sql("PRAGMA table_info(historical)", con)["name"].tolist()
        missing = [c for c in EXPORT_COLS if c not in avail]
        if missing:
            sys.exit(f"[ERR] historical 에 없는 컬럼: {missing}")

        col_sql = ", ".join(f'"{c}"' for c in EXPORT_COLS)
        where = []
        if start:
            where.append(f"timestamp >= '{start} 00:00:00'")
        if end:
            where.append(f"timestamp <= '{end} 23:00:00'")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        q = f"SELECT {col_sql} FROM historical{where_sql} ORDER BY timestamp"
        df = pd.read_sql(q, con, parse_dates=["timestamp"])
    finally:
        con.close()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 요약 출력 (ASCII only — Windows CP949 안전)
    print(f"[OK] rows={len(df)}  range {df['timestamp'].min()} -> {df['timestamp'].max()}")
    tgt = df[TARGET_COLS].notna().mean() * 100
    print(f"     solar_util nonnull={tgt['real_solar_utilization_jeju']:.1f}%  "
          f"wind_util nonnull={tgt['real_wind_utilization_jeju']:.1f}%")
    # 핵심 피처 결측 점검
    for c in ["solar_rad_west", "solar_rad_south",
              "wind_spd_west", "wind_spd_east", "wind_spd_south"]:
        print(f"     {c:18s} nonnull={df[c].notna().mean()*100:5.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Colab(pandas)에서 한글 day_type 등 안전하게 읽힘
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="solar/wind 3지점 학습용 CSV 추출")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    args = ap.parse_args()
    export(args.db, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
