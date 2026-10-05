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

import numpy as np
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


FUT_STATIONS = ["west", "south"]   # solar 모델이 쓰는 지점만 (east 제외, SOLAR_COLS 와 동일)
KIMG_VARS = ["total_cloud", "midlow_cloud", "radiation", "rainfall"]   # radiation -> solar_rad 로 개명
KIMG_ARCHIVE_START = "2025-12-20"   # forecast_horizon 실존 시작 (이 전은 QM(JMA) 대체)
RAIN_HOURLY_CAP = 100.0             # mm/h 물리적 상한 (KIMG 강수 sentinel ~1310.7mm 방지)
OLD_MODEL_EVAL_HORIZONS = [2, 3, 4, 5]   # 현행 서빙 모델 D+n 비교용 *_kimg_d{n}
REUSE_EVAL_HORIZON = 2                   # 새 모델 D+1 가중치 재사용 평가 — JMA 운량 예보가 D+2 까지뿐


REFDATA_DIR = REPO_ROOT / "data" / "refdata"
SEOGWIPO_ASOS_CSV = REFDATA_DIR / "meteo_data" / "Seogwipo_Weather_Final_Cleaned.csv"   # 189 = south, 일조(hr) 제공
SATELLITE_SOUTH_CSV = REFDATA_DIR / "meteo_data" / "satellite_south.csv"
GAP_REFERENCE_START = "2024-05-01"   # 이 날부터 남부 일사에 결손이 없다 — 위성 환산 비율의 기준 구간
GAP_SATELLITE_MIN = 0.1              # MJ/m^2/h — 위성이 이보다 밝아야 "해가 있었다"로 본다


def _fill_south_radiation_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """남부(서귀포 189) ASOS 일사의 '측정 안 함 → 0' 결손을 위성으로 채운다 (2026-10-02 사용자 확정).

    2020-01 ~ 2024-04 에 매달 50~80시간, 주로 오전 7~11시가 0 이다. 일출 탓이 아니다 —
    한여름 09시(태양고도 ~50°)에도 일조 1.0시간·위성 2.0MJ 인데 일사만 0 이고, 2024-05 이후엔 0건.
    조건: ASOS 일사 == 0  AND  일조 > 0  AND  위성 > 0.1MJ  → 확실히 빠진 시간만.
    값:   위성 × (2024-05~ ASOS/위성 비율, 월·시각별) — 지금 ASOS 눈금으로 환산.
    나머지 ASOS 값은 그대로 둔다. 원본은 solar_rad_south_raw, 채운 시각은 solar_rad_south_filled=1.
    서부(고산)는 해당 시간이 2020년 30여 시간뿐이고 일조 자료가 없어 손대지 않는다.
    """
    seogwipo = pd.read_csv(SEOGWIPO_ASOS_CSV, encoding="cp949")
    sunshine = pd.Series(pd.to_numeric(seogwipo.iloc[:, 7], errors="coerce").values,   # 7번째 열 = 일조(hr)
                         index=pd.to_datetime(seogwipo.iloc[:, 0]))
    satellite_raw = pd.read_csv(SATELLITE_SOUTH_CSV, parse_dates=["time_kst"])
    satellite = pd.Series(satellite_raw["shortwave_radiation"].values * 0.0036, index=satellite_raw["time_kst"])

    frame = df.set_index("timestamp")
    asos = frame["solar_rad_south"]
    sat = satellite.reindex(frame.index)

    reference = pd.DataFrame({"asos": asos, "sat": sat})[frame.index >= pd.Timestamp(GAP_REFERENCE_START)]
    reference = reference[(reference["asos"] > 0) & (reference["sat"] > GAP_SATELLITE_MIN)]
    sums = reference.groupby([reference.index.month, reference.index.hour]).sum()
    ratio = (sums["asos"] / sums["sat"]).rename("ratio")
    ratio_by_row = pd.Series(
        ratio.reindex(list(zip(frame.index.month, frame.index.hour))).values, index=frame.index)

    is_gap = (asos == 0) & (sunshine.reindex(frame.index) > 0) & (sat > GAP_SATELLITE_MIN) & ratio_by_row.notna()
    df = df.copy()
    df["solar_rad_south_raw"] = asos.values
    df["solar_rad_south"] = np.where(is_gap.values, (sat * ratio_by_row).round(2).values, asos.values)
    df["solar_rad_south_filled"] = is_gap.values.astype(int)
    by_year = is_gap[is_gap].groupby(is_gap[is_gap].index.year).size().to_dict()
    print(f"[남부 일사 결손 채움] {int(is_gap.sum())}시간 (연도별 {by_year})")
    return df


def _read_kimg_horizon(con, horizon_d: int) -> pd.DataFrame:
    """forecast_horizon 에서 12z 발표·지평 horizon_d 인 행만 (timestamp 당 1행).

    ★freshest(horizon_d 최소·base 최신)를 쓰지 않는다 — 2026-08-27 부터 18z 당일예보(hd=0)가
    섞여 들어와 D+1 보다 좋은 정보로 학습·평가하게 된다. 서빙 모델은 12z-origin 이므로 12z 만.
    """
    cols = ", ".join(f'"{v}_{st}"' for v in KIMG_VARS for st in FUT_STATIONS)
    q = f"""
    SELECT timestamp, {cols} FROM forecast_horizon
    WHERE substr(base, 12) = '21:00:00' AND horizon_d = {horizon_d}
      AND timestamp >= '{KIMG_ARCHIVE_START}'
    ORDER BY timestamp
    """
    kimg = pd.read_sql(q, con, parse_dates=["timestamp"])
    assert not kimg["timestamp"].duplicated().any(), f"hd={horizon_d}: timestamp 중복 (base 가 겹침)"
    for st in FUT_STATIONS:
        kimg[f"rainfall_{st}"] = pd.to_numeric(kimg[f"rainfall_{st}"], errors="coerce").clip(upper=RAIN_HOURLY_CAP)
    return kimg.rename(columns={f"radiation_{st}": f"solar_rad_{st}" for st in FUT_STATIONS})


def _load_future_source(db_path: Path, qm_dir: Path) -> pd.DataFrame:
    """"미래(decoder)" 슬라이스용 예보 시계열 (2026-10-02 사용자 확정 구성).

    `*_fut` (학습·서빙 입력):
      total_cloud = **JMA 만** (KIMG 운량은 부적절 판정) — ~2022-07-01 분석치 월별QM, 이후 JMA D+1 예보
      solar_rad / rainfall = ~2025-12-19 JMA→KIMG 월별QM, 2025-12-20~ 실제 KIMG 12z D+1
      midlow_cloud 는 없다 (JMA previous run 에 층별운량 없음 → 피처에서 제외)
    평가 전용:
      `*_fut_d2`      새 모델 D+1 가중치 재사용 검증 (JMA 운량 D+2 + KIMG 12z D+2)
      `*_kimg`, `*_kimg_d{n}`  현행 서빙 모델 비교용 KIMG 원본 4변수 (KIMG 구간만)
    입력: fit_jma_monthly_qm.py 의 jma_future_{st}.csv
    """
    con = sqlite3.connect(str(db_path))
    try:
        kimg_by_horizon = {n: _read_kimg_horizon(con, n).set_index("timestamp")
                           for n in [1] + OLD_MODEL_EVAL_HORIZONS}
    finally:
        con.close()

    jma_parts = []
    for st in FUT_STATIONS:
        path = qm_dir / f"jma_future_{st}.csv"
        if not path.exists():
            sys.exit(f"[ERR] {path} 없음 — 먼저 fit_jma_monthly_qm.py 를 실행할 것")
        jma = pd.read_csv(path, parse_dates=["time_kst"]).set_index("time_kst")
        jma_parts.append(pd.DataFrame({
            f"total_cloud_{st}_fut": jma["total_cloud"],
            f"solar_rad_{st}_fut": jma["radiation"],
            f"rainfall_{st}_fut": jma["rainfall"],
            f"total_cloud_{st}_fut_d{REUSE_EVAL_HORIZON}": jma["total_cloud_d2"],
        }))
    fut = pd.concat(jma_parts, axis=1)
    fut.index.name = "timestamp"

    # 일사·강수: KIMG 구간은 실제 KIMG 로 덮어쓴다 (QM 값은 이 구간에 이미 비어 있다)
    kimg_d1 = kimg_by_horizon[1]
    kimg_rows = fut.index >= pd.Timestamp(KIMG_ARCHIVE_START)
    for st in FUT_STATIONS:
        for var in ("solar_rad", "rainfall"):
            fut.loc[kimg_rows, f"{var}_{st}_fut"] = kimg_d1[f"{var}_{st}"].reindex(fut.index[kimg_rows]).values
            fut[f"{var}_{st}_fut_d{REUSE_EVAL_HORIZON}"] = \
                kimg_by_horizon[REUSE_EVAL_HORIZON][f"{var}_{st}"].reindex(fut.index)

    # 현행 서빙 모델 비교용 KIMG 원본
    for horizon_d, kimg in kimg_by_horizon.items():
        suffix = "_kimg" if horizon_d == 1 else f"_kimg_d{horizon_d}"
        fut = fut.join(kimg.add_suffix(suffix), how="left")
    return fut.reset_index()


def export(db_path: Path, out_path: Path, start: str | None, end: str | None,
          with_future: bool = False, qm_dir: Path | None = None) -> None:
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

    if with_future:
        df = _fill_south_radiation_gaps(df)
        # 눈금 맞춤(변화 시점 이전 ASOS 일사에 월·시각 배율)은 시도 후 철회 — 인위적 수정 (사용자 결정 2026-10-02)
        fut = _load_future_source(db_path, qm_dir)
        before = len(df)
        df = df.merge(fut, on="timestamp", how="left")
        assert len(df) == before, "merge 로 행수가 바뀌면 안 된다(timestamp 유일성 깨짐)"
        fut_cols = [c for c in fut.columns if c != "timestamp"]
        na_rate = df[fut_cols].isna().mean()
        print("[future 소스] 결측률:")
        print(na_rate.round(4).to_string())

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
    ap.add_argument("--with-future", action="store_true",
                    help="과거(ASOS)=기존 그대로 + 미래(KIMG-proxy) *_fut 컬럼 추가")
    ap.add_argument("--qm-dir", type=Path,
                    default=REPO_ROOT / "data" / "refdata" / "meteo_data" / "processed",
                    help="fit_jma_monthly_qm.py 산출물(jma_future_{st}.csv) 위치")
    args = ap.parse_args()
    export(args.db, args.out, args.start, args.end, with_future=args.with_future, qm_dir=args.qm_dir)


if __name__ == "__main__":
    main()
