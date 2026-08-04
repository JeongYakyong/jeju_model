# -*- coding: utf-8 -*-
"""collect_archive.py -- KIMR/KIMG 소스 분리 기상 수집 -> 메인 DB(input_data_jeju.db).

목적
----
① KIMR·KIMG 를 **소스 분리 테이블**로 동시 수집: forecast_kimr / forecast_kimg.
   두 테이블 모두 표준 컬럼명(temp_*, radiation_*, total_cloud_* ...) --
   출처는 테이블이 말한다.  지평 = 둘 다 D+1~D+5, 1h.
② 재학습이 소스 분리 테이블에서 변수별 소스(일사·운량 KIMR vs KIMG)를 고른다.
③ 3h 결손 안전장치: 1h 그리드 reindex 후 시간 보간(연속 2h 까지, 내부만) --
   2026-07-02~11 KIMG 운량이 3h 로 떨어졌던 사고 유형을 수집 단계에서 치유.
   보간 셀 수는 로그로 남긴다 (원자료가 아님을 추적).

2026-07-30: 격리 아카이브(weather_kim.db)를 메인 DB 로 흡수했다 -- 소스 비교가 끝나
  격리 이유(2026-07-17)가 소멸.  두 테이블은 이제 forecast_horizon(구 GRIB 서빙 입력)과
  같은 DB 에 나란히 산다.  재학습이 이 둘에서 서빙 입력을 새로 만든 뒤 GRIB 은 폐기된다.

소스별 수집 경로
----
forecast_kimr (KIMR/R030 3km -- 12z 기준 lead 상한 120h = 118시각, 꼬리 2h 정상 결손):
  ★NC 단일 스택 (2026-07-21 사용자 결정 -- KIMR 자체완결).  전에는 met=GRIB /
   일사=NC 로 프로토콜이 갈렸는데, 일사를 받으려면 어차피 NC per-hf 를 도니까
   같은 호출에 met 을 함께 요청한다 -> 호출수 증가 0, GRIB 3콜 소멸.
  - met + 일사: std NC per-hf 병렬 8워커, name = R030_NAME_LAND + TSKIN
    (T2 RH2 U10 V10 U80 V80 GUST MCAPE MCIN PBLH MSLP RAINC RAINNC GRAUPEL
     + SWDDIR2 SWDDIF2 ACSWDNB) -> temp/wind/.../radiation_*(GHI=직달+산란)
    + radiation_direct/diffuse_* + radiation_acswdnb_*(누적 diff)
  - 운량: 등압면 CLDFRA 24레벨 결합 (k2.fetch_r030_cldfra_long) -> total/midlow_cloud_*
forecast_kimg (KIMG/NE57 전구 8km, D+5 전 구간 1h -- hf<=135 안):
  - 현행 운영 fetch 재사용: cf.fetch_kimg_long + ci.kimg_one_point + ci.kimg_solar

사용 예
    python collectors/collect_archive.py                   # 최신 12z, 두 소스
    python collectors/collect_archive.py --kimr-only       # KIMR 만 (재취득/속도)
    python collectors/collect_archive.py --backfill 10     # 과거 10개 12z (소스별 resume-skip)
    python collectors/collect_archive.py --base 20260716   # 지정일 12z
    python collectors/collect_archive.py --backfill 12 --kimr-only --force
                                                # 이미 찬 base 도 행 교체 후 재수집

소스 태그
----
`forecast_kimr.src_met_proto` = 'NC' | 'GRIB' -- 그 행의 met 을 어느 프로토콜로
받았는지.  **새로 쌓이는 행은 항상 'NC'** 다 (2026-08-04 GRIB 폐기).  과거 행 중
'GRIB'/NULL 이 남아 있어 컬럼 자체는 유지한다 -- 옛 GRIB 행에는 cape/cinn 의
9999 sentinel 과 2바이트 랩어라운드(655.36 배수) 결함이 있으니 재학습 때 이 태그로
걸러낼 것.  NULL = 2026-07-22 태그 도입 이전 = 전부 GRIB.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import collect_forecast as cf     # base 선택·창 산식·적재 엔진·KIMG fetch 전부 여기
import pivot as ci
import kma_kimr_nc as k2
import postprocess as pp

KST = cf.KST
UTC = cf.UTC

DB_PATH = cf.DEFAULT_DB                          # 메인 DB (2026-07-30 격리 흡수)
TABLE_KIMR = "forecast_kimr"
TABLE_KIMG = "forecast_kimg"
DAYS_DEFAULT = 5
COVERAGE_WARN = 0.95
RAD_WORKERS = 8                                   # CLDFRA 와 동일 (std NC 병렬 실측 안전)

P_JEJU = k2.POINTS_JEJU_V2
SUFFIX_MAP = {p["name"]: p["sfx"] for p in P_JEJU}
# 건전성 sentinel: 각 테이블에서 소스가 공급해야 하는 대표 컬럼 (지점 x 3계열)
KEY_COLS = [f"{p}_{s}" for s in SUFFIX_MAP.values()
            for p in ("temp", "radiation", "total_cloud")]


def _window(base_utc: datetime, days: int) -> tuple[datetime, datetime]:
    """day-aligned KST 윈도우 (kma_kimg.window_bounds 위임).
    기본 [D+1 00시, D+days 24시); SAMEDAY_18Z + 18z 는 당일 창(04시부터)."""
    return cf.ckg.window_bounds(base_utc, days)


# ── KIMR: met + 일사 (std NC per-hf 병렬, 한 호출에 같이) ─────────────────────
# ★2026-07-21 사용자 결정: KIMR 은 **NC 단일 스택**으로 자체완결시킨다.
#   전에는 met=GRIB / 일사=NC / 운량=NC(등압면) 로 두 프로토콜이 섞여 있었다.
#   일사를 받으려면 어차피 NC per-hf 를 돌아야 하므로, 같은 호출에 met 을
#   함께 요청하면 **호출수 증가 0** 으로 GRIB 3콜이 통째로 사라진다.
#   2026-08-04: GRIB 폴백(--met grib)까지 제거하고 NC 단일 경로가 됐다 — met 은
#   GRIB↔NC 가 r 0.998~0.9997 로 사실상 같은 값임이 실측됐고(서빙 A/B 수요 차이
#   0.002%), 오히려 GRIB 쪽에만 cape/cinn 결함이 있었다.
# TSKIN(지표온)은 R030_NAME_LAND 에 없어 따로 붙인다 — 응답에 없으면
# derive_r030_categories 가 조용히 건너뛰므로 아래 커버리지 로그로 확인한다.
NAME_KIMR_FULL = k2.R030_NAME_LAND + ",TSKIN"


def fetch_kimr_nc_long(points: list[dict], base_utc: datetime, days: int,
                       names: str = NAME_KIMR_FULL,
                       workers: int = RAD_WORKERS) -> pd.DataFrame:
    """R030 std NC per-hf -> long (met 15종 + 일사 4종).

    수집 스택 자체는 `kma_kimr_nc.fetch_nc_long` 한 벌이다 (운영 met 경로와 공용 --
    2026-08-04 통합).  이 함수는 아카이브용 파생(derive_r030_categories: TEMP_C 등
    표준 컬럼 관례)을 지정하는 얇은 껍데기다.  누적 변수 diff 앵커(hf start-1)도
    엔진이 처리한다.
    """
    return k2.fetch_nc_long(points, base_utc, days, names,
                            k2.derive_r030_categories, workers=workers)


# 구 이름 호환 (일사만 받던 시절 호출부가 남아 있으면 그대로 동작)
def fetch_kimr_rad_long(points, base_utc, days, workers=RAD_WORKERS):
    """[구] 일사 3변수만 받는 판.  NC 전환 후에는 fetch_kimr_nc_long 이 met 까지 받는다."""
    return fetch_kimr_nc_long(points, base_utc, days,
                              names="SWDDIR2,SWDDIF2,ACSWDNB", workers=workers)


# ── 소스별 wide 빌더 ─────────────────────────────────────────────────────────
def kimr_wide_one(base_utc: datetime, days: int) -> pd.DataFrame:
    """KIMR-only wide -> 표준 컬럼명.

    met+일사 한 스택(std NC per-hf) + 운량(등압면 CLDFRA) — KIMR 자체완결.
    (구 `--met grib` 폴백은 2026-08-04 GRIB 폐기와 함께 제거.  met 은 GRIB↔NC 가
     r 0.998~0.9997 로 사실상 같은 값이었고, GRIB 쪽에만 cape/cinn 의 9999 sentinel
     과 2바이트 랩어라운드(655.36 배수) 결함이 있었다.)
    """
    met_long = fetch_kimr_nc_long(P_JEJU, base_utc, days)       # met + 일사 한 번에
    cld = k2.fetch_r030_cldfra_long(P_JEJU, base_utc, days, k2.R030_MAX_HF)
    longs = [d for d in (met_long, cld) if not d.empty]
    if not longs:
        return pd.DataFrame()
    wide = k2.long_to_wide_v2(pd.concat(longs, ignore_index=True),
                              SUFFIX_MAP, radiation_round=2)
    if wide.empty:
        return wide
    # 등압면 결합 운량 -> 표준 컬럼명 (출처는 테이블이 구분)
    wide = wide.rename(columns={c: c.replace("_r030", "")
                                for c in wide.columns if "_r030_" in c})
    return pp.clip_ranges(wide)


def kimg_wide_one(base_utc: datetime, days: int, point_workers: int = 1) -> pd.DataFrame:
    """KIMG-only wide (backfill_jeju_forecast.kimg_wide_one 의 D+5 판)."""
    with cf.forecast_days_override(days):
        window_start, window_end = cf._window_for([base_utc])
        kimg_long = cf.fetch_kimg_long([base_utc], point_workers=point_workers)
    if kimg_long.empty:
        return pd.DataFrame()
    start_s = window_start.strftime("%Y-%m-%d %H:%M")
    end_s = window_end.strftime("%Y-%m-%d %H:%M")
    parts = []
    for point, suffix in ci.POINT_SUFFIX.items():
        p = ci.kimg_one_point(kimg_long, point, suffix, window_start, window_end)
        sub = kimg_long[
            (kimg_long["category"] == "SOLAR_RAD") &
            (kimg_long["point_name"] == point) &
            (kimg_long["fcst_datetime"] >= start_s) &
            (kimg_long["fcst_datetime"] < end_s)
        ]
        rad = ci.kimg_solar(sub, suffix)
        if not rad.empty:
            p = rad.to_frame() if p.empty else p.join(rad, how="outer")
        if not p.empty:
            parts.append(p)
    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index, format="%Y-%m-%d %H:%M").strftime(
        "%Y-%m-%d %H:%M:%S")
    wide.index.name = "timestamp"
    return pp.clip_ranges(wide)


# ── 공통: 그리드 정규화 + 3h 결손 보간 + upsert ─────────────────────────────
def to_grid(wide: pd.DataFrame, ws: datetime, we: datetime) -> tuple[pd.DataFrame, int]:
    """윈도우 트림 + 1h 그리드 reindex + 시간 보간(연속 2h 까지, 내부만).

    KIMR/KIMG 아카이브 창은 전 구간 1h 라 기대 인덱스가 곧 1h date_range 다.
    보간 자체는 `postprocess.fill_short_gaps` 한 벌(운영 수집 경로와 공용 --
    2026-08-04 통합).  반환: (wide, 보간 셀 수).
    """
    if wide.empty:
        return wide, 0
    idx = pd.date_range(ws, we, freq="h", inclusive="left").strftime("%Y-%m-%d %H:%M:%S")
    return pp.fill_short_gaps(wide, idx)


def upsert_wide_coalesce(df: pd.DataFrame, db_path: Path,
                         table: str = cf.RUNS_TABLE) -> int:
    """(base,timestamp) 충돌 시 새 값이 NULL 이 아니면 갱신, NULL 이면 기존 유지.

    INSERT ... ON CONFLICT(base,timestamp) DO UPDATE SET col=COALESCE(excluded.col, col).
    스키마 자동 확장(ALTER)·유니크 인덱스 생성은 cf._upsert_df 와 동일 패턴.
    cf._upsert_df 의 INSERT OR REPLACE 는 재실행 때 기존 컬럼을 NULL 로 덮으므로,
    소스를 나눠 여러 번 쓰는 아카이브(KIMR 패스 → KIMG 패스)에는 이쪽을 쓴다.
    (2026-07-21 collect_forecast_v2 폐지 시 이 파일로 이주 — 유일한 사용처였다.)
    """
    if df.empty:
        return 0
    drop = [c for c in df.columns if cf.is_non_kma(c)]
    if drop:
        df = df.drop(columns=drop)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"_tmp_{table}"
    with sqlite3.connect(db_path) as c:
        df.to_sql(tmp, c, if_exists="replace", index=True)
        existing = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        tmp_cols = [r[1] for r in c.execute(f"PRAGMA table_info({tmp})").fetchall()]
        if not existing:
            c.execute(f"CREATE TABLE {table} AS SELECT * FROM {tmp} WHERE 0")
            existing = set(tmp_cols)
        for col in tmp_cols:
            if col not in existing:
                c.execute(f'ALTER TABLE {table} ADD COLUMN "{col}"')
        c.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_base_ts "
            f"ON {table}(base, timestamp)"
        )
        col_list = ", ".join(f'"{col}"' for col in tmp_cols)
        updates = ", ".join(
            f'"{col}"=COALESCE(excluded."{col}", "{table}"."{col}")'
            for col in tmp_cols if col not in ("base", "timestamp")
        )
        c.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {tmp} "
            f"WHERE true ON CONFLICT(base, timestamp) DO UPDATE SET {updates}"
        )
        n = c.execute("SELECT changes()").fetchone()[0]
        c.execute(f"DROP TABLE {tmp}")
    return n


def upsert_tagged(wide: pd.DataFrame, base_utc: datetime, table: str,
                  met_proto: str | None = None) -> int:
    """base + horizon_d (+ src_met_proto) 태그 후 컬럼 보존 upsert (전-NaN 행 제외).

    `met_proto` = KIMR met 을 어느 프로토콜로 받았는지 ('NC'/'GRIB').  wide 저장은
    출처를 지우므로 본 DB 의 `src_met_*`(모델 마스크)과 같은 취지로 남긴다 —
    다만 여기선 테이블이 이미 모델을 말하므로 갈리는 축은 **프로토콜뿐**이다
    (일사·운량은 KIMR 도 항상 NC).  KIMG 는 경로가 하나라 태그하지 않는다.
    NULL = 태그 도입(2026-07-22) 이전에 쌓인 행 = 전부 GRIB met.
    ※ --force 없이 재수집하면 COALESCE 라 옛 컬럼이 남는데 태그만 새 값으로
      바뀐다.  프로토콜을 바꿔 다시 받을 때는 --force 를 쓸 것.
    """
    wide = wide.dropna(how="all")
    if wide.empty:
        return 0
    base_kst = base_utc.astimezone(KST)
    df = wide.copy()
    ts_dates = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S").date
    if met_proto:
        df.insert(0, "src_met_proto", met_proto)
    df.insert(0, "horizon_d", [(d - base_kst.date()).days for d in ts_dates])
    df.insert(0, "base", base_kst.strftime("%Y-%m-%d %H:%M:%S"))
    return upsert_wide_coalesce(df, DB_PATH, table=table)


def _coverage(wide: pd.DataFrame, expected: int) -> tuple[float, dict[str, int]]:
    """KEY_COLS 기준 최저 커버리지와 컬럼별 확보 수."""
    if wide.empty or expected == 0:
        return 0.0, {}
    counts = {c: int(wide[c].notna().sum()) for c in KEY_COLS if c in wide.columns}
    if len(counts) < len(KEY_COLS):                # 컬럼 자체가 빠짐 = 0 취급
        return 0.0, counts
    return min(counts.values()) / expected, counts


# ── 단일 base 수집 ───────────────────────────────────────────────────────────
def collect_one(base_utc: datetime, days: int, do_kimr: bool, do_kimg: bool,
                point_workers: int, force: bool = False) -> int:
    ws, we = _window(base_utc, days)
    base_label = base_utc.strftime("%Y%m%d%H") + " UTC"
    rc = 0
    for src, do, builder, table, expected in (
        ("KIMR", do_kimr, lambda: kimr_wide_one(base_utc, days), TABLE_KIMR,
         len(k2.hf_range_1h(base_utc, days, k2.R030_MAX_HF))),
        ("KIMG", do_kimg, lambda: kimg_wide_one(base_utc, days, point_workers),
         TABLE_KIMG, int((we - ws).total_seconds() // 3600)),
    ):
        if not do:
            continue
        t0 = time.time()
        base_day = base_utc.astimezone(KST).date()
        hd_lo = (ws.date() - base_day).days                      # 12z=1, 18z 당일 모드=0
        hd_hi = ((we - timedelta(hours=1)).date() - base_day).days
        print(f"[archive] {src} base {base_label} horizon_d {hd_lo}~{hd_hi} -> {table}")
        wide = builder()
        if wide.empty:
            print(f"[archive] {src} 확보 0행 -- 실패")
            rc = max(rc, 2)
            continue
        wide, filled = to_grid(wide, ws, we)
        cov, _counts = _coverage(wide, expected)
        # ★삭제는 수집이 성공한 뒤에만 — 여기까지 왔으면 새로 넣을 행이 확보돼 있다.
        if force:
            removed = delete_base(table, base_utc)
            if removed:
                print(f"[archive] {src} --force: 기존 {removed}행 삭제 후 재적재")
        n = upsert_tagged(wide, base_utc, table,
                          met_proto="NC" if src == "KIMR" else None)
        interp_note = f"  보간 {filled}셀" if filled else ""
        print(f"[archive] {src} upsert {n}행  최저 커버리지 {cov:.0%}"
              f" (기대 {expected}시각){interp_note}  ({time.time() - t0:.0f}s)")
        if cov < COVERAGE_WARN:
            print(f"[archive] {src} 커버리지 {COVERAGE_WARN:.0%} 미만 -- "
                  f"재실행 권장 (재실행 = 부족분만 채움)")
            rc = max(rc, 1)
    return rc


def delete_base(table: str, base_utc: datetime) -> int:
    """--force 전용: 그 base 의 기존 행 제거.  **수집 성공 후에만** 호출한다.

    COALESCE upsert 는 새 값이 NULL 인 컬럼에 옛 값을 남긴다.  그래서 met 경로를
    GRIB -> NC 로 바꾼 뒤 재수집하면 한 행에 두 소스가 섞인다 (NC 가 안 주는
    컬럼에 GRIB 잔재가 남음).  소스를 통일하려면 지우고 새로 넣어야 한다.
    forecast_kimr / forecast_kimg 가 별도 테이블이라 한쪽만 지워도 안전하다.
    """
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return 0
    base_kst_str = base_utc.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as c:
        try:
            cur = c.execute(f"DELETE FROM {table} WHERE base=?", (base_kst_str,))
        except sqlite3.OperationalError:        # 테이블이 아직 없음
            return 0
        return cur.rowcount


def existing_rows(table: str, base_kst_str: str, sentinel: str) -> int:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return 0
    with sqlite3.connect(DB_PATH) as c:
        try:
            return c.execute(
                f'select count("{sentinel}") from {table} where base=?',
                (base_kst_str,)).fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="KIMR/KIMG 소스 분리 기상 수집 (메인 DB, D+5 1h)")
    ap.add_argument("--base", help="YYYYMMDD -- 그 날짜(UTC)의 --utc 발표 (기본: 최신 가용 12z). "
                                   "18z 는 UTC 날짜 기준: 20260717 18z = 07-18 03시 KST 발표")
    ap.add_argument("--utc", type=int, default=12, choices=[0, 6, 12, 18])
    ap.add_argument("--days", type=int, default=None,
                    help="윈도우 길이 (기본: 12z 등 5 / 18z 당일 모드 3)")
    ap.add_argument("--backfill", type=int, metavar="N",
                    help="최근 N 개 base(--utc 시각), 오래된 것부터 (소스별 resume-skip)")
    ap.add_argument("--force", action="store_true",
                    help="이미 찬 base 도 다시 받고, 그 base 의 기존 행을 지운 뒤 새로 적재. "
                         "met 프로토콜을 바꿔 재수집할 때 필수 -- 기본 COALESCE upsert 는 "
                         "새 값이 NULL 인 컬럼에 옛 소스를 남긴다 (삭제는 수집 성공 후에만).")
    ap.add_argument("--kimr-only", action="store_true")
    ap.add_argument("--kimg-only", action="store_true")
    ap.add_argument("--point-workers", type=int, default=1,
                    help="KIMG 지점 동시성 (기본 1 -- parallel-safe 규칙)")
    args = ap.parse_args()
    do_kimr = not args.kimg_only
    do_kimg = not args.kimr_only

    # 18z = 당일예보 모드 (opt-in): 창 = 당일 04시(hf=1)부터 days 일, horizon_d=0 생성.
    # 기본 days=3 -- KIMG expected 산정이 부풀어 커버리지 rc=1 오탐이 나지 않게 한다
    # (KIMR 18z lead 72h 라 days=5 면 D+3~4 가 통째로 결손 취급됨).
    if args.utc == 18:
        cf.ckg.SAMEDAY_18Z = True
        print("[archive] 18z 당일예보 모드 (창 = 당일 04시~, horizon_d 0..)")
    if args.days is None:
        args.days = 3 if args.utc == 18 else DAYS_DEFAULT

    if args.backfill:
        # ★소스 분리 2-패스 (2026-07-17 사용자 지적 + backfill_jeju_forecast 교훈):
        # base 마다 KIMR->KIMG 를 교대 호출하면 apihub 부담으로 매우 느려지고
        # KIMR 504 가 잦다.  KIMR 패스 전체 -> KIMG 패스 전체로 돌린다 (소스별 resume-skip).
        bases = cf.backfill_12z_bases(args.backfill)
        if args.utc != 12:                         # 18z 등 다른 발표의 백필: 시각만 치환
            cut = datetime.now(tz=UTC) - timedelta(hours=cf.PUBLISH_DELAY_HOURS)
            bases = [b for b in (b.replace(hour=args.utc) for b in bases) if b <= cut]
        rcs = []
        for src_label, table, do in (("KIMR", TABLE_KIMR, do_kimr),
                                     ("KIMG", TABLE_KIMG, do_kimg)):
            if not do:
                continue
            print(f"[archive] == {src_label} 패스: {len(bases)} base ==")
            for base_utc in bases:
                b_kst = base_utc.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
                if src_label == "KIMR":
                    expected = len(k2.hf_range_1h(base_utc, args.days, k2.R030_MAX_HF))
                else:
                    ws, we = _window(base_utc, args.days)
                    expected = int((we - ws).total_seconds() // 3600)
                if (not args.force
                        and existing_rows(table, b_kst, "temp_west") >= expected * COVERAGE_WARN):
                    print(f"[archive] {src_label} {base_utc:%Y%m%d%H} UTC 이미 참 -- skip")
                    continue
                rcs.append(collect_one(base_utc, args.days,
                                       src_label == "KIMR", src_label == "KIMG",
                                       args.point_workers, force=args.force))
        return max(rcs) if rcs else 0

    if args.base:
        base_utc = datetime.strptime(args.base, "%Y%m%d").replace(
            hour=args.utc, tzinfo=UTC)
    elif args.utc == 12:
        base_utc = cf.latest_12z()
    else:
        base_utc = cf.latest_12z().replace(hour=args.utc)
        if base_utc > datetime.now(tz=UTC) - timedelta(hours=cf.PUBLISH_DELAY_HOURS):
            base_utc -= timedelta(days=1)
    return collect_one(base_utc, args.days, do_kimr, do_kimg, args.point_workers,
                       force=args.force)


if __name__ == "__main__":
    try:
        # line_buffering: 백그라운드/리다이렉트 실행에서도 로그가 실시간으로 흐르게
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    sys.exit(main())
