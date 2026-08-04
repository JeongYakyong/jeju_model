"""
collect_historical.py -- (제주) 실측 수집 진입점 -> input_data_jeju.db::historical.

    python collectors/collect_historical.py                     # 최근 2일 탑업
    python collectors/collect_historical.py --historical-days 7 # 최근 7일
    python collectors/collect_historical.py --backfill 30       # 과거 30일 일괄
    python collectors/collect_historical.py --no-save           # dry-run

담는 것 (전부 `historical` 한 테이블로)
    KPX 제주 수급(chejusukub)     -> *_jeju
    KPX 하루전 DA SMP·예상수요    -> smp_*_da / *_est_demand_da
    KPX 제주 실시간시장 RT SMP     -> smp_rt_g1..g4 / smp_jeju_rt / smp_rt_neg_num
    KMA ASOS 3지점 관측           -> *_west / *_east / *_south
    + recompute_jeju_capacity 파생 (real_*_capacity / real_*_utilization_jeju)

★`*_da` 만 수집 창이 다르다 — 실측은 today 까지지만 DA 는 발행이 미래를 향하므로
  today + DA_FUTURE_DAYS 까지 받는다 (build_historical 주석 참고).

기상 예보는 이 파일 소관이 아니다 -> collect_forecast.py (forecast_horizon).
쓰기는 전부 partial_upsert 라 다른 경로가 채운 컬럼이 NULL 로 덮이지 않는다.
(2026-07-21: 구 collect_historical.py 에서 예보 기상부를 collect_forecast.py 로 넘기고
 실측 전용으로 개명. 구 `forecast` 테이블은 2026-06-20 폐기.)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# 통합 모듈 (2026-06-01 compaction).  별칭은 과거 모듈명을 유지해 호출부 변경 최소화.
# kpx_asos = 실측 소스 fetch (KPX 전력시장 + KMA ASOS 관측).
# + fetch_asos + KPX(수급/est).  아래 kim/ci/kpx 는 모두 이 한 모듈을 가리킨다.
import kpx_asos as kpx           # KPX 수급·DA·RT SMP + ASOS 관측
import postprocess as pp
from kma_kimg import partial_upsert


KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

# data/ 는 repo 루트 한 곳만 사용 (모든 collector 공통 규칙).
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "input_data_jeju.db"

# 이 모듈이 쓰는 테이블은 historical 하나뿐이다.
# - historical: *_jeju (관측 수급) + asos (관측 weather) + *_da (DA SMP/수요)
# 구 forecast 테이블(timestamp 단일키)은 폐기됐다(2026-06-20) — 기상 예보는
# forecast_horizon(collect_forecast)이 base x horizon_d 로 보관한다.
HISTORICAL_TABLE = "historical"

# KPX 하루전(*_da) 을 today 기준 며칠 앞까지 받을지.  DA SMP·예상수요는 전일
# 15시경 익일분이 발행되므로 2 면 내일·모레를 커버한다(모레는 대개 빈 응답 → 무해).
DA_FUTURE_DAYS = 2

# 제주(_jeju) 발전 capacity / utilization 파생 컬럼 (historical 전체 기간 누적 max 기반).
# 육지 collect_data_land.recompute_kr_capacity 와 동일 로직.  capacity ~= 설비용량 근사
# = 발전량의 running cummax (첫 해는 그 해 peak 로 평탄화).  utilization = 발전량/capacity.
# 제주는 태양광이 단일 컬럼(real_solar_gen_jeju)이라 분해 불필요.  컬럼명은 기반 발전
# 컬럼(real_*_gen_jeju)과 짝이 맞는 real_*_capacity_jeju / real_*_utilization_jeju.
# (구 legacy 추정설비 Solar_Capacity_Est_jeju / *_Utilization_jeju 컬럼은 제거됨.)
_JEJU_CAPACITY_SPEC = {
    # capacity 컬럼              : 기반 발전 컬럼
    "real_wind_capacity_jeju":   "real_wind_gen_jeju",
    "real_solar_capacity_jeju":  "real_solar_gen_jeju",
}
_JEJU_UTILIZATION_SPEC = {
    # utilization 컬럼              : (발전 컬럼, capacity 컬럼)
    "real_wind_utilization_jeju":  ("real_wind_gen_jeju",  "real_wind_capacity_jeju"),
    "real_solar_utilization_jeju": ("real_solar_gen_jeju", "real_solar_capacity_jeju"),
}


def write_to_historical(wide: pd.DataFrame, db_path: Path) -> int:
    """historical 테이블에 partial UPSERT.

    partial_upsert 사용 이유: build_historical() 의 새 batch (*_jeju/asos/*_da)가
    같은 timestamp 행의 다른 컬럼 -- 특히 recompute_jeju_capacity 가 별도로 채우는
    real_*_capacity_jeju / real_*_utilization_jeju -- 를 덮어쓰지 않도록 (배치에 없는
    컬럼은 보존).
    """
    return partial_upsert(HISTORICAL_TABLE, wide, db_path)


# ── _jeju capacity / utilization 파생 (historical 전체 기간 누적 max) ─────
def recompute_jeju_capacity(db_path: Path = DEFAULT_DB) -> int:
    """historical 의 real_wind_gen_jeju / real_solar_gen_jeju 전체 시계열로 capacity +
    utilization 4개 파생 컬럼을 계산해 historical 에 다시 UPSERT.

    육지 collect_data_land.recompute_kr_capacity 와 같은 로직:
    capacity ~= 설비용량 근사 = 발전량의 running max(cummax), 단 첫 해(2020) 전체는
    그 해 peak 로 평탄화(첫 행 야간 저점에서 시작하는 cummax 왜곡 방지).  이후 설비
    증설을 반영해 단조 증가.  utilization = 발전량 / capacity (이용률).

    cummax 는 전체 기간 문맥이 필요하므로 batch postprocess 가 아니라 여기서 historical
    전체를 읽어 재계산한다.  매번 전체 재계산이라 idempotent (cummax 는 단조라 과거 행은
    안 바뀌고, 새 발전량 peak 만 이후 capacity 를 갱신).
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        print("  [jeju-capacity] DB 없음 -- skip")
        return 0
    base_cols = sorted(set(_JEJU_CAPACITY_SPEC.values()))
    with sqlite3.connect(db_path) as c:
        existing = {r[1] for r in c.execute(
            f"PRAGMA table_info({HISTORICAL_TABLE})").fetchall()}
        if not existing:
            print("  [jeju-capacity] historical 테이블 없음 -- skip")
            return 0
        avail = [col for col in base_cols if col in existing]
        if not avail:
            print("  [jeju-capacity] real_wind_gen_jeju / real_solar_gen_jeju 부재 -- skip")
            return 0
        cols_sql = ", ".join(f'"{col}"' for col in ["timestamp", *avail])
        df = pd.read_sql(
            f"SELECT {cols_sql} FROM {HISTORICAL_TABLE} ORDER BY timestamp", c,
        )
    if df.empty:
        print("  [jeju-capacity] historical 비어있음 -- skip")
        return 0

    df = df.set_index("timestamp")
    out = pd.DataFrame(index=df.index)
    # capacity 하한 = 첫 해(2020) 발전량 최대값 -> 첫 해 전체를 그 해 peak 로 평탄화.
    first_year = df.index[0][:4]
    in_first_year = df.index.str[:4] == first_year
    for cap_col, gen_col in _JEJU_CAPACITY_SPEC.items():
        if gen_col not in df.columns:
            continue
        gen = pd.to_numeric(df[gen_col], errors="coerce")
        floor = gen[in_first_year].max()
        # cummax 는 NaN 위치를 NaN 으로 남긴다 -> ffill 로 중간 결측을 직전 max 로 채우고,
        # 선두(첫 valid 이전) NaN 은 floor 로 채운 뒤 clip 으로 첫 해를 floor 로 평탄화.
        # (clip 은 NaN 을 올리지 못하고 ffill 은 선두 NaN 을 못 채우므로 순서가 중요.)
        cap = gen.cummax().ffill()
        if pd.notna(floor):
            cap = cap.fillna(floor).clip(lower=floor)
        out[cap_col] = cap
    for util_col, (gen_col, cap_col) in _JEJU_UTILIZATION_SPEC.items():
        if gen_col not in df.columns or cap_col not in out.columns:
            continue
        gen = pd.to_numeric(df[gen_col], errors="coerce")
        cap = out[cap_col]
        out[util_col] = (gen / cap.where(cap > 0)).replace([np.inf, -np.inf], np.nan)

    out = out.dropna(how="all")
    if out.empty:
        print("  [jeju-capacity] 계산 결과 없음 -- skip")
        return 0
    out.index.name = "timestamp"
    n = partial_upsert(HISTORICAL_TABLE, out, db_path)
    print(
        f"  [jeju-capacity] {list(out.columns)} over {len(out):,} rows -> UPSERT {n:,}"
    )
    return n


# ── Historical (관측 데이터) ────────────────────────────────────────────
def build_historical(
    n_days_back: int = 2,
    end_date: str | None = None,
    save: bool = True,
    db_path: Path = DEFAULT_DB,
) -> pd.DataFrame:
    """과거 N 일치 관측 + DA 가격/수요 데이터를 wide 로 합쳐 historical 에 UPSERT.

    소스 (collect_kpx_asos_data 의 fetcher 재사용; 제주 전용 — 육지 수급/발전은
    collect_data_land.py 로 분리됨):
        kpx.fetch_kpx_jeju    : 제주(chejusukub) 계통 수급     -> *_jeju cols
        kpx.fetch_asos        : KMA ASOS 3지점 관측             -> *_west/_east/_south
        kpx.fetch_kpx_est     : DA SMP + 예상수요(제주/육지)    -> smp_*_da,
                                *_est_demand_da (forecast 와 동일 컬럼을 historical
                                에도 누적, legacy ingest 정책 정합).
        kpx.fetch_kpx_jeju_rt_smp : 제주 실시간시장 RT SMP      -> smp_rt_g1..g4,
                                smp_jeju_rt, smp_rt_neg_num (4단계 SMP 모델 타깃;
                                구간 원시값 + 파생, historical 전용).

    파라미터
    - n_days_back : end_date 로부터 거꾸로 몇 일치를 받을지.  기본 2 일
                    (D-2 ~ today, daily 탑업).  --backfill N 에서는 N 을 그대로 전달.
    - end_date    : 'YYYY-MM-DD'.  None 이면 KST 기준 today.
    - save        : True 면 historical 테이블에 UPSERT.  False 면 wide 만 반환.
    - db_path     : 기본 data/input_data_jeju.db.

    ★*_da(KPX 하루전) 만 창이 다르다 — 실측(수급·ASOS·RT SMP)은 today 까지지만
    DA SMP·예상수요는 **발행이 미래를 향하므로 today+DA_FUTURE_DAYS 까지** 받는다.
    (2026-07-21: 구 build() 가 forecast 윈도우에서 미래 *_da 를 공급했으나
    운영 파이프라인이 --no-forecast 라 한 번도 실행되지 않아 내일치 smp_jeju_da /
    jeju_est_demand_da 가 계속 비어 있었다.  build() 폐지와 함께 이리로 이관.)

    Returns: wide DataFrame (timestamp 인덱스, ~47 cols).
    """
    today_kst = datetime.now(tz=KST).date()
    end = (
        today_kst if end_date is None
        else datetime.strptime(end_date, "%Y-%m-%d").date()
    )
    start = end - timedelta(days=n_days_back)
    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")
    # DA 전용 종료일 — 미발행 구간은 빈 응답이라 무해하고, 발행되면 다음 실행이 채운다.
    da_end_str = (end + timedelta(days=DA_FUTURE_DAYS)).strftime("%Y-%m-%d")

    print(
        f"[collect_historical] historical window={s_str} ~ {e_str} "
        f"(*_da 는 ~{da_end_str})  target table='{HISTORICAL_TABLE}' (UPSERT)"
    )

    print("\n[H1/4] fetch *_jeju cols (chejusukub)")
    try:
        jeju = kpx.fetch_kpx_jeju(s_str, e_str)
    except Exception as e:
        print(f"  [WARN] *_jeju failed: {e}")
        jeju = pd.DataFrame()
    print(f"  *_jeju:    {len(jeju):,} rows x {len(jeju.columns)} cols")

    print("\n[H2/4] fetch KMA ASOS (3 stations)")
    try:
        asos = kpx.fetch_asos(s_str, e_str)   # ASOS 는 kma_fetcher_jeju 로 이동
    except Exception as e:
        print(f"  [WARN] asos failed: {e}")
        asos = pd.DataFrame()
    print(f"  asos:      {len(asos):,} rows x {len(asos.columns)} cols")

    # *_da (DA SMP + 제주/육지 예상수요).  _da 접미사 덕분에 historical 의 다른
    # (관측치) 컬럼과 충돌 없다.  ★여기만 미래까지 받는다 — DA 는 하루전 발행이라
    # 내일치가 이미 나와 있고, 화면의 SMP 검증 참조선·수요 비교선이 이 값을 쓴다.
    print(f"\n[H3/4] fetch *_da (smp_*_da + jeju/land_est_demand_da) ~{da_end_str}")
    try:
        kest = kpx.fetch_kpx_est(s_str, da_end_str)
    except Exception as e:
        print(f"  [WARN] *_da failed: {e}")
        kest = pd.DataFrame()
    print(f"  *_da:      {len(kest):,} rows x {len(kest.columns)} cols")

    # 제주 실시간시장 RT SMP (smp_rt_g1..g4 + smp_jeju_rt + smp_rt_neg_num).  매일
    # 23:00 KST 발행이라 지연·미발행 날짜는 빈 응답(누락) -- partial_upsert COALESCE 가
    # 기존값 보존.  historical 전용 (RT 는 실현치).  4단계 SMP 모델의 타깃.
    print("\n[H4/4] fetch RT SMP (smp_rt_g1..g4 + smp_jeju_rt + smp_rt_neg_num)")
    try:
        rt_smp = kpx.fetch_kpx_jeju_rt_smp(s_str, e_str)
    except Exception as e:
        print(f"  [WARN] rt_smp failed: {e}")
        rt_smp = pd.DataFrame()
    print(f"  rt_smp:    {len(rt_smp):,} rows x {len(rt_smp.columns)} cols")

    parts = [df for df in (jeju, asos, kest, rt_smp) if not df.empty]
    if not parts:
        print("  [WARN] all historical sources empty -- nothing to write")
        return pd.DataFrame()

    # 컬럼이 disjoint 하므로 axis=1 concat 안전.  index(=timestamp 문자열)는
    # outer-aligned -> 일부 소스에만 있는 시간은 다른 컬럼이 NaN 으로 채워짐.
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "timestamp"
    print(
        f"\n  historical wide: {len(wide):,} rows x {len(wide.columns)} cols "
        f"(NaN ratio = {wide.isna().mean().mean():.2%})"
    )

    print("\n[postprocess] range clip + day_type")
    wide = pp.clip_ranges(wide)
    wide = pp.add_day_type(wide)

    if save:
        n = write_to_historical(wide, db_path)
        print(f"\n  UPSERT historical: {n:,} rows -> {db_path}")
        # 제주 발전실적(real_*_gen_jeju)이 갱신됐으니 capacity/utilization 파생을 전체
        # 기간 기준 재계산 (cummax 는 단조라 과거 행은 그대로, 새 peak 만 이후 갱신).
        print("\n[jeju-capacity] recompute wind/solar capacity + utilization (_jeju)")
        recompute_jeju_capacity(db_path)
    return wide


# ── CLI ─────────────────────────────────────────────────────────────────
def main() -> None:
    """제주 실측 수집 CLI — KPX 수급·DA·RT SMP + KMA ASOS → historical 적재.

    기상 예보는 이 파일 소관이 아니다 (collect_forecast.py → forecast_horizon).
    이 모듈이 forecast 기상을 위해 제공하는 것은 라이브러리 표면
    (fetch_kimr_long / fetch_kimg_long / build_wide) 뿐이다.
    """
    p = argparse.ArgumentParser(
        description=(
            "Direct API -> post-processing -> input_data_jeju.db::historical.  "
            "KPX 제주 수급 + DA SMP/예상수요 + RT SMP + KMA ASOS 3지점.  "
            "(기상 예보는 collect_forecast.py)"
        ),
    )
    p.add_argument(
        "--historical-days", type=int, default=2, metavar="N",
        help="수집 창 길이(일).  기본 2 = D-2~today 탑업.  *_da 는 today+2 까지.",
    )
    p.add_argument(
        "--backfill", type=int, metavar="N_DAYS",
        help="과거 N 일치를 한 번에 (D-N~today 단일 fetch, UPSERT 라 idempotent)",
    )
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"SQLite path (default {DEFAULT_DB})",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="DB 쓰기 생략 (dry-run).  --backfill 과는 함께 쓸 수 없다.",
    )
    args = p.parse_args()

    t0 = time.time()
    if args.backfill is not None:
        if args.no_save:
            sys.exit("--no-save is not supported with --backfill")
        print(f"\n=== historical backfill (N={args.backfill}) ===")
        build_historical(n_days_back=args.backfill, save=True, db_path=args.db)
        print(f"\n[collect_historical] done in {(time.time()-t0)/60:.1f}m")
        return

    print(f"\n=== historical build (last {args.historical_days} days) ===")
    build_historical(
        n_days_back=args.historical_days, save=not args.no_save, db_path=args.db,
    )
    print(f"\n[collect_historical] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
