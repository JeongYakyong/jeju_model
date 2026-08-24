"""
collect_forecast.py -- 기상예보(KMA) 수집 진입점 + forecast_horizon 지평 아카이브 엔진.

    python collectors/collect_forecast.py --region jeju --days 5            # 12z (운영 cron)
    python collectors/collect_forecast.py --region jeju --days 3 --utc 18   # 18z 당일예보
    python collectors/collect_forecast.py --backfill 30                     # 과거 30일 (resume-skip)
    python collectors/collect_forecast.py --backfill 10 --out bf            # data/bf_jeju.db 격리
    python collectors/collect_forecast.py --merge bf                        # 격리본 -> 본 DB 병합
    python collectors/collect_forecast.py --verify                          # 완전성 검사 (fetch 없음)

2026-07-21 통합: 구 collect_forecast(진입점) + collect_forecast(적재 엔진)
+ collect_forecast(wide 래퍼) + collect_historical 의 기상부(KIMR/KIMG fetch·병합)
가 4겹 래퍼 체인이었다.  같은 일을 4파일에 나눠 두느라 "어느 게 현역인지" 알 수 없었다.

구성 (위에서 아래로 = 데이터 흐름)
  ① fetch      : fetch_kimr_long / fetch_kimg_long      -> long 5컬럼
  ② 병합·피벗  : build_wide (지점별 KIMR combine_first KIMG) -> wide
  ③ wide 라이브러리 : build_forecast_wide (KPX·DB 없는 순수 KMA 경로)
  ④ 적재 엔진  : upsert_runs / _upsert_df / verify_runs / merge_runs
  ⑤ 진입점     : base 선택 + 완결성 skip + CLI

long 5컬럼(base_datetime/point_name/fcst_datetime/category/fcst_value)이 정식 중간
계약이고, 피벗은 pivot._pivot_point 한 벌이다 (selftest_pivot.py 로 보호).

병합 규칙 (build_wide)
  - 지점별 kimr_part.combine_first(kimg_part): KIMR 우선, KIMG 는 못 채운 칸만.
  - 일사·운량은 **KIMG(NE57) 단독**(전 지평).  KIMR 에 그 변수가 없어서가 아니라,
    서빙 모델이 NE57 분포로 학습됐고 2026-07-31 실측에서 KIMR 일사·운량이 열세로
    확인됐기 때문이다(일사 r 0.64~0.70, 전운량 r 0.47 — 다른 모델이다).
    소스 전환은 재학습과 함께만.
  - temp/wind/gust/reh/rainfall 은 KIMR(D+1~5, 1h)이 채우고 D+6~ 는 KIMG 가 보충.
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

CORE = Path(__file__).resolve().parent
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import kma_kimr_nc as kimr        # KIMR(R030) std NC — KIMR 유일 경로 (2026-08-04 GRIB 폐기)
import pivot as ci                # long -> wide 피벗 (KIMR·KIMG 공용 1벌)
import kpx_asos as kpx            # KPX 하루전(DA) -- 차단 몽키패치 대상
import kma_kimg as kimg           # KIMG(NE57) http/parse/derive core
import kma_kimg as ckg            # (같은 모듈 -- 창 산식·MIN_HF 등 글로벌 접근용 별칭)
import postprocess as pp

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

RUNS_TABLE = "forecast_horizon"   # KMA 기상 예보 전용 지평 아카이브
# data/ 는 repo 루트 한 곳만 사용 (모든 collector 공통 규칙).
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "input_data_jeju.db"
DATA_DIR = DEFAULT_DB.parent
PUBLISH_DELAY_HOURS = ckg.PUBLISH_DELAY_HOURS

# ═══ ① fetch: API -> long 5컬럼 ═══════════════════════════════════════
class NoUsableForecastRows(Exception):
    """build_wide 가 윈도우 안에서 쓸 만한 KIMR 행을 못 만들었을 때.

    sys.exit 대신 이 예외를 던져 forecast 단계만 건너뛰고 historical 등 후속 단계는
    계속 진행할 수 있게 한다 (일시적 KIMR 결손이 전체 실행을 죽이지 않도록).
    """

# ── API → in-memory long DataFrame ──────────────────────────────────────
def fetch_kimr_long(bases: list[datetime], point_workers: int = 3) -> pd.DataFrame:
    """주어진 bases 에 대해 KIMR(R030 std NC)을 호출하고 long 5컬럼 DF 반환.

    수집 스택은 `kma_kimr_nc.fetch_kimr_met_long` 한 벌(지점 순차 × hf 병렬 8워커 +
    결측 hf 재시도 2라운드)이고, 카테고리 라벨은 구 GRIB 관례 그대로라 뒤따르는
    피벗(`pivot._SPEC_KIMR`)이 무수정으로 성립한다.

    윈도우 길이는 호출 시점의 `kma_kimg.FORECAST_DAYS`(= forecast_days_override 가
    세팅한 값)를 읽는다 -- KIMG 와 같은 창을 쓰기 위한 SSOT 경유.
    실패한 base 는 경고만 출력하고 건너뛴다 (전체 흐름은 계속).
    """
    parts: list[pd.DataFrame] = []
    for base in bases:
        base_label = base.strftime("%Y%m%d%H") + " UTC"
        try:
            part = kimr.fetch_kimr_met_long(
                kimr.POINTS_JEJU_V2, base, ckg.FORECAST_DAYS,
                point_workers=point_workers,
            )
        except Exception as e:
            print(f"  [WARN] KIMR {base_label}: {e}")
            continue
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=[
            "base_datetime", "point_name", "fcst_datetime", "category", "fcst_value"])
    return pd.concat(parts, ignore_index=True)


def _fetch_kimg_one_point(base: datetime, base_kst, base_dt_str: str,
                          base_label: str, pt: dict) -> list[tuple]:
    """단일 (base, point) 의 hf 를 workers=6 으로 fanout 해 row 튜플 리스트 반환.

    point-local list 에만 append 하므로 여러 point 를 병렬로 돌려도 thread-safe
    (호출자가 결과 리스트를 메인 스레드에서 합친다).  요약 1줄도 여기서 출력.
    """
    hf_list = list(kimg.collection_hf_range(base))
    t0 = time.time()
    failed = empty = n_kept = 0
    rows: list[tuple] = []
    with ThreadPoolExecutor(max_workers=kimg.MAX_WORKERS) as ex:
        fut_to_hf = {ex.submit(kimg.fetch_one_hf, base, pt, hf): hf for hf in hf_list}
        for fut in as_completed(fut_to_hf):
            hf = fut_to_hf[fut]
            body = fut.result()
            if body is None:
                failed += 1
                continue
            raw = kimg.parse_response(body)
            if not raw:
                empty += 1
                continue
            cats = kimg.derive_categories(raw)
            fcst_kst = base_kst + timedelta(hours=hf)
            fcst_dt_str = fcst_kst.strftime("%Y-%m-%d %H:%M")
            for cat, val in cats.items():
                rows.append((base_dt_str, pt["name"], fcst_dt_str, cat, float(val)))
                n_kept += 1
    elapsed = time.time() - t0
    print(
        f"  KIMG {base_label} {pt['name']:<22}  hfs={len(hf_list):2d}  "
        f"kept_rows={n_kept:4d}  failed={failed:2d}  empty={empty:2d}  "
        f"({elapsed:.1f}s)"
    )
    return rows


def fetch_kimg_long(bases: list[datetime], point_workers: int = 1) -> pd.DataFrame:
    """주어진 bases × KIMG.POINTS 에 대해 hf 별 API 호출 → long DF.

    collect_kimg.collect_one_point 의 DB-없는 버전.  (base, point) 마다 hf 를
    workers=6 으로 fanout, 결과를 메인 스레드에서 모아 리스트로 누적.  실패 hf 는
    [WARN] 후 스킵.  derive_categories 변환식이 그대로 적용되므로 SOLAR_RAD /
    TEMP_C / WIND_U/V_10M 등이 라벨 그대로 나옴.

    point_workers
    - 1(기본): (base,point) 외부 루프 순차 -- CLAUDE.md 의 "KIMG parallel-safe 규칙"
      (hf 만 6 workers, 지점은 순차로 서버 동시성 ~6 유지, 504 회피).  레거시
      build_forecast_wide 는 이 안전 기본값을 그대로 쓴다.
    - >1: 지점도 동시에 받는다(동시성 = point_workers × MAX_WORKERS).  3 지점·7일
      윈도우에서 base당 KIMG 시간을 ~1/3 로 줄이는 forecast_horizon 백필용 opt-in
      (build_forecast_wide 가 지점 수만큼 켠다).  KMA 504 위험을
      감수하는 경로라 기본값이 아니다.
    """
    kimg.warmup()

    rows: list[tuple] = []
    for base in bases:
        base_kst = base.astimezone(kimg.KST)
        base_dt_str = base_kst.strftime("%Y-%m-%d %H:%M")
        base_label = base.strftime("%Y%m%d%H") + " UTC"
        if point_workers <= 1:
            for pt in kimg.POINTS:
                rows.extend(
                    _fetch_kimg_one_point(base, base_kst, base_dt_str, base_label, pt)
                )
        else:
            with ThreadPoolExecutor(max_workers=point_workers) as ex:
                futs = {
                    ex.submit(
                        _fetch_kimg_one_point, base, base_kst, base_dt_str, base_label, pt
                    ): pt
                    for pt in kimg.POINTS
                }
                for fut in as_completed(futs):
                    rows.extend(fut.result())
    if not rows:
        return pd.DataFrame(
            columns=["base_datetime", "point_name", "fcst_datetime",
                     "category", "fcst_value"]
        )
    df = pd.DataFrame(
        rows,
        columns=["base_datetime", "point_name", "fcst_datetime",
                 "category", "fcst_value"],
    )
    df["fcst_value"] = pd.to_numeric(df["fcst_value"], errors="coerce")
    return df


# ═══ ② 병합·피벗: long -> wide ════════════════════════════════════════
# 소스 마스크 값 — DB 에서 눈으로 읽히도록 축약하지 않는다 (30k행 규모라 비용 무시).
SRC_KIMR = "KIMR"
SRC_KIMG = "KIMG"


def _met_source(kimr_part: pd.DataFrame, kimg_part: pd.DataFrame,
                suffix: str, index) -> pd.Series:
    """지점별 met 블록 출처 -> `src_met_<suffix>` ('KIMR' / 'KIMG' / NULL).

    왜 필요한가: 병합이 `combine_first`(셀 단위)라 wide 만 봐서는 그 칸이 어느
    모델에서 왔는지 알 수 없다.  소스별 성능을 비교하거나 재학습 때 "이 기온은
    무엇으로 학습된 값인가"를 답하려면 출처가 남아야 한다 (이 마스크와 별개로
    소스 분리 테이블 forecast_kimr/forecast_kimg 도 같은 메인 DB 에 있다).

    sentinel 은 temp: 두 모델 스펙(_SPEC_KIMR/_SPEC_KIMG)이 모두 temp 를 내고,
    verify_runs / base_complete 도 같은 sentinel 을 쓴다.

    ※ **이 파일 경로에서만** 일사(radiation_*)·운량(total/midlow_cloud_*)이 항상
      KIMG 다 — 그래서 마스크 대상이 아니다.  이유는 "KIMR 에 그 변수가 없어서"가
      아니라 **여기서 쓰는 KIMR 엔드포인트가 GRIB 단일면(data=U)이라서**다.
      KIMR 도 일사는 std NC(SWDDIR2+SWDDIF2), 운량은 **등압면 CLDFRA(data=P, 24레벨
      결합)** 로 받을 수 있고, 실제로 collect_archive 가 그렇게 받아
      메인 DB 의 forecast_kimr 에 넣는다 (2026-07-17 채택, 2026-07-30 메인 DB 흡수).
      이 GRIB 경로(forecast_horizon)를 NC(forecast_kimr)로 바꾸려면 재학습이 필요하다
      (서빙 모델이 GRIB met + NE57 분포로 학습됨). 재학습 시점 = GRIB 폐기 시점.
    """
    col = f"temp_{suffix}"

    def has(df: pd.DataFrame) -> pd.Series:
        if df.empty or col not in df.columns:
            return pd.Series(False, index=index)
        return df[col].reindex(index).notna()

    return pd.Series(
        np.where(has(kimr_part), SRC_KIMR, np.where(has(kimg_part), SRC_KIMG, None)),
        index=index, name=f"src_met_{suffix}",
    )


def build_wide(
    kimr_long: pd.DataFrame, kimg_long: pd.DataFrame,
    window_start: datetime, window_end: datetime,
) -> pd.DataFrame:
    """collect_input 의 후처리 함수들을 그대로 사용해 wide DataFrame 생성.

    출력 컬럼은 collect_input.build_csv 와 정확히 동일 (location-suffixed,
    timestamp index).  KIMG 가 비어있으면 일사(radiation_*)·구름(*_cloud_*) 컬럼이 빠진다.

    지점별 KIMR + KIMG 병합 (2026-06-13, 장지평 확장):
    KIMR(지역모델, 고해상도)이 있는 시간은 KIMR 값을 쓰고, KIMR lead 한계(120h)
    이후나 결손 시간은 kimg_one_point 가 만든 동일 스키마 컬럼으로 채운다
    (combine_first).  KIMR 전용 변수(cape 등)는 KIMG-only 구간에서 NaN.
    출처는 `src_met_<suffix>` 컬럼으로 함께 남긴다 (_met_source 참고).
    """
    parts: list[pd.DataFrame] = []
    for point, suffix in ci.POINT_SUFFIX.items():
        kimr_part = ci.kimr_one_point(kimr_long, point, suffix, window_start, window_end)
        kimg_part = ci.kimg_one_point(kimg_long, point, suffix, window_start, window_end)
        if kimr_part.empty and kimg_part.empty:
            print(f"  [WARN] no KIMR/KIMG rows for {point} in window -- column group skipped")
            continue
        if kimr_part.empty:
            part = kimg_part
        elif kimg_part.empty:
            part = kimr_part
        else:
            part = kimr_part.combine_first(kimg_part)  # KIMR 우선, KIMG 는 빈 곳만
        src = _met_source(kimr_part, kimg_part, suffix, part.index)
        n_r, n_g = (src == SRC_KIMR).sum(), (src == SRC_KIMG).sum()
        print(f"  met 출처 {suffix:5}: KIMR {n_r}행 / KIMG {n_g}행")
        parts.append(part.join(src, how="left"))

    if not parts:
        raise NoUsableForecastRows("no usable KIMR/KIMG rows after pivoting")

    wide = pd.concat(parts, axis=1)

    # KIMG SOLAR_RAD → 지점별 radiation_<suffix> 컬럼.  일사는 KIMR 에 없어 KIMG 단독
    # 소스다.  POINT_SUFFIX 의 각 지점(west/east/south)에 대해 그 지점 SOLAR_RAD 만
    # window 안으로 필터링한 뒤 ci.kimg_solar 에 넘긴다(그 함수는 freshest 만).
    # 태양광 모델은 radiation_west + radiation_south 를 입력으로 쓴다.
    start_s = window_start.strftime("%Y-%m-%d %H:%M")
    end_s = window_end.strftime("%Y-%m-%d %H:%M")
    solar_all = kimg_long[
        (kimg_long["category"] == "SOLAR_RAD") &
        (kimg_long["fcst_datetime"] >= start_s) &
        (kimg_long["fcst_datetime"] < end_s)
    ]
    for point, suffix in ci.POINT_SUFFIX.items():
        rad = ci.kimg_solar(solar_all[solar_all["point_name"] == point], suffix)
        if rad.empty:
            print(f"  [WARN] KIMG SOLAR_RAD empty for {point} -- radiation_{suffix} omitted")
            continue
        wide = wide.join(rad, how="left")
        n_rad = wide[f"radiation_{suffix}"].notna().sum()
        print(f"  joined radiation_{suffix} ({n_rad}/{len(wide)} hours have value)")

    # collect_input 과 동일한 timestamp 문자열 (초 단위 포함, downstream 호환).
    wide.index = pd.to_datetime(wide.index, format="%Y-%m-%d %H:%M").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    wide.index.name = "timestamp"
    wide = wide.sort_index()
    return wide


# ═══ 발표(base)·윈도우 선택 ═══════════════════════════════════════════
@contextmanager
def forecast_days_override(days: int | None):
    """forecast 수집 윈도우 길이를 임시로 바꾼다 (육지 경로 시절부터의 동형).

    창 산식의 SSOT 는 `kma_kimg`(window_bounds/collection_window/collection_hf_range)
    하나뿐이라 여기서 바꾸는 글로벌도 하나다.  KIMR(std NC)은 `days` 를 인자로 받는
    `hf_range_1h` 를 쓰므로 글로벌에 의존하지 않고, 대신 `fetch_kimr_long` 이 이
    override 가 세팅한 `kma_kimg.FORECAST_DAYS` 를 읽어 같은 창을 쓴다.
    days=None 이면 no-op (기본 2일 유지).

    소스별 lead 한계 (00/12 UTC 발표 probe, 2026-06-12~13):
    - KIMR(지역): 120h=D+5 까지, 전 구간 1h.
    - KIMG(전구): 288h=D+12 까지, 단 1h 는 135h 까지고 이후는 3h 간격만 존재.
    D+5 초과 윈도우에서는 build_wide 가 KIMG 변수(kimg_one_point)로 빈 구간을
    채운다 (KIMR 전용 변수는 NaN, D+6~ 는 3h 행만).
    """
    if days is None:
        yield
        return
    prev_kimg = kimg.FORECAST_DAYS
    kimg.FORECAST_DAYS = days
    try:
        yield
    finally:
        kimg.FORECAST_DAYS = prev_kimg


def _pick_bases(base: datetime | None, n_bases: int) -> list[datetime]:
    """--base 주어졌으면 그것 한 개, 아니면 최근 n_bases 발표 (오래된→최신)."""
    if base is not None:
        return [base]
    now_kst = datetime.now(tz=KST)
    latest = ckg.latest_published_base(now_kst)  # KIMR/KIMG 공통 publish 시각(SSOT)
    out: list[datetime] = []
    cur = latest
    for _ in range(max(1, n_bases)):
        out.append(cur)
        cur -= timedelta(hours=6)
    out.sort()
    return out


def _window_for(bases: list[datetime]) -> tuple[datetime, datetime]:
    """입력 발표들의 KST 윈도우 합집합 (가장 빠른 start ~ 가장 늦은 end).

    창 산식 SSOT = kma_kimg.collection_window (KIMR·KIMG 가 같은 창을 쓴다).
    """
    starts = [ckg.collection_window(b)[0] for b in bases]
    ends = [ckg.collection_window(b)[1] for b in bases]
    return min(starts), max(ends)


# ═══ ③ wide 라이브러리 (KPX·DB 없는 순수 KMA 경로) ═════════════════════
def build_forecast_wide(
    base: datetime | None = None,
    n_bases: int = 2,
    forecast_days: int | None = None,
    point_workers: int = 3,
) -> pd.DataFrame:
    """제주 KIMR+KIMG 병합 기상만 메모리 wide 로 반환.  KPX(*_da) 호출 없음, DB 쓰기 없음.

    collect_historical.build() 에서 KPX(*_da) join 과 forecast 테이블 쓰기를 제거한 순수
    KMA 경로다 (= collect_forecast 의 disable_kpx 패치판을 네이티브로 정리).
    collect_forecast.py 가 이 wide 에 base/horizon_d 태그를 붙여 forecast_horizon 에 적재.

    base/n_bases  : 발표 선택 (--base 단일 / 기본 최근 n_bases 발표).
    forecast_days : 수집 윈도우 길이(일).  None=기본(kma_kimg.FORECAST_DAYS).
                    7 = D+7 까지 (D+1~D+5 KIMR 1h, D+6~D+7 KIMG: hf135 까지 1h 이후 3h).
    """
    bases = _pick_bases(base, n_bases)
    with forecast_days_override(forecast_days):
        # 윈도우는 반드시 override 안에서 계산한다 -- _window_for 가 FORECAST_DAYS
        # 글로벌을 읽으므로 밖에서 부르면 기본 2일로 잡혀 build_wide 가 D+2 까지로
        # 잘라버린다(KIMG 는 7일치를 받아도 결과가 48행이 되는 함정).
        window_start, window_end = _window_for(bases)
        days_eff = ckg.FORECAST_DAYS   # ★override 밖에서는 기본값(2)으로 돌아간다
        # 기대 timestamp 집합도 여기서 뽑는다 (아래 결손 보간의 그리드).
        expected = sorted(set().union(
            *(expected_timestamps(b, ckg.FORECAST_DAYS) for b in bases)))
        # KIMR = std NC per-hf.  아카이브(collect_archive)가 161일 백필로 검증한
        # 것과 같은 스택이다.  여기가 오래 순차였고 base 당 시간의 대부분을 먹었다
        # (2026-08-24 측정: forecast 6.4분/base vs archive 1.7분/base).
        # point_workers 기본 3 = 3 지점 동시 (사용자 결정 2026-08-24).
        # 동시성 = N x 8.  hf 결손이 보이면 --point-workers 2 또는 1 로 내린다 --
        # `--verify` 와 수집 로그의 "n/n시각" 커버리지가 판정 근거다.
        kimr_long = fetch_kimr_long(bases, point_workers=point_workers)
        # KIMG 지점 병렬 opt-in.  3 지점 전부 동시(동시성 3×6=18)는 KMA 504 로 hf 가
        # 빠져(2026-06-16 실측) 2 지점 동시(동시성 12)로 낮춘다 -- 1 point(=6)가 원래
        # 안전했던 기준선과의 절충.  여전히 빠지면 1(순차)로 더 내릴 것.
        kimg_long = fetch_kimg_long(bases, point_workers=2)
    try:
        wide = build_wide(kimr_long, kimg_long, window_start, window_end)
    except NoUsableForecastRows:
        return pd.DataFrame()
    if wide.empty:
        return wide
    # 3h 결손 보간 (연속 2개까지, 내부만) -- 데이터센터 장애로 예보가 1h 가 아니라
    # 3h 간격으로만 오는 사고 대비.  ★clip_ranges 앞에 두어야 한다: clip_ranges 가
    # radiation/rainfall 의 NaN 을 0 으로 채우므로 뒤에 두면 메울 결손이 사라진다.
    # 기대 인덱스는 expected_timestamps(=collection_hf_range) 를 그대로 쓴다 --
    # 1h/3h 해상도 전환·MIN_HF 를 이미 반영한 SSOT 라 그리드를 여기서 추측하지 않는다.
    wide, _filled = pp.fill_short_gaps(wide, expected)
    if wide.empty:
        return wide
    # ③ 그래도 빈 일사·운량은 KIMR(NC)로 대체 -- 사다리의 마지막 칸.
    wide = _substitute_solar_cloud(wide, bases, window_start, window_end, days_eff)
    # ④ sentinel 정정 -- 9999 는 이상치가 아니라 "값 없음"이라 NaN 이 정직하다.
    #    clip_ranges 앞에 둔다 (뒤에 두면 9999 가 범위검사·통계에 그대로 섞인다).
    wide, _ = pp.drop_sentinels(wide)
    # postprocess: 범위 clip + day_type.  day_type 은 forecast_horizon 적재 시
    # _upsert_df 의 is_non_kma 필터가 떼므로 무해하지만 빌더 일관성 위해 적용.
    wide = pp.clip_ranges(wide)
    wide = pp.add_day_type(wide)
    return wide

# ── KIMG 실패 시 일사·운량 대체 (사다리 ③) ─────────────────────────────────
SOLAR_CLOUD_PREFIX = ("radiation", "total_cloud", "midlow_cloud")
SRC_SOLAR_CLOUD = "src_solar_cloud"   # 'KIMG'(정상) / 'KIMR'(대체) -- 출처 통보용


def _substitute_solar_cloud(wide, bases, window_start, window_end, days):
    """보간 뒤에도 비어 있는 일사·운량을 KIMR(NC)로 채운다.  출처를 컬럼에 남긴다.

    ★순서가 핵심이다 -- `fill_short_gaps` **다음**에 온다.
      짧은 결손(연속 2개 = 3h 사고)은 시간 보간이 KIMR 대체보다 훨씬 낫다.
      2026-07-02~11 사고를 정답(재수집본)과 대조한 실측, 결손 780칸:
          시간 보간   MAE 0.063~0.106  r 0.81~0.90  큰오차 3.8~9.9%
          KIMR 대체   MAE 0.346~0.490  r 0.25~0.43  큰오차 43~59%
      두 모델이 다른 구름을 본다(전운량 r 0.47).  그래서 KIMR 은 **보간이 못 메운
      자리에만** 들어간다.
    ★그래도 두는 이유: KIMG 가 통째로 안 오면 보간할 이웃이 없다.  그때는
      "예측 없음"보다 "오차 있는 예측 + 출처 통보"가 낫다 (사용자 결정 2026-08-12).
    ★KIMR lead 는 120h 라 D+1~5 를 거의 덮는다(꼬리 2h 는 못 채운다).
    ★`days` 는 **반드시 인자로 받는다** -- 이 함수는 `forecast_days_override` 밖에서
      불리므로 `ckg.FORECAST_DAYS` 를 직접 읽으면 기본값 2 로 돌아가 창이 48시각으로
      잘린다 (2026-08-12 실패 시나리오 테스트에서 잡았다).
    """
    # 기대 컬럼 = 3지점 × (일사·전운량·중하운량).  **컬럼이 아예 없는 것도 결손**이다
    # (KIMG 가 통째로 실패하면 build_wide 가 그 컬럼을 만들지 못한다).
    want = [f"{p}_{s}" for p in SOLAR_CLOUD_PREFIX for s in ci.POINT_SUFFIX.values()]
    have = wide.reindex(columns=want)      # 없는 컬럼은 전부 NaN 으로 채워 셈한다
    n_missing = int(have.isna().sum().sum())
    if n_missing == 0:
        wide[SRC_SOLAR_CLOUD] = SRC_KIMG
        return wide

    print(f"  [사다리③] 일사·운량 결손 {n_missing}셀 -- KIMR(NC)로 대체 시도")
    try:
        sub_long = pd.concat(
            [kimr.fetch_kimr_solar_cloud_long(kimr.POINTS_JEJU_V2, b, days)
             for b in bases], ignore_index=True)
    except Exception as e:
        print(f"  [WARN] KIMR 대체 fetch 실패: {e} -- 결손을 그대로 둔다")
        wide[SRC_SOLAR_CLOUD] = SRC_KIMG
        return wide
    if sub_long.empty:
        print("  [WARN] KIMR 대체분이 비었다 -- 결손을 그대로 둔다")
        wide[SRC_SOLAR_CLOUD] = SRC_KIMG
        return wide

    # 라벨을 KIMG 관례로 냈으므로 같은 피벗(_SPEC_KIMG)·kimg_solar 가 그대로 쓰인다.
    parts = []
    for point, suffix in ci.POINT_SUFFIX.items():
        part = ci.kimg_one_point(sub_long, point, suffix, window_start, window_end)
        rad = ci.kimg_solar(
            sub_long[(sub_long["category"] == "SOLAR_RAD")
                     & (sub_long["point_name"] == point)], suffix)
        if not rad.empty:
            part = rad.to_frame() if part.empty else part.join(rad, how="outer")
        if not part.empty:
            parts.append(part)
    if not parts:
        wide[SRC_SOLAR_CLOUD] = SRC_KIMG
        return wide
    sub = pd.concat(parts, axis=1)
    sub.index = pd.to_datetime(sub.index, format="%Y-%m-%d %H:%M").strftime(
        "%Y-%m-%d %H:%M:%S")
    sub = sub[[c for c in sub.columns
               if any(c.startswith(p + "_") for p in SOLAR_CLOUD_PREFIX)]]

    was_na = wide.reindex(columns=want).isna()        # 대체 **전** 결손 (컬럼 없음 포함)
    wide = wide.combine_first(sub.reindex(wide.index))   # 빈 칸만 채운다
    now_na = wide.reindex(columns=want).isna()           # 대체 **후** 결손 (같은 컬럼 집합)
    filled_cells = was_na & ~now_na
    filled, after = int(filled_cells.sum().sum()), int(now_na.sum().sum())
    print(f"  [사다리③] KIMR 로 {filled}셀 채움 (남은 결손 {after}셀)")
    # 실제로 채워진 시각만 'KIMR' 로 표기 -- 화면·검증이 출처를 볼 수 있게.
    wide[SRC_SOLAR_CLOUD] = np.where(filled_cells.any(axis=1), SRC_KIMR, SRC_KIMG)
    return wide


# ═══ ④ 적재 엔진 (forecast_horizon) ═══════════════════════════════════
def is_non_kma(col: str) -> bool:
    """이 아카이브에서 제외할 컬럼: KPX 익일(`*_da`: smp_*_da/*_est_demand_da) +
    달력(`day_type`).  KPX 는 D+1 한계라 지평 아카이브에 해당 없음(historical/
    forecast 테이블에만 존재), day_type 은 timestamp 에서 재산출 가능.
    migrate_forecast_horizon.is_kma_weather 와 동일 기준(지역 무관)."""
    return col.endswith("_da") or col == "day_type"

# 권역별 본 DB 와 기본 윈도우 길이.  region 축은 원본 이식 흔적으로 남겨 두되
# (호출부 시그니처 보존) 이 저장소의 권역은 제주 하나뿐이다 (육지 경로 삭제 2026-07-21).
# days 기본값은 **운영값 5** 다 (2026-07-31 운영 지평 D+5 통일, 2026-08-04 반영).
# 구 기본값 7 은 D+7 시절 잔재로, --days 없이 부르면 D+6~7 에 KIMG 3h 행만 쌓였다.
REGIONS: dict[str, dict] = {
    "jeju": {"db": DEFAULT_DB, "days": 5},
}

# ── 발표(base) 선택 ──────────────────────────────────────────────────────
def latest_12z(now_utc: datetime | None = None) -> datetime:
    """공개 지연을 감안한 가장 최근 가용 12 UTC 발표."""
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    cutoff = now_utc - timedelta(hours=PUBLISH_DELAY_HOURS)
    cand = cutoff.replace(hour=12, minute=0, second=0, microsecond=0)
    if cand > cutoff:
        cand -= timedelta(days=1)
    return cand


def backfill_12z_bases(n_days: int) -> list[datetime]:
    """가장 최근 가용 12z 부터 거꾸로 N 개 (오래된 것부터 적재하도록 reverse)."""
    latest = latest_12z()
    return [latest - timedelta(days=k) for k in range(n_days)][::-1]

# ── DB ───────────────────────────────────────────────────────────────────
def region_db(region: str, out: str | None) -> Path:
    """--out NAME 이면 data/<NAME>_<region>.db, 아니면 권역 본 DB."""
    if out:
        return DATA_DIR / f"{out}_{region}.db"
    return REGIONS[region]["db"]


def _upsert_df(df: pd.DataFrame, db_path: Path) -> int:
    """base + horizon_d 가 이미 붙은 DF 를 forecast_horizon 에 (base,timestamp) UPSERT.

    collect_historical.upsert_wide_to 와 같은 temp-table / INSERT OR REPLACE 패턴이되
    키가 복합.  본 테이블이 없으면 생성, 새 컬럼은 ALTER 로 확장.
    """
    if df.empty:
        return 0
    # 아카이브는 KMA 기상 전용 -- 빌더가 끼워넣는 KPX(*_da)/day_type 는 적재 전 제거.
    drop = [c for c in df.columns if is_non_kma(c)]
    if drop:
        df = df.drop(columns=drop)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"_tmp_{RUNS_TABLE}"
    with sqlite3.connect(db_path) as c:
        df.to_sql(tmp, c, if_exists="replace", index=True)

        existing = {r[1] for r in c.execute(f"PRAGMA table_info({RUNS_TABLE})").fetchall()}
        tmp_cols = [r[1] for r in c.execute(f"PRAGMA table_info({tmp})").fetchall()]
        if not existing:
            c.execute(f"CREATE TABLE {RUNS_TABLE} AS SELECT * FROM {tmp} WHERE 0")
            existing = set(tmp_cols)
        for col in tmp_cols:
            if col not in existing:
                c.execute(f'ALTER TABLE {RUNS_TABLE} ADD COLUMN "{col}"')

        c.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{RUNS_TABLE}_base_ts "
            f"ON {RUNS_TABLE}(base, timestamp)"
        )

        full_cols = [r[1] for r in c.execute(f"PRAGMA table_info({RUNS_TABLE})").fetchall()]
        select_exprs = [f'"{col}"' if col in tmp_cols else "NULL" for col in full_cols]
        col_list = ", ".join(f'"{col}"' for col in full_cols)
        c.execute(
            f"INSERT OR REPLACE INTO {RUNS_TABLE} ({col_list}) "
            f"SELECT {', '.join(select_exprs)} FROM {tmp}"
        )
        n = c.execute("SELECT changes()").fetchone()[0]
        c.execute(f"DROP TABLE {tmp}")
    return n


def upsert_runs(wide: pd.DataFrame, base_utc: datetime, db_path: Path) -> int:
    """wide(단일 base)에 base + horizon_d 태그를 붙여 forecast_horizon 에 UPSERT."""
    if wide.empty:
        return 0
    base_kst = base_utc.astimezone(KST)
    base_date = base_kst.date()

    df = wide.copy()
    ts_dates = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S").date
    df.insert(0, "horizon_d", [(d - base_date).days for d in ts_dates])
    df.insert(0, "base", base_kst.strftime("%Y-%m-%d %H:%M:%S"))
    df.index.name = "timestamp"
    return _upsert_df(df, db_path)


def verify_runs(db_path: Path, region: str, out: str | None,
                days: int | None = None) -> list[str]:
    """base 별 완전성 검사: 행수 + 지점별 sentinel 컬럼(temp*)의 NULL 셀.

    hf 단위 실패는 행 누락이 아니라 해당 지점 컬럼의 NULL 셀로 남는다 (다른
    지점이 그 시각 행을 만들기 때문) -- 그래서 행수(COUNT(*))만으론 못 잡고
    지점별 non-null 을 행수와 비교한다.  temp* 를 sentinel 로 쓰는 이유: 모든
    지점·모든 hf 가 항상 포함하는 변수라 'NULL = 그 (지점,시각) fetch 실패'.
    (제주 KIMR 전용 변수(cape 등)는 D+6+ 가 정상 NULL 이라 sentinel 부적합.)

    반환: 재수집이 필요한 base 날짜(YYYYMMDD) 목록.  출력으로 상세와 redo 명령을
    찍는다.
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        print(f"[verify:{region}] {db_path.name} 없음")
        return []
    # 기대 행수는 **수집 창에서 계산**한다 (구 하드코딩 144 는 D+7 시절 값이라,
    # D+5 로 정상 수집된 base 를 전부 불완전으로 오판했다 -- 2026-08-04 수정).
    days = days if days is not None else REGIONS[region]["days"]
    with sqlite3.connect(db_path) as c:
        try:
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({RUNS_TABLE})")]
        except sqlite3.OperationalError:
            print(f"[verify:{region}] {RUNS_TABLE} 테이블 없음")
            return []
        # sentinel = 모든 지점·모든 hf 가 항상 가져야 하는 컬럼.
        # · temp_*      : KIMR/KIMG 둘 다 내므로 NULL = 그 (지점,시각) fetch 실패
        #   (temp_skin_* 은 KIMR 전용이라 D+5 꼬리 NULL 이 정상 -- 제외)
        # · total_cloud_*: **2026-08-04 추가.** 2026-07-02~11 에 KIMG 운량만 3h 로 떨어진
        #   사고가 있었는데 temp 는 멀쩡해서 10 base 동안 못 잡았다. 운량은 전 지평
        #   KIMG 단독 공급이라 D+1~5 에서 NULL 이면 실패가 맞다.
        sentinels = [
            col for col in cols
            if (col.startswith("temp") and not col.startswith("temp_skin"))
            or col.startswith("total_cloud")
        ]
        if not sentinels:
            print(f"[verify:{region}] temp* sentinel 컬럼 없음 -- 검사 불가")
            return []
        agg = ", ".join(f'COUNT("{col}") AS "{col}"' for col in sentinels)
        # ★운영 지평(D+1~days)만 본다.  기대 행수를 그 창에서 계산하므로 NULL 검사도
        #   같은 범위여야 일관된다 -- D+7 시절에 쌓인 잔재 행(3h 간격, KIMG-only)까지
        #   세면 정상 base 가 영원히 불완전으로 남는다 (2026-08-04).
        rows = c.execute(
            f"SELECT base, COUNT(*) AS n_rows, {agg} FROM {RUNS_TABLE} "
            f"WHERE horizon_d <= ? GROUP BY base ORDER BY base",
            (days,),
        ).fetchall()

    expected_rows = len(expected_timestamps(latest_base(12), days))
    bad: list[str] = []
    print(f"[verify:{region}] {db_path.name}::{RUNS_TABLE} -- "
          f"{len(rows)} bases, sentinel {len(sentinels)} cols, 기대 행수 {expected_rows}")
    for row in rows:
        base, n_rows, counts = row[0], row[1], row[2:]
        defects = []
        if n_rows < expected_rows:
            defects.append(f"rows {n_rows}/{expected_rows}")
        defects += [
            f"{col} null x{n_rows - cnt}"
            for col, cnt in zip(sentinels, counts) if cnt < n_rows
        ]
        if defects:
            bad.append(base[:10].replace("-", ""))
            print(f"  {base[:10]}  INCOMPLETE: {', '.join(defects)}")
    if bad:
        out_opt = f" --out {out}" if out else ""
        print(f"\n[verify:{region}] 불완전 base {len(bad)}개 -- 재수집:")
        for d in bad:
            print(f"  python collectors/collect_forecast.py "
                  f"--base {d} --region {region}{out_opt} --no-kpx --force")
    else:
        print(f"[verify:{region}] 모든 base 완전 (OK)")
    return bad


def merge_runs(src_db: Path, dst_db: Path) -> int:
    """src 의 forecast_horizon 전체를 dst 의 forecast_horizon 에 (base,timestamp) UPSERT."""
    if not src_db.exists():
        print(f"  [merge] {src_db.name} 없음 -- skip")
        return 0
    with sqlite3.connect(src_db) as c:
        try:
            df = pd.read_sql(f"SELECT * FROM {RUNS_TABLE}", c)
        except Exception as e:
            print(f"  [merge] {src_db.name} 읽기 실패: {e} -- skip")
            return 0
    if df.empty:
        print(f"  [merge] {src_db.name} 비어있음 -- skip")
        return 0
    df = df.set_index("timestamp")
    n = _upsert_df(df, dst_db)
    print(
        f"  [merge] {src_db.name} -> {dst_db.name}::{RUNS_TABLE}  "
        f"{n:,} rows ({df['base'].nunique()} bases)"
    )
    return n


# ── KPX 차단 (forecast_horizon 는 KMA 기상 전용) ──────────────────────────
def disable_kpx() -> None:
    """KPX *_da (smp_*_da / *_est_demand_da) 호출을 끈다 -- 기상청(KIMR/KIMG)만 수집.

    forecast_horizon 는 KMA 전용이라 항상 호출(=불필요한 data.go.kr 호출/429)을 끈다.
    기존 수집기 무수정 원칙대로 이 프로세스 안에서만 패치 -- fetch_kpx_est 가 빈 DF.
    설령 *_da 가 들어와도 _upsert_df 의 is_non_kma 필터가 적재 전 제거하므로 이중 안전.
    """
    kpx.fetch_kpx_est = lambda s, e: pd.DataFrame()
    print("[collect_forecast] KMA 전용: KPX(*_da) 호출 비활성 (기상청만)")

# ═══ ⑤ 진입점: base 선택 + 완결성 skip + CLI ═══════════════════════════
# ── fetch: 네이티브 clean wide (KMA 전용, KPX 경로 자체가 없음) ──────────────
def fetch_one(region: str, base_utc: datetime, days: int,
              point_workers: int = 3) -> pd.DataFrame:
    """단일 base 의 기상 wide 를 메모리로 반환 (KMA 전용).

    jeju : build_forecast_wide (KIMR+KIMG 병합 3 지점, 일사 3지점).
    KPX(*_da) 호출이 경로에 없어 forecast_horizon 에 KPX 가 섞일 여지가 원천 차단된다.
    """
    if region != "jeju":
        raise SystemExit(f"[collect_forecast] 지원하지 않는 권역 '{region}' (제주 전용)")
    return build_forecast_wide(base=base_utc, forecast_days=days,
                               point_workers=point_workers)


# ── base 선택: 발행시각(UTC) 일반화 ────────────────────────────────────────
def latest_base(utc_hour: int, now_utc: datetime | None = None) -> datetime:
    """공개 지연을 감안한 가장 최근 가용 발표 (latest_12z 의 발행시각 일반화).

    KIMG/KIMR 은 00/06/12/18 UTC 발표.  운영 cron 은 12z 고정이고, 12z 발표에
    문제가 있을 때(예: 장지평 빈 응답) 관리자 메뉴에서 06z·18z 를 수동 수집한다.
    utc_hour=12 이면 latest_12z 와 동일한 base 를 돌려준다.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    cutoff = now_utc - timedelta(hours=PUBLISH_DELAY_HOURS)
    cand = cutoff.replace(hour=utc_hour, minute=0, second=0, microsecond=0)
    if cand > cutoff:
        cand -= timedelta(days=1)
    return cand


# ── 완결성 기반 skip ───────────────────────────────────────────────────────
def expected_timestamps(base_utc: datetime, days: int) -> set[str]:
    """수집기가 실제로 받는 timestamp 집합 = {base_kst + hf : hf ∈ collection_hf_range}.

    (region 인자는 제주 단일 권역이 된 뒤로 쓰이지 않아 2026-08-04 제거.)

    collection_hf_range 가 윈도우(forecast_days_override)·1h/3h 해상도 전환·MIN_HF 를 모두
    반영하므로, 완전 base 의 저장 timestamp 집합과 정확히 일치한다.
    병합 wide 의 timestamp 집합 = KIMG collection_hf_range 그리드와 같다
    (KIMR 1h D+1~5 는 KIMG 1h ≤hf135 의 부분집합이라 새 시각을 더하지 않음, 7일=144행).
    override 는 kma_kimg.FORECAST_DAYS 를 세팅하고 collection_hf_range 가 그 글로벌을
    읽는다 (jeju override 는 KIMR 글로벌도 세팅하나 hf 그리드 산출엔 무관).
    """
    base_kst = base_utc.astimezone(KST)
    with forecast_days_override(days):
        hfs = list(ckg.collection_hf_range(base_utc))
    return {(base_kst + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in hfs}


def base_complete(region: str, db_path: Path, base_utc: datetime, days: int) -> bool:
    """그 base 가 완전한가 = (기대 timestamp 전부 존재) AND (지점 sentinel temp* NULL 셀 없음).

    hf 단위 실패는 행 누락(timestamp 부재) 또는 그 지점 컬럼의 NULL 셀로 남으므로 둘 다 본다.
    temp_skin* 은 KIMR 전용(장지평 정상 NULL)이라 sentinel 에서 제외(verify_runs 와 동일).
    sentinel = temp_west/east/south (3지점 모두 KIMR/KIMG temp 보유).
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False
    base_str = base_utc.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
    exp = expected_timestamps(base_utc, days)
    with sqlite3.connect(db_path) as c:
        try:
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({RUNS_TABLE})").fetchall()]
        except sqlite3.OperationalError:
            return False
        if "timestamp" not in cols:
            return False
        present = {r[0] for r in c.execute(
            f"SELECT timestamp FROM {RUNS_TABLE} WHERE base=?", (base_str,)).fetchall()}
        if not exp.issubset(present):
            return False
        sentinels = [col for col in cols
                     if col.startswith("temp") and not col.startswith("temp_skin")]
        if sentinels:
            agg = ", ".join(f'SUM("{s}" IS NULL)' for s in sentinels)
            nullcnt = c.execute(
                f"SELECT {agg} FROM {RUNS_TABLE} WHERE base=?", (base_str,)).fetchone()
            if any((v or 0) > 0 for v in nullcnt):
                return False
    return True


def run_region(
    region: str, bases: list[datetime], days: int, force: bool, db_path: Path,
    point_workers: int = 3,
) -> int:
    """base 목록을 차례로 수집 → forecast_horizon 적재 (완결성 기반 skip).

    skip 판정 = base 가 완전(base_complete)하면 건너뛰고, **불완전하면 다시 받는다**.
    쓰기가 (base,timestamp) upsert 라 부분 적재 base 는 재실행만으로 부족분이 채워진다(auto-resume).
    --force 는 "완전해도 다시 받기".  네이티브 build_forecast_wide 경로라 완결성 기반으로
    통일(과거 count 기준은 부분 적재 base 를 못 채웠다).
    """
    total = 0
    warnings: list[str] = []
    for i, b in enumerate(bases, 1):
        base_str = b.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
        label = f"[runs:{region}] base {b.strftime('%Y%m%d %HZ')} ({base_str} KST)"
        if not force and base_complete(region, db_path, b, days):
            print(f"{label} -- skip (완전; --force 로 재수집)")
            continue
        print(f"\n{'='*70}\n{label}  ({i}/{len(bases)}, window={days}d)\n{'='*70}")
        try:
            wide = fetch_one(region, b, days, point_workers=point_workers)
        except Exception as e:
            print(f"{label} -- [WARN] fetch failed: {e} (skip)")
            continue
        if wide.empty:
            print(f"{label} -- [WARN] empty wide, nothing to write")
            continue
        # ★적재 직전 건전성 검사.  값은 고치지 않고 **드러내기만** 한다 --
        #   무엇이 이상인지는 사람이 판단해야 하고, 잘못 고치면 더 나쁘다.
        #   (2026-07-02~11 운량 사고가 10 base 동안 조용했던 게 이 검사가 없어서다.)
        for w in pp.sanity_check(wide, label=f"[sanity {b:%Y%m%d}]"):
            print(f"  ⚠ {w}")
            warnings.append(w)
        n = upsert_runs(wide, b, db_path)
        total += n
        h = pd.Series(
            [(d - b.astimezone(KST).date()).days
             for d in pd.to_datetime(wide.index).date]
        )
        print(
            f"{label} -- UPSERT {n:,} rows -> {db_path.name}::{RUNS_TABLE} "
            f"(horizon D+{h.min()}~D+{h.max()})"
        )
    if warnings:
        print(f"\n[sanity:{region}] 건전성 경고 {len(warnings)}건 -- 위 로그 확인")
    return total, warnings


# ── CLI ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "KMA 발표(기본 12 UTC, --utc 로 변경)를 horizon-tagged 로 forecast_horizon 에 적재 "
            "(KMA 기상 전용, 제주 clean wide = build_forecast_wide)."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--base", metavar="YYYYMMDD",
        help="특정 날짜의 발표 (기본: 최신 가용 발표, 시각은 --utc)",
    )
    mode.add_argument(
        "--backfill", type=int, metavar="N_DAYS",
        help="과거 N 일치 발표를 차례로 적재 (resume-skip, 시각은 --utc)",
    )
    mode.add_argument(
        "--merge", metavar="NAME",
        help="data/<NAME>_<region>.db 의 forecast_horizon 를 본 DB 에 병합 (fetch 없음)",
    )
    mode.add_argument(
        "--verify", action="store_true",
        help="base 별 완전성 검사 (행수 + 지점별 NULL 셀) -- fetch 없음",
    )
    p.add_argument(
        "--utc", type=int, choices=[0, 6, 12, 18], default=12,
        help="발표 시각 UTC (default 12 -- 운영 cron 과 동일). "
             "--utc 18 은 '당일예보' 모드(창 = 당일 04시부터, horizon_d=0 생성). "
             "--base 는 UTC 날짜: 20260717 의 18z = 07-18 03시 KST 발표(당일=07-18)",
    )
    p.add_argument(
        "--region", choices=["jeju"], default="jeju",
        help="대상 권역 (제주 전용 -- 육지 경로 삭제 2026-07-21)",
    )
    p.add_argument(
        "--days", type=int, default=None, metavar="N",
        help="윈도우 길이 override (기본 7, 운영 cron 은 12z 5일 / 18z 3일)",
    )
    p.add_argument(
        "--out", metavar="NAME", default=None,
        help="본 DB 대신 data/<NAME>_jeju.db 에 적재 (백필 격리용)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="완전한 base 도 다시 받는다 (기본은 완결성 기반: 불완전 base 는 재실행만으로 부족분 auto-resume)",
    )
    p.add_argument(
        "--min-hf", type=int, default=0, metavar="H",
        help="이 hf(시간) 이상만 수집 -- 장지평 증분 백필용 (기존 짧은 지평 재호출 생략)",
    )
    p.add_argument(
        "--point-workers", type=int, default=3, metavar="N",
        help="KIMR 지점 병렬 수 (기본 3 = 3 지점 동시).  동시성 = N x 8.  "
             "1 로 내리면 순차(가장 안전), 2 는 절충.",
    )
    args = p.parse_args()

    if args.min_hf:
        ckg.MIN_HF = args.min_hf
        print(f"[collect_forecast] 증분 모드: hf >= {args.min_hf} 만 수집")

    # 위임 경로를 위해 KPX 호출을 끈다.  build_forecast_wide 는 KPX 를 부르지
    # 않지만 _upsert_df 의 is_non_kma 필터와 함께 이중 안전.
    disable_kpx()

    regions = [args.region]

    # 18z 당일예보 모드 (opt-in -- kma_kimg.SAMEDAY_18Z).
    if args.utc == 18:
        ckg.SAMEDAY_18Z = True
        if args.days is None:
            args.days = 3          # 당일(hd0)~모레(hd2) -- KIMR 18z lead 72h 와 짝
        print(f"[collect_forecast] 18z 당일예보 모드 (창 = 당일 04시~, days={args.days})")

    if args.verify:
        bad = []
        for region in regions:
            bad += verify_runs(region_db(region, args.out), region, args.out, args.days)
            print()
        return 1 if bad else 0      # 스크립트에서 rc 로 판정할 수 있게

    if args.merge:
        print(f"[collect_forecast] merge '{args.merge}' -> main DBs")
        for region in regions:
            merge_runs(region_db(region, args.merge), REGIONS[region]["db"])
        return 0

    if args.base:
        bases = [datetime.strptime(args.base, "%Y%m%d").replace(hour=args.utc, tzinfo=UTC)]
    elif args.backfill:
        bases = [latest_base(args.utc) - timedelta(days=k)
                 for k in range(args.backfill)][::-1]
    else:
        bases = [latest_base(args.utc)]

    print(
        f"[collect_forecast] regions={regions}  bases={len(bases)} "
        f"({bases[0].strftime('%Y%m%d')}~{bases[-1].strftime('%Y%m%d')} {args.utc:02d}Z)  "
        f"out={'main DB' if not args.out else args.out + '_<region>.db'}  "
        f"table='{RUNS_TABLE}' key=(base,timestamp)"
    )

    t0 = time.time()
    all_warnings: list[str] = []
    for region in regions:
        days = args.days if args.days is not None else REGIONS[region]["days"]
        db_path = region_db(region, args.out)
        n, warns = run_region(region, bases, days, args.force, db_path,
                              point_workers=args.point_workers)
        all_warnings += warns
        print(f"\n[runs:{region}] total UPSERT {n:,} rows -> {db_path.name}")
    print(f"\n[collect_forecast] done in {(time.time()-t0)/60:.1f}m")
    if all_warnings:
        # rc=1 로 올려 run_pipeline 의 단계별 rc 요약에 남게 한다.  적재는 이미 끝났고
        # (부분 결과라도 남기는 게 낫다) "확인이 필요하다"는 신호만 준다.
        print(f"[collect_forecast] ⚠ 건전성 경고 {len(all_warnings)}건 -- rc=1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
