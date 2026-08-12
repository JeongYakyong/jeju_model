"""postprocess.py -- forecast / historical wide DataFrame 정제 + day_type 부착.

수집 진입점(collect_forecast / collect_archive / collect_historical)이 UPSERT 직전에
이 모듈의 함수들을 호출한다.  소스 fetcher(KIMR/KIMG/KPX/ASOS)는 손대지 않는다.

    wide = build_wide(...)            # API -> 메모리 wide
    wide, _ = postprocess.fill_short_gaps(wide, expected_index)   # 3h 결손 보간
    wide = postprocess.clip_ranges(wide)
    wide = postprocess.add_day_type(wide)
    upsert_wide_to(table, wide, db)

★순서 주의: `fill_short_gaps` 는 반드시 `clip_ranges` **앞**이다 -- clip_ranges 가
  radiation/rainfall 의 NaN 을 0 으로 채우므로(_ZERO_FILL_PREFIXES), 뒤에 두면
  메워야 할 결손이 이미 0 으로 위조돼 보간할 것이 없어진다.

규칙은 컬럼명 prefix 기준 dispatch -- 새 컬럼이 추가돼도 매핑 안 된 컬럼은 그대로
통과(안전한 fallback).  forecast / historical 둘 다 같은 함수로 처리한다.

물리적 범위
- 온도 (temp/temp_c/temp_skin)              : -50 ~ 50 °C, 밖은 NaN.
- 습도 (humidity/reh)                       : 0 ~ 100 %, 밖은 NaN.
- 운량 (total/midlow/low/mid/high_cloud)    : 0 ~ 1, 밖은 NaN.  (KIMR 의 tcog/tcoh
  는 cloud fraction 이 아니라 kg/m^2 column water 라 여기 안 들어감.)
- 풍속 / 돌풍 (wind_spd/gust)               : 0 ~ 100 m/s, 밖은 NaN.
- 풍향 sin/cos (wd_sin/wd_cos)              : -1 ~ 1, 밖은 NaN.
- 일사(예보) / 강수 (radiation/rainfall)    : <0 -> 0 clip, NaN -> 0.
  (예보 결측/무강수가 사실상 0 인 변수 -- KIMG 야간 radiation, 무강수 시간 등.)
  ASOS solar_rad 는 제외 -- fetch_asos 가 센서 가동일 야간만 0 으로 채우고 무센서
  구간은 NaN 으로 두므로, 여기서 일괄 0 채우면 '관측 없음'을 '일사 0' 으로 위조.
- KPX 전력량 / 수요 (supply_cap_*/real_demand_*/max_pred_demand_*/*_reserve/
  jeju_est_demand_da/land_est_demand_da/재래식 gen_*: 화력·원자력·수력·가스·
  gen_total_kr 등)                       : 음수 NaN (음의 수요/재래식 발전은 불가능).
  *_land / *_jeju / *_kr 접미사가 prefix 매칭으로 자동 흡수.
  예외(음수 허용, 클립 제외):
    - gen_pumped_* (양수발전): 충전(펌핑) 시 음수 정상.
    - 태양광/풍력/재생합계 발전 (gen_solar_*/gen_wind_*/renew_gen_total_* +
      제주 real_solar_gen/real_wind_gen/real_renew_gen): ESS 충방전·계량보정으로
      소폭 음수 실측 -> 음수 허용 (하한 클립 안 함; 큰 이상 음수 가드는 미적용).
    - net_load_* (=total-renewables): 잔차라 음수 가능 -> 통과.
    - *_utilization_* (이용률, ESS 로 소폭 음수 가능) / *_capacity_* (파생): 통과.
- 예비율 (*_pct, *_pct_land)                : 음수 NaN (운영예비율 200% 까지 정상이라
  상한은 두지 않음).
- 그 외 (cape/cinn/hpbl/tcog/tcoh/smp_*)   : 통과 (음의 SMP, 음의 CINN 은 정상).
  제주 RT SMP 의 구간 원시값(smp_rt_g1..g4)·시간평균(smp_jeju_rt)도 음수 가능이라
  통과.  smp_rt_neg_num(음수권 구간 개수 0..4)도 카운트라 그대로 통과.

day_type ('weekday' / 'weekend' / 'holiday')
- holidays.SouthKorea() 가 인정하는 한국 공휴일이면 'holiday' (대체공휴일 포함).
- 공휴일 아니고 토/일이면 'weekend'.
- 그 외는 'weekday'.
- 우선순위: holiday > weekend > weekday.  (토/일에 겹친 공휴일도 'holiday'.)
"""
from __future__ import annotations

import pandas as pd
import holidays


# ── 컬럼 prefix 매핑 ────────────────────────────────────────────────────
# 모든 location suffix(west/east/south/land/무) 에 자동 적용되도록 prefix 매칭.
_TEMP_PREFIXES = ("temp", "temp_skin", "temp_c")
_HUMID_PREFIXES = ("humidity", "reh")
# low/mid/high 는 2026-08-04 KIMG 확장분 (층별 원시 운량).  전부 0~1 분율.
_CLOUD_PREFIXES = ("total_cloud", "midlow_cloud", "low_cloud", "mid_cloud", "high_cloud")
_WIND_PREFIXES = ("wind_spd", "gust")
_TRIG_PREFIXES = ("wd_sin", "wd_cos")
# NaN -> 0 채움 대상.  ASOS solar_rad 는 여기 넣지 않는다 -- fetch_asos 가 이미
# 센서 가동일의 야간만 0 으로 채웠고, 무센서 구간(성산/남쪽 2024 이전)의 NaN 을
# 여기서 0 으로 덮으면 '관측 없음'이 '일사 0' 으로 위조되기 때문.  radiation(KIMG
# 예보)은 야간 0 이 맞아 유지, rainfall 은 무강수=0 으로 유지.  snow_depth 는 미수집.
_ZERO_FILL_PREFIXES = ("radiation", "rainfall")
_KPX_POWER_PREFIXES = (
    "supply_cap", "real_demand", "max_pred_demand", "supply_reserve",
    "oper_reserve", "jeju_est_demand", "land_est_demand",
    # 발전원별 발전실적 컬럼은 모두 'gen_' prefix -- 한 항목으로 흡수.
    # gen_total_kr / gen_nre_kr / 화력·원자력 등 재래식 발전은 모두 >=0.
    # (태양광/풍력/재생합계/양수 + 제주 실측발전은 아래 _KPX_SIGNED_PREFIXES 로 음수 허용.)
    "gen",
    # net_load_kr(=total-renewables)은 음수 가능성 있어 클립 대상에서 제외(통과).
)
# 음수가 물리적으로 정상이라 클립(음수->NaN)에서 제외하는 발전 컬럼.
# - gen_pumped(양수발전): 충전(펌핑) 시 음의 발전량.
# - 태양광/풍력/재생합계 발전: ESS 충방전·계량보정으로 소폭 음수가 실측된다 -> 음수 허용
#   (단 _KPX_SIGNED_PREFIXES 는 하한 클립을 아예 안 하므로 큰 이상 음수도 통과; 필요시
#   별도 임계값 가드 추가).  land(_kr) gen_solar_*/gen_wind + renew_gen_total,
#   제주 실측 real_solar_gen/real_wind_gen/real_renew_gen 모두 포함.
_KPX_SIGNED_PREFIXES = (
    "gen_pumped",
    "gen_solar_btm", "gen_solar_ppa", "gen_solar_market", "gen_wind",
    "renew_gen_total",
    "real_solar_gen", "real_wind_gen", "real_renew_gen",
)


def _match(col: str, prefixes: tuple[str, ...]) -> bool:
    """col 이 prefixes 중 하나와 정확히 같거나 'prefix_' 로 시작하는지."""
    for p in prefixes:
        if col == p or col.startswith(p + "_"):
            return True
    return False


def _is_pct_col(col: str) -> bool:
    """예비율 % 컬럼 식별 (supply_reserve_pct, oper_reserve_pct_land 등)."""
    return "_pct" in col


def clip_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """물리적 범위 밖 값 정제.  복사본 반환 (원본 안 건드림).

    매핑 안 된 컬럼은 그대로 통과 -- KPX/ASOS 가 늘어나도 안전.
    """
    if df.empty:
        return df
    out = df.copy()
    n_clipped = 0
    n_zerofilled = 0
    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        # 이용률(*_utilization_*: ESS 로 소폭 음수 가능)/용량(*_capacity_*: 파생)은
        # 물리범위 클립 대상이 아니므로 통과.  (현재 recompute 가 partial_upsert 로 직접
        # 써서 이 경로를 안 타지만, 일관성/안전을 위해 명시.)
        if "_utilization" in col or "_capacity" in col:
            continue
        before_nan = s.isna().sum()
        if _match(col, _TEMP_PREFIXES):
            out[col] = s.where((s >= -50) & (s <= 50))
        elif _match(col, _HUMID_PREFIXES):
            out[col] = s.where((s >= 0) & (s <= 100))
        elif _match(col, _CLOUD_PREFIXES):
            out[col] = s.where((s >= 0) & (s <= 1))
        elif _match(col, _WIND_PREFIXES):
            out[col] = s.where((s >= 0) & (s <= 100))
        elif _match(col, _TRIG_PREFIXES):
            out[col] = s.where((s >= -1) & (s <= 1))
        elif _match(col, _ZERO_FILL_PREFIXES):
            out[col] = s.clip(lower=0).fillna(0)
            n_zerofilled += int(before_nan)
            continue  # n_clipped 카운트에서 제외 (별도 카운트).
        elif _match(col, _KPX_SIGNED_PREFIXES):
            continue  # 양수발전 등 음수 정상 -> 클립 없이 통과.
        elif _match(col, _KPX_POWER_PREFIXES):
            out[col] = s.where(s >= 0)
        elif _is_pct_col(col):
            out[col] = s.where(s >= 0)
        else:
            continue
        after_nan = out[col].isna().sum()
        n_clipped += int(after_nan - before_nan)
    if n_clipped or n_zerofilled:
        print(
            f"  postprocess.clip_ranges: out-of-range -> NaN ({n_clipped}), "
            f"NaN -> 0 in zero-fill cols ({n_zerofilled})"
        )
    return out


# ── sentinel: "값이 아니라 없음" ─────────────────────────────────────────────
# ★9999 는 **이상치가 아니라 결측**이다.  cape/cinn 에서 9999 = "대류불안정 없음"
#   (Training EDA 3cmp-2 기록).  숫자로 두면 평균·상관·모델 입력이 통째로 망가진다
#   -- 구 GRIB 구간의 forecast_horizon 은 cape 57% / cinn 68% 가 9999 다.
#   NC 경로는 9999 를 내지 않으므로(2026-08-04 실측 0건) 신규 수집분엔 무영향이고,
#   API 가 다시 내보내기 시작해도 여기서 막힌다.
SENTINEL_VALUES = (9999.0, -9999.0)
SENTINEL_PREFIXES = ("cape", "cinn")


def drop_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """cape/cinn 의 sentinel(9999)을 NaN 으로.  반환 (df, 바꾼 셀 수).

    "이상치 제거"가 아니라 **결측 표기 정정**이다 -- 그 시각에 값이 없었다는 뜻이므로
    NaN 이 정직한 표현이고, 다운스트림(평균·상관·학습)이 알아서 빼고 센다.
    """
    if df.empty:
        return df, 0
    out = df.copy()
    n = 0
    for col in out.columns:
        if not _match(col, SENTINEL_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        hit = out[col].isin(SENTINEL_VALUES)
        if hit.any():
            n += int(hit.sum())
            out.loc[hit, col] = pd.NA
    if n:
        print(f"  postprocess.drop_sentinels: 9999 sentinel -> NaN ({n}셀) "
              f"— 이상치가 아니라 '값 없음'이다")
    return out, n


# ── 건전성 검사 (수집 직후) ──────────────────────────────────────────────────
# clip_ranges 는 범위 **밖**만 잡는다.  범위 **안**의 이상 -- 얼어붙은 값, 급변,
# 변수끼리의 모순 -- 은 아무도 못 잡아서 2026-07-02~11 운량 사고가 10 base 동안
# 조용히 지나갔다.  여기서 잡아 경고로 올린다 (값은 건드리지 않는다: 무엇이
# 이상인지 사람이 판단해야 하고, 잘못 고치면 더 나쁘다).
_MUST_VARY = ("temp", "reh", "wind_spd_10m")     # 5일 창에서 상수면 수집 이상
_JUMP_LIMIT = {"temp": 5.0, "reh": 40.0, "wind_spd_10m": 15.0, "dewpoint": 5.0}


def sanity_check(df: pd.DataFrame, label: str = "") -> list[str]:
    """수집 직후 건전성 검사.  경고 문자열 목록 반환 (빈 목록 = 이상 없음).

    값을 고치지 않는다 -- **드러내는 것**이 목적이다.  호출자가 rc 에 반영한다.
    sentinel(9999)은 이미 drop_sentinels 가 NaN 으로 바꾼 뒤라 결측으로 세어진다.
    """
    if df.empty:
        return [f"{label} wide 가 비었다"]
    warn: list[str] = []
    num = df.select_dtypes(include="number")

    # ⓪ 결측률 -- 소스가 공급해야 하는 대표 컬럼이 비면 수집 사고다.
    #    (2026-07-02~11 은 운량만 3h 로 떨어졌는데 temp 는 멀쩡해서 아무도 몰랐다.)
    for pref in ("temp", "total_cloud", "radiation"):
        cols = [c for c in num.columns if _match(c, (pref,))
                and not c.startswith("temp_skin")]
        if not cols:
            warn.append(f"{label} {pref}_* 컬럼이 없다 -- 소스 실패 의심")
            continue
        na = num[cols].isna().mean().mean()
        if na > 0.02:
            warn.append(f"{label} {pref}_* 결측 {na*100:.1f}% "
                        f"({int(num[cols].isna().sum().sum())}셀) -- 수집 사고 의심")

    # ① 얼어붙음 -- 5일 창에서 상수인 변수는 수집이 멈춘 것이다
    for col in num.columns:
        if not _match(col, _MUST_VARY):
            continue
        v = num[col].dropna()
        if len(v) > 10 and v.nunique() == 1:
            warn.append(f"{label} {col}: 전 구간 상수({v.iloc[0]}) — 수집 이상 의심")

    # ② 급변 -- 1시간 사이 물리적으로 어려운 변화
    for col in num.columns:
        base_pref = next((p for p in _JUMP_LIMIT if _match(col, (p,))), None)
        if base_pref is None:
            continue
        d = num[col].diff().abs()
        n = int((d > _JUMP_LIMIT[base_pref]).sum())
        if n:
            warn.append(f"{label} {col}: 1h 급변 {n}회 "
                        f"(>{_JUMP_LIMIT[base_pref]}, 최대 {d.max():.1f})")

    # ③ 변수끼리의 모순 -- 하나라도 있으면 파생이나 소스가 어긋난 것이다
    for sfx in ("west", "east", "south"):
        t, td = f"temp_{sfx}", f"dewpoint_{sfx}"
        if t in num and td in num:
            n = int((num[td] > num[t] + 0.5).sum())
            if n:
                warn.append(f"{label} 이슬점 > 기온 ({sfx}) {n}회")
        w, g = f"wind_spd_10m_{sfx}", f"gust_{sfx}"
        if w in num and g in num:
            n = int((num[g] < num[w] - 0.1).sum())
            if n:
                warn.append(f"{label} 돌풍 < 풍속 ({sfx}) {n}회")
        tc = f"total_cloud_{sfx}"
        layers = [f"{p}_{sfx}" for p in ("low_cloud", "mid_cloud", "high_cloud")
                  if f"{p}_{sfx}" in num]
        if tc in num and layers:
            n = int((num[layers].max(axis=1) > num[tc] + 0.02).sum())
            if n:
                warn.append(f"{label} 층별운량 > 전운량 ({sfx}) {n}회")
    return warn


def fill_short_gaps(
    df: pd.DataFrame, index, limit: int = 2,
) -> tuple[pd.DataFrame, int]:
    """기대 timestamp 인덱스로 재색인 + 연속 `limit` 개까지 시간보간 (내부만).

    무엇을 고치나: KMA 데이터센터 장애로 예보가 1h 가 아니라 **3h 간격으로만** 오는
    사고가 있다 (2026-07-02~11 KIMG 운량 실측: 10 base × 78행 결손).  행이 통째로
    빠지기도 하고, 행은 있는데 특정 컬럼만 NULL 로 비기도 한다 -- 둘 다 여기서 메운다.
    3h 결손은 1h 그리드에서 연속 NaN 2개라 기본값 limit=2 로 정확히 덮인다.

    ★외삽 금지(`limit_area='inside'`).  양끝을 지어내지 않는다 -- KIMR lead 꼬리
      (D+5 22~23시)는 NaN 으로 남겨 결손임이 드러나야 한다(백업 소스 몫).
    ★`index` 는 호출자가 자기 SSOT 로 만든다 -- 수집 창이 1h 인지 3h 인지를 이 함수가
      추측하지 않기 위해서다 (collect_forecast 는 expected_timestamps, collect_archive
      는 1h date_range).  인덱스가 3h 간격이면 limit 은 '행 수'라 6시간을 뜻하게 되니
      주의 (운영 창 --days 5 는 전 구간 1h 라 해당 없음).

    반환: (재색인·보간된 df, 보간으로 채운 셀 수).  전부 NaN 인 행은 떨어낸다.
    """
    if df.empty:
        return df, 0
    idx = pd.Index(list(index), name="timestamp")
    out = df.reindex(idx)
    out.index = pd.to_datetime(out.index)
    # 숫자 컬럼만 보간한다.  src_met_* 같은 문자열 출처 마스크는 보간 대상이 아니고
    # (없는 시각의 출처를 지어낼 수 없다), object dtype 보간은 pandas 가 경고한다.
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    before = int(out[num_cols].notna().sum().sum())
    out[num_cols] = out[num_cols].interpolate(
        method="time", limit=limit, limit_area="inside")
    filled = int(out[num_cols].notna().sum().sum()) - before
    out.index = idx
    out = out.dropna(how="all")
    if filled:
        print(f"  postprocess.fill_short_gaps: 결손 보간 {filled}셀 (연속 {limit}개까지)")
    return out, filled


# holidays 패키지 인스턴스는 전 모듈에서 한 번만 (반복 호출 시 캐시 효과).
_KR_HOLIDAYS = holidays.SouthKorea()


def add_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp 인덱스 기준 'day_type' 컬럼 추가.

    값: 'weekday' / 'weekend' / 'holiday' (셋 중 하나, 우선순위 holiday > weekend).
    인덱스는 'YYYY-MM-DD HH:MM:SS' 문자열 또는 DatetimeIndex 둘 다 지원.
    """
    if df.empty or len(df.index) == 0:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        dates = [d.date() for d in out.index]
    else:
        dates = [d.date() for d in pd.to_datetime(out.index)]

    # 하루에 24 행이 있으므로 날짜별 캐시로 holidays 조회를 24배 절약.
    cache: dict = {}
    def _classify(d) -> str:
        if d not in cache:
            if d in _KR_HOLIDAYS:
                cache[d] = "holiday"
            elif d.weekday() >= 5:
                cache[d] = "weekend"
            else:
                cache[d] = "weekday"
        return cache[d]

    out["day_type"] = [_classify(d) for d in dates]
    counts = pd.Series(out["day_type"]).value_counts().to_dict()
    print(f"  postprocess.add_day_type: {counts}")
    return out
