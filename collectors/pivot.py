"""pivot.py -- long -> wide 피벗 (KIMR·KIMG 공용 1벌).

수집 계층의 정식 중간 계약인 long 5컬럼
(`base_datetime / point_name / fcst_datetime / category / fcst_value`)을
지점별 wide 컬럼으로 접는다.  **모델과 무관한 단계**(freshest 선택·강수 누적 diff·
윈도우 트림)는 `_pivot_point` 한 벌이고, 모델 차이는 `_SPEC_KIMR`/`_SPEC_KIMG`
스펙표 두 개에만 있다.

이 파일을 고치면 반드시 `python collectors/selftest_pivot.py` 를 돌릴 것
(네트워크 0회, 17항목 회귀망 -- 컬럼 순서·반올림 관례까지 잡는다).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from kma_kimg import freshest

# ════════════════════════════════════════════════════════════════════════
# 제주 long->wide 파생 (구 collect_input).  collect_data_jeju 가 이 함수들로
# KIMR long DF 와 KIMG SOLAR_RAD 를 지점별 wide 컬럼으로 후처리한다.
# ════════════════════════════════════════════════════════════════════════

# DB 의 point_name → wide 컬럼 suffix.
POINT_SUFFIX: dict[str, str] = {
    "West(Gosan)":       "west",
    "East(Seongsan)":    "east",
    "solar_farm(south)": "south",
}

# day-ahead 발표 선택 base 시각 (KST): KIMR 00 UTC(09 KST) / KIMG 18 UTC(03 KST).
DAY_AHEAD_BASE_HOUR_KST: dict[str, int] = {
    "kimr": 9,
    "kimg": 3,
}


def restrict_to_day_ahead_issue(df: pd.DataFrame, base_hour_kst: int) -> pd.DataFrame:
    """base_datetime 의 KST 시(hour)가 base_hour_kst 인 발표만 남긴다 (day-ahead 선택)."""
    if df.empty:
        return df
    base_hours = pd.to_datetime(
        df["base_datetime"], format="%Y-%m-%d %H:%M"
    ).dt.hour
    return df[base_hours == base_hour_kst]


# ── long → wide 피벗 (KIMR·KIMG 공용 1벌) ────────────────────────────────────
# long 5컬럼(base_datetime/point_name/fcst_datetime/category/fcst_value)이 두 모델의
# 정식 중간 계약이라, 피벗·강수 누적diff·윈도우 트림은 모델과 무관하게 같다.
# 모델별로 다른 건 **마지막 컬럼 파생 스펙뿐** — 아래 표 두 개가 그 차이의 전부다.
# (2026-07-21 통합: 구 kimr_one_point / kimg_one_point 는 1~3단계가 복붙이었다.)
#
# 스펙 항목 = (long 카테고리, 출력 컬럼 접두, K→°C 변환 여부, 반올림 자리)
#           또는 센티널 _WIND — U/V 성분에서 풍속·풍향(sin/cos)을 푸는 블록의 자리.
# ★순서가 곧 출력 컬럼 순서다.  _WIND 위치까지 기존 동작 그대로 보존할 것
#   (2026-07-21 통합 시 wind 를 끝으로 몰았다가 컬럼 순서가 바뀌어 회귀 검출됨).
_WIND = "__WIND__"

_SPEC_KIMR = [
    ("TEMP",      "temp",       True,  2),
    ("TEMP_SKIN", "temp_skin",  True,  2),
    _WIND,
    ("GUST",      "gust",       False, 4),
    ("CAPE",      "cape",       False, 4),
    ("CINN",      "cinn",       False, 4),
    ("HPBL",      "hpbl",       False, 4),
    ("TCOG",      "tcog",       False, 4),
    ("TCOH",      "tcoh",       False, 4),
    ("REH",       "reh",        False, 4),
]
# KIMG 는 저장 시점에 이미 °C 로 변환돼 있어 TEMP_C 를 그대로 쓴다.
# ★reh 반올림이 KIMR(4자리)과 다르다(2자리) — 기존 동작 그대로 보존.
_SPEC_KIMG = [
    ("TEMP_C",        "temp",          False, 2),
    _WIND,
    ("GUST",          "gust",          False, 4),
    ("REH",           "reh",           False, 2),
    ("TCLD",          "total_cloud",   False, 4),
    ("MIDLOW_CLOUD",  "midlow_cloud",  False, 4),
    # ── 2026-08-04 확장 (KIMG 22변수).  **뒤에만 붙인다** — 앞을 건드리면 기존 컬럼
    #    순서가 바뀌어 selftest ① 이 잡는다.  KIMR 에도 있는 것(temp_skin/hpbl)은
    #    **같은 컬럼명**을 쓴다: 소스 비교가 join 한 번으로 되게 하려는 것이다.
    ("TEMP_SKIN",     "temp_skin",     False, 2),   # KIMG 는 이미 °C (KIMR 은 K→°C)
    ("LOW_CLOUD",     "low_cloud",     False, 4),
    ("MID_CLOUD",     "mid_cloud",     False, 4),
    ("HIGH_CLOUD",    "high_cloud",    False, 4),
    ("LW_DOWN",       "lwdown",        False, 2),   # W/m^2 (radiation_* 는 MJ/m^2/h)
    ("HEAT_SENS",     "heat_sens",     False, 2),   # W/m^2
    ("HEAT_LAT",      "heat_lat",      False, 2),   # W/m^2
    ("DEWPOINT",      "dewpoint",      False, 2),   # °C
    ("SHUM",          "shum",          False, 6),   # kg/kg
    ("USTAR",         "ustar",         False, 4),   # m/s
    ("HPBL",          "hpbl",          False, 4),   # m -- KIMR HPBL 과 같은 컬럼
]


def _pivot_point(
    df_long: pd.DataFrame, point: str,
    window_start: datetime, window_end: datetime,
    day_ahead_hour: int | None = None,
) -> pd.DataFrame:
    """단일 point 의 long → 카테고리 컬럼 wide (index=fcst_datetime 문자열).

    ① 비강수: freshest per (fcst_datetime, category) → pivot
    ② 강수  : per-base 누적 → 시간차 diff → freshest of the diff (RAIN_HOURLY)
    ③ 윈도우 트림 (누적 diff 용 lookback 여분 제거)
    day_ahead_hour 가 주어지면 그 시각 발표만 사용(globally-freshest 대신).
    """
    sub = df_long[df_long["point_name"] == point]
    if sub.empty:
        return pd.DataFrame()
    if day_ahead_hour is not None:
        sub = restrict_to_day_ahead_issue(sub, day_ahead_hour)
        if sub.empty:
            return pd.DataFrame()

    non_rain = sub[~sub["category"].isin(["RAIN_CONV", "RAIN_STRAT"])]
    non_rain = freshest(non_rain, ["fcst_datetime", "category"])
    wide = non_rain.pivot(index="fcst_datetime", columns="category", values="fcst_value")

    rain = sub[sub["category"].isin(["RAIN_CONV", "RAIN_STRAT"])]
    if not rain.empty:
        acc = (
            rain.groupby(["base_datetime", "fcst_datetime"], as_index=False)["fcst_value"]
                .sum()
                .rename(columns={"fcst_value": "acc"})
                .sort_values(["base_datetime", "fcst_datetime"])
        )
        # base 단위로 diff (첫 행은 NaN → drop → freshest 가 직전 base 값 선택).
        acc["hourly"] = (
            acc.groupby("base_datetime")["acc"].diff().clip(lower=0).round(2)
        )
        acc = acc.dropna(subset=["hourly"])
        acc = freshest(acc, ["fcst_datetime"])
        rain_s = acc.set_index("fcst_datetime")["hourly"].rename("RAIN_HOURLY")
        wide = wide.join(rain_s, how="outer")

    start_s = window_start.strftime("%Y-%m-%d %H:%M")
    end_s = window_end.strftime("%Y-%m-%d %H:%M")
    return wide.loc[(wide.index >= start_s) & (wide.index < end_s)]


def _derive_point(wide: pd.DataFrame, suffix: str, spec: list) -> pd.DataFrame:
    """카테고리 wide → 최종 컬럼명·단위 (지점 접미사 부착).  스펙 순서 = 출력 순서."""
    out = pd.DataFrame(index=wide.index)
    for item in spec:
        if item is _WIND:
            # U/V → 풍속 + 풍향 sin/cos.  두 모델 공통 변환식.
            for height in ("10m", "80m"):
                u_col = f"WIND_U_{height.upper()}"
                v_col = f"WIND_V_{height.upper()}"
                if u_col in wide and v_col in wide:
                    u, v = wide[u_col], wide[v_col]
                    spd = np.sqrt(u**2 + v**2)
                    wdir = (270 - np.degrees(np.arctan2(v, u))) % 360
                    out[f"wind_spd_{height}_{suffix}"] = spd.round(2)
                    out[f"wd_sin_{height}_{suffix}"]   = np.sin(np.radians(wdir)).round(4)
                    out[f"wd_cos_{height}_{suffix}"]   = np.cos(np.radians(wdir)).round(4)
            continue
        raw, name, to_celsius, ndigits = item
        if raw not in wide:
            continue
        col = wide[raw] - 273.15 if to_celsius else wide[raw]
        out[f"{name}_{suffix}"] = col.round(ndigits)

    if "RAIN_HOURLY" in wide:
        out[f"rainfall_{suffix}"] = wide["RAIN_HOURLY"]
    return out


def kimr_one_point(
    df_long: pd.DataFrame, point: str, suffix: str,
    window_start: datetime, window_end: datetime,
    day_ahead: bool = False,
) -> pd.DataFrame:
    """단일 point 의 KIMR long → 후처리된 wide (index=fcst_datetime 문자열).

    day_ahead=True 면 00 UTC(09 KST) 발표만 사용.  False(기본) 면 globally-freshest.
    """
    wide = _pivot_point(
        df_long, point, window_start, window_end,
        day_ahead_hour=DAY_AHEAD_BASE_HOUR_KST["kimr"] if day_ahead else None,
    )
    if wide.empty:
        return wide
    return _derive_point(wide, suffix, _SPEC_KIMR)


def kimg_one_point(
    df_long: pd.DataFrame, point: str, suffix: str,
    window_start: datetime, window_end: datetime,
) -> pd.DataFrame:
    """단일 point 의 KIMG long → kimr_one_point 와 동일한 컬럼 스키마의 wide.

    KIMR lead 한계(120h=D+5) 이후 장지평 구간(D+6~)을 KIMG 로 잇기 위한 함수
    (2026-06-13).  호출자(collect_data_jeju.build_wide)가 KIMR 우선 combine_first
    로 합친다.  KIMR 전용 변수(temp_skin/cape/cinn/hpbl/tcog/tcoh)는 _SPEC_KIMG 에
    없어 KIMG-only 구간에서 NaN 으로 남는다.  구름(total_cloud/midlow_cloud)은
    KIMR 단일면에 없어 KIMG 단독 소스다(수요 모델의 구름 4 feature 가 이걸 읽는다).

    주의: hf>135(kma_kimg.KIMG_HOURLY_MAX_HF) 구간은 3h 간격만 존재하므로 rainfall
    diff 가 그 구간에서 시간당이 아니라 3시간 누적값이 된다 (저장은 원본 그대로 정책).
    """
    wide = _pivot_point(df_long, point, window_start, window_end)
    if wide.empty:
        return wide
    return _derive_point(wide, suffix, _SPEC_KIMG)


def kimg_solar(
    df_long: pd.DataFrame, suffix: str = "south", day_ahead: bool = False,
) -> pd.Series:
    """KIMG SOLAR_RAD freshest → radiation_{suffix} (저장 단위 그대로, MJ/m^2/h).

    일사는 KIMR 에 없어 KIMG 단독 소스다.  호출자(build_wide)가 POINT_SUFFIX 의 각
    지점(west/east/south)에 대해 그 지점 SOLAR_RAD 만 필터해 넘기면 지점별
    radiation_<suffix> 컬럼을 만든다.  태양광 모델은 radiation_west + radiation_south
    를 입력으로 쓴다 (serve_solarwind_lgbm.FORE_MAP).
    """
    name = f"radiation_{suffix}"
    if df_long.empty:
        return pd.Series(dtype="float64", name=name)
    if day_ahead:
        df_long = restrict_to_day_ahead_issue(df_long, DAY_AHEAD_BASE_HOUR_KST["kimg"])
        if df_long.empty:
            return pd.Series(dtype="float64", name=name)
    fresh = freshest(df_long, ["fcst_datetime"])
    s = fresh.set_index("fcst_datetime")["fcst_value"]
    return s.round(2).rename(name)
