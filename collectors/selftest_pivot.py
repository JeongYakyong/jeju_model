"""selftest_pivot.py -- long → wide 피벗 불변조건 검사 (네트워크 0회).

    python collectors/selftest_pivot.py

왜 있나
-------
`_pivot_point` / `_derive_point` 는 KIMR·KIMG 두 모델이 공유하는 **유일한** 피벗이다
(2026-07-21 통합 — 그전엔 kimr_one_point / kimg_one_point 가 1~3단계 복붙이었다).
한 곳을 고치면 두 모델이 같이 움직이므로, 조용히 깨지기 쉬운 지점들을 못박아 둔다.

여기서 잡는 것 (전부 합성 데이터, API 호출 없음)
  ① 컬럼 순서   — 스펙 순서 = 출력 순서.  wind 블록이 중간에 낀다.
                  (통합 때 wind 를 끝으로 몰았다가 실제로 이 검사에 걸렸다.)
  ② 반올림 자리 — reh 가 KIMR 4자리 / KIMG 2자리로 **다르다**.  모델별 스펙의 핵심.
  ③ 단위 변환   — KIMR TEMP 는 K→°C, KIMG TEMP_C 는 변환 없음.
  ④ freshest-wins — 같은 (시각, 카테고리)에 두 발표가 있으면 최신 base 값.
  ⑤ 강수 누적 diff — base 단위 누적을 시간차로 풀고 음수는 0 클립.
  ⑥ 경계 입력   — 빈 DF / 윈도우 밖 / 강수 없음에서 죽지 않는다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CORE = Path(__file__).resolve().parent
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import pivot as aj

KST = ZoneInfo("Asia/Seoul")
LONG_COLS = ["base_datetime", "point_name", "fcst_datetime", "category", "fcst_value"]
PT = "West(Gosan)"
W0 = datetime(2026, 7, 3, 0, 0, tzinfo=KST)
W1 = datetime(2026, 7, 5, 0, 0, tzinfo=KST)

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        _failures.append(label)
        print(f"  FAIL {label}  {detail}")


def long_rows(pairs, base="2026-07-02 21:00", point=PT, start="2026-07-03 00:00", n=3):
    """pairs = {category: value or [values...]} -> long DF (n 시각)."""
    rows = []
    for k in range(n):
        f = (pd.Timestamp(start) + pd.Timedelta(hours=k)).strftime("%Y-%m-%d %H:%M")
        for cat, val in pairs.items():
            v = val[k] if isinstance(val, (list, tuple)) else val
            rows.append((base, point, f, cat, float(v)))
    return pd.DataFrame(rows, columns=LONG_COLS)


# ① 컬럼 순서 — wind 가 temp 뒤, gust/cape 앞에 온다 (KIMR)
full_kimr = {c: 1.0 for c in (
    "TEMP", "TEMP_SKIN", "GUST", "CAPE", "CINN", "HPBL", "TCOG", "TCOH", "REH",
    "WIND_U_10M", "WIND_V_10M", "WIND_U_80M", "WIND_V_80M")}
out = aj.kimr_one_point(long_rows(full_kimr), PT, "west", W0, W1)
expected = ["temp_west", "temp_skin_west",
            "wind_spd_10m_west", "wd_sin_10m_west", "wd_cos_10m_west",
            "wind_spd_80m_west", "wd_sin_80m_west", "wd_cos_80m_west",
            "gust_west", "cape_west", "cinn_west", "hpbl_west",
            "tcog_west", "tcoh_west", "reh_west"]
check("① KIMR 컬럼 순서 (wind 가 temp 와 gust 사이)",
      list(out.columns) == expected, f"got {list(out.columns)}")

full_kimg = {c: 1.0 for c in (
    "TEMP_C", "GUST", "REH", "TCLD", "MIDLOW_CLOUD",
    "WIND_U_10M", "WIND_V_10M", "WIND_U_80M", "WIND_V_80M")}
out_g = aj.kimg_one_point(long_rows(full_kimg), PT, "west", W0, W1)
expected_g = ["temp_west",
              "wind_spd_10m_west", "wd_sin_10m_west", "wd_cos_10m_west",
              "wind_spd_80m_west", "wd_sin_80m_west", "wd_cos_80m_west",
              "gust_west", "reh_west", "total_cloud_west", "midlow_cloud_west"]
check("① KIMG 컬럼 순서", list(out_g.columns) == expected_g, f"got {list(out_g.columns)}")

# ② 반올림 — reh 는 KIMR 4자리 / KIMG 2자리 (모델별로 다르다)
r = aj.kimr_one_point(long_rows({"REH": 55.123456}), PT, "west", W0, W1)
g = aj.kimg_one_point(long_rows({"REH": 55.123456}), PT, "west", W0, W1)
check("② reh 반올림 KIMR=4자리", r["reh_west"].iloc[0] == 55.1235, f"got {r['reh_west'].iloc[0]}")
check("② reh 반올림 KIMG=2자리", g["reh_west"].iloc[0] == 55.12, f"got {g['reh_west'].iloc[0]}")

# ③ 단위 — KIMR TEMP 는 K→°C, KIMG TEMP_C 는 그대로
r = aj.kimr_one_point(long_rows({"TEMP": 300.0}), PT, "west", W0, W1)
g = aj.kimg_one_point(long_rows({"TEMP_C": 26.85}), PT, "west", W0, W1)
check("③ KIMR TEMP K→°C", r["temp_west"].iloc[0] == 26.85, f"got {r['temp_west'].iloc[0]}")
check("③ KIMG TEMP_C 변환 없음", g["temp_west"].iloc[0] == 26.85, f"got {g['temp_west'].iloc[0]}")

# ④ freshest-wins — 같은 시각에 두 발표가 있으면 최신 base
old_b = long_rows({"TEMP": 300.0}, base="2026-07-02 09:00")
new_b = long_rows({"TEMP": 310.0}, base="2026-07-02 21:00")
mixed = pd.concat([new_b, old_b], ignore_index=True)   # 일부러 역순 배치
r = aj.kimr_one_point(mixed, PT, "west", W0, W1)
check("④ freshest-wins (최신 base 채택)",
      bool((r["temp_west"] == 36.85).all()), f"got {r['temp_west'].tolist()}")

# ⑤ 강수 누적 → 시간차 diff, 음수는 0 클립
#    누적 [0, 2.0, 1.5] -> diff [NaN, 2.0, -0.5→0.0]; 첫 행은 drop 되어 2시각만 남는다.
rain = long_rows({"RAIN_CONV": [0.0, 2.0, 1.5], "RAIN_STRAT": [0.0, 0.0, 0.0]})
r = aj.kimr_one_point(rain, PT, "west", W0, W1)
check("⑤ 강수 누적→시간차 diff + 음수 0 클립",
      r["rainfall_west"].tolist() == [2.0, 0.0], f"got {r['rainfall_west'].tolist()}")

# ⑥ 경계 입력 — 죽지 않고 빈 DF
empty = pd.DataFrame(columns=LONG_COLS)
check("⑥ 빈 입력 → 빈 DF", aj.kimr_one_point(empty, PT, "west", W0, W1).empty)
far = long_rows({"TEMP": 300.0}, start="2026-08-01 00:00")
check("⑥ 윈도우 밖 입력 → 빈 DF", aj.kimr_one_point(far, PT, "west", W0, W1).empty)
check("⑥ 없는 지점 → 빈 DF",
      aj.kimr_one_point(long_rows({"TEMP": 300.0}), "NoSuchPoint", "west", W0, W1).empty)
norain = aj.kimr_one_point(long_rows({"TEMP": 300.0}), PT, "west", W0, W1)
check("⑥ 강수 없는 입력 → rainfall 컬럼 없음", "rainfall_west" not in norain.columns)

# ⑦ 소스 마스크 — combine_first 로 어느 모델이 이겼는지 src_met_* 가 정확히 기록하나
import collect_forecast as cf

PTS = {"West(Gosan)": "west", "East(Seongsan)": "east", "solar_farm(south)": "south"}


def multi_point(pairs, hours, start="2026-07-03 00:00", base="2026-07-02 21:00"):
    out = []
    for pt in PTS:
        for k in hours:
            f = (pd.Timestamp(start) + pd.Timedelta(hours=k)).strftime("%Y-%m-%d %H:%M")
            for cat, val in pairs.items():
                out.append((base, pt, f, cat, float(val)))
    return pd.DataFrame(out, columns=LONG_COLS)


# KIMR 은 0~23시만 (lead 한계 흉내), KIMG 는 0~47시 전부 → 앞은 KIMR, 뒤는 KIMG 여야 한다
_kimr = multi_point({"TEMP": 300.0, "REH": 80.0}, range(0, 24))
_kimg = multi_point({"TEMP_C": 26.0, "REH": 80.0, "SOLAR_RAD": 1.0,
                     "TCLD": 0.5, "MIDLOW_CLOUD": 0.3}, range(0, 48))
_w = cf.build_wide(_kimr, _kimg, W0, datetime(2026, 7, 5, 0, 0, tzinfo=KST))
_s = _w["src_met_west"]
check("⑦ src_met: KIMR 구간 표기", _s.iloc[:24].unique().tolist() == ["KIMR"],
      f"got {_s.iloc[:24].unique().tolist()}")
check("⑦ src_met: KIMG 구간 표기", _s.iloc[24:].unique().tolist() == ["KIMG"],
      f"got {_s.iloc[24:].unique().tolist()}")
check("⑦ src_met 과 실제 값 출처 일치 (KIMR 300K→26.85 / KIMG 26.0)",
      _w["temp_west"].iloc[0] == 26.85 and _w["temp_west"].iloc[30] == 26.0,
      f"got {_w['temp_west'].iloc[0]} / {_w['temp_west'].iloc[30]}")
check("⑦ 일사는 마스크 대상 아님 (항상 KIMG, 전 구간 존재)",
      int(_w["radiation_west"].isna().sum()) == 0)
check("⑦ 3지점 모두 마스크 생성",
      all(f"src_met_{s}" in _w.columns for s in PTS.values()))

print()
if _failures:
    print(f"[selftest_pivot] 실패 {len(_failures)}건: {', '.join(_failures)}")
    sys.exit(1)
print("[selftest_pivot] 전부 통과")
