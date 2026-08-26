# -*- coding: utf-8 -*-
"""기상개황 — 제주 3구역(서부/동부/남부) 단색(초록) 지도, Leaflet HTML 임베드.

(원본: forecastmodel/08_streamlit/weather_map.py 의 8권역판을 제주 3구역으로 개조)

- 구역 = data/refdata/jeju_zones_3.json — **읍면동(행정동) 43개**를 사용자 확정 명단으로
  west/east/south 에 배정해 **구역별로 병합(dissolve)** 한 것 (tools/make_jeju_zones.py).
  구역 안 읍면동 경계선은 없고 **구역 사이 경계선만** 그려진다 (사용자 피드백 2026-07-17).
- 표시 모드(신재생 강도/일사/풍속)와 발전원 토글은 HTML 내부 JS — 실데이터는 렌더 시 주입.
- 기준 시간 = 09–15시 평균(일사·기온·풍속·강수·운량). 별도 시각 선택 없음.
- 좌측 패널 = 기온·습도 통계 + 섬 전체 태양광·풍력 이용률 카드(est_horizon_jeju).
- 우측 패널 = 순 부하 스파크라인(예측+실측 겹침)·최대/최소·SMP 최저·음수가격 경보 시간수.
- 동부(성산)는 일사계가 없다 — 예보(radiation_east)는 있으므로 예보 모드는 정상 표시,
  실측 모드는 '관측 없음' 폴백.
- 라이트/다크 테마 겸용 — build_html 이 활성 테마(common.theme_type)에 맞춰 타일(CARTO
  light_all/dark_all)·패널 토큰을 주입한다.
"""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다
from pages import common as C

ZONES_GEOJSON = Path(P.REF_JEJU_ZONES)

# 3구역 — 표시명·관측소·KIM 풍력 격자·일사계 유무·라벨 좌표(지도 위 이모지+이름 위치)
ZONES = {
    "west":  dict(name="서부", stname="고산", stn_id=185, kim_grid="X530/Y251",
                  rad_obs=True,  lat=33.32, lon=126.26),
    "east":  dict(name="동부", stname="성산", stn_id=188, kim_grid="X553/Y254",
                  rad_obs=False, lat=33.43, lon=126.80),
    "south": dict(name="남부", stname="서귀포", stn_id=189, kim_grid=None,
                  rad_obs=True,  lat=33.31, lon=126.57),
}
SUFFIXES = list(ZONES)            # DB 컬럼 접미사 = zone 키와 동일 (west/east/south)

HOURS = ("09:00:00", "15:00:00")   # 기준 시간대 — 09–15시 평균

# ---- 활성도 BIN — 원본(전국 실측 역산 교정값) 재사용: 구역 간 상대 신호로만 쓴다 ----
SOLAR_BINS = [(0.80, 61, "매우 좋음"), (0.60, 49, "좋음"), (0.40, 36, "보통"),
              (0.20, 23, "낮음"), (0.00, 12, "매우 낮음")]
WIND_BINS = [
    (0, 4,    0,   "무풍-이용불가"),
    (4, 7,   30,   "낮음"),
    (7, 10,  50,   "양호"),
    (10, 15, 70,   "좋음"),
    (15, 25, 100,  "최적"),
    (25, 999,  0,  "강풍-이용불가")
]
SA_MAX = max(p for _, p, _ in SOLAR_BINS)
WA_MAX = max(p for _, _, p, _ in WIND_BINS)

# 청천 일사(09–15시 평균, MJ/m²·h) — 월별 관측 P97 (원본 historical 산출값 재사용)
CLEARSKY_0915 = {1: 1.64, 2: 2.09, 3: 2.50, 4: 2.93, 5: 3.10, 6: 3.06,
                 7: 2.79, 8: 2.69, 9: 2.59, 10: 2.20, 11: 1.83, 12: 1.47}

RAIN_MMH = 0.3            # 09–15 평균 강수 ≥ 이 값이면 강수로 판정
CLOUD_OVC, CLOUD_BKN = 0.85, 0.50   # 흐림 / 약간흐림 경계
SNOW_TEMP = 1.0           # 강수 시 기온 < 이 값이면 눈
WIND_FULL = 6.0           # 풍속 모드 정규화 상한(ASOS 스케일, m/s)


def solar_act(ratio: float | None, rainy: bool) -> dict | None:
    if ratio is None or pd.isna(ratio):
        return None
    if rainy:
        return {"pct": 12, "lab": "강수"}
    for mn, pct, lab in SOLAR_BINS:
        if ratio >= mn:
            return {"pct": pct, "lab": lab}
    return {"pct": 12, "lab": "매우 낮음"}


def wind_act(ws: float | None) -> dict | None:
    if ws is None or pd.isna(ws):
        return None
    for lo, hi, pct, lab in WIND_BINS:
        if lo <= ws < hi:
            return {"pct": pct, "lab": lab}
    return {"pct": 0, "lab": "차단"}


def sky_of(cloud: float | None, rain: float | None, temp: float | None,
           ratio: float | None) -> dict:
    """하늘상태 4분류 — 강수 우선, 운량 기준. 운량 결측 시 일사 비율로 근사(폴백)."""
    rain = 0.0 if rain is None or pd.isna(rain) else rain
    if rain >= RAIN_MMH:
        if temp is not None and not pd.isna(temp) and temp < SNOW_TEMP:
            return {"emo": "🌨️", "t": "눈"}
        return {"emo": "🌧️", "t": "비"}
    if cloud is not None and not pd.isna(cloud):
        if cloud >= CLOUD_OVC:
            return {"emo": "☁️", "t": "흐림"}
        if cloud >= CLOUD_BKN:
            return {"emo": "⛅", "t": "약간흐림"}
        return {"emo": "☀️", "t": "맑음"}
    if ratio is not None and not pd.isna(ratio):    # 운량 결측 — 일사로 근사
        if ratio >= 0.65:
            return {"emo": "☀️", "t": "맑음"}
        if ratio >= 0.40:
            return {"emo": "⛅", "t": "약간흐림"}
        return {"emo": "☁️", "t": "흐림"}
    return {"emo": "", "t": "—"}


# ---------------------------------------------------------------- 데이터 레이어
# 테이블별 컬럼 접두사 — forecast_horizon(KMA 예보)와 historical(관측)은 이름만 다름.
_PREFIX = {
    "forecast":   {"temp": "temp_", "rad": "radiation_", "wind": "wind_spd_10m_",
                   "rain": "rainfall_", "cloud": "total_cloud_"},
    "historical": {"temp": "temp_c_", "rad": "solar_rad_", "wind": "wind_spd_",
                   "rain": "rainfall_", "cloud": "total_cloud_"},
}


def _station_means_fh(suffixes: list[str], date: str) -> dict[str, dict]:
    """forecast_horizon(지평 아카이브) 09–15시 평균 — 예보 전용.

    timestamp 당 여러 base 가 있으므로 최신 base 행만 골라 가장 새 예보를 쓴다.
    """
    px = _PREFIX["forecast"]
    s, e = f"{date} {HOURS[0]}", f"{date} {HOURS[1]}"
    cols = [f"{p}{sx}" for sx in suffixes for p in px.values()]
    sel = ", ".join(f'"{c}"' for c in cols)
    df = C.query("jeju",
                 f"SELECT {sel} FROM forecast_horizon fh WHERE timestamp BETWEEN ? AND ? "
                 f"AND base=(SELECT MAX(base) FROM forecast_horizon WHERE timestamp=fh.timestamp)",
                 (s, e))
    out = {}
    for sx in suffixes:
        if df.empty:
            out[sx] = {k: float("nan") for k in px}
            continue
        m = df.mean(numeric_only=True)
        out[sx] = {k: m.get(f"{p}{sx}") for k, p in px.items()}
    return out


def _station_means_hist(suffixes: list[str], date: str) -> dict[str, dict]:
    """historical(관측) 09–15시 평균 — 동부(성산)는 solar_rad_east 컬럼 자체가 없어
    존재하는 컬럼만 SELECT 하고 없는 항목은 NaN 으로 채운다('관측 없음' 폴백)."""
    px = _PREFIX["historical"]
    have = set(C.table_columns("jeju", "historical"))
    s, e = f"{date} {HOURS[0]}", f"{date} {HOURS[1]}"
    cols = [f"{p}{sx}" for sx in suffixes for p in px.values() if f"{p}{sx}" in have]
    if not cols:
        return {sx: {k: float("nan") for k in px} for sx in suffixes}
    df = C.query("jeju", f"SELECT {', '.join(cols)} FROM historical "
                         "WHERE timestamp BETWEEN ? AND ?", (s, e))
    out = {}
    for sx in suffixes:
        if df.empty:
            out[sx] = {k: float("nan") for k in px}
            continue
        m = df.mean(numeric_only=True)
        out[sx] = {k: (m.get(f"{p}{sx}") if f"{p}{sx}" in have else float("nan"))
                   for k, p in px.items()}
    return out


def _util_label(pct: float) -> str:
    """이용률 %를 정성 라벨로 (제주 분포 기준)."""
    if pct >= 45:
        return "매우 강함"
    if pct >= 28:
        return "강함"
    if pct >= 15:
        return "보통"
    if pct >= 7:
        return "약함"
    return "미약"


@st.cache_data(ttl=C.CACHE_TTL)
def jeju_util(date: str, forecast: bool) -> dict:
    """섬 전체 태양광·풍력 이용률(%) — 좌측 패널 카드용.

    forecast=True : est_horizon_jeju 최신 base — 평균(태양광 09–15시·풍력 24시간) + 시간별 최대.
    forecast=False: historical 실측 — real_*_utilization_jeju 같은 집계.
    없으면 값 None (카드 '—').
    """
    none = {"solar": None, "solar_max": None, "wind": None, "wind_max": None}
    s, e = f"{date} 00:00:00", f"{date} 23:00:00"
    if forecast:
        df = C.query("jeju",
                     "SELECT timestamp, est_solar_util_jeju su, est_wind_util_jeju wu "
                     "FROM est_horizon_jeju eh WHERE timestamp BETWEEN ? AND ? "
                     "AND base=(SELECT MAX(base) FROM est_horizon_jeju "
                     "WHERE timestamp=eh.timestamp AND est_solar_util_jeju IS NOT NULL)", (s, e))
    else:
        df = C.query("jeju",
                     "SELECT timestamp, real_solar_utilization_jeju su, "
                     "real_wind_utilization_jeju wu FROM historical "
                     "WHERE timestamp BETWEEN ? AND ?", (s, e))
    if df.empty or df["su"].isna().all():
        return none
    h = df["timestamp"].dt.hour

    def pct(v):
        return None if pd.isna(v) else round(float(v) * 100, 1)

    return {"solar": pct(df.loc[h.between(9, 15), "su"].mean()),
            "solar_max": pct(df["su"].max()),
            "wind": pct(df["wu"].mean()),
            "wind_max": pct(df["wu"].max())}


@st.cache_data(ttl=C.CACHE_TTL)
def jeju_humidity(date: str) -> float | None:
    """제주 평균 습도(%) — forecast_horizon 09–15시 3지점 평균(최신 base)."""
    cols = ", ".join(f"reh_{s}" for s in SUFFIXES)
    s, e = f"{date} {HOURS[0]}", f"{date} {HOURS[1]}"
    df = C.query("jeju", f"SELECT {cols} FROM forecast_horizon fh WHERE timestamp BETWEEN ? AND ? "
                 "AND base=(SELECT MAX(base) FROM forecast_horizon WHERE timestamp=fh.timestamp)",
                 (s, e))
    if df.empty:
        return None
    m = df.mean(numeric_only=True).mean()
    return None if pd.isna(m) else float(m)


def _build_zones(date: str, table: str) -> dict[str, dict]:
    """3구역 기상(09–15시 평균)·하늘상태·활성도 — 예보/실측 공용 계산."""
    if table == "forecast":
        wx_all = _station_means_fh(SUFFIXES, date)
    else:
        wx_all = _station_means_hist(SUFFIXES, date)
    clear = CLEARSKY_0915[int(date[5:7])]

    zones = {}
    for key, z in ZONES.items():
        w = wx_all[key]
        ok = w["temp"] is not None and not pd.isna(w["temp"])
        ratio = None
        if w["rad"] is not None and not pd.isna(w["rad"]):
            ratio = float(min(1.0, max(0.0, w["rad"] / clear)))
        rain = w["rain"] if w["rain"] is not None and not pd.isna(w["rain"]) else 0.0
        rainy = ok and rain >= RAIN_MMH
        zones[key] = {
            "name": z["name"], "stname": z["stname"], "stn_id": z["stn_id"],
            "kim_grid": z["kim_grid"], "rad_obs": z["rad_obs"],
            "lat": z["lat"], "lon": z["lon"], "ok": bool(ok),
            "temp": None if not ok else round(float(w["temp"]), 1),
            "wind_ms": None if pd.isna(w["wind"]) else round(float(w["wind"]), 1),
            "rain": None if not ok else round(float(rain), 1),
            "ratio": None if ratio is None else round(ratio, 2),
            "sky": sky_of(w["cloud"], w["rain"], w["temp"], ratio) if ok
                   else {"emo": "", "t": "—"},
            "sa": solar_act(ratio, rainy),
            "wa": wind_act(w["wind"]),
        }
    return zones


@st.cache_data(ttl=C.CACHE_TTL)
def zone_day(date: str) -> dict[str, dict]:
    """선택일 3구역 — KMA 예보(forecast_horizon 최신 base) 기준."""
    return _build_zones(date, "forecast")


@st.cache_data(ttl=C.CACHE_TTL)
def zone_actual(date: str) -> dict[str, dict]:
    """과거 날짜 3구역 실측(관측) — 예보와 같은 계산·같은 bin. 동부 일사는 '관측 없음'."""
    return _build_zones(date, "historical")


@st.cache_data(ttl=C.CACHE_TTL)
def netload_panel_data(date: str) -> dict | None:
    """우측 패널 데이터 — 순 부하 예측 24h 스파크(+실측 겹침)·최대/최소·SMP 최저·경보 시간수."""
    day = pd.Timestamp(date)
    df = C.jeju_range_compare(day, day, use_live=False, mode="latest")
    nl_est = df["est_net_load_jeju"]
    if nl_est.isna().all():
        return None

    def _series(s):
        return [None if pd.isna(v) else round(float(v), 1) for v in s]

    smp = C.jeju_smp_frame(day, day, mode="latest")
    smp_min = None if smp["est_smp"].isna().all() else round(float(smp["est_smp"].min()), 1)
    danger_hours = int(smp["smp_danger"].fillna(0).sum())
    dem = df["est_demand_jeju"]
    return {
        "nl_spark": _series(nl_est),
        "nl_real_spark": _series(df["real_net_load_jeju"]),
        "nl_max": None if nl_est.isna().all() else round(float(nl_est.max())),
        "nl_min": None if nl_est.isna().all() else round(float(nl_est.min())),
        "demand_peak": None if dem.isna().all() else round(float(dem.max())),
        "demand_spark": _series(dem),
        "demand_real_spark": _series(df["real_demand_jeju"]),
        "smp_min": smp_min,
        "danger_hours": danger_hours,
    }


# ---------------------------------------------------------------- HTML 임베드
@st.cache_resource
def _geo_text() -> str:
    return ZONES_GEOJSON.read_text(encoding="utf-8")


_CONF = {"obs": ("과거 실측", "#16a34a"), "today": ("당일 예보", "#16a34a"),
         "high": ("신뢰도 높음", "#16a34a"), "med": ("신뢰도 보통", "#d97706"),
         "low": ("신뢰도 낮음 · 참고용", "#dc2626")}


def conf_of(dplus: int) -> tuple[str, str]:
    """달력 거리(선택일−오늘) 기준 신뢰도 라벨 — 발표 배지(_issue_badge)와 함께 표시된다."""
    key = ("obs" if dplus < 0 else "today" if dplus == 0 else
           "high" if dplus <= 3 else "med" if dplus <= 7 else "low")
    return _CONF[key]


def _issue_badge(date: str, dplus: int) -> str:
    """부제·툴팁용 발표 표기 — 화면 D+N 금지 방침(2026-07-17 용어 체계).

    예보 뷰는 그 날 09–15시 구간을 실제로 채운 최신 발표(base)를 '8/25 21시 발표 (어제)'
    처럼 실제 시각 + 선택일 기준 경과로 적고, 과거 뷰는 '실측 관측'.
    구 표기('전일 밤 발표')는 5일후를 봐도 똑같이 나와 언제 만든 예측인지 알 수 없었다.
    """
    if dplus < 0:
        return "실측 관측"
    df = C.query("jeju", "SELECT MAX(base) b FROM forecast_horizon "
                         "WHERE timestamp BETWEEN ? AND ?",
                 (f"{date} {HOURS[0]}", f"{date} {HOURS[1]}"))
    b = None if df.empty else df.loc[0, "b"]
    return "예보 없음" if (b is None or pd.isna(b)) else C.base_stamp(b, date)


# 단색 강도맵 상수 — 투명도 상한 40% → opacity ≤ 0.60
GREEN = "#059669"
OP_MIN, OP_MAX = 0.06, 0.60

# 테마 토큰 — build_html 이 활성 테마에 맞춰 템플릿에 주입 (좌우 패널·타일·경계선 색)
_MAP_THEMES = {
    "light": dict(TILES="light_all", MAPBG="#e8edf2", PANEL="#ffffff", INK="#0f172a",
                  SUB="#475569", LINE="#e2e8f0", WELL="#f8fafc", STROKE="#cbd5e1",
                  HOVER="#0f172a", NODATA="#94a3b8",
                  CHIP_BG="rgba(255,255,255,.85)", CHIP_INK="#0f172a", CHIP_SUB="#475569"),
    "dark":  dict(TILES="dark_all", MAPBG="#10151f", PANEL="#1b2130", INK="#e8edf5",
                  SUB="#94a3b8", LINE="#333c4f", WELL="#151b28", STROKE="#3a465c",
                  HOVER="#e8edf5", NODATA="#475569",
                  CHIP_BG="rgba(15,23,42,.82)", CHIP_INK="#f1f5f9", CHIP_SUB="#94a3b8"),
}

_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" />
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<style>
  :root{ --ink:__INK__; --sub:__SUB__; --line:__LINE__; --panel:__PANEL__; --well:__WELL__;
    --solar:#e11d48; --solar-soft:#fb7185; --wind:#1d4ed8; --wind-soft:#60a5fa; --green:#059669; }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;height:100%;width:100%;
    font-family:'Pretendard',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink);}
  #map{position:absolute;inset:0;}
  .leaflet-container{font-family:inherit;background:__MAPBG__;}

  .panel{position:fixed;top:12px;left:12px;z-index:1000;width:min(31%,410px);background:var(--panel);
    border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px rgba(15,23,42,.10);overflow:hidden;}
  .panel__head{padding:14px 18px 12px;border-bottom:1px solid var(--line);}
  .panel__eyebrow{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--sub);}
  .panel__title{font-size:20px;font-weight:800;margin:0;line-height:1.25;}
  .panel__title .unit{font-size:13px;font-weight:600;color:var(--sub);margin-left:2px;}
  .panel__sub{font-size:13px;font-weight:600;color:var(--sub);margin-top:4px;line-height:1.35;}
  .conf{font-size:12px;font-weight:600;}
  .panel__body{padding:13px 18px 15px;}

  .wxstats{display:flex;gap:9px;margin-top:11px;}
  .wxstat{flex:1;text-align:center;background:var(--well);border:1px solid var(--line);border-radius:11px;padding:8px 6px;}
  .wxstat__k{font-size:11px;font-weight:700;color:var(--sub);white-space:nowrap;}
  .wxstat__v{font-size:18px;font-weight:800;color:var(--ink);margin-top:2px;font-variant-numeric:tabular-nums;}

  .modes{display:flex;gap:5px;margin-bottom:4px;}
  .mchip{flex:1;text-align:center;padding:7px 4px;border-radius:9px;border:1px solid var(--line);
    background:var(--panel);cursor:pointer;font-size:12px;font-weight:700;transition:.12s;line-height:1.1;}
  .mchip:hover{border-color:var(--sub);}
  .mchip.active{background:var(--ink);border-color:var(--ink);color:__ACTIVE_TEXT__;}

  .toggle{display:flex;align-items:center;gap:9px;padding:7px 10px;border-radius:9px;cursor:pointer;user-select:none;transition:background .12s;}
  .toggle:hover{background:var(--well);}
  .toggle input{position:absolute;opacity:0;width:0;height:0;}
  .swatch{width:15px;height:15px;border-radius:5px;flex:none;border:2px solid __STROKE__;background:var(--panel);display:grid;place-items:center;transition:.12s;}
  .toggle[data-k="solar"] input:checked + .swatch{background:var(--solar);border-color:var(--solar);}
  .toggle[data-k="wind"]  input:checked + .swatch{background:var(--wind); border-color:var(--wind);}
  .swatch svg{opacity:0;width:9px;height:9px;}
  .toggle input:checked + .swatch svg{opacity:1;}
  .toggle .lab{font-size:13px;font-weight:600;flex:1;}
  .toggle .dot{width:10px;height:10px;border-radius:50%;}
  .toggle[data-k="solar"] .dot{background:var(--solar);}
  .toggle[data-k="wind"]  .dot{background:var(--wind);}

  .divider{height:1px;background:var(--line);margin:10px 0 10px;}

  .verdict{padding:14px 16px;border-radius:12px;background:var(--well);border:1px solid var(--line);}
  .verdict__top{display:flex;align-items:center;gap:8px;font-size:15px;font-weight:800;color:var(--ink);}
  .verdict__bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:11px 0 12px;background:var(--line);}
  .verdict__bar > span{display:block;height:100%;transition:width .3s;background:var(--green);}
  .verdict__msg{font-size:15px;line-height:1.75;color:var(--sub);}
  .verdict__msg b{font-weight:800;color:var(--ink);}

  .ucards{display:flex;gap:9px;margin-top:11px;}
  .ucard{flex:1;background:var(--well);border:1px solid var(--line);border-radius:12px;padding:10px 12px;}
  .ucard__k{font-size:12px;font-weight:700;color:var(--sub);white-space:nowrap;}
  .ucard__v{font-size:23px;font-weight:800;color:var(--ink);margin-top:3px;font-variant-numeric:tabular-nums;}
  .ucard__v small{font-size:13px;font-weight:600;color:var(--sub);margin-left:1px;}
  .ucard__s{font-size:11.5px;color:#94a3b8;font-weight:600;margin-top:2px;}

  .legend{margin-top:10px;font-size:11px;color:var(--sub);line-height:1.55;}
  .legend b{color:var(--ink);}

  .more{margin-top:11px;}
  .more > summary{cursor:pointer;list-style:none;font-size:12.5px;font-weight:700;color:var(--sub);
    padding:8px 11px;border:1px solid var(--line);border-radius:10px;background:var(--well);user-select:none;}
  .more > summary::-webkit-details-marker{display:none;}
  .more > summary:before{content:"⚙ ";opacity:.7;}
  .more > summary:hover{border-color:#cbd5e1;color:var(--ink);}
  .more[open] > summary{margin-bottom:9px;}
  .more .modes{margin-bottom:4px;}

  /* 우측 순 부하 패널 */
  .panel--netload{left:auto;right:12px;width:min(27%,330px);}
  .panel--netload .panel__title{font-size:27px;}
  .panel--netload .divider{margin:11px 0;}
  .nlrow{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:14px;margin:9px 0;}
  .nlrow .gk{display:flex;align-items:center;gap:8px;color:var(--sub);font-weight:600;}
  .nlrow .gk i{width:10px;height:10px;border-radius:50%;display:inline-block;}
  .nlrow .gv{font-weight:800;font-variant-numeric:tabular-nums;color:var(--ink);}
  .nlrow .gv small{font-weight:600;color:#94a3b8;margin-left:3px;font-size:11.5px;}
  .nlrow .gv.warn{color:#dc2626;}

  .dem__top{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:14px;}
  .dem__top .gk{color:var(--sub);font-weight:700;}
  .dem__top .gv{font-weight:800;color:var(--ink);font-variant-numeric:tabular-nums;}
  .dem__top .gv small{font-weight:600;color:#94a3b8;margin-left:3px;font-size:11.5px;}
  .sparklab{font-size:11.5px;font-weight:700;color:var(--sub);margin-top:9px;}
  .spark{display:block;width:100%;height:44px;margin-top:7px;}

  .wxwrap{background:none!important;border:none!important;}
  .wx{transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:1px;pointer-events:none;}
  .wx .emo{font-size:22px;line-height:1;filter:drop-shadow(0 1px 3px rgba(0,0,0,.3));}
  .wx .nm{font-size:11px;font-weight:800;color:__CHIP_INK__;background:__CHIP_BG__;
    padding:1px 6px;border-radius:7px;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,.14);}
  .wx .nm small{font-weight:600;color:__CHIP_SUB__;}

  .leaflet-tooltip.rt{background:#0f172a;border:none;border-radius:10px;box-shadow:0 6px 20px rgba(0,0,0,.25);padding:0;color:#fff;font-family:inherit;}
  .leaflet-tooltip.rt:before{display:none;}
  .tip{padding:11px 13px;min-width:200px;}
  .tip__name{font-size:14px;font-weight:800;}
  .tip__mem{font-size:10.5px;color:#94a3b8;margin:1px 0 7px;}
  .tip__wx{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:700;margin-bottom:7px;
    padding-bottom:7px;border-bottom:1px solid rgba(255,255,255,.14);}
  .tip__wx .ws{margin-left:auto;color:#94a3b8;font-weight:500;font-size:11px;}
  .tip__row{display:flex;align-items:center;justify-content:space-between;gap:14px;font-size:12px;margin:4px 0;}
  .tip__row .k{display:flex;align-items:center;gap:6px;color:#cbd5e1;}
  .tip__row .k i{width:8px;height:8px;border-radius:50%;display:inline-block;}
  .tip__row .v{font-weight:700;font-variant-numeric:tabular-nums;}
  .tip__row .v small{font-weight:500;color:#94a3b8;margin-left:3px;}
  .tip__act{margin-top:7px;padding-top:7px;border-top:1px solid rgba(255,255,255,.14);}
  .tip__act .lvl{font-size:11px;color:#94a3b8;}
  .mini{height:5px;border-radius:3px;background:rgba(255,255,255,.16);margin-top:3px;overflow:hidden;}
  .mini > span{display:block;height:100%;}
</style>
</head>
<body>
<div id="map"></div>

<div class="panel">
  <div class="panel__head">
    <h1 class="panel__title">__DATE_LABEL__ 제주 브리핑</h1>
    <div class="panel__sub">기상개황 및 순 부하 · __ISSUE_BADGE__<span class="conf" style="color:__CONF_C__"> · __CONF_T__</span></div>
  </div>
  <div class="panel__body">
    <div class="verdict">
      <div class="verdict__top"><span id="v-ico">☀️</span><span id="v-name">—</span></div>
      <div class="verdict__bar"><span id="v-strength"></span></div>
      <div class="verdict__msg" id="v-msg">—</div>
    </div>
    __WX_STATS__
    __UTIL_CARDS__
    <details class="more">
      <summary>표시 설정 · 일사 / 풍속 보기</summary>
      <div class="modes">
        <div class="mchip active" data-m="gen">신재생 강도</div>
        <div class="mchip" data-m="rad">일사</div>
        <div class="mchip" data-m="wind">풍속</div>
      </div>
      <div id="toggles">
        <label class="toggle" data-k="solar">
          <input type="checkbox" id="ck-solar" checked />
          <span class="swatch"><svg viewBox="0 0 10 10"><path d="M1 5l2.5 2.5L9 2" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
          <span class="lab">태양광 발전</span><span class="dot"></span>
        </label>
        <label class="toggle" data-k="wind">
          <input type="checkbox" id="ck-wind" checked />
          <span class="swatch"><svg viewBox="0 0 10 10"><path d="M1 5l2.5 2.5L9 2" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
          <span class="lab">풍력 발전</span><span class="dot"></span>
        </label>
      </div>
      <div class="legend" id="legend"></div>
    </details>
  </div>
</div>

__NETLOAD_PANEL__

<script>
const GEO = __GEO__;
const Z = __ZONES__;            /* 구역별 기상(09–15 평균)·활성도 — Python 주입 */
const META = __META__;          /* dplus·하늘상태 대표 */
const GREEN = "__GREEN__", OP_MIN = __OP_MIN__, OP_MAX = __OP_MAX__, WIND_FULL = __WIND_FULL__;
const SA_MAX = __SA_MAX__, WA_MAX = __WA_MAX__;   /* 교정 활성도 상한(최상 bin) */

const LEGEND = {
  gen: "<b>면 색</b> = 신재생 발전 강도 — 진할수록 강함 · 체크된 발전원만 반영",
  rad: "<b>면 색</b> = 일사 비율 — 진할수록 강함",
  wind:"<b>면 색</b> = 풍속 — 진할수록 강함",
};

const map = L.map("map", {zoomControl:false, scrollWheelZoom:false, zoomSnap:0.25, zoomDelta:0.5});
L.control.zoom({position:'bottomright'}).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/__TILES__/{z}/{x}/{y}{r}.png",{
  subdomains:"abcd", maxZoom:18, attribution:"&copy; OpenStreetMap &copy; CARTO"}).addTo(map);
const STROKE = "__STROKE__", HOVER = "__HOVER__", NODATA = "__NODATA__";

/* hover 카드가 지도 가장자리에서 잘리지 않게 — 구역별 '여는 방향' */
const TIP_DIR = {"west":"right", "east":"left", "south":"top"};

/* 구역 폴리곤 — 전처리에서 읍면동이 구역별로 병합돼 피처 3개뿐. 경계선 = 구역 사이만. */
const gj = L.geoJSON(GEO, {
  style: ()=>({color:STROKE, weight:1.1, fillColor:NODATA, fillOpacity:0.08}),
  onEachFeature: (f, lyr)=>{
    lyr._zone = f.properties.zone;
    lyr.bindTooltip("", {className:"rt", sticky:true, opacity:1,
                         direction:(TIP_DIR[lyr._zone] || "top")});
    lyr.on('mouseover', ()=> lyr.setStyle({weight:2.4, color:HOVER}));
    lyr.on('mouseout',  ()=> lyr.setStyle({weight:1.1, color:STROKE}));
  }
}).addTo(map);

/* 제주 전체에 화면 맞춤 — 좌우 패널 폭만큼 패딩, 섬 밖 이동·축소 잠금.
   숨은 탭에서 크기 0으로 초기화되면 지도가 비어 보이므로, 컨테이너가 실제 크기를
   가질 때까지 맞춤을 미루고(ResizeObserver) 보이는 순간 invalidateSize 후 fit한다. */
const fullB = gj.getBounds();
map.setMaxBounds(fullB.pad(1.2));
const mapEl = document.getElementById("map");
let needFit = true;
function fitJeju(){
  if (!needFit || !mapEl.clientWidth || !mapEl.clientHeight) return;
  map.invalidateSize();
  const sidePad = Math.round(mapEl.clientWidth * 0.26);
  map.fitBounds(fullB, {paddingTopLeft:[sidePad,16], paddingBottomRight:[sidePad,20]});
  map.setMinZoom(map.getZoom() - 0.5);
  needFit = false;
}
fitJeju();
new ResizeObserver(()=>{ map.invalidateSize(); fitJeju(); }).observe(mapEl);

/* 구역 라벨 — 하늘상태 이모지 + 구역명·기온 */
const labelLayer = L.layerGroup().addTo(map);
Object.keys(Z).forEach(key=>{
  const d = Z[key];
  const t = d.ok ? Math.round(d.temp) + "°" : "—";
  L.marker([d.lat, d.lon], {interactive:false, icon: L.divIcon({
    className:"wxwrap", iconSize:[0,0], iconAnchor:[0,0],
    html:`<div class="wx"><span class="emo">${d.sky.emo}</span>`+
         `<span class="nm">${d.name} <small>${t}</small></span></div>`})}).addTo(labelLayer);
});

function fmt(v, suf){ return (v===null||v===undefined) ? "—" : v.toLocaleString() + (suf||""); }
function tipHTML(key, d){
  if (!d.ok){
    return `<div class="tip"><div class="tip__name">${d.name}</div>`+
      `<div class="tip__mem">관측: ${d.stname}(${d.stn_id})</div>`+
      `<div class="tip__wx">기상 데이터 없음 <span class="ws">수집 범위 밖</span></div></div>`;
  }
  const sa = d.sa ? d.sa : {pct:0, lab:"—"}, wa = d.wa ? d.wa : {pct:0, lab:"—"};
  const rad = d.ratio===null ? (d.rad_obs ? "—" : "관측 없음") : Math.round(d.ratio*100)+"%";
  const rain = d.rain>0 ? ` · 강수 ${d.rain} mm/h` : "";
  const grid = d.kim_grid ? `<div class="tip__row"><span class="k"><i style="background:var(--wind)"></i>KIM 풍력 격자</span><span class="v">${d.kim_grid}</span></div>` : "";
  return `<div class="tip"><div class="tip__name">${d.name} <span style="font-size:10px;color:#64748b">${META.badge}</span></div>`+
    `<div class="tip__mem">관측: ${d.stname}(${d.stn_id}) · 일사계 ${d.rad_obs?"있음":"없음"}</div>`+
    `<div class="tip__wx">${d.sky.emo} ${d.sky.t} · ${d.temp}° · 풍속 ${fmt(d.wind_ms," m/s")}<span class="ws">${d.stname}</span></div>`+
    `<div class="tip__row"><span class="k">일사 비율 (09–15)</span><span class="v">${rad}<small>${rain}</small></span></div>`+
    grid+
    `<div class="tip__act">`+
      `<div class="tip__row"><span class="lvl">☀️ 태양광 활성도</span><span class="v">${sa.pct}%<small>${sa.lab}</small></span></div>`+
      `<div class="mini"><span style="width:${sa.pct}%;background:var(--solar-soft)"></span></div>`+
      `<div class="tip__row" style="margin-top:6px"><span class="lvl">🌀 풍력 활성도</span><span class="v">${wa.pct}%<small>${wa.lab}</small></span></div>`+
      `<div class="mini"><span style="width:${wa.pct}%;background:var(--wind-soft)"></span></div>`+
    `</div></div>`;
}

let mode = "gen";
function render(){
  const ckS=document.getElementById("ck-solar").checked, ckW=document.getElementById("ck-wind").checked;
  document.getElementById("toggles").style.display = (mode==="gen") ? "" : "none";
  document.getElementById("legend").innerHTML = LEGEND[mode];

  /* 신재생 강도: 구역별 설비 분리치가 없으므로 활성도(교정 이용률 %)만으로 상대 비교 */
  const refGen = (ckS?SA_MAX:0) + (ckW?WA_MAX:0) || 1;
  let sumScore=0, nOk=0, top=[];
  Object.keys(Z).forEach(key=>{
    const d=Z[key];
    if (!d.ok){ d._score=null; return; }
    d._gen = (ckS?(d.sa?d.sa.pct:0):0) + (ckW?(d.wa?d.wa.pct:0):0);
    const genScore = Math.min(1, d._gen/refGen);
    d._score = (mode==="gen") ? genScore
             : (mode==="rad") ? (d.ratio===null ? null : d.ratio)
             : (d.wind_ms===null ? null : Math.min(1, d.wind_ms/WIND_FULL));
    sumScore += genScore; nOk += 1; top.push([d.name, d._gen]);
  });
  top.sort((a,b)=>b[1]-a[1]);

  gj.getLayers().forEach(lyr=>{
    const d=Z[lyr._zone]; if(!d) return;
    if (d._score===null || d._score===undefined){
      lyr.setStyle({fillColor:NODATA, fillOpacity:0.08, color:STROKE, weight:1.1});
    } else {
      lyr.setStyle({fillColor:GREEN, fillOpacity:OP_MIN+(OP_MAX-OP_MIN)*d._score,
                    color:STROKE, weight:1.1});
    }
    lyr.setTooltipContent(tipHTML(lyr._zone, d));
  });

  document.getElementById("v-ico").textContent = META.rep.emo || "·";
  document.getElementById("v-name").textContent = `제주 대체로 ${META.rep.t}`;
  if (nOk){
    const avg = Math.round(sumScore/nOk*100);
    document.getElementById("v-strength").style.width = Math.min(100, avg*1.4)+"%";
    const lead = top.length>=2
      ? `<br>가장 활발한 구역 — <b>${top[0][0]}</b> · <b>${top[1][0]}</b>` : "";
    document.getElementById("v-msg").innerHTML =
      `제주 신재생 가동 강도 <b>${avg}%</b>${lead}`;
  } else {
    document.getElementById("v-strength").style.width = "0%";
    document.getElementById("v-msg").innerHTML = "이 날짜의 기상·이용률 데이터가 없습니다.";
  }
}

document.querySelectorAll(".mchip").forEach(b=>{
  b.onclick = ()=>{ mode=b.dataset.m;
    document.querySelectorAll(".mchip").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); render(); };
});
document.getElementById("ck-solar").onchange=render;
document.getElementById("ck-wind").onchange=render;
render();
</script>
</body>
</html>"""


def _sparkline_svg(vals: list, color: str = "#1f77b4", w: int = 264, h: int = 44,
                   fill: bool = True, overlay: list | None = None,
                   overlay_color: str = "#94a3b8") -> str:
    """미니 시계열 스파크라인 SVG. vals=주선(점선+옅은 채움), overlay=보조선(실측, 같은 y범위)."""
    n = len(vals)
    allv = [v for s in (vals, overlay or []) for v in s if v is not None]
    main_pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if n < 2 or len(main_pts) < 2 or not allv:
        return ""
    ymin, ymax = min(allv), max(allv)
    rx = (n - 1) or 1
    ry = (ymax - ymin) or 1
    pad = 4

    def X(x):
        return pad + x / rx * (w - 2 * pad)

    def Y(y):
        return h - pad - (y - ymin) / ry * (h - 2 * pad)

    def poly(s):
        return " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s) if v is not None)

    main = poly(vals)
    xs = [i for i, v in enumerate(vals) if v is not None]
    parts = [f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">']
    if fill:
        parts.append(f'<polygon points="{X(xs[0]):.1f},{h - pad} {main} '
                     f'{X(xs[-1]):.1f},{h - pad}" fill="{color}" fill-opacity="0.10"/>')
    if overlay and poly(overlay):
        parts.append(f'<polyline points="{poly(overlay)}" fill="none" stroke="{overlay_color}" '
                     'stroke-width="1.8" stroke-opacity="0.9" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<polyline points="{main}" fill="none" stroke="{color}" stroke-width="1.4" '
                 'stroke-dasharray="3 2" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append('</svg>')
    return "".join(parts)


def _wx_stats_html(tmax: float | None, tmin: float | None, humidity: float | None) -> str:
    """왼쪽 패널 — 3구역 최고/최저 기온·습도 통계 3칸."""
    def cell(k, v, suf):
        vv = "—" if v is None else f"{v:.0f}{suf}"
        return f'<div class="wxstat"><div class="wxstat__k">{k}</div><div class="wxstat__v">{vv}</div></div>'

    return ('<div class="wxstats">'
            + cell("🌡 최고기온", tmax, "°")
            + cell("최저기온", tmin, "°")
            + cell("💧 습도", humidity, "%")
            + '</div>')


def _util_cards_html(util: dict, util_act: dict | None = None) -> str:
    """왼쪽 패널 — 섬 전체 태양광·풍력 이용률 카드 2개(예측값 + 최대/실측 보조)."""
    util_act = util_act or {}

    def card(emoji, label, avg, mx, act):
        if avg is None:
            v, sub = "—", ""
        else:
            v = f"{avg:.1f}"
            sub = (f"실측 {act:.1f}%" if act is not None
                   else f"최대 {mx:.1f}%" if mx is not None else "")
        return (f'<div class="ucard"><div class="ucard__k">{emoji} {label}</div>'
                f'<div class="ucard__v">{v}<small>%</small></div>'
                f'<div class="ucard__s">{sub}</div></div>')

    return ('<div class="ucards">'
            + card("☀️", "태양광 이용률", util.get("solar"), util.get("solar_max"),
                   util_act.get("solar"))
            + card("🌀", "풍력 이용률", util.get("wind"), util.get("wind_max"),
                   util_act.get("wind"))
            + '</div>')


def _netload_panel_html(nl: dict | None) -> str:
    """오른쪽 패널 — 순 부하 예측(스파크·최대/최소) + 수요 + SMP 최저·음수가격 경보."""
    if not nl:
        return ""

    def f(v):
        return "—" if v is None else f"{v:,.0f}"

    # 스파크 색 = 차트 팔레트와 동일 규약 (inject_style 이 테마별로 갱신한 C.COLOR)
    nl_chart = ""
    if nl.get("nl_spark"):
        nl_chart = ('<div class="sparklab">순 부하(예상) · 점선=예측 / 실선=실측</div>'
                    + _sparkline_svg(nl["nl_spark"], color=C.COLOR["net_load"],
                                     overlay=nl.get("nl_real_spark")))
    dem = ""
    if nl.get("demand_spark") and nl.get("demand_peak") is not None:
        dem = ('<div class="divider"></div>'
               '<div class="dem__top"><span class="gk">전력수요(예상)</span>'
               f'<span class="gv">{f(nl["demand_peak"])}<small>MW 최대</small></span></div>'
               + _sparkline_svg(nl["demand_spark"], color=C.COLOR["demand"],
                                overlay=nl.get("demand_real_spark")))
    smp_min = nl.get("smp_min")
    danger = nl.get("danger_hours", 0)
    danger_cls = ' class="gv warn"' if danger else ' class="gv"'
    return (
        '<div class="panel panel--netload">'
        '<div class="panel__head">'
        '<div class="panel__eyebrow">오늘의 예상 순 부하</div>'
        f'<h1 class="panel__title">{f(nl["nl_max"])} <span class="unit">MW 최대</span></h1>'
        '</div><div class="panel__body">'
        f'<div class="nlrow"><span class="gk"><i style="background:{C.COLOR["net_load"]}"></i>'
        f'순 부하 최소</span><span class="gv">{f(nl["nl_min"])}<small>MW</small></span></div>'
        f'{nl_chart}'
        '<div class="divider"></div>'
        f'<div class="nlrow"><span class="gk"><i style="background:{C.COLOR["smp"]}"></i>SMP 최저(예측)</span>'
        f'<span class="gv">{"—" if smp_min is None else f"{smp_min:,.1f}"}<small>원/kWh</small></span></div>'
        '<div class="nlrow"><span class="gk">음수가격 경보</span>'
        f'<span{danger_cls}>{danger}<small>시간</small></span></div>'
        f'{dem}'
        '</div></div>')


def build_html(day: pd.Timestamp, dplus: int, zones: dict, util: dict,
               netload: dict | None = None, util_act: dict | None = None,
               humidity: float | None = None) -> str:
    """임베드 HTML — 선택일 데이터 주입.

    netload(순 부하 패널)·util_act(이용률 실측)·humidity(평균 습도)를 주면 좌우 패널을 채운다.
    """
    conf_t, conf_c = conf_of(dplus)
    issue_badge = _issue_badge(day.strftime("%Y-%m-%d"), dplus)
    skies = [z["sky"]["t"] for z in zones.values() if z["ok"]]
    rep = ({"emo": "", "t": "—"} if not skies else
           next(z["sky"] for z in zones.values()
                if z["ok"] and z["sky"]["t"] == max(set(skies), key=skies.count)))
    weekday = "월화수목금토일"[day.weekday()]
    temps = [z["temp"] for z in zones.values() if z["ok"] and z["temp"] is not None]
    tmax = max(temps) if temps else None
    tmin = min(temps) if temps else None
    meta = {"dplus": dplus, "rep": rep, "badge": issue_badge}
    theme = _MAP_THEMES[C.theme_type()]
    active_text = theme["PANEL"] if C.theme_type() == "dark" else "#ffffff"
    html = _TEMPLATE
    for k, v in ([("__GEO__", _geo_text()),
                  ("__ZONES__", json.dumps(zones, ensure_ascii=False)),
                  ("__META__", json.dumps(meta, ensure_ascii=False)),
                  ("__DATE_LABEL__", f"{day:%m-%d} ({weekday})"),
                  ("__ISSUE_BADGE__", issue_badge),
                  ("__CONF_T__", conf_t), ("__CONF_C__", conf_c),
                  ("__GREEN__", GREEN), ("__OP_MIN__", str(OP_MIN)),
                  ("__OP_MAX__", str(OP_MAX)), ("__WIND_FULL__", str(WIND_FULL)),
                  ("__SA_MAX__", str(SA_MAX)), ("__WA_MAX__", str(WA_MAX)),
                  ("__NETLOAD_PANEL__", _netload_panel_html(netload)),
                  ("__UTIL_CARDS__", _util_cards_html(util, util_act)),
                  ("__WX_STATS__", _wx_stats_html(tmax, tmin, humidity)),
                  ("__ACTIVE_TEXT__", active_text)]
                 + [(f"__{key}__", value) for key, value in theme.items()]):
        html = html.replace(k, v)
    return html


def build_day_html(day: pd.Timestamp) -> str:
    """종합 화면 hero 용 원스톱 빌더 — 선택일 기준 예보/실측을 골라 HTML 을 만든다.

    미래·오늘 = 예보(forecast_horizon) + 이용률 예측.  과거 = 실측 관측 + 이용률 실측 보조.
    """
    today = pd.Timestamp.now().normalize()
    dplus = int((day.normalize() - today).days)
    date = day.strftime("%Y-%m-%d")
    if dplus >= 0:
        zones = zone_day(date)
        util = jeju_util(date, forecast=True)
        util_act = None
    else:
        zones = zone_actual(date)
        util = jeju_util(date, forecast=True)          # 그날 예측(발행본 아카이브)
        util_act = jeju_util(date, forecast=False)     # 실측 보조 표기
    return build_html(day, dplus, zones, util,
                      netload=netload_panel_data(date),
                      util_act=util_act, humidity=jeju_humidity(date))
