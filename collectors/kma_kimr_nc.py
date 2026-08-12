"""kma_kimr_nc.py -- KIMR(R030) 표준 경로.  신 KIM 표준 API(nph-kim_nc_pt_txt2_std).

★KIMR 의 **유일 경로**다 (2026-08-04).  구 GRIB 경로(`kma_kimr_grib.py`)는
  같은 날 파일째 삭제됐다 -- 아카이브(`collect_archive`)에 이어 서빙 입력
  (`forecast_horizon`)까지 이 경로로 넘어오면서 유예 조건이 소멸했다.
  KIMR 관련 코드는 전부 여기 있다.

2026-07-01 KMA 개편 대응 v2 수집기의 fetch 계층 (new_kma 연구로 검증, REPORT_01~03).
기존 core/ 파일은 수정하지 않는다 -- kma_kimg 의 검증된 HTTP/키풀 스택을 import 재사용.

핵심 사실 (실측 검증 2026-07-04)
- 엔드포인트: typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std -- 위경도 직접 입력,
  콤마 멀티변수 1콜, 3모델(KIMG/NE57·KIMR/R030·KIML/L010) 지원, 디코더 정상
  (격자 방식 xy_txt2_std 의 전구 화면고도 버그 없음).
- R030(지역 3km): 00/12z 최대 120h·06/18z 72h, 전 구간 1h.  운량 없음.
- L010(국지 1.3km): 4주기 48h, 전 구간 1h.
- GHI(전천일사) = SWDDIR2(수평면 직달) + SWDDIF2(산란) -- 물리 검증 완료.

지점 확장 규약 (사용자 요구: 5->17지점 같은 확장 + 그 지점만 백필)
- POINTS_*_V2 리스트가 SSOT: [{"name","sfx","lat","lon"}].  지점 추가 = 1줄.
- 컬럼 접미사(sfx)는 명시 필드 -- 이름 파싱 없음.
- 지점 단위 부분 수집은 collect_forecast_v2 의 --points 와 컬럼 보존 upsert 가 담당.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import kma_kimg as ckg
from kma_kimg import (
    KST,
    UTC,
    RETRY_MAX,
    _kma_session,
    kma_quota_exceeded,
    freshest,
)

URL_PT_STD = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std"

# 발표시각(UTC)별 최대 리드(h) -- 2026-07-04 전수 스캔 실측 (new_kma REPORT_02 §3).
R030_MAX_HF = {0: 120, 6: 72, 12: 120, 18: 72}
L010_MAX_HF = {0: 48, 6: 48, 12: 48, 18: 48}

# ── 지점 SSOT (sfx 명시 -- 지점 확장 시 여기에 1줄 추가) ────────────────────
# 좌표는 구 GRIB 수집기가 쓰던 것과 동일 (2026-05-25 '바다 셀 회피' 보정 계승).
# grib 시계열 경로도 위경도 직접 입력을 지원함을 실측 확인(2026-07-04) -- 위경도로
# 주면 std(pt_txt2_std)와 같은 셀로 해석돼 소스 간 셀 일관성이 보장된다.
# (자체 X/Y 를 주면 최근접 산정 차이로 한 셀 어긋날 수 있음.)  지점 추가 = 이 리스트에
# 위경도 1줄이면 끝, 격자 변환 불필요.
POINTS_LAND_V2 = [
    {"name": "Daegwallyeong(100)", "sfx": "daegwallyeong", "lat": 37.6772, "lon": 128.7185},
    {"name": "Wonju(114)",         "sfx": "wonju",         "lat": 37.3376, "lon": 127.9466},
    {"name": "Seosan(129)",        "sfx": "seosan",        "lat": 36.7766, "lon": 126.4939},
    {"name": "Pohang(138)",        "sfx": "pohang",        "lat": 36.0327, "lon": 129.3799},
    {"name": "Yeonggwang(252)",    "sfx": "yeonggwang",    "lat": 35.2807, "lon": 126.4750},
]
# ★제주는 x/y 고정 필수: 운영 grib 좌표(2026-05-25 "바다 셀 회피" 수동 보정 계승).
#   위경도 최근접은 서/동쪽이 바다 셀로 떨어져 met 값이 크게 어긋난다
#   (temp_skin 최대 9.5°C 차 실측 -- 해수면/지표 차이).  x/y 가 있으면 fetch 가
#   X/Y 인자로 셀을 고정한다(pt_txt2_std·grib 둘 다 지원 확인).
POINTS_JEJU_V2 = [
    {"name": "West(Gosan)",        "sfx": "west",  "lat": 33.4427, "lon": 126.1713, "x": 530, "y": 251},
    {"name": "East(Seongsan)",     "sfx": "east",  "lat": 33.3868, "lon": 126.8802, "x": 553, "y": 254},
    {"name": "solar_farm(south)",  "sfx": "south", "lat": 33.3284, "lon": 126.8366, "x": 550, "y": 250},
]

# 요청 변수 묶음 (콤마 멀티변수 1콜).  대소문자 엄격 -- KIM 변수정보 PDF 기준.
# MSLP = 해면기압(급변동 감지: 기압 경향 = 전선·기압골 접근, 2026-07-05 사용자 승인)
# GRAUPEL = 구 GRIB varn 1074(TCOG, 싸락눈) 의 NC 이름 (2026-08-04 프로브로 확인).
#   태양광·풍력 대류일 후처리(serve_solarwind._apply_tcog)가 이 값을 쓴다.
#   ※ 우박(구 TCOH)은 NC 에 대응 이름이 없다 -- 다만 실측 32,012행 × 3지점이 전부
#     정확히 0 이라(Training EDA 도 "상수 0" 으로 제외) 잃을 정보가 없다.
R030_NAME_LAND = ("T2,RH2,U10,V10,U80,V80,GUST,SWDDIR2,SWDDIF2,"
                  "RAINC,RAINNC,MCAPE,MCIN,PBLH,MSLP,ACSWDNB,GRAUPEL")
# 누적 변수 diff 기준점(hf=start-1) 전용.  강수(RAINC/RAINNC)와 ACSWDNB(누적 총일사)는
# 모두 base 부터 누적이라 시작 직전값이 있어야 첫 시각 diff 가 옳다.
R030_NAME_DIFF_ANCHOR = "RAINC,RAINNC,ACSWDNB"
R030_NAME_RAIN_ANCHOR = R030_NAME_DIFF_ANCHOR   # (구명 호환)
L010_NAME = R030_NAME_LAND                  # L010 도 동일 세트 (가용성 스윕 확인)
# ※ 제주 met 도 2026-08-04 부터 이 NC 경로가 유일하다 (구 GRIB 시계열 폐기).
#   2026-07-05 에 "NC per-hf 는 혼잡 밤 마비" 로 한 번 기각됐던 우려는 아카이브가
#   161일 백필 + 매일 00:20 KST cron 으로 뒤집었다 -- 지점당 ~10s.


# ── kim2 전용 키 풀 ─────────────────────────────────────────────────────────
# ★신규 std API 는 키(계정)별 "활용신청"이 필요하다 (2026-07-04 실측: 미신청 키는
# 403 {"message": "활용신청이 필요한 API 입니다..."}).  kma_kimg 의 프로세스 공용
# 키 회전(rotate_kma_key)을 여기서 돌리면 구 엔드포인트(NE57 part)까지 예비 키로
# 끌려가므로, kim2 는 자기만의 키 인덱스를 쓴다 -- 등록 안 된 키는 건너뛰고
# 처음 성공한 키에 고정(pin), 한도 초과 시 다음 키로.
_K2_KEYS: list[str] = []
for _k in (os.getenv("KMA_API_KEY"), os.getenv("KMA_API_KEY_SUB"),
           os.getenv("KMA_API_KEY_SUB2"), os.getenv("KMA_API_KEY_SUB3")):
    if _k and _k not in _K2_KEYS:
        _K2_KEYS.append(_k)
_k2_idx = 0
_k2_lock = threading.Lock()
_NOT_REGISTERED_MARKER = "활용신청"   # "활용신청" (u-escape: 소스 인코딩 무관)


def _k2_current() -> str | None:
    return _K2_KEYS[_k2_idx] if _K2_KEYS else None


def _k2_rotate(bad_key: str | None, why: str) -> bool:
    """bad_key 가 여전히 현재 키일 때만 다음 키로.  남은 키 없으면 False."""
    global _k2_idx
    with _k2_lock:
        if _k2_current() != bad_key:
            return True
        if _k2_idx + 1 >= len(_K2_KEYS):
            return False
        _k2_idx += 1
        print(f"  [kim2-key] {why} -> 다음 키로 전환 ({_k2_idx + 1}/{len(_K2_KEYS)})")
        return True


# ── HTTP fetch (구 kma_kimg.fetch_one_hf 미러 -- 세션·백오프 동일) ────────────
def fetch_pt_std(
    group: str, nwp: str, name: str, base_utc: datetime, hf: int,
    lat: float, lon: float, x: int | None = None, y: int | None = None,
    data: str = "U",
) -> str | None:
    """단일 (모델, 발표, 지점, hf) 호출.  plaintext body 반환 (실패 시 None).

    x/y 가 주어지면 격자 셀 고정(X/Y 인자 -- 제주 바다 셀 회피), 아니면 위경도
    최근접.  5xx/timeout 은 exponential backoff(2s,4s) 재시도.  '활용신청 필요'
    403 과 한도 초과는 kim2 전용 키 풀에서 다음 키로 전환 후 즉시 재시도.
    data="P" 면 등압면(레벨별 행 반환 -- CLDFRA 등), 기본 "U"=단일면.
    ※ nwp 는 대문자 필수(R030/NE57) -- 소문자는 "Input variable error" (2026-07-17 실측).
    """
    params = {
        "group": group, "nwp": nwp, "data": data, "name": name,
        "tmfc": base_utc.strftime("%Y%m%d%H"), "hf": str(hf),
        "disp": "A", "help": "0",
    }
    if x is not None and y is not None:
        params["X"], params["Y"] = str(x), str(y)
    else:
        params["lat"], params["lon"] = f"{lat:.4f}", f"{lon:.4f}"
    attempt = 0
    while attempt < RETRY_MAX:
        key = _k2_current()
        if key is None:
            return None
        params["authKey"] = key
        try:
            r = _kma_session.get(URL_PT_STD, params=params, timeout=30)
            if r.status_code == 403 and _NOT_REGISTERED_MARKER in r.text:
                # 이 키(계정)는 신규 API 미신청 -- 재시도 카운트 소모 없이 키만 전환.
                if _k2_rotate(key, "활용신청 안 된 키"):
                    continue
                return None
            if kma_quota_exceeded(r.status_code, r.text):
                if _k2_rotate(key, "한도 초과"):
                    continue
                return None
            if r.status_code == 200:
                return r.text
            if 500 <= r.status_code < 600 and attempt < RETRY_MAX - 1:
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            if attempt < RETRY_MAX - 1:
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def parse_pt_std(body: str) -> dict[str, float]:
    """응답 -> {변수명: 값}.  행 = 'TMFC TMEF VARN LEVEL VALUS NAME(단위)'.

    변수명은 NAME 토큰에서 '(' 앞부분 -- VARN 숫자코드에 의존하지 않는다
    (모델·파일 개편 때 코드가 바뀌어도 이름은 유지되는 것을 신뢰).
    """
    out: dict[str, float] = {}
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        try:
            val = float(parts[4])
        except ValueError:
            continue
        out[parts[5].split("(")[0]] = val
    return out


# ── 카테고리 파생 (kma_kimg.derive_categories 의 R030/L010 판) ────────────────
# 라벨·반올림을 KIMG long 관례에 정확히 맞춰 기존 피벗 수식이 그대로 성립하게 한다.
def derive_r030_categories(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "T2" in raw:
        out["TEMP_C"] = round(raw["T2"] - 273.15, 2)
    if "TSKIN" in raw:
        out["TEMP_SKIN"] = round(raw["TSKIN"] - 273.15, 2)
    if "RH2" in raw:
        out["REH"] = round(raw["RH2"], 2)
    for src, cat in (("U10", "WIND_U_10M"), ("V10", "WIND_V_10M"),
                     ("U80", "WIND_U_80M"), ("V80", "WIND_V_80M")):
        if src in raw:
            out[cat] = round(raw[src], 3)
    if "GUST" in raw:
        out["GUST"] = round(raw["GUST"], 3)
    if "SWDDIR2" in raw and "SWDDIF2" in raw:
        ghi = raw["SWDDIR2"] + raw["SWDDIF2"]          # GHI = 직달수평 + 산란
        out["SOLAR_RAD"] = round(ghi * 0.0036, 4)       # W/m^2 -> MJ/m^2/h (현행식)
        out["RAD_DIRECT"] = round(raw["SWDDIR2"] * 0.0036, 4)
        out["RAD_DIFFUSE"] = round(raw["SWDDIF2"] * 0.0036, 4)
    if "RAINC" in raw:
        out["RAIN_CONV"] = round(raw["RAINC"], 4)       # mm 누적 (kg/m^2 동치)
    if "RAINNC" in raw:
        out["RAIN_STRAT"] = round(raw["RAINNC"], 4)
    if "ACSWDNB" in raw:
        out["SOLAR_ACC"] = round(raw["ACSWDNB"], 4)     # 누적 총일사 MJ/m^2 (실측=시간누적과 동일 성질)
    if "MCAPE" in raw:
        out["CAPE"] = raw["MCAPE"]                      # 피벗에서 r4 (제주 KIMR 관례)
    if "MCIN" in raw:
        out["CINN"] = raw["MCIN"]
    if "PBLH" in raw:
        out["HPBL"] = raw["PBLH"]
    if "MSLP" in raw:
        out["PRESS_MSL"] = round(raw["MSLP"] / 100.0, 2)   # Pa -> hPa
    if "GRAUPEL" in raw:
        out["TCOG"] = raw["GRAUPEL"]                    # 구 GRIB varn 1074 와 같은 라벨로
    return out


# ── hf 그리드 ────────────────────────────────────────────────────────────────
def hf_range_1h(base_utc: datetime, days: int, max_hf_map: dict[int, int]) -> list[int]:
    """day-aligned KST 윈도우를 1h hf 리스트로 (kma_kimg.window_bounds 위임).

    기본 [D+1 00시, D+days 24시); ckg.SAMEDAY_18Z + 18z 는 당일 창(04시=hf1 부터).
    발표시각별 모델 리드 상한(max_hf_map)으로 절단 -- --utc 6/18 이면 자동 축소.
    (days 인자를 직접 받아 FORECAST_DAYS 글로벌에 의존하지 않는 설계는 유지.)
    """
    base_kst = base_utc.astimezone(KST)
    start_kst, end_kst = ckg.window_bounds(base_utc, days)
    start_hf = int((start_kst - base_kst).total_seconds() // 3600)
    end_hf = int((end_kst - base_kst).total_seconds() // 3600)   # exclusive
    cap = max_hf_map[base_utc.hour]
    return [hf for hf in range(max(start_hf, 1), end_hf) if hf <= cap]


# ── std NC per-hf -> long 5컬럼 (수집 진입점 공용 엔진) ──────────────────────
# 아카이브(collect_archive -> forecast_kimr)와 운영(collect_forecast ->
# forecast_horizon)이 **같은 스택**을 쓴다.  둘의 차이는 요청 변수 목록과 카테고리
# 파생 함수뿐이라 그 둘만 인자로 받는다.
NC_WORKERS = 8      # CLDFRA 와 같은 값 (std NC per-hf 병렬 안전 실측 2026-07-17)

LONG_COLS = ["base_datetime", "point_name", "fcst_datetime", "category", "fcst_value"]


def fetch_nc_long(points: list[dict], base_utc: datetime, days: int,
                  names: str, derive, max_hf_map: dict[int, int] | None = None,
                  workers: int = NC_WORKERS, label: str = "KIMR-nc") -> pd.DataFrame:
    """(지점 × hf) std NC 호출 -> long 5컬럼 DataFrame.

    누적 변수(RAINC/RAINNC/ACSWDNB)의 diff 기준점을 위해 **hf=start-1 앵커**를 함께
    받는다 -- 앵커 행은 피벗의 윈도우 트림에서 빠지고, 덕분에 창 첫 시각의 diff 가
    옳다.  (앵커 hf=0 도 유효: ACSWDNB(hf=0)=0.0 실측 확인 2026-07-18 18z 프로브.
    18z 당일 창은 hf_list[0]=1 이라 앵커가 0 이 된다.)

    지점 순차 × hf 병렬(8워커) + 결측 hf 순차 재시도 2라운드.  fetch_pt_std 자체가
    5xx 백오프·키 회전을 하므로 여기서는 라운드만 더한다.  그래도 남으면 호출자가
    커버리지로 경고 -> 재실행 치유(COALESCE upsert 는 idempotent).

    derive : raw{변수명: 값} -> {카테고리: 값}.  소비처별 라벨 관례를 여기서 가른다.
    """
    hf_list = hf_range_1h(base_utc, days, max_hf_map or R030_MAX_HF)
    if not hf_list:
        return pd.DataFrame(columns=LONG_COLS)
    hfs = ([hf_list[0] - 1] + hf_list) if hf_list[0] >= 1 else hf_list
    base_dt_str = base_utc.astimezone(KST).strftime("%Y-%m-%d %H:%M")

    rows: list[tuple] = []
    seen_names: set[str] = set()
    for pt in points:
        t0 = time.time()

        def one(hf: int) -> tuple[int, dict[str, float]]:
            body = fetch_pt_std("KIMR", "R030", names, base_utc, hf,
                                pt.get("lat"), pt.get("lon"), pt.get("x"), pt.get("y"))
            return hf, (parse_pt_std(body) if body else {})

        got: dict[int, dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for hf, raw in ex.map(one, hfs):
                if raw:
                    got[hf] = raw
        for rnd in range(2):
            missing = [h for h in hfs if h not in got]
            if not missing:
                break
            if rnd:
                time.sleep(2)
            for hf in missing:
                hf2, raw = one(hf)
                if raw:
                    got[hf2] = raw

        for hf, raw in got.items():
            seen_names |= set(raw)
            fcst = (base_utc + timedelta(hours=hf)).astimezone(KST).strftime(
                "%Y-%m-%d %H:%M")
            for cat, val in derive(raw).items():
                rows.append((base_dt_str, pt["name"], fcst, cat, float(val)))
        print(f"  {label}  {pt['name']:<22} {len(got)}/{len(hfs)}시각 "
              f"({time.time() - t0:.1f}s)")

    missing_names = sorted(set(names.split(",")) - seen_names)
    if missing_names:
        print(f"  [주의] R030 std NC 가 주지 않은 변수: {missing_names} "
              f"(해당 컬럼은 비게 된다)")
    df = pd.DataFrame(rows, columns=LONG_COLS)
    if not df.empty:
        df["fcst_value"] = pd.to_numeric(df["fcst_value"], errors="coerce")
    return df


# ── 운영 met (collect_forecast -> forecast_horizon) ──────────────────────────
# ★출력 카테고리를 **구 GRIB 라벨·원단위 그대로** 낸다 (TEMP 는 K, 반올림 없음).
#   피벗(pivot._SPEC_KIMR)이 그 라벨을 기대하고 K->°C 변환·반올림도 거기서 하므로,
#   여기서 이름만 맞춰 주면 `pivot.py`·`selftest_pivot.py`·`forecast_horizon` 스키마가
#   전부 불변이다 -- GRIB->NC 전환 검증이 "값이 같은가" 하나로 단순해진다.
#   (derive_r030_categories 는 아카이브용이라 TEMP_C(°C)·일사 파생까지 내므로 안 쓴다.)
# ※ 구 GRIB 15종 중 TCOH(우박)만 NC 에 대응 이름이 없다 -- 실측 전부 0 이라 손실 없음.
_MET_OPS_RENAME = {
    "T2":      "TEMP",          # K 그대로 (pivot 이 -273.15 후 r2)
    "TSKIN":   "TEMP_SKIN",     # K 그대로
    "RH2":     "REH",
    "U10":     "WIND_U_10M",
    "V10":     "WIND_V_10M",
    "U80":     "WIND_U_80M",
    "V80":     "WIND_V_80M",
    "GUST":    "GUST",
    "MCAPE":   "CAPE",
    "MCIN":    "CINN",
    "PBLH":    "HPBL",
    "GRAUPEL": "TCOG",          # 구 GRIB varn 1074
    "RAINC":   "RAIN_CONV",     # 누적 -- 피벗이 base 단위 diff
    "RAINNC":  "RAIN_STRAT",
}
NAME_MET_OPS = ",".join(_MET_OPS_RENAME)


def derive_met_ops_categories(raw: dict[str, float]) -> dict[str, float]:
    """raw{NC 변수: 값} -> {구 GRIB 카테고리: 값}.  단위 변환·반올림 없이 이름만 바꾼다."""
    return {cat: raw[src] for src, cat in _MET_OPS_RENAME.items() if src in raw}


def fetch_kimr_met_long(points: list[dict], base_utc: datetime, days: int,
                        workers: int = NC_WORKERS) -> pd.DataFrame:
    """운영 met long (collect_forecast -> forecast_horizon).  구 GRIB 라벨로 낸다."""
    return fetch_nc_long(points, base_utc, days, NAME_MET_OPS,
                         derive_met_ops_categories, workers=workers,
                         label="KIMR-met")


# ── KIMG 대체용 일사·운량 (KIMG 수집 실패 시 마지막 사다리) ──────────────────
# ★언제 쓰나: `collect_forecast` 가 KIMG 를 통째로(또는 길게) 못 받았을 때.
#   짧은 결손(연속 2개 = 3h 사고)은 시간 보간이 훨씬 낫다 — 2026-07-02~11 사고를
#   정답과 대조한 실측: 보간 MAE 0.063~0.106 (r 0.81~0.90) vs KIMR 대체 0.346~0.490
#   (r 0.25~0.43, 큰 오차 43~59%).  두 모델이 **다른 구름을 본다**(2026-07-31: 전운량
#   r 0.47).  그래서 이 함수는 **보간이 못 메운 자리에만** 쓴다.
# ★그래도 쓰는 이유: base 통째 실패면 보간할 이웃이 없다.  "예측 없음" 보다
#   "오차 있는 예측 + 출처 통보" 가 운영상 낫다 (사용자 결정 2026-08-12).
# 라벨을 **KIMG 관례**(SOLAR_RAD / TCLD / MIDLOW_CLOUD)로 내보내므로 뒤따르는
# 피벗(`pivot._SPEC_KIMG`)과 `kimg_solar` 가 무수정으로 성립한다.
NAME_SOLAR_SUB = "SWDDIR2,SWDDIF2"


def _derive_solar_kimg_labels(raw: dict[str, float]) -> dict[str, float]:
    """SWDDIR2+SWDDIF2 -> SOLAR_RAD (MJ/m^2/h).  KIMG dswrsfc 와 같은 단위·라벨."""
    if "SWDDIR2" in raw and "SWDDIF2" in raw:
        return {"SOLAR_RAD": round((raw["SWDDIR2"] + raw["SWDDIF2"]) * 0.0036, 4)}
    return {}


def fetch_kimr_solar_cloud_long(points: list[dict], base_utc: datetime, days: int,
                                workers: int = NC_WORKERS) -> pd.DataFrame:
    """KIMG 대체용 일사·운량 long (라벨은 KIMG 관례).

    일사 = std NC 순시 GHI(SWDDIR2+SWDDIF2), 운량 = 등압면 CLDFRA 24레벨 결합.
    반환 카테고리: SOLAR_RAD / TCLD / MIDLOW_CLOUD.
    """
    rad = fetch_nc_long(points, base_utc, days, NAME_SOLAR_SUB,
                        _derive_solar_kimg_labels, workers=workers,
                        label="KIMR-sub-rad")
    cld = fetch_r030_cldfra_long(points, base_utc, days, R030_MAX_HF)
    if not cld.empty:
        cld = cld.replace({"category": {"CLOUD_TOTAL_R030": "TCLD",
                                        "CLOUD_MIDLOW_R030": "MIDLOW_CLOUD"}})
    parts = [d for d in (rad, cld) if not d.empty]
    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    return pd.concat(parts, ignore_index=True)


# ── long -> wide 피벗 (cl.kimg_land_long_to_wide 의 파라미터화 사본) ─────────
# 원본(collect_data_land.py:191-273)은 POINT_SUFFIX 5지점 dict 가 하드코딩이라
# 지점 확장이 안 된다.  수식·반올림을 자구 그대로 옮기고 suffix 맵만 주입 --
# 동일성은 스모크 테스트(아래 __main__)가 원본과 출력 비교로 기계 검증한다.
_SCALAR_CATS = {
    "SOLAR_RAD":    "radiation",
    "TCLD":         "total_cloud",
    "MIDLOW_CLOUD": "midlow_cloud",
    "TEMP_C":       "temp",
    "TEMP_SKIN":    "temp_skin",
}
_RAIN_CATS = ("RAIN_CONV", "RAIN_STRAT")
# v2 확장 카테고리 -> 컬럼 prefix (제주 KIMR 관례 r4).
_EXT_R4_CATS = {"CAPE": "cape", "CINN": "cinn", "HPBL": "hpbl",
                "TCOG": "tcog", "TCOH": "tcoh",
                "CLOUD_TOTAL_R030": "total_cloud_r030",
                "CLOUD_MIDLOW_R030": "midlow_cloud_r030"}
_EXT_RAD_CATS = {"RAD_DIRECT": "radiation_direct", "RAD_DIFFUSE": "radiation_diffuse",
                 "RAD_TOA": "radiation_toa"}
# 파생 시 이미 반올림 완료 -- 피벗은 그대로 통과 (mslp: hPa r2)
_EXT_PASS_CATS = {"PRESS_MSL": "mslp"}


def long_to_wide_v2(df: pd.DataFrame, suffix_map: dict[str, str],
                    radiation_round: int | None = None) -> pd.DataFrame:
    """long -> wide.  기존 컬럼 수식은 cl.kimg_land_long_to_wide 와 동일,
    + 확장 카테고리(cape/cinn/hpbl r4, radiation_direct/diffuse).

    suffix_map       : {point_name: 컬럼 접미사} -- 지점 확장 SSOT 주입.
    radiation_round  : 제주 관례(r2) 적용 시 2.  None 이면 육지 관례(r4 유지).
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["fcst_value"] = pd.to_numeric(df["fcst_value"], errors="coerce")

    parts: list[pd.DataFrame] = []
    for point, suffix in suffix_map.items():
        sub = df[df["point_name"] == point]
        if sub.empty:
            print(f"  [WARN] no rows for {point} -- skipped")
            continue

        # 1) 비누적: (fcst_datetime, category) 별 freshest -> pivot.
        non_rain = sub[~sub["category"].isin(_RAIN_CATS + ("SOLAR_ACC",))]
        non_rain = freshest(non_rain, ["fcst_datetime", "category"])
        piv = non_rain.pivot(index="fcst_datetime", columns="category",
                             values="fcst_value")
        out = pd.DataFrame(index=piv.index)
        for cat, prefix in _SCALAR_CATS.items():
            if cat in piv:
                col = piv[cat]
                out[f"{prefix}_{suffix}"] = (
                    col.round(2) if cat in ("TEMP_C", "TEMP_SKIN") else col)
        if "REH" in piv:
            out[f"reh_{suffix}"] = piv["REH"].round(2)
        for height in ("10m", "80m"):
            u_col = f"WIND_U_{height.upper()}"
            v_col = f"WIND_V_{height.upper()}"
            if u_col in piv and v_col in piv:
                u, v = piv[u_col], piv[v_col]
                spd = np.sqrt(u**2 + v**2)
                wdir = (270 - np.degrees(np.arctan2(v, u))) % 360
                out[f"wind_spd_{height}_{suffix}"] = spd.round(2)
                out[f"wd_sin_{height}_{suffix}"]   = np.sin(np.radians(wdir)).round(4)
                out[f"wd_cos_{height}_{suffix}"]   = np.cos(np.radians(wdir)).round(4)
        if "GUST" in piv:
            out[f"gust_{suffix}"] = piv["GUST"].round(4)
        # 확장: cape/cinn/hpbl (제주 KIMR 관례 = raw round 4)
        for cat, prefix in _EXT_R4_CATS.items():
            if cat in piv:
                out[f"{prefix}_{suffix}"] = piv[cat].round(4)
        # 확장: 파생 단계에서 반올림 끝난 카테고리 (mslp hPa r2 등)
        for cat, prefix in _EXT_PASS_CATS.items():
            if cat in piv:
                out[f"{prefix}_{suffix}"] = piv[cat]
        # 확장: 일사 직달/산란 (derive 에서 r4 완료; 제주 관례면 r2 재적용)
        for cat, prefix in _EXT_RAD_CATS.items():
            if cat in piv:
                col = piv[cat]
                out[f"{prefix}_{suffix}"] = (
                    col.round(radiation_round) if radiation_round else col)
        if radiation_round and f"radiation_{suffix}" in out:
            out[f"radiation_{suffix}"] = out[f"radiation_{suffix}"].round(radiation_round)

        # 2) 강수: per-base 누적(conv+strat) -> diff -> clip(0) -> freshest.
        rain = sub[sub["category"].isin(_RAIN_CATS)]
        if not rain.empty:
            acc = (
                rain.groupby(["base_datetime", "fcst_datetime"], as_index=False)["fcst_value"]
                    .sum()
                    .rename(columns={"fcst_value": "acc"})
                    .sort_values(["base_datetime", "fcst_datetime"])
            )
            acc["hourly"] = (
                acc.groupby("base_datetime")["acc"].diff().clip(lower=0).round(2)
            )
            acc = acc.dropna(subset=["hourly"])
            acc = freshest(acc, ["fcst_datetime"])
            rain_s = acc.set_index("fcst_datetime")["hourly"].rename(f"rainfall_{suffix}")
            out = out.join(rain_s, how="outer")

        # 2b) ACSWDNB(누적 총일사) -> per-base diff -> clip(0) -> radiation_acswdnb.
        #     실측(기상청 시간누적 일사)과 단위·정의가 동일 -- 순시 SWDDIR2+SWDDIF2 대비
        #     한낮 과대편향이 작다(EDA new_kma/REPORT_05 §5-보강).  사용은 +보정 예정.
        sw = sub[sub["category"] == "SOLAR_ACC"]
        if not sw.empty:
            swacc = (
                sw.groupby(["base_datetime", "fcst_datetime"], as_index=False)["fcst_value"]
                  .sum()
                  .rename(columns={"fcst_value": "acc"})
                  .sort_values(["base_datetime", "fcst_datetime"])
            )
            swacc["hourly"] = (
                swacc.groupby("base_datetime")["acc"].diff().clip(lower=0)
                     .round(radiation_round if radiation_round else 4)
            )
            swacc = swacc.dropna(subset=["hourly"])
            swacc = freshest(swacc, ["fcst_datetime"])
            sw_s = swacc.set_index("fcst_datetime")["hourly"].rename(f"radiation_acswdnb_{suffix}")
            out = out.join(sw_s, how="outer")

        if out.empty:
            print(f"  [WARN] {point} produced no rows -- skipped")
            continue
        parts.append(out)
        print(f"  pivot {point:<20} -> {len(out):4d} rows x {len(out.columns)} cols")

    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts, axis=1)
    wide.index = pd.to_datetime(wide.index, format="%Y-%m-%d %H:%M").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    wide.index.name = "timestamp"
    return wide.sort_index()


# ── R030 등압면 CLDFRA(std NC, data=P) 결합 -> 1h 운량 ───────────────────────
# 지역·국지 단일면엔 운량이 없지만 등압면 CLDFRA(24레벨)로 결합 운량을 만들 수 있다
# (new_kma nb05 검증: max-random 오버랩이 자체 일사 투과율과 정합).
# 서빙은 기존 NE57 운량을 계속 쓰고, 이 컬럼들은 EDA·재훈련 대비 아카이브다.
# (구 GRIB varn 6032(frcc) 경로는 2026-08-04 GRIB 폐기와 함께 제거 -- 같은 필드임은
#  값 대조로 검증돼 있었다: NC 결합 0.978 == GRIB 결합 0.978.)




def _frcc_combine(levels: dict[int, float]) -> tuple[float, float]:
    """{hPa: cloud fraction} -> (total, midlow).

    total  = max-random 오버랩: 붙어 있는 구름층(연속 레벨)은 한 덩어리(최대값),
             떨어진 덩어리끼리 랜덤 결합 -- nb05 채택식.
    midlow = low + mid(1-low), low=max(>=800hPa)/mid=max(450~799hPa)
             -- 기존 midlow_cloud(lcld+mcld(1-lcld)) 결합식과 동일 철학.
    """
    cs = [levels[p] for p in sorted(levels, reverse=True)]   # 지면(1000)→상공(50)
    blocks: list[float] = []
    cur = 0.0
    for c in cs:
        if c > 0.01:
            cur = max(cur, c)
        elif cur > 0:
            blocks.append(cur)
            cur = 0.0
    if cur > 0:
        blocks.append(cur)
    total = 1.0
    for bk in blocks:
        total *= (1 - bk)
    total = 1 - total
    low = max((v for p, v in levels.items() if p >= 800), default=0.0)
    mid = max((v for p, v in levels.items() if 450 <= p < 800), default=0.0)
    return round(total, 4), round(low + mid * (1 - low), 4)


# ── R030 등압면 CLDFRA (std NC per-hf) -> 1h 운량 -- 유일 경로 (2026-07-17 채택) ──
# GRIB 시계열(data=P) 경로는 게이트웨이 30s 컷 신설로 대청크가 불가능해졌다
# (위 _GRIB_EF_CHUNK 주석).  std NC per-hf 는 콜당 ~1s 에 24레벨이 한 번에 오고
# 병렬 안전이 실측됨: 8워커로 지점당 16.4s, 118/118 완전 (13시대 낮 측정).
# CLDFRA(std) == frcc(GRIB) 동일 필드는 new_kma nb05 값 대조 검증(0.978==0.978).
# GRIB 폴백은 두지 않는다(2026-07-17 사용자 결정: data=P GRIB 은 순차 강제 +
# 콜당 ~10s 라 폴백 가치가 낮음).  결측 안전장치 = 순차 재시도 2라운드 +
# 수집기 커버리지 경고(rc=1) + 재실행 치유(COALESCE upsert, idempotent).
CLDFRA_WORKERS = 8


def _parse_cldfra_levels(body: str) -> dict[int, float]:
    """std data=P 응답 -> {hPa: cloud fraction}.  행 'TMFC TMEF VARN LEVEL VALUS NAME'."""
    levels: dict[int, float] = {}
    for ln in (body or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        q = s.split()
        if len(q) < 6:
            continue
        try:
            levels[int(q[3])] = float(q[4])
        except ValueError:
            continue
    return levels


def fetch_r030_cldfra_long(points: list[dict], base_utc: datetime, days: int,
                           max_hf_map: dict[int, int],
                           workers: int = CLDFRA_WORKERS) -> pd.DataFrame:
    """R030 등압면 CLDFRA per-hf 병렬 -> 결합 운량 long (전운량 유일 경로).

    카테고리·스키마는 fetch_r030_frcc_long 과 동일(CLOUD_TOTAL_R030 /
    CLOUD_MIDLOW_R030) -- 피벗·upsert 쪽은 두 경로를 구분하지 않는다.
    지점 순차 x hf 병렬(현행 fetch_kimg_land_long 패턴).
    """
    hf_list = hf_range_1h(base_utc, days, max_hf_map)
    if not hf_list:
        return pd.DataFrame(columns=[
            "base_datetime", "point_name", "fcst_datetime", "category", "fcst_value"])
    base_kst = base_utc.astimezone(KST)
    base_dt_str = base_kst.strftime("%Y-%m-%d %H:%M")
    base_label = base_utc.strftime("%Y%m%d%H") + " UTC"

    rows: list[tuple] = []
    for pt in points:
        t0 = time.time()
        prof: dict[int, dict[int, float]] = {}     # hf -> {hPa: frcc}

        def one(hf: int) -> tuple[int, dict[int, float]]:
            body = fetch_pt_std("KIMR", "R030", "CLDFRA", base_utc, hf,
                                pt.get("lat"), pt.get("lon"),
                                pt.get("x"), pt.get("y"), data="P")
            return hf, (_parse_cldfra_levels(body) if body else {})

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for hf, levels in ex.map(one, hf_list):
                if len(levels) >= 12:               # 반쪽 프로파일 배제 (frcc 와 동일)
                    prof[hf] = levels
        n_par = len(prof)
        # 결측 안전장치: 순차 재시도 최대 2라운드 (병렬 burst 에 밀린 hf 회수).
        # fetch_pt_std 자체가 5xx 백오프·키 회전을 하므로 여기는 라운드만 더한다.
        # 그래도 남으면 호출자(collect_cloud_kimr)가 커버리지 경고 -> 재실행 치유.
        for rnd in range(2):
            missing = [h for h in hf_list if h not in prof]
            if not missing:
                break
            if rnd:
                time.sleep(2)
            for hf in missing:
                hf2, levels = one(hf)
                if len(levels) >= 12:
                    prof[hf2] = levels

        for hf in sorted(prof):
            total, midlow = _frcc_combine(prof[hf])
            fcst_dt_str = (base_utc + timedelta(hours=hf)).astimezone(KST).strftime(
                "%Y-%m-%d %H:%M")
            rows.append((base_dt_str, pt["name"], fcst_dt_str,
                         "CLOUD_TOTAL_R030", total))
            rows.append((base_dt_str, pt["name"], fcst_dt_str,
                         "CLOUD_MIDLOW_R030", midlow))
        retry_note = f" +재시도 {len(prof) - n_par}" if len(prof) > n_par else ""
        print(f"  CLDFRA {base_label} {pt['name']:<22} 병렬 {n_par}/{len(hf_list)}"
              f"{retry_note}  최종 {len(prof)}/{len(hf_list)} ({time.time() - t0:.1f}s)")
    df = pd.DataFrame(rows, columns=[
        "base_datetime", "point_name", "fcst_datetime", "category", "fcst_value"])
    if not df.empty:
        df["fcst_value"] = pd.to_numeric(df["fcst_value"], errors="coerce")
    return df


# ── 스모크 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 실호출 스모크: R030/L010 각 1지점 1hf -- 파싱·GHI 검산
    # (구 2번 항목 '피벗 동일성' 은 육지 kimg_land_long_to_wide 대조였으므로
    #  육지 경로 삭제와 함께 폐지 — 2026-07-21)
    base = ckg.latest_published_base(datetime.now(KST)).replace(hour=12)
    if base > datetime.now(UTC) - timedelta(hours=3):
        base -= timedelta(days=1)
    print(f"[smoke] base = {base}")
    for grp, nwp, nm in (("KIMR", "R030", R030_NAME_LAND), ("KIML", "L010", L010_NAME)):
        body = fetch_pt_std(grp, nwp, nm, base, 18,
                            POINTS_JEJU_V2[2]["lat"], POINTS_JEJU_V2[2]["lon"])
        raw = parse_pt_std(body or "")
        cats = derive_r030_categories(raw)
        missing = [v for v in nm.split(",") if v not in raw]
        print(f"[smoke] {nwp}: raw {len(raw)}/{len(nm.split(','))} vars"
              f"{' (누락: ' + ','.join(missing) + ')' if missing else ''}, "
              f"cats={sorted(cats)}")
        assert not missing, f"{nwp} 변수 누락: {missing}"
        assert abs(cats["SOLAR_RAD"] - (cats["RAD_DIRECT"] + cats["RAD_DIFFUSE"])) < 2e-4, \
            "GHI != DIR + DIF"
    print("[smoke] 실호출 파싱·GHI 검산 OK")
