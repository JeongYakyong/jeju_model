"""kpx_asos.py -- 실측(`historical`) 소스 fetch: KPX 전력시장 + KMA ASOS 관측.

예보가 아니라 **일어난 일**을 받는 경로다.  목적지는 전부 `historical` 테이블이고
`collect_historical.py` 가 유일한 호출자다 (예보 수집기는 KPX DA 만 쓴다).

  [KPX]  fetch_kpx_jeju        chejusukub 수급·신재생 (*_jeju)
         fetch_kpx_est         하루전 SMP + 예상수요 (*_da, 제주·육지)
         fetch_kpx_jeju_rt_smp 제주 실시간시장 SMP (구간 4개 + 평균)
  [KMA]  fetch_asos            제주 ASOS 3지점 관측 (kma_kimg 의 station primitive 사용)

ASOS 는 KMA 소스지만 예보가 아니라 관측이라 예보 fetcher(kma_kimr_*/kma_kimg)가 아니라
여기 있다 -- 파일을 가르는 축이 "출처"가 아니라 **예보냐 실측이냐**이기 때문이다.
"""
from __future__ import annotations

import io
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from kma_kimg import (
    _fetch_asos_one_station_chunk,
    _chunk_date_range,
    _decode_kpx,
    current_kma_key,
    KPX_API_KEY,
    KPX_BASE_HEADERS,
    MAX_CHUNK_DAYS,
    DEFAULT_SLEEP_SEC,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# 제주 ASOS 3지점 관측 (구 collect_kpx_asos_data.fetch_asos).  kma_kimg 의
# station primitive(_fetch_asos_one_station_chunk)를 3 지점에 적용해 wide 로 반환.
# ════════════════════════════════════════════════════════════════════════
# solar: 일사(SI) 센서 유무.  고산(185)/남쪽(189) O, 성산(188) 없음.
ASOS_STATIONS = [
    {"stn_id": 185, "suffix": "west",  "solar": True},   # 고산
    {"stn_id": 188, "suffix": "east",  "solar": False},  # 성산 (일사 센서 없음)
    {"stn_id": 189, "suffix": "south", "solar": True},   # 남쪽 태양광 단지 인근 ASOS
]


def fetch_asos(
    start_date: str,
    end_date: str,
    auth_key: str | None = None,
    chunk_days: int = MAX_CHUNK_DAYS,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    progress: bool = True,
) -> pd.DataFrame:
    """3 ASOS 지점 관측 -> wide DataFrame.  컬럼명: <var>_<west|east|south>.

    수집 변수 (지점별 9개; 적설 제외): temp_c / humidity / total_cloud /
    midlow_cloud / wind_spd / wd_sin / wd_cos / solar_rad / rainfall.
    solar_rad 는 센서가 있는 지점만 (west/south).  성산(east)은 컬럼 자체가 없다.
    결측 정책: rainfall->0, solar_rad 는 센서 가동일의 야간 결측만 0, 그 외 NaN 유지.
    """
    key = auth_key or current_kma_key()
    if not key:
        sys.exit("KMA_API_KEY is not set (check .env)")

    per_station: dict[str, list[pd.DataFrame]] = {
        st["suffix"]: [] for st in ASOS_STATIONS
    }
    for s, e in _chunk_date_range(start_date, end_date, chunk_days):
        for st in ASOS_STATIONS:
            try:
                df = _fetch_asos_one_station_chunk(s, e, st["stn_id"], key)
                if progress:
                    print(
                        f"  [asos] {s} ~ {e}  stn={st['stn_id']} "
                        f"({st['suffix']:<5})  rows={len(df)}"
                    )
                if not df.empty:
                    per_station[st["suffix"]].append(df)
            except Exception as ex:
                print(
                    f"  [asos] {s} ~ {e}  stn={st['stn_id']} "
                    f"({st['suffix']})  FAIL: {ex}"
                )
            time.sleep(sleep_sec)

    parts: list[pd.DataFrame] = []
    for st in ASOS_STATIONS:
        chunks = per_station[st["suffix"]]
        if not chunks:
            continue
        df_st = (
            pd.concat(chunks)
              .reset_index()
              .drop_duplicates(subset="timestamp")
              .set_index("timestamp")
              .sort_index()
        )
        # 일사(SI) 처리: 센서 없는 지점(성산)은 컬럼을 버린다.  센서 있는 지점은
        # '그 날 일사를 한 번이라도 보고했는가'로 가동일을 판정해 가동일의 NaN(=야간)
        # 만 0 으로 채우고, 비가동일(무센서 기간)의 NaN 은 그대로 둔다.
        if not st["solar"]:
            df_st = df_st.drop(columns="solar_rad", errors="ignore")
        else:
            sr = df_st["solar_rad"]
            day_key = pd.to_datetime(df_st.index).normalize()
            sensor_day = sr.notna().groupby(day_key).transform("any")
            df_st["solar_rad"] = sr.mask(sr.isna() & sensor_day, 0.0)
        df_st.columns = [f"{c}_{st['suffix']}" for c in df_st.columns]
        parts.append(df_st)

    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "timestamp"
    return wide


# ========================================================================
# KPX 제주 fetcher (구 kpx_fetcher_jeju 통합).  수급(*_jeju) + DA SMP/est(*_da).
# ========================================================================
KPX_JEJU_URL = "https://openapi.kpx.or.kr/downloadChejuSukubCSV.do"
KPX_EST_URL = (
    "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
)
# 제주 실시간시장 SMP/수요 (data.go.kr B552115).  date 별 96행(24h x 4구간) 반환.
# jsmpRt = 구간별 실시간 SMP -> 시간평균이 모델 타깃(smp_jeju_rt).
KPX_JEJU_RT_URL = (
    "https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2"
)


# ── 1. KPX 제주 (chejusukub 수급 + 신재생, historical) ─────────────────
# 컬럼명은 모두 *_jeju suffix -- 'sukub' 의 *_land 와 짝을 이뤄 동일 변수가 두 계통에
# 충돌 없이 같은 wide DF / DB 에 공존 (supply_cap_land vs supply_cap_jeju).
_KPX_JEJU_RENAME = {
    "공급능력(MW)":     "supply_cap_jeju",
    "현재수요(MW)":     "real_demand_jeju",
    "신재생총합(MW)":   "real_renew_gen_jeju",
    "신재생태양광(MW)": "real_solar_gen_jeju",
    "신재생풍력(MW)":   "real_wind_gen_jeju",
}
_KPX_JEJU_POWER_COLS = list(_KPX_JEJU_RENAME.values())


def _fetch_kpx_jeju_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    headers = {**KPX_BASE_HEADERS, "Referer": "https://openapi.kpx.or.kr/chejusukub.do"}
    resp = requests.post(
        KPX_JEJU_URL,
        data={"startDate": start_date, "endDate": end_date},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(_decode_kpx(resp.content)))
    df.columns = df.columns.str.strip()
    if df.empty or "기준일시" not in df.columns:
        return pd.DataFrame()
    df = df[df["기준일시"].astype(str).str.endswith("0000")].copy()
    df["timestamp"] = pd.to_datetime(
        df["기준일시"].astype(str), format="%Y%m%d%H%M%S"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.rename(columns=_KPX_JEJU_RENAME)
    cols = ["timestamp"] + [c for c in _KPX_JEJU_POWER_COLS if c in df.columns]
    df = df[cols].copy()
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_kpx_jeju(
    start_date: str,
    end_date: str,
    chunk_days: int = MAX_CHUNK_DAYS,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    progress: bool = True,
) -> pd.DataFrame:
    """KPX 제주(chejusukub) 1h 수급 + 신재생을 wide DataFrame 으로 반환.

    Returns 컬럼 (5 cols, 모두 *_jeju suffix):
        supply_cap_jeju / real_demand_jeju / real_renew_gen_jeju /
        real_solar_gen_jeju / real_wind_gen_jeju.

    real_demand_jeju == 0 은 계측 오류 (제주 수요가 0 이 될 수 없음) -> 해당 행
    전체를 NaN 처리 후 양방향 시간보간 (최대 3 연속).
    """
    chunks: list[pd.DataFrame] = []
    for s, e in _chunk_date_range(start_date, end_date, chunk_days):
        try:
            df = _fetch_kpx_jeju_chunk(s, e)
            if progress:
                print(f"  [*_jeju] {s} ~ {e}  rows={len(df)}")
            if not df.empty:
                chunks.append(df)
        except Exception as ex:
            print(f"  [*_jeju] {s} ~ {e}  FAIL: {ex}")
        time.sleep(sleep_sec)
    if not chunks:
        return pd.DataFrame()
    df = (
        pd.concat(chunks, ignore_index=True)
          .drop_duplicates(subset="timestamp")
          .sort_values("timestamp")
          .set_index("timestamp")
    )

    # demand=0 보정 (전체 컬럼 NaN -> 시간 보간).
    zero_mask = df["real_demand_jeju"] == 0
    if zero_mask.any():
        print(
            f"  [*_jeju] demand=0 sensor errors: {zero_mask.sum()} rows "
            f"-> NaN + time interpolate (limit=3)"
        )
        df.loc[zero_mask, _KPX_JEJU_POWER_COLS] = np.nan
        df.index = pd.to_datetime(df.index)
        df = df.interpolate(method="time", limit=3, limit_direction="both")
        df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    df.index.name = "timestamp"
    return df


# ── 2. KPX est (일전 SMP + 예상수요 제주/육지, FORECAST 테이블) ─────────
# 이 fetcher 의 출력은 collect_data_new 의 forecast 테이블로 들어간다.
# 컬럼은 모두 _da 접미사 (day-ahead 의미) -- smp_jeju_da / smp_land_da /
# jeju_est_demand_da / land_est_demand_da.  jeju_/land_ 접두사를 붙여 두 권역의
# 예상수요를 같은 wide DF / DB 에 충돌 없이 공존시킨다.  historical 테이블의
# 실현치(별도 컬럼)와도 이름이 안 겹치고, legacy DB 의 SMP/예상수요 컬럼을 같은
# 이름으로 양 테이블에 매핑할 수 있다.
def _fetch_kpx_est_one_day(target_date: str, service_key: str) -> pd.DataFrame:
    """하루치 (24h × 제주+육지) 호출 -> wide DF (timestamp index).

    target_date : 'YYYY-MM-DD'.  API 는 'YYYYMMDD' 로 받으므로 내부에서 변환.
    API 발행 시점: 전날 23:00 KST 이후 다음날치가 올라온다 (예: 05-27 23:00 ->
    05-28 데이터).  미래 날짜는 빈 응답.
    """
    params = {
        "serviceKey": service_key,
        "dataType": "json",
        "date": target_date.replace("-", ""),
        "numOfRows": "100",  # 24h x 2 areas = 48 rows, 100 이면 충분.
    }
    resp = requests.get(KPX_EST_URL, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json().get("response", {}).get("body", {}) or {}
    items = (body.get("items") or {}).get("item")
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    if df.empty or "areaName" not in df.columns:
        return pd.DataFrame()
    # 값 컬럼(smp/jlfd/mlfd) 중 일부가 응답에서 빠져도 KeyError 로 그 날짜 전체를
    # 잃지 않도록 방어적으로 채운다 (없으면 NaN -> 해당 _da 만 결측).
    for _col in ("smp", "jlfd", "mlfd"):
        if _col not in df.columns:
            df[_col] = pd.NA

    # 제주 행: smp_jeju_da + jeju_est_demand_da(jlfd) 만 사용.
    # 육지 행: smp_land_da + land_est_demand_da(mlfd) 만 사용.
    # slfd = jlfd + mlfd 라 별도 저장 안 함.
    df_jeju = df[df["areaName"] == "제주"][["date", "hour", "smp", "jlfd"]].rename(
        columns={"smp": "smp_jeju_da", "jlfd": "jeju_est_demand_da"}
    )
    df_land = df[df["areaName"] == "육지"][["date", "hour", "smp", "mlfd"]].rename(
        columns={"smp": "smp_land_da", "mlfd": "land_est_demand_da"}
    )
    merged = pd.merge(df_jeju, df_land, on=["date", "hour"], how="outer")
    # KPX 의 hour 는 1..24 -> 00:00 ~ 23:00 으로 매핑 (hour-1).
    merged["timestamp"] = (
        pd.to_datetime(merged["date"], format="%Y%m%d")
        + pd.to_timedelta(merged["hour"].astype(int) - 1, unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    out_cols = [
        "smp_jeju_da", "smp_land_da",
        "jeju_est_demand_da", "land_est_demand_da",
    ]
    out = merged[["timestamp", *out_cols]].copy()
    for c in out_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.set_index("timestamp").sort_index()


def fetch_kpx_est(
    start_date: str,
    end_date: str,
    service_key: str | None = None,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    progress: bool = True,
) -> pd.DataFrame:
    """일전(DA) SMP(제주/육지) + 제주/육지 예상수요를 wide DataFrame 으로 반환.

    API 는 일 단위 호출(target_date 1개) 이라 [start..end] 의 각 날짜에 1회씩
    호출한다.  발행되지 않은 미래 일자는 빈 응답 -> 해당 날짜는 결과에서 누락.

    Returns 컬럼 (4 cols, 모두 *_da suffix):
        smp_jeju_da / smp_land_da / jeju_est_demand_da / land_est_demand_da
    -> forecast 테이블 행.  build_historical 도 같은 컬럼을 historical 에 누적.
    """
    key = service_key or KPX_API_KEY
    if not key:
        sys.exit("KPX_API_KEY is not set (check .env)")
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    if s > e:
        raise ValueError(f"start_date ({start_date}) > end_date ({end_date})")

    days: list[pd.DataFrame] = []
    cur = s
    while cur <= e:
        d_str = cur.strftime("%Y-%m-%d")
        try:
            df = _fetch_kpx_est_one_day(d_str, key)
            if progress:
                print(f"  [*_da] {d_str}  rows={len(df)}")
            if not df.empty:
                days.append(df)
        except Exception as ex:
            print(f"  [*_da] {d_str}  FAIL: {ex}")
        cur += timedelta(days=1)
        if cur <= e:
            time.sleep(sleep_sec)
    if not days:
        return pd.DataFrame()
    out = pd.concat(days).sort_index()
    out.index.name = "timestamp"
    return out


# ── 3. KPX 제주 실시간시장 SMP (RT SMP, HISTORICAL 타깃) ─────────────────
# 제주 실시간시장(시범사업)의 구간별(15분=4구간/h) 실시간 SMP/수요.  getJejuSmpLfd2.
# 저장 정책 (2026-06-03 변경): 구간별 원시값(smp_rt_g1..g4)을 그대로 저장하고,
# 파생은 *필요할 때* 계산해 쓴다.  저장 단계에서 함께 넣는 파생 2종:
#   smp_jeju_rt    = mean(g1..g4)                 (시간평균 RT SMP, 4단계 모델 타깃)
#   smp_rt_neg_num = count(g1..g4 < NEG_THRESHOLD) (음수권 구간 개수 0..4)
# (구 smp_rt_neg_flag = any(g<0) boolean 은 폐기.  음수 기준도 0 -> 5 로 변경:
#  실시간시장 SMP 가 5 미만이면 사실상 바닥/음수권으로 보고 구간 개수를 센다.)
# 출력은 historical 테이블 전용 (RT 는 실현치라 forecast 엔 불필요).
#
# 구간 식별: 응답 gugan 라벨은 EUC-KR 깨짐이나 선두 숫자(1..4)가 구간 인덱스라
# 그 숫자만 뽑아 g1..g4 로 피벗한다 (라벨 본문은 무시).
#
# 제약(사용자 메모, 2026-06-02):
#   - 매일 23:00 KST 발행, 단 KPX API 불안정으로 지연 가변(최대 익일 18:00 관측).
#     -> 미발행/지연 날짜는 빈 응답.  partial_upsert 의 COALESCE 가 기존값 보존하므로
#        빈 날짜를 매 실행 재시도해도 안전 (clobber 없음).
#   - smp_jeju_rt 가 없을 때의 da 대체(smp_jeju_da)는 *서빙/학습 레이어*의 제한적
#     사용이며, 저장 단계에선 RT 를 순수 유지(없으면 컬럼 NULL).  여기선 대체 안 함.
_JEJU_RT_GUGAN = ["smp_rt_g1", "smp_rt_g2", "smp_rt_g3", "smp_rt_g4"]
_JEJU_RT_MEAN = "smp_jeju_rt"
_JEJU_RT_NEG_NUM = "smp_rt_neg_num"
# 한 구간이 이 값 미만이면 '음수권'으로 카운트 (smp_rt_neg_num).  과거 boolean
# flag 는 <0 기준이었으나, 바닥권(0~5) 도 음수 위험 신호라 임계를 5 로 올렸다.
_JEJU_RT_NEG_THRESHOLD = 5.0


def _fetch_jeju_rt_smp_one_day(
    target_date: str, service_key: str, retry: int = 3,
) -> pd.DataFrame:
    """하루치(24h x 4구간) 호출 -> 구간별 + 파생 wide DF (timestamp index).

    target_date : 'YYYY-MM-DD'.  API 는 'YYYYMMDD'.
    반환 컬럼: smp_rt_g1..g4(구간 원시 RT SMP), smp_jeju_rt(시간평균),
              smp_rt_neg_num(구간 중 <NEG_THRESHOLD 개수 0..4).
    미발행/빈 응답이면 빈 DF.  5xx/네트워크 오류는 backoff 재시도(KPX 불안정 대응).
    """
    params = {
        "serviceKey": service_key,
        "dataType": "json",
        "date": target_date.replace("-", ""),
        "pageNo": "1",
        "numOfRows": "200",   # 24h x 4구간 = 96 < 200, 한 번에 전량.
    }
    body = None
    for attempt in range(retry):
        try:
            resp = requests.get(KPX_JEJU_RT_URL, params=params, timeout=30)
            if resp.status_code == 200:
                # data.go.kr 은 오류 시 XML 을 주기도 함 -> JSON 파싱 실패는 빈 응답 취급.
                try:
                    body = resp.json().get("response", {}).get("body", {}) or {}
                except ValueError:
                    return pd.DataFrame()
                break
            if 500 <= resp.status_code < 600 and attempt < retry - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return pd.DataFrame()
        except (requests.Timeout, requests.ConnectionError):
            if attempt < retry - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    if not body:
        return pd.DataFrame()

    items = (body.get("items") or {}).get("item")
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    needed = {"hour", "jsmpRt", "gugan"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["jsmpRt"] = pd.to_numeric(df["jsmpRt"], errors="coerce")
    # gugan 라벨은 EUC-KR 깨짐('1����' 등)이나 선두 숫자가 구간 인덱스(1..4).
    df["gnum"] = pd.to_numeric(
        df["gugan"].astype(str).str.extract(r"^\s*(\d)")[0], errors="coerce"
    )
    df = df.dropna(subset=["hour", "jsmpRt", "gnum"])
    df = df[df["gnum"].between(1, 4)]
    if df.empty:
        return pd.DataFrame()
    df["gnum"] = df["gnum"].astype(int)

    # hour x 구간 피벗 -> g1..g4 (한 시간에 4구간).  중복 구간이 와도 평균으로 합침.
    pivot = df.pivot_table(
        index="hour", columns="gnum", values="jsmpRt", aggfunc="mean",
    ).reindex(columns=[1, 2, 3, 4])
    pivot.columns = _JEJU_RT_GUGAN

    out = pivot.round(4)
    # 파생: 시간평균(타깃) + 음수권 구간 개수.  g 가 일부 NaN 이어도 안전하게 집계.
    out[_JEJU_RT_MEAN] = pivot.mean(axis=1).round(4)
    out[_JEJU_RT_NEG_NUM] = (
        (pivot < _JEJU_RT_NEG_THRESHOLD).sum(axis=1).astype(int)
    )

    # KPX hour 1..24 -> 00:00 ~ 23:00 (hour-1).
    out.index = (
        pd.to_datetime(target_date)
        + pd.to_timedelta(out.index.astype(int) - 1, unit="h")
    ).strftime("%Y-%m-%d %H:%M:%S")
    out.index.name = "timestamp"
    return out.sort_index()


def fetch_kpx_jeju_rt_smp(
    start_date: str,
    end_date: str,
    service_key: str | None = None,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    progress: bool = True,
) -> pd.DataFrame:
    """제주 실시간시장 RT SMP 를 [start..end] 일별 호출 -> wide DataFrame.

    Returns 컬럼 (6 cols, historical 전용):
        smp_rt_g1..g4    : 구간별 원시 실시간 SMP
        smp_jeju_rt      : 구간 평균 실시간 SMP (모델 타깃)
        smp_rt_neg_num   : 그 시간 4구간 중 <NEG_THRESHOLD 인 구간 개수 (0..4)

    미발행 미래/지연 일자는 빈 응답 -> 결과에서 누락(해당 시간 컬럼 NULL 유지).
    """
    key = service_key or KPX_API_KEY
    if not key:
        sys.exit("KPX_API_KEY is not set (check .env)")
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    if s > e:
        raise ValueError(f"start_date ({start_date}) > end_date ({end_date})")

    days: list[pd.DataFrame] = []
    cur = s
    while cur <= e:
        d_str = cur.strftime("%Y-%m-%d")
        try:
            df = _fetch_jeju_rt_smp_one_day(d_str, key)
            if progress:
                # 음수권(neg_num>0) 시간 수 -- 발행 품질 한눈 확인용.
                n_neg = int((df[_JEJU_RT_NEG_NUM] > 0).sum()) if not df.empty else 0
                print(f"  [rt_smp] {d_str}  rows={len(df)}  neg_hours={n_neg}")
            if not df.empty:
                days.append(df)
        except Exception as ex:
            print(f"  [rt_smp] {d_str}  FAIL: {ex}")
        cur += timedelta(days=1)
        if cur <= e:
            time.sleep(sleep_sec)
    if not days:
        return pd.DataFrame()
    out = pd.concat(days).sort_index()
    out.index.name = "timestamp"
    return out


# NOTE: ASOS 는 KMA 관측이라 예보 fetcher 가 아니라 이 파일(실측 소스)에 있다.
# 저수준 chunk 호출은 kma_kimg._fetch_asos_one_station_chunk 가 맡는다.
