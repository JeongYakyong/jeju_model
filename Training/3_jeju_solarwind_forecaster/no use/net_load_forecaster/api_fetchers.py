import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone
import time
import io
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger('jejucr.api')
logger.setLevel(logging.DEBUG)


# ============================================================================
# KMA NCM API — persistent session with connection pooling
# ============================================================================
_kma_session = requests.Session()
_kma_session.mount("https://apihub.kma.go.kr/", HTTPAdapter(
    pool_connections=4,
    pool_maxsize=20,
))

def warmup_kma_session(auth_key):
    """
    Pre-establish TCP+TLS connection to KMA API server.
    Cheap when connections are alive (~100ms), valuable when
    they've expired after idle (~500ms vs cold parallel burst).
    Called before every NCM batch — no flag needed.
    """
    try:
        _kma_session.get(
            "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-kim_nc_pt_txt2",
            params={'help': '1', 'authKey': auth_key},
            timeout=10,
        )
    except Exception:
        pass


# ============================================================================
# 1. KPX API - 계통 데이터
# ============================================================================

def fetch_kpx_past(start_date, end_date):
    url = "https://openapi.kpx.or.kr/downloadChejuSukubCSV.do"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://openapi.kpx.or.kr/chejusukub.do"
    }
    payload = {
        'startDate': start_date,  # 하이픈 포함 그대로
        'endDate': end_date
    }
    
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        
        # 1시간 단위 필터링 (0000으로 끝나는 행)
        df = df[df['기준일시'].astype(str).str.endswith('0000')].copy()
        
        # timestamp 생성
        df['timestamp'] = pd.to_datetime(
            df['기준일시'].astype(str), 
            format='%Y%m%d%H%M%S'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 컬럼명 변경
        df = df.rename(columns={
            '공급능력(MW)': 'supply_cap',
            '현재수요(MW)': 'real_demand',
            '신재생총합(MW)': 'real_renew_gen',
            '신재생태양광(MW)': 'real_solar_gen',
            '신재생풍력(MW)': 'real_wind_gen'
        })
        
        # 필요한 컬럼만 선택 및 숫자 변환
        power_cols = ['supply_cap', 'real_demand', 'real_renew_gen',
                      'real_solar_gen', 'real_wind_gen']
        result = df.set_index('timestamp')[power_cols].apply(pd.to_numeric, errors='coerce')

        # demand=0 은 계측 오류 (실제 수요가 0이 될 수 없음)
        # → 해당 행 전체를 NaN 처리 후 양방향 보간 (최대 3개 연속)
        zero_mask = result['real_demand'] == 0
        if zero_mask.any():
            result.loc[zero_mask, power_cols] = np.nan
            logger.warning(f"[KPX Past] demand=0 오류 {zero_mask.sum()}행 → NaN 양방향 보간")
        result.index = pd.to_datetime(result.index)
        result = result.interpolate(method='time', limit=3, limit_direction='both')
        result.index = result.index.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"[KPX Past] {len(result)}행 수집")
        return result
        
    except Exception as e:
        logger.error(f"[KPX Past] 실패: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=32)
def fetch_kpx_past_15min(start_date, end_date):
    """
    KPX 실측 발전량 15분 단위 데이터 (시각화 전용, 30분 캐시).
    fetch_kpx_past와 동일한 API를 호출하되 15분 간격(00·15·30·45분)만 유지.
    DB 저장·모델 입력에는 fetch_kpx_past를 사용.
    """
    url = "https://openapi.kpx.or.kr/downloadChejuSukubCSV.do"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://openapi.kpx.or.kr/chejusukub.do"
    }
    payload = {'startDate': start_date, 'endDate': end_date}

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        # 15분 단위 필터링 (00·15·30·45분)
        df['기준일시'] = df['기준일시'].astype(str)
        df = df[df['기준일시'].str[-4:].isin(['0000', '1500', '3000', '4500'])].copy()

        df['timestamp'] = pd.to_datetime(
            df['기준일시'], format='%Y%m%d%H%M%S'
        )

        df = df.rename(columns={
            '공급능력(MW)': 'supply_cap',
            '현재수요(MW)': 'real_demand',
            '신재생총합(MW)': 'real_renew_gen',
            '신재생태양광(MW)': 'real_solar_gen',
            '신재생풍력(MW)': 'real_wind_gen'
        })

        power_cols = ['supply_cap', 'real_demand', 'real_renew_gen',
                      'real_solar_gen', 'real_wind_gen']
        result = df.set_index('timestamp')[power_cols].apply(pd.to_numeric, errors='coerce')

        # demand=0 오류 → NaN 양방향 보간
        zero_mask = result['real_demand'] == 0
        if zero_mask.any():
            result.loc[zero_mask, power_cols] = np.nan
            logger.warning(f"[KPX 15min] demand=0 오류 {zero_mask.sum()}행 → NaN 양방향 보간")
        result = result.interpolate(method='time', limit=3, limit_direction='both')

        logger.info(f"[KPX 15min] {len(result)}행 수집")
        return result

    except Exception as e:
        logger.error(f"[KPX 15min] 실패: {e}")
        return pd.DataFrame()


def fetch_kpx_future(target_date, service_key):
    url = 'https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand'
    params = {
        'serviceKey': service_key, 
        'dataType': 'json', 
        'date': target_date.replace('-', ''), 
        'numOfRows': '100'
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        items = resp.json()['response']['body']['items']['item']
        df = pd.DataFrame(items)
        
        if df.empty:
            logger.warning(f"[KPX Future] {target_date} 데이터가 없습니다.")
            return pd.DataFrame()

        # 1. 제주 데이터와 육지 데이터를 각각 분리합니다.
        df_jeju = df[df['areaName'] == '제주'].copy()
        df_land = df[df['areaName'] == '육지'].copy()
        
        # 2. 필요한 컬럼만 남기고 이름 변경 (제주수요: jlfd -> est_demand)
        df_jeju = df_jeju[['date', 'hour', 'smp', 'jlfd']].rename(columns={'smp': 'smp_jeju', 'jlfd': 'est_demand'})
        df_land = df_land[['date', 'hour', 'smp']].rename(columns={'smp': 'smp_land'})
        
        # 3. 날짜와 시간을 기준으로 두 데이터를 가로로 예쁘게 합칩니다.
        df_merged = pd.merge(df_jeju, df_land, on=['date', 'hour'], how='outer')
        
        # 4. timestamp 생성 (hour 1~24 -> 00~23)
        df_merged['timestamp'] = (
            pd.to_datetime(df_merged['date'], format='%Y%m%d') + 
            pd.to_timedelta(df_merged['hour'].astype(int) - 1, unit='h')
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 5. 최종 데이터 정리 및 숫자형 변환
        
        result = df_merged[['timestamp', 'smp_jeju', 'smp_land', 'est_demand']].copy()
        result['smp_jeju'] = pd.to_numeric(result['smp_jeju'], errors='coerce')
        result['smp_land'] = pd.to_numeric(result['smp_land'], errors='coerce')
        result['est_demand'] = pd.to_numeric(result['est_demand'], errors='coerce')
        
        result = result.set_index('timestamp')
        
        return result
        
    except Exception as e:
        logger.error(f"[KPX Future] 실패: {e}")
        return pd.DataFrame()
    
def fetch_kpx_historical(start_date, end_date, service_key):
    all_smp = []
    
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_days = (end - current).days + 1
    logger.info(f"[KPX Historical] {total_days}일치 수집 시작...")
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        try:
            # 과거 날짜로 future API 호출
            df = fetch_kpx_future(date_str, service_key)
            
            if not df.empty and 'smp_jeju' in df.columns and 'smp_land' in df.columns and 'est_demand' in df.columns:
                all_smp.append(df[['est_demand', 'smp_jeju', 'smp_land']])
                logger.info(f"  {date_str}: {len(df)}행")
            else:
                logger.warning(f"  {date_str}: 데이터 없음")
                
        except Exception as e:
            logger.error(f"  {date_str}: 실패 ({e})")
        
        current += timedelta(days=1)
    
    if not all_smp:
        logger.warning("[KPX SMP] 전체 데이터 없음")
        return pd.DataFrame()
    
    result = pd.concat(all_smp)
    logger.info(f"[KPX SMP] 총 {len(result)}행 수집 완료")
    return result


# ============================================================================
# 2. KMA API - 기상 데이터
# ============================================================================
# ============================================================================
# ASOS 실측 풍속 2권역 가중평균 (api_fetchers.py 에 추가)
# ============================================================================

_ZONE_EAST = "동쪽"
_ZONE_WEST = "서쪽"

# 예보(WIND_ZONES)와 동일한 용량 비율 사용
ASOS_WIND_STATIONS = {
    _ZONE_EAST: {"stn_id": 188, "capacity_mw": 201.1},   # 성산
    _ZONE_WEST: {"stn_id": 185, "capacity_mw": 217.8},   # 고산
}


def _apply_zone_data(zone_data, row_dict):
    """zone_data의 동쪽/서쪽 값을 row_dict의 east/west 컬럼으로 기록."""
    for key, direction in [(_ZONE_EAST, "east"), (_ZONE_WEST, "west")]:
        if key in zone_data:
            spd, sin_v, cos_v = zone_data[key]
            row_dict[f"wind_spd_{direction}"] = spd
            row_dict[f"wd_sin_{direction}"]   = sin_v
            row_dict[f"wd_cos_{direction}"]   = cos_v
ASOS_WIND_TOTAL = sum(s["capacity_mw"] for s in ASOS_WIND_STATIONS.values())


def fetch_kma_past_asos_wind(start_date, end_date, auth_key):
    """
    성산(188) + 고산(185) ASOS 풍속을 용량 가중 평균하여
    wind_spd_north, wd_sin_north, wd_cos_north 를 산출.
    
    출력 형태는 기존과 동일 — 3개 컬럼의 DataFrame (timestamp 인덱스).
    """
    station_dfs = []

    for name, info in ASOS_WIND_STATIONS.items():
        logger.info(f"[ASOS Wind-{name}] stn_id={info['stn_id']} 수집 중...")
        df = fetch_kma_past_asos(start_date, end_date, auth_key, stn_id=info["stn_id"])
        if df.empty:
            logger.warning(f"[ASOS Wind-{name}] 데이터 없음 — 스킵")
            continue
        weight = info["capacity_mw"] / ASOS_WIND_TOTAL
        station_dfs.append((name, weight, df))

    if not station_dfs:
        logger.error("[ASOS Wind] 모든 관측소 수집 실패")
        return pd.DataFrame()

    # 벡터 가중 평균 (예보와 동일 방식)
    # wind_spd + wd_sin/cos → u, v 복원 → 가중합산 → 재합성
    base_index = station_dfs[0][2].index
    for _, _, df in station_dfs[1:]:
        base_index = base_index.union(df.index)
    base_index = base_index.sort_values()

    result_rows = []
    for ts in base_index:
        u_sum = 0.0
        v_sum = 0.0
        w_sum = 0.0
        zone_data = {}

        for name, weight, df in station_dfs:
            if ts in df.index:
                row = df.loc[ts]
                spd = row["wind_spd"]
                sin_val = row["wd_sin"]
                cos_val = row["wd_cos"]

                u_sum += weight * spd * sin_val
                v_sum += weight * spd * cos_val
                w_sum += weight
                zone_data[name] = (round(spd, 2), round(sin_val, 4), round(cos_val, 4))

        if w_sum == 0:
            continue

        u_avg = u_sum / w_sum
        v_avg = v_sum / w_sum

        wind_spd = np.sqrt(u_avg**2 + v_avg**2)
        wind_dir_rad = np.arctan2(u_avg, v_avg)

        row_dict = {
            "timestamp": ts,
            "wind_spd_north": round(wind_spd, 2),
            "wd_sin_north": round(np.sin(wind_dir_rad), 4),
            "wd_cos_north": round(np.cos(wind_dir_rad), 4),
        }
        _apply_zone_data(zone_data, row_dict)
        result_rows.append(row_dict)

    if not result_rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(result_rows).set_index("timestamp").sort_index()
    zone_names = ", ".join(n for n, _, _ in station_dfs)
    logger.info(f"[ASOS Wind] {len(station_dfs)}개 관측소 가중평균 완료 ({zone_names}), {len(df_result)}행")
    return df_result


def fetch_kma_past_asos(start_date, end_date, auth_key, stn_id=189):
    url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
    
    params = {
        "tm1": f"{start_date}0000",  
        "tm2": f"{end_date}2300",    
        "stn": str(stn_id),          # 파라미터로 받은 관측소 ID 사용
        "help": "0",
        "authKey": auth_key
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        # 주석(#) 제거하고 데이터만 추출
        lines = [l for l in resp.text.split('\n') if l.strip() and not l.startswith('#')]
        
        if not lines:
            logger.warning("[KMA ASOS] 데이터 없음")
            return pd.DataFrame()
        
        # 🔥 공백으로 split (고정폭 아님!)
        df_raw = pd.DataFrame([l.split() for l in lines])
        
        # 결측치 처리 함수
        def clean(val):
            try:
                v = float(val)
                return np.nan if v <= -9 else v
            except:
                return np.nan
        
        df = pd.DataFrame()
        
        # Timestamp (0번 컬럼)
        df['timestamp'] = pd.to_datetime(df_raw[0], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 기상 변수 (인덱스 기반)
        df['temp_c'] = df_raw[11].apply(clean)  # 기온
        df['humidity'] = df_raw[13].apply(clean)  # 습도
        
        # 구름 (0~10 → 0~1)
        df['total_cloud'] = df_raw[25].apply(clean) / 10
        df['midlow_cloud'] = df_raw[26].apply(clean) / 10
        
        # 풍향/풍속
        wind_spd = df_raw[3].apply(clean).fillna(0)
        wind_dir = df_raw[2].apply(clean).fillna(0) * 10  # 36방위 → 360도
        
        df['wind_spd'] = wind_spd.round(2)
        df['wd_sin'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos'] = np.cos(np.radians(wind_dir)).round(4)
        
        # 일사량 (MJ/m² → W/m²는 나중에 처리)
        solar_raw = df_raw[34].apply(clean).fillna(0)
        df['solar_rad'] = solar_raw.clip(lower=0)  # 음수 제거
        
        # 강수량
        df['rainfall'] = df_raw[15].apply(clean).fillna(0)
        
        # 적설량 (cm → m)
        df['snow_depth'] = df_raw[21].apply(clean).fillna(0) / 100
        
        result = df.set_index('timestamp')
        
        logger.info(f"[KMA ASOS] {len(result)}행 수집")
        return result
        
    except Exception as e:
        logger.error(f"[KMA ASOS] 실패: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_kma_future_ncm(lat, lon, auth_key, base_date_kst, as_of_kst=None):
    """
    지정 시점(as_of_kst) 기준 가장 최신 예보 사이클을 자동 선택하여
    해당 사이클이 커버하는 미래 시간대만 수집.

    예보 사이클: 00, 06, 12, 18 UTC (배포 지연 ~2h)
    호출할 때마다 forecast 테이블에 UPSERT되므로,
    여러 사이클로 반복 호출하면 각 시간대가 가장 가까운 예보로 갱신됨.

    Args:
        lat, lon: 관측 좌표
        auth_key: KMA API 인증키
        base_date_kst: "YYYY-MM-DD" 형식의 대상 날짜
        as_of_kst: 기준 시점 (None이면 현재시간, 과거 소급 시 지정)
                   "YYYY-MM-DD HH:MM" 또는 datetime 객체
    """
    from datetime import datetime, timezone, timedelta

    VARN_MAP = {
        51: 'solar_rad',
        25: 'temp_k',
        37: 'total_cloud',
        35: 'mid_cloud',
        34: 'low_cloud',
        20: 'u_wind',
        21: 'v_wind',
        41: 'snow_depth',
        26: 'humidity',
        65: 'rain_conv',
        66: 'rain_strat',
    }

    def parse_raw_text_by_varn(raw_text):
        parsed_dict = {}
        lines = raw_text.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                varn_code = int(parts[2])
                value = float(parts[4])
                if varn_code in VARN_MAP:
                    parsed_dict[VARN_MAP[varn_code]] = value
            except (ValueError, IndexError):
                continue
        return parsed_dict

    # ── 1) 기준 시점 결정 ──
    KST = timezone(timedelta(hours=9))

    if as_of_kst is None:
        ref_kst = datetime.now(KST)
    elif isinstance(as_of_kst, str):
        ref_kst = datetime.strptime(as_of_kst, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    else:
        ref_kst = as_of_kst if as_of_kst.tzinfo else as_of_kst.replace(tzinfo=KST)

    ref_utc = ref_kst.astimezone(timezone.utc)
    target_dt = pd.to_datetime(base_date_kst).tz_localize(KST)

    # ── 2) 최신 사용 가능 예보 사이클 계산 ──
    CYCLES_UTC = [0, 6, 12, 18]
    PUBLISH_DELAY_H = 2

    available_utc = ref_utc - timedelta(hours=PUBLISH_DELAY_H)

    candidates = []
    for day_offset in [0, -1, -2]:
        base_day = available_utc.date() + timedelta(days=day_offset)
        for cycle_h in CYCLES_UTC:
            cycle_dt = datetime(base_day.year, base_day.month, base_day.day,
                                cycle_h, 0, 0, tzinfo=timezone.utc)
            if cycle_dt <= available_utc:
                candidates.append(cycle_dt)

    if not candidates:
        logger.warning("[KMA NCM] 사용 가능한 예보 사이클 없음")
        return pd.DataFrame()

    best_cycle_utc = max(candidates)
    base_tmfc = best_cycle_utc.strftime('%Y%m%d%H')
    base_time_kst = best_cycle_utc.astimezone(KST)

    logger.info(f"[KMA NCM] 예보 기준: {base_tmfc} UTC "
                f"(기준시점: {ref_kst.strftime('%Y-%m-%d %H:%M')} KST)")

    # ── 3) 수집 대상 시간 범위 계산 ──
    # 항상 00시부터 전체 수집 (forecast 테이블을 빈틈없이 채움)
    # 모델 예측 시 ASOS 우선은 get_historical_and_forecast의 combine_first가 처리
    collect_from_h = 0

    target_hours = []
    for h in range(24):
        kst_time = target_dt.replace(hour=h, minute=0, second=0)

        if target_dt.date() == ref_kst.date() and h < collect_from_h:
            continue

        offset_h = int((kst_time - base_time_kst).total_seconds() / 3600)
        if offset_h < 0:
            continue

        target_hours.append((offset_h, kst_time))

    if not target_hours:
        logger.warning("[KMA NCM] 수집 대상 시간 없음")
        return pd.DataFrame()

    logger.info(f"[KMA NCM] 수집 범위: "
                f"{target_hours[0][1].strftime('%H')}시 ~ "
                f"{target_hours[-1][1].strftime('%H')}시 KST "
                f"({len(target_hours)}시간, offset +{target_hours[0][0]}h ~ +{target_hours[-1][0]}h)")

    # ── 4) API 호출 (병렬, 공유 세션) ──
    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-kim_nc_pt_txt2"
    rows = []

    def fetch_hour(hour_offset, kst_time):
        params = {
            'group': 'KIMG', 'nwp': 'NE57', 'data': 'U',
            'name': 'dswrsfc,t2m,tcld,mcld,lcld,u10m,v10m,snowd,rh2m,rainc_acc,rainl_acc',
            'tmfc': base_tmfc, 'hf': str(hour_offset),
            'lat': str(lat), 'lon': str(lon),
            'disp': 'A', 'help': '0', 'authKey': auth_key
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = _kma_session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = parse_raw_text_by_varn(resp.text)
                    if data:
                        data['timestamp'] = kst_time.strftime('%Y-%m-%d %H:%M:%S')
                        return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.debug(f"[Retry] +{hour_offset}h: {attempt+1}/{max_retries} ({wait}s 대기)")
                    time.sleep(wait)
                else:
                    logger.warning(f"[Fail] +{hour_offset}h: {max_retries}회 시도 실패")
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_hour = {
            executor.submit(fetch_hour, h, kst_t): h
            for h, kst_t in target_hours
        }
        for future in as_completed(future_to_hour):
            res = future.result()
            if res:
                rows.append(res)

    if not rows:
        # 최신 사이클에서 데이터 없음 → 이전 사이클로 폴백 시도
        candidates.remove(best_cycle_utc)
        if candidates:
            fallback_cycle_utc = max(candidates)
            fallback_tmfc = fallback_cycle_utc.strftime('%Y%m%d%H')
            logger.warning(f"[KMA NCM] 데이터 없음 → 이전 사이클 {fallback_tmfc} UTC로 재시도")

            # as_of를 폴백 사이클 배포 시점으로 조정하여 재귀 호출
            fallback_kst = fallback_cycle_utc.astimezone(KST) + timedelta(hours=PUBLISH_DELAY_H)
            return fetch_kma_future_ncm(
                lat, lon, auth_key, base_date_kst,
                as_of_kst=fallback_kst
            )

        logger.error("[KMA NCM] 모든 사이클 시도 실패")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── 5) 후처리 ──
    if 'solar_rad' in df.columns:
        df['solar_rad'] = (df['solar_rad'] * 0.0036).round(2)

    if 'temp_k' in df.columns:
        df['temp_c'] = (df['temp_k'] - 273.15).round(2)
        df = df.drop(columns=['temp_k'])

    if 'u_wind' in df.columns and 'v_wind' in df.columns:
        df['wind_spd'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2).round(2)
        wind_dir = (270 - np.degrees(np.arctan2(df['v_wind'], df['u_wind']))) % 360
        df['wd_sin'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos'] = np.cos(np.radians(wind_dir)).round(4)
        df = df.drop(columns=['u_wind', 'v_wind'])

    if 'low_cloud' in df.columns and 'mid_cloud' in df.columns:
        df['midlow_cloud'] = df['low_cloud'] + df['mid_cloud'] * (1 - df['low_cloud'])
        df = df.drop(columns=['low_cloud', 'mid_cloud'])

    #if 'rain_conv' in df.columns and 'rain_strat' in df.columns:
    #    df['rainfall'] = (df['rain_conv'].fillna(0) + df['rain_strat'].fillna(0)).round(2)
    #    df = df.drop(columns=['rain_conv', 'rain_strat'])
    if 'rain_conv' in df.columns and 'rain_strat' in df.columns:
            # 1. 두 변수를 합쳐 누적 강수량(Accumulated Rainfall) 생성
            df['acc_rainfall'] = (df['rain_conv'].fillna(0) + df['rain_strat'].fillna(0))
            
            # 2. 누적 강수량을 시간당 강수량으로 변환 (현재 시간 값 - 직전 시간 값)
            # diff()를 사용하면 첫 번째 행은 NaN이 되므로 0으로 채움
            df['rainfall'] = df['acc_rainfall'].diff().fillna(df['acc_rainfall']).round(2)
            
            # 3. 모델 초기화 등의 이유로 간혹 음수가 나올 수 있으므로 0 미만은 0으로 강제
            df['rainfall'] = df['rainfall'].clip(lower=0)
            
            # 사용한 원본 컬럼 삭제
            df = df.drop(columns=['rain_conv', 'rain_strat', 'acc_rainfall'])

    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')

    logger.info(f"[KMA NCM] {len(df)}행 수집 완료 (예보 기준: {base_tmfc} UTC)")
    return df


def fetch_kma_future_ncm_north(lat, lon, auth_key, base_date_kst, as_of_kst=None):
    from datetime import datetime, timezone, timedelta

    VARN_MAP = {
        20: 'u_wind',
        21: 'v_wind',
    }

    def parse_raw_text_by_varn(raw_text):
        parsed_dict = {}
        lines = raw_text.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                varn_code = int(parts[2])
                value = float(parts[4])
                if varn_code in VARN_MAP:
                    parsed_dict[VARN_MAP[varn_code]] = value
            except (ValueError, IndexError):
                continue
        return parsed_dict

    # ── 1) 기준 시점 결정 ──
    KST = timezone(timedelta(hours=9))

    if as_of_kst is None:
        ref_kst = datetime.now(KST)
    elif isinstance(as_of_kst, str):
        ref_kst = datetime.strptime(as_of_kst, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    else:
        ref_kst = as_of_kst if as_of_kst.tzinfo else as_of_kst.replace(tzinfo=KST)

    ref_utc = ref_kst.astimezone(timezone.utc)
    target_dt = pd.to_datetime(base_date_kst).tz_localize(KST)

    # ── 2) 최신 사용 가능 예보 사이클 계산 ──
    CYCLES_UTC = [0, 6, 12, 18]
    PUBLISH_DELAY_H = 2

    available_utc = ref_utc - timedelta(hours=PUBLISH_DELAY_H)

    candidates = []
    for day_offset in [0, -1, -2]:
        base_day = available_utc.date() + timedelta(days=day_offset)
        for cycle_h in CYCLES_UTC:
            cycle_dt = datetime(base_day.year, base_day.month, base_day.day,
                                cycle_h, 0, 0, tzinfo=timezone.utc)
            if cycle_dt <= available_utc:
                candidates.append(cycle_dt)

    if not candidates:
        logger.warning("[KMA NCM] 사용 가능한 예보 사이클 없음")
        return pd.DataFrame()

    best_cycle_utc = max(candidates)
    base_tmfc = best_cycle_utc.strftime('%Y%m%d%H')
    base_time_kst = best_cycle_utc.astimezone(KST)

    logger.info(f"[KMA NCM] 예보 기준: {base_tmfc} UTC "
                f"(기준시점: {ref_kst.strftime('%Y-%m-%d %H:%M')} KST)")

    # ── 3) 수집 대상 시간 범위 계산 ──
    # 항상 00시부터 전체 수집 (forecast 테이블을 빈틈없이 채움)
    # 모델 예측 시 ASOS 우선은 get_historical_and_forecast의 combine_first가 처리
    collect_from_h = 0

    target_hours = []
    for h in range(24):
        kst_time = target_dt.replace(hour=h, minute=0, second=0)

        if target_dt.date() == ref_kst.date() and h < collect_from_h:
            continue

        offset_h = int((kst_time - base_time_kst).total_seconds() / 3600)
        if offset_h < 0:
            continue

        target_hours.append((offset_h, kst_time))

    if not target_hours:
        logger.warning("[KMA NCM] 수집 대상 시간 없음")
        return pd.DataFrame()

    logger.info(f"[KMA NCM] 수집 범위: "
                f"{target_hours[0][1].strftime('%H')}시 ~ "
                f"{target_hours[-1][1].strftime('%H')}시 KST "
                f"({len(target_hours)}시간, offset +{target_hours[0][0]}h ~ +{target_hours[-1][0]}h)")

    # ── 4) API 호출 (병렬, 공유 세션) ──
    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-kim_nc_pt_txt2"
    rows = []

    def fetch_hour(hour_offset, kst_time):
        params = {
            'group': 'KIMG', 'nwp': 'NE57', 'data': 'U',
            'name': 'u10m,v10m',
            'tmfc': base_tmfc, 'hf': str(hour_offset),
            'lat': str(lat), 'lon': str(lon),
            'disp': 'A', 'help': '0', 'authKey': auth_key
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = _kma_session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = parse_raw_text_by_varn(resp.text)
                    if data:
                        data['timestamp'] = kst_time.strftime('%Y-%m-%d %H:%M:%S')
                        return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.debug(f"[Retry] +{hour_offset}h: {attempt+1}/{max_retries} ({wait}s 대기)")
                    time.sleep(wait)
                else:
                    logger.warning(f"[Fail] +{hour_offset}h: {max_retries}회 시도 실패")
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_hour = {
            executor.submit(fetch_hour, h, kst_t): h
            for h, kst_t in target_hours
        }
        for future in as_completed(future_to_hour):
            res = future.result()
            if res:
                rows.append(res)

    if not rows:
        # 최신 사이클에서 데이터 없음 → 이전 사이클로 폴백 시도
        candidates.remove(best_cycle_utc)
        if candidates:
            fallback_cycle_utc = max(candidates)
            fallback_tmfc = fallback_cycle_utc.strftime('%Y%m%d%H')
            logger.warning(f"[KMA NCM] 데이터 없음 → 이전 사이클 {fallback_tmfc} UTC로 재시도")

            # as_of를 폴백 사이클 배포 시점으로 조정하여 재귀 호출
            fallback_kst = fallback_cycle_utc.astimezone(KST) + timedelta(hours=PUBLISH_DELAY_H)
            return fetch_kma_future_ncm(
                lat, lon, auth_key, base_date_kst,
                as_of_kst=fallback_kst
            )

        logger.error("[KMA NCM] 모든 사이클 시도 실패")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 후처리 (수정된 부분)
    if 'u_wind' in df.columns and 'v_wind' in df.columns:
        # 아예 처음부터 _north를 붙여서 컬럼 생성
        df['wind_spd_north'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2).round(2)
        wind_dir = (270 - np.degrees(np.arctan2(df['v_wind'], df['u_wind']))) % 360
        df['wd_sin_north'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos_north'] = np.cos(np.radians(wind_dir)).round(4)
        
        # 임시 변수 삭제
        df = df.drop(columns=['u_wind', 'v_wind'])
    
    # timestamp 포맷 변환
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')
    
    logger.info(f"[KMA NCM north] {len(df)}행 수집 완료")
    return df


# ============================================================================
# 풍력 3권역 좌표 설정 (임시 - 추후 정밀 조정)
# ============================================================================
WIND_ZONES = [
    {
        "name": "동쪽",  # 구좌·성산 + 가시리·수망리 70%
        "lat": 33.5913,
        "lon": 126.7930,
        "capacity_mw": 201.1,
    },
    {
        "name": "서쪽",  # 한경·한림·애월 + 가시리·수망리 30%
        "lat": 33.4427,
        "lon": 126.1713,
        "capacity_mw": 217.8,
    },
]

TOTAL_WIND_CAPACITY = sum(z["capacity_mw"] for z in WIND_ZONES)


def fetch_kma_future_ncm_wind(auth_key, base_date_kst, as_of_kst=None):
    """
    3개 권역의 NCM 풍속 예보를 각각 수집한 뒤,
    설비용량 가중 평균으로 wind_spd_north / wd_sin_north / wd_cos_north 를 산출.
    
    기존 fetch_kma_future_ncm_north 의 drop-in replacement.
    출력 DataFrame 형태가 동일하므로 _merge_south_north 와 호환됨.
    """
    from datetime import datetime, timezone, timedelta

    KST = timezone(timedelta(hours=9))

    # ── 1) 권역별 수집 (병렬) ──
    def _fetch_zone(zone):
        logger.info(f"[Wind-{zone['name']}] ({zone['lat']}, {zone['lon']}) 수집 중...")
        df_zone = fetch_kma_future_ncm_north_single(
            zone["lat"], zone["lon"], auth_key, base_date_kst, as_of_kst=as_of_kst
        )
        if df_zone.empty:
            logger.warning(f"[Wind-{zone['name']}] 데이터 없음 — 스킵")
            return None
        weight = zone["capacity_mw"] / TOTAL_WIND_CAPACITY
        return (zone["name"], weight, df_zone)

    with ThreadPoolExecutor(max_workers=len(WIND_ZONES)) as executor:
        results = list(executor.map(_fetch_zone, WIND_ZONES))
    zone_dfs = [r for r in results if r is not None]

    if not zone_dfs:
        logger.error("[Wind] 모든 권역 수집 실패")
        return pd.DataFrame()

    # ── 2) 용량 가중 평균 합산 ──
    # 모든 권역의 timestamp를 합집합으로 잡고, 가중 평균
    # 일부 권역만 성공한 시간대는 성공한 권역끼리 재정규화
    
    all_timestamps = set()
    for name, w, df_z in zone_dfs:
        all_timestamps.update(df_z.index.tolist())
    all_timestamps = sorted(all_timestamps)

    result_rows = []
    for ts in all_timestamps:
        u_sum = 0.0
        v_sum = 0.0
        w_sum = 0.0
        zone_data = {}

        for name, weight, df_z in zone_dfs:
            if ts in df_z.index:
                row = df_z.loc[ts]
                spd = row["wind_spd_north"]
                wd_sin = row["wd_sin_north"]
                wd_cos = row["wd_cos_north"]

                u_sum += weight * spd * wd_sin
                v_sum += weight * spd * wd_cos
                w_sum += weight
                zone_data[name] = (round(spd, 2), round(wd_sin, 4), round(wd_cos, 4))

        if w_sum == 0:
            continue

        u_avg = u_sum / w_sum
        v_avg = v_sum / w_sum

        wind_spd = np.sqrt(u_avg**2 + v_avg**2)
        wind_dir_rad = np.arctan2(u_avg, v_avg)

        row_dict = {
            "timestamp": ts,
            "wind_spd_north": round(wind_spd, 2),
            "wd_sin_north": round(np.sin(wind_dir_rad), 4),
            "wd_cos_north": round(np.cos(wind_dir_rad), 4),
        }
        _apply_zone_data(zone_data, row_dict)
        result_rows.append(row_dict)

    if not result_rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(result_rows)
    df_result = df_result.set_index("timestamp").sort_index()

    actual_zones = len(zone_dfs)
    zone_names = ", ".join(n for n, _, _ in zone_dfs)
    logger.info(f"[Wind] {actual_zones}개 권역 가중평균 완료 ({zone_names}), {len(df_result)}행")
    
    return df_result


def fetch_kma_future_ncm_north_single(lat, lon, auth_key, base_date_kst, as_of_kst=None):
    """
    단일 좌표에 대한 NCM 풍속(u, v) 수집.
    기존 fetch_kma_future_ncm_north 로직을 그대로 유지하되,
    이름만 _single 로 변경하여 내부 전용으로 사용.
    """
    from datetime import datetime, timezone, timedelta

    VARN_MAP = {
        20: 'u_wind',
        21: 'v_wind',
    }

    def parse_raw_text_by_varn(raw_text):
        parsed_dict = {}
        lines = raw_text.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                varn_code = int(parts[2])
                value = float(parts[4])
                if varn_code in VARN_MAP:
                    parsed_dict[VARN_MAP[varn_code]] = value
            except (ValueError, IndexError):
                continue
        return parsed_dict

    KST = timezone(timedelta(hours=9))

    if as_of_kst is None:
        ref_kst = datetime.now(KST)
    elif isinstance(as_of_kst, str):
        ref_kst = datetime.strptime(as_of_kst, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    else:
        ref_kst = as_of_kst if as_of_kst.tzinfo else as_of_kst.replace(tzinfo=KST)

    ref_utc = ref_kst.astimezone(timezone.utc)
    target_dt = pd.to_datetime(base_date_kst).tz_localize(KST)

    CYCLES_UTC = [0, 6, 12, 18]
    PUBLISH_DELAY_H = 2

    available_utc = ref_utc - timedelta(hours=PUBLISH_DELAY_H)

    candidates = []
    for day_offset in [0, -1, -2]:
        base_day = available_utc.date() + timedelta(days=day_offset)
        for cycle_h in CYCLES_UTC:
            cycle_dt = datetime(base_day.year, base_day.month, base_day.day,
                                cycle_h, 0, 0, tzinfo=timezone.utc)
            if cycle_dt <= available_utc:
                candidates.append(cycle_dt)

    if not candidates:
        return pd.DataFrame()

    best_cycle_utc = max(candidates)
    base_tmfc = best_cycle_utc.strftime('%Y%m%d%H')
    base_time_kst = best_cycle_utc.astimezone(KST)

    collect_from_h = 0
    target_hours = []
    for h in range(24):
        kst_time = target_dt.replace(hour=h, minute=0, second=0)
        if target_dt.date() == ref_kst.date() and h < collect_from_h:
            continue
        offset_h = int((kst_time - base_time_kst).total_seconds() / 3600)
        if offset_h < 0:
            continue
        target_hours.append((offset_h, kst_time))

    if not target_hours:
        return pd.DataFrame()

    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-kim_nc_pt_txt2"
    rows = []

    def fetch_hour(hour_offset, kst_time):
        params = {
            'group': 'KIMG', 'nwp': 'NE57', 'data': 'U',
            'name': 'u10m,v10m',
            'tmfc': base_tmfc, 'hf': str(hour_offset),
            'lat': str(lat), 'lon': str(lon),
            'disp': 'A', 'help': '0', 'authKey': auth_key
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = _kma_session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = parse_raw_text_by_varn(resp.text)
                    if data:
                        data['timestamp'] = kst_time.strftime('%Y-%m-%d %H:%M:%S')
                        return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_hour = {
            executor.submit(fetch_hour, h, kst_t): h
            for h, kst_t in target_hours
        }
        for future in as_completed(future_to_hour):
            res = future.result()
            if res:
                rows.append(res)

    if not rows:
        candidates.remove(best_cycle_utc)
        if candidates:
            fallback_cycle_utc = max(candidates)
            fallback_kst = fallback_cycle_utc.astimezone(KST) + timedelta(hours=PUBLISH_DELAY_H)
            return fetch_kma_future_ncm_north_single(
                lat, lon, auth_key, base_date_kst, as_of_kst=fallback_kst
            )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    if 'u_wind' in df.columns and 'v_wind' in df.columns:
        df['wind_spd_north'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2).round(2)
        wind_dir = (270 - np.degrees(np.arctan2(df['v_wind'], df['u_wind']))) % 360
        df['wd_sin_north'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos_north'] = np.cos(np.radians(wind_dir)).round(4)
        df = df.drop(columns=['u_wind', 'v_wind'])

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')

    return df


# ============================================================================
# KMA VilageFcst 2.0 — 북쪽 풍속 예보 (빠른 단기 API)
# ============================================================================
_VILAGE_BASE_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/"
    "VilageFcstInfoService_2.0/getVilageFcst"
)
_VILAGE_RELEASE_HOURS_KST  = [2, 5, 8, 11, 14, 17, 20, 23]  # KST 발표 시각
_VILAGE_PUBLISH_DELAY_MIN  = 10                               # 발표 후 최소 대기 (분)
_VILAGE_MAX_HISTORY_DAYS   = 1                                # 수집 허용 범위 (D-1): API가 fcstDate >= 어제 데이터만 제공
_VILAGE_API_LIMIT_DAYS     = 3                                # API 실제 3일 제한 (사이클 후보 컷오프용)
_VILAGE_MAX_FUTURE_DAYS    = 2                                # API 예보 한계 (D+2)
_VILAGE_GRID_POINTS = [
    {"name": "서쪽(고산)", "nx": 46, "ny": 35},
    {"name": "동쪽(성산)", "nx": 59, "ny": 38},
]


def _fetch_vilage_raw(nx, ny, base_date_str, base_time_str, auth_key):
    """단일 격자점·단일 사이클 API 호출. items 리스트 반환, 오류 시 None."""
    params = {
        "pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
        "base_date": base_date_str, "base_time": base_time_str,
        "nx": nx, "ny": ny, "authKey": auth_key,
    }
    try:
        resp = _kma_session.get(_VILAGE_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") != "00":
            msg = header.get("resultMsg", "")
            logger.debug(f"[VilageFcst] nx={nx},ny={ny} 응답코드: {msg}")
            if "3일" in msg:
                return "LIMIT"   # 3일 제한 도달 → 이하 모든 사이클도 실패 확정
            return None
        return data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    except Exception as e:
        logger.debug(f"[VilageFcst] nx={nx},ny={ny} 요청실패: {e}")
        return None


def _parse_vilage_wind(items, target_date_str):
    """
    UUU/VVV 카테고리를 target_date_str 기준으로 필터링.
    3시간 간격 구간은 선형 보간으로 1시간 단위로 채움.
    반환: UUU, VVV 컬럼을 가진 DatetimeIndex DataFrame (빈 DataFrame 가능)
    """
    rows = {}
    for item in items:
        if item.get("fcstDate") != target_date_str:
            continue
        cat = item.get("category")
        if cat not in ("UUU", "VVV"):
            continue
        t = item["fcstTime"]
        if t not in rows:
            rows[t] = {}
        try:
            rows[t][cat] = float(item["fcstValue"])
        except (ValueError, TypeError):
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    y, mo, d = int(target_date_str[:4]), int(target_date_str[4:6]), int(target_date_str[6:])
    df.index = pd.to_datetime(
        [f"{target_date_str[:4]}-{target_date_str[4:6]}-{target_date_str[6:]} {t[:2]}:{t[2:]}"
         for t in df.index]
    )

    full_idx = pd.date_range(
        start=pd.Timestamp(year=y, month=mo, day=d),
        periods=24, freq='h'
    )
    df = df.reindex(full_idx).interpolate(method='time', limit=2)
    return df[['UUU', 'VVV']].dropna(how='all')


def fetch_kma_vilage_wind(auth_key, base_date_kst, as_of_kst=None):
    """
    KMA VilageFcst 2.0 API로 북쪽 풍속 예보(고산·성산) 수집.

    설계 원칙
    - 발표 시각: KST 02/05/08/11/14/17/20/23  (UTC 변환 없이 KST 직접 사용)
    - 발표 지연: 가변적 → 최신 사이클부터 순차 시도, 첫 성공 사이클 사용
    - 스레드 없음: 포인트당 1회 호출로 전체 시간대 일괄 반환, 순차 2회면 충분
    - 수집 가능 범위: D-3 ~ D+3 (API 소급 한계 3일)

    반환
        timestamp 인덱스(문자열)의 DataFrame:
            wind_spd_north_v, wd_sin_north_v, wd_cos_north_v
        모든 사이클 실패 시 빈 DataFrame
    """
    from datetime import datetime, timezone, timedelta

    KST = timezone(timedelta(hours=9))

    if as_of_kst is None:
        ref_kst = datetime.now(KST)
    elif isinstance(as_of_kst, str):
        ref_kst = datetime.strptime(as_of_kst, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    else:
        ref_kst = as_of_kst if as_of_kst.tzinfo else as_of_kst.replace(tzinfo=KST)

    target_date     = pd.to_datetime(base_date_kst).date()
    target_date_str = target_date.strftime('%Y%m%d')

    # 날짜 범위 검사: D-3 ~ D+2 만 VilageFcst 사용, 외부는 KIMG 담당
    today = ref_kst.date()
    if not (today - timedelta(days=_VILAGE_MAX_HISTORY_DAYS)
            <= target_date
            <= today + timedelta(days=_VILAGE_MAX_FUTURE_DAYS)):
        logger.debug(f"[VilageFcst] {base_date_kst} 범위 외(D-3~D+2) → KIMG 사용")
        return pd.DataFrame()

    # 후보 사이클: 최신순, 발표 후 10분 경과 기준
    # 실제 현재 시각 기준 API 3일 제한 적용 (as_of가 과거여도 실제 API 한계는 지금 기준)
    real_now_kst   = datetime.now(KST)
    api_earliest   = real_now_kst.date() - timedelta(days=_VILAGE_API_LIMIT_DAYS)
    cutoff_kst     = ref_kst - timedelta(minutes=_VILAGE_PUBLISH_DELAY_MIN)

    candidates = []
    for day_offset in range(_VILAGE_MAX_HISTORY_DAYS + 2):   # 약간 넉넉하게 탐색
        base_day = ref_kst.date() - timedelta(days=day_offset)
        if base_day < api_earliest:
            break                                              # 이 이상 과거는 API가 차단
        for h in sorted(_VILAGE_RELEASE_HOURS_KST, reverse=True):
            cycle_dt = datetime(base_day.year, base_day.month, base_day.day, h, 0, tzinfo=KST)
            if cycle_dt <= cutoff_kst:
                candidates.append(cycle_dt)

    for cycle_dt in candidates:
        b_date = cycle_dt.strftime('%Y%m%d')
        b_time = f"{cycle_dt.hour:02d}00"

        point_uvs = []
        point_data = {}
        api_error  = False

        for pt in _VILAGE_GRID_POINTS:
            items = _fetch_vilage_raw(pt['nx'], pt['ny'], b_date, b_time, auth_key)
            if items == "LIMIT":
                logger.debug(f"[VilageFcst] {b_date} {b_time} 3일 한계 → 수집 중단")
                return pd.DataFrame()
            if items is None:
                api_error = True
                break
            df_uv = _parse_vilage_wind(items, target_date_str)
            if not df_uv.empty:
                point_uvs.append(df_uv)
                point_data[pt['name']] = df_uv

        if api_error:
            logger.debug(f"[VilageFcst] {b_date} {b_time} API 오류 → 이전 사이클")
            continue
        if not point_uvs:
            logger.debug(f"[VilageFcst] {b_date} {b_time} 대상 날짜({target_date_str}) 없음 → 이전 사이클")
            continue

        # 등가중 평균 (blended 출력 — 기존 방식 유지)
        if len(point_uvs) == 1:
            df_avg = point_uvs[0]
        else:
            df_avg = pd.concat(point_uvs).groupby(level=0).mean()

        u = df_avg['UUU']
        v = df_avg['VVV']
        wind_spd = np.sqrt(u**2 + v**2).round(2)
        wind_dir = (270 - np.degrees(np.arctan2(v, u))) % 360
        wd_sin   = np.sin(np.radians(wind_dir)).round(4)
        wd_cos   = np.cos(np.radians(wind_dir)).round(4)

        result = pd.DataFrame({
            'wind_spd_north_v': wind_spd,
            'wd_sin_north_v':   wd_sin,
            'wd_cos_north_v':   wd_cos,
        })

        # 권역별 개별 저장 (미래 재학습용)
        for pt_name, df_uv in point_data.items():
            suffix = 'east_v' if _ZONE_EAST in pt_name else 'west_v'
            u_z = df_uv['UUU']
            v_z = df_uv['VVV']
            spd_z = np.sqrt(u_z**2 + v_z**2).round(2)
            dir_z = (270 - np.degrees(np.arctan2(v_z, u_z))) % 360
            result[f'wind_spd_{suffix}'] = spd_z
            result[f'wd_sin_{suffix}']   = np.sin(np.radians(dir_z)).round(4)
            result[f'wd_cos_{suffix}']   = np.cos(np.radians(dir_z)).round(4)

        result.index = result.index.strftime('%Y-%m-%d %H:%M:%S')
        result.index.name = 'timestamp'

        logger.info(f"[VilageFcst] 사이클 {b_date} {b_time} KST 사용 → {len(result)}행")
        return result

    logger.warning(f"[VilageFcst] 모든 사이클 실패 ({base_date_kst})")
    return pd.DataFrame()
