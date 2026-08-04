from .db_manager import JejuEnergyDB
from .api_fetchers import (
    fetch_kpx_past,
    fetch_kpx_future,
    fetch_kpx_historical,
    fetch_kma_past_asos,
    fetch_kma_future_ncm,
    fetch_kma_future_ncm_wind,
    fetch_kma_past_asos_wind,
    fetch_kma_vilage_wind,
    warmup_kma_session,
)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pvlib
import numpy as np
import torch
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

logger = logging.getLogger('jejucr.pipeline')
logger.setLevel(logging.DEBUG)

load_dotenv()
KMA_KEY = os.getenv("KMA_API_KEY")
KPX_KEY = os.getenv("KPX_API_KEY")

_NCM_CYCLE_KST    = ["03:00", "11:00", "17:00"]
_VILAGE_CYCLE_KST = ["01:00", "11:00", "17:00"]

# ============================================================================
# 시나리오 1: 초기 데이터 수집 (5년치)
# ============================================================================

def add_capacity_features(df):
    df = df.copy()
    
    # 1. Capacity 추정 (Rolling Cummax 방식)
    if 'real_solar_gen' in df.columns:
        #df['Solar_Capacity_Est'] = df['real_solar_gen'].expanding().max()
        df['Solar_Capacity_Est'] = df['real_solar_gen'].rolling(720, min_periods=1).max()
        # 또는 720시간 윈도우 유지하려면:

    
    if 'real_wind_gen' in df.columns:
        #df['Wind_Capacity_Est'] = df['real_wind_gen'].expanding().max()
        df['Wind_Capacity_Est'] = df['real_wind_gen'].rolling(720, min_periods=1).max()
    # 2. Utilization 계산
    if 'real_solar_gen' in df.columns and 'Solar_Capacity_Est' in df.columns:
        df['Solar_Utilization'] = df['real_solar_gen'] / df['Solar_Capacity_Est']
        df['Solar_Utilization'] = df['Solar_Utilization'].fillna(0)
    
    if 'real_wind_gen' in df.columns and 'Wind_Capacity_Est' in df.columns:
        df['Wind_Utilization'] = df['real_wind_gen'] / df['Wind_Capacity_Est']
        df['Wind_Utilization'] = df['Wind_Utilization'].fillna(0)
    
    return df



# ============================================================================
# daily_historical_update 수정 부분 (변경 전/후)
# ============================================================================

# ── 변경 전 ──
"""
        asos_north = fetch_kma_past_asos(
            start_date.replace('-', ''), 
            end_date.replace('-', ''), 
            KMA_KEY,
            stn_id=185
        )
        
        if not asos_south.empty and not asos_north.empty:
            asos_north = asos_north[['wind_spd', 'wd_sin', 'wd_cos']].rename(
                columns={
                    'wind_spd': 'wind_spd_north', 
                    'wd_sin': 'wd_sin_north', 
                    'wd_cos': 'wd_cos_north'
                }
            )
            asos_data = pd.concat([asos_south, asos_north], axis=1)
        else:
            asos_data = asos_south
            print(f"북쪽 ASOS 데이터 로드 실패 : 남쪽 ASOS 데이터 적용")
"""

# ── 변경 후 ── (daily_historical_update 와 daily_historical_kma 둘 다 동일)
"""

"""

# ============================================================================
# 시나리오 2: 실측 업데이트(사용자가 기간 설정)
# ============================================================================
def daily_historical_update(start_date, end_date):
    """
    실측 데이터 업데이트 (최대 30일 제한, 미래 날짜 제한, 독립적 API 호출)
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 시간 정보를 제외한 오늘 날짜 (0시 0분 0초 기준)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # ==========================================
    # [날짜 제한 검증 로직]
    # ==========================================
    # 1. 시작일이 종료일보다 늦은 경우 방지
    if start_dt > end_dt:
        logger.error("시작일이 종료일보다 늦을 수 없습니다.")
        return

    # 2. 미래 날짜 선택 제한 (오늘보다 앞선 날짜만 가능)
    if end_dt > today or start_dt > today:
        logger.error(f"오늘({today.strftime('%Y-%m-%d')}) 이후의 미래 날짜는 조회할 수 없습니다.")
        return

    # 3. 최대 30일 간격 제한
    if (end_dt - start_dt).days > 30:
        logger.error("실측 데이터 조회는 최대 30일까지만 가능합니다.")
        return
    # ==========================================

    logger.info(f"일일 실측 업데이트 시작: {start_date} ~ {end_date}")
    
    db = JejuEnergyDB()
    
    # 1. API 각각 독립 호출
    kpx_data = pd.DataFrame()
    try:
        kpx_data = fetch_kpx_past(start_date, end_date)
    except Exception as e:
        logger.error(f"KPX 실측 API 호출 실패: {e}")

    asos_data = pd.DataFrame()
    try:
        # 남쪽 베이스 호출 (189)
        asos_south = fetch_kma_past_asos(
            start_date.replace('-', ''), 
            end_date.replace('-', ''), 
            KMA_KEY,
            stn_id=189
        )
        
        # 2권역 가중평균 풍속 (성산 + 고산)
        asos_wind = fetch_kma_past_asos_wind(
            start_date.replace('-', ''),
            end_date.replace('-', ''),
            KMA_KEY
        )

        if not asos_south.empty and not asos_wind.empty:
            asos_data = pd.concat([asos_south, asos_wind], axis=1)
        elif not asos_south.empty:
            asos_data = asos_south
            logger.warning("풍속 2권역 데이터 없음 — 남쪽 ASOS만 사용")
        else:
            asos_data = pd.DataFrame()

    except Exception as e:
        logger.error(f"KMA 실측 기상 API 호출 실패: {e}")

    smp_data = pd.DataFrame()
    try:
        smp_data = fetch_kpx_historical(start_date, end_date, KPX_KEY)
    except Exception as e:
        logger.error(f"KPX SMP API 호출 실패: {e}")

    # 2. 병합 준비 (성공해서 데이터가 있는 것만 모음)
    df_list = [df for df in [kpx_data, asos_data, smp_data] if not df.empty]

    if not df_list:
        logger.error("수집된 실측 데이터가 없어 업데이트를 종료합니다.")
        db.close()
        return

    try:
        # 3. 데이터 병합
        actual_df = pd.concat(df_list, axis=1)
        
        # 4. Capacity 및 Utilization 계산
        lookback_date = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        historical_for_calc = db.get_historical(lookback_date, end_date)
        
        if not historical_for_calc.empty:
            combined = pd.concat([historical_for_calc, actual_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            combined['Solar_Capacity_Est'] = combined['real_solar_gen'].rolling(720, min_periods=1).max()
            combined['Wind_Capacity_Est'] = combined['real_wind_gen'].rolling(720, min_periods=1).max()
            
            actual_df = combined.loc[actual_df.index]
        else:
            if 'real_solar_gen' in actual_df.columns:
                actual_df['Solar_Capacity_Est'] = actual_df['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in actual_df.columns:
                actual_df['Wind_Capacity_Est'] = actual_df['real_wind_gen'].rolling(720, min_periods=1).max()
                
        if 'real_solar_gen' in actual_df.columns and 'Solar_Capacity_Est' in actual_df.columns:
            actual_df['Solar_Utilization'] = actual_df['real_solar_gen'] / actual_df['Solar_Capacity_Est']
        if 'real_wind_gen' in actual_df.columns and 'Wind_Capacity_Est' in actual_df.columns:
            actual_df['Wind_Utilization'] = actual_df['real_wind_gen'] / actual_df['Wind_Capacity_Est']
        
        # 5. DB 저장
        db.save_historical(actual_df)
        logger.info(f"실측 데이터 {len(actual_df)}행 업데이트 완료")

    except Exception as e:
        logger.error(f"실측 데이터 병합 또는 저장 실패: {e}")
        
    db.close()

def daily_historical_kma(start_date, end_date):
    """ KMA 종관기상관측(ASOS) 실측 데이터 수집 """
    logger.info(f"KMA 기상 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    
    asos_data = pd.DataFrame()
    try:
        # 남쪽 베이스 호출 (189)
        asos_south = fetch_kma_past_asos(
            start_date.replace('-', ''), 
            end_date.replace('-', ''), 
            KMA_KEY,
            stn_id=189
        )
        
        # 2권역 가중평균 풍속 (성산 + 고산)
        asos_wind = fetch_kma_past_asos_wind(
            start_date.replace('-', ''),
            end_date.replace('-', ''),
            KMA_KEY
        )

        if not asos_south.empty and not asos_wind.empty:
            asos_data = pd.concat([asos_south, asos_wind], axis=1)
        elif not asos_south.empty:
            asos_data = asos_south
            logger.warning("풍속 2권역 데이터 없음 — 남쪽 ASOS만 사용")
        else:
            asos_data = pd.DataFrame()

        # DB 저장 로직 추가
        if not asos_data.empty:
            db.save_historical(asos_data)
            logger.info("KMA 실측 기상 데이터 병합 및 DB 저장 완료")
        else:
            logger.warning("KMA 실측 기상 데이터가 비어있어 저장하지 않았습니다.")

    except Exception as e:
        logger.error(f"KMA 실측 기상 API 호출 실패: {e}")
        
    finally:
        db.close()
        

def daily_historical_kpx(start_date, end_date):
    """ KPX 실측 발전량 데이터 수집 및 파생변수 계산 """
    logger.info(f"KPX 발전량 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()

    try:
        actual_df = fetch_kpx_past(start_date, end_date)  # ← fetch 범위만 +1일
        if actual_df.empty:
            logger.error("수집된 KPX 데이터가 없습니다.")
            return

        # Capacity 및 Utilization 계산을 위해 과거 30일 데이터 불러오기
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        lookback_date = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        historical_for_calc = db.get_historical(lookback_date, end_date)
        
        if not historical_for_calc.empty:
            combined = pd.concat([historical_for_calc, actual_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            if 'real_solar_gen' in combined.columns:
                combined['Solar_Capacity_Est'] = combined['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in combined.columns:
                combined['Wind_Capacity_Est'] = combined['real_wind_gen'].rolling(720, min_periods=1).max()
            
            actual_df = combined.loc[actual_df.index]
        else:
            if 'real_solar_gen' in actual_df.columns:
                actual_df['Solar_Capacity_Est'] = actual_df['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in actual_df.columns:
                actual_df['Wind_Capacity_Est'] = actual_df['real_wind_gen'].rolling(720, min_periods=1).max()
                
        if 'real_solar_gen' in actual_df.columns and 'Solar_Capacity_Est' in actual_df.columns:
            actual_df['Solar_Utilization'] = actual_df['real_solar_gen'] / actual_df['Solar_Capacity_Est']
        if 'real_wind_gen' in actual_df.columns and 'Wind_Capacity_Est' in actual_df.columns:
            actual_df['Wind_Utilization'] = actual_df['real_wind_gen'] / actual_df['Wind_Capacity_Est']
        
        db.save_historical(actual_df)
        logger.info(f"KPX 발전량 데이터 {len(actual_df)}행 업데이트 완료")

    except Exception as e:
        logger.error(f"KPX 실측 데이터 처리 실패: {e}")
    finally:
        db.close()

def daily_historical_kpx_smp(start_date, end_date):
    """ KPX SMP 실측 가격 데이터 수집 """
    logger.info(f"KPX SMP 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    
    try:
        smp_data = fetch_kpx_historical(start_date, end_date, KPX_KEY) # 전역 변수 필요
        if not smp_data.empty:
            db.save_historical(smp_data)
            logger.info(f"KPX SMP 데이터 {len(smp_data)}행 업데이트 완료")
        else:
            logger.warning("수집된 SMP 데이터가 없습니다.")

    except Exception as e:
        logger.error(f"KPX SMP 데이터 처리 실패: {e}")
    finally:
        db.close()
        

def _save_vilage_cycles(db, target_date, target_d, today, now_kst, KST):
    """VilageFcst 사이클 저장 — past/today/future 분기. 실패 시 KIMG 폴백."""
    try:
        if target_d < today:
            for time_str in _VILAGE_CYCLE_KST:
                v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
        elif target_d == today:
            for time_str in _VILAGE_CYCLE_KST:
                ct = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)
                if ct > now_kst:
                    continue
                v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
            v = fetch_kma_vilage_wind(KMA_KEY, target_date)
            if not v.empty:
                db.save_forecast(v, auto_add_capacity=False)
        else:
            v = fetch_kma_vilage_wind(KMA_KEY, target_date)
            if not v.empty:
                db.save_forecast(v, auto_add_capacity=False)
    except Exception as e:
        logger.warning(f"[{target_date}] VilageFcst 수집 실패 (KIMG 사용): {e}")


# ============================================================================
# 시나리오 3: 예측 정보 업데이트 (현재 기준 과거는 3일전, 미래는 1일 후 까지)
# ============================================================================
def daily_forecast_and_predict(start_date, end_date):
    """ 예보 데이터 통합 업데이트 (KPX + KMA 병합 처리) """
    from datetime import datetime, timezone, timedelta
    
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)
    today = now_kst.date()

    NCM_CYCLE_KST    = ["03:00", "11:00", "17:00"]  # KIMG NCM 사이클 (기존)
    VILAGE_CYCLE_KST = ["01:00", "11:00", "17:00"]  # VilageFcst 사이클 (D-1 23:00 커버 포함)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    logger.info(f"통합 예보 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt

    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        target_d = current_dt.date()
        logger.info(f"[{target_date}] 데이터 수집 시작...")

        # 1. KPX
        try:
            kpx_forecast = fetch_kpx_future(target_date, KPX_KEY)
            if not kpx_forecast.empty:
                db.save_forecast(kpx_forecast, auto_add_capacity=True)
                logger.info(f"[{target_date}] KPX 저장 완료")
        except Exception as e:
            logger.error(f"[{target_date}] KPX API 실패: {e}")

        # 2. KIMG NCM (남쪽+북쪽 동시, 기존 사이클 유지)
        try:
            if target_d < today:
                logger.info(f"[{target_date}] KMA NCM 과거 소급 ({len(NCM_CYCLE_KST)} 사이클)")
                for time_str in NCM_CYCLE_KST:
                    kma_data = _fetch_kma_concurrent(target_date, as_of_kst=f"{target_date} {time_str}")
                    if not kma_data.empty:
                        db.save_forecast(kma_data, auto_add_capacity=True)
            elif target_d == today:
                logger.info(f"[{target_date}] KMA NCM 오늘 멀티사이클")
                for time_str in NCM_CYCLE_KST:
                    ct = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)
                    if ct > now_kst:
                        continue
                    kma_data = _fetch_kma_concurrent(target_date, as_of_kst=f"{target_date} {time_str}")
                    if not kma_data.empty:
                        db.save_forecast(kma_data, auto_add_capacity=True)
                kma_data = _fetch_kma_concurrent(target_date)
                if not kma_data.empty:
                    db.save_forecast(kma_data, auto_add_capacity=True)
            else:
                kma_data = _fetch_kma_concurrent(target_date)
                if not kma_data.empty:
                    db.save_forecast(kma_data, auto_add_capacity=True)
        except Exception as e:
            logger.error(f"[{target_date}] KMA NCM API 실패: {e}")

        # 3. VilageFcst 북쪽 풍속 (별도 사이클 — D-3~D+2 범위 내에서만 동작)
        try:
            if target_d < today:
                for time_str in VILAGE_CYCLE_KST:
                    v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                    if not v.empty:
                        db.save_forecast(v, auto_add_capacity=False)
            elif target_d == today:
                for time_str in VILAGE_CYCLE_KST:
                    ct = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)
                    if ct > now_kst:
                        continue
                    v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                    if not v.empty:
                        db.save_forecast(v, auto_add_capacity=False)
                v = fetch_kma_vilage_wind(KMA_KEY, target_date)
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
            else:
                v = fetch_kma_vilage_wind(KMA_KEY, target_date)
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
        except Exception as e:
            logger.warning(f"[{target_date}] VilageFcst 수집 실패 (KIMG 사용): {e}")

        current_dt += timedelta(days=1)
        
    db.close()
    logger.info("전체 예보 통합 업데이트 완료")


def daily_forecast_kma(start_date, end_date):
    """
    KMA 예보 데이터 수집 (멀티사이클 지원)
    
    - 과거 날짜: 당시 가용했던 사이클을 시간순 호출하여 소급 수집
      → save_forecast의 UPSERT로 각 시간대에 가장 가까운 예보가 최종 반영됨
    - 오늘 날짜: 이미 지난 사이클 순차 호출 + 최신 사이클 → 00시부터 23시까지 빈틈없이 커버
    - 미래 날짜: 현재 사용 가능한 최신 사이클로 호출
    
    [수정 이력]
    - CYCLE_AVAILABLE_KST 첫 사이클을 "02:00"으로 변경
      → PUBLISH_DELAY_H=2이므로 02:00 기준 available_utc = 전날 15:00 UTC
      → 전날 12:00 UTC 사이클 선택 (= 전날 21:00 KST)
      → 이 사이클은 당일 00~02시 KST를 양수 offset(+3h~+5h)으로 커버 가능
      → 기존 "05:00"은 당일 18:00 UTC 사이클 선택 → 00~02시가 음수 offset → 누락
    """
    from datetime import datetime, timezone, timedelta
    
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)
    today = now_kst.date()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # ── 사이클 시간 설계 ──
    # KMA NCM 예보 사이클: 00, 06, 12, 18 UTC (배포 지연 약 2시간)
    # 
    # "02:00" KST → ref_utc=전날17UTC → available=전날15UTC → 전날12UTC 사이클
    #   → base_time_kst = 전날 21:00 KST → 당일 00시 offset=+3h ✓
    # "05:00" KST → ref_utc=전날20UTC → available=전날18UTC → 전날18UTC 사이클
    #   → base_time_kst = 당일 03:00 KST → 당일 03시부터 커버 (00~02시 보완)
    # "11:00" KST → 당일00UTC 사이클 → 09시부터
    # "17:00" KST → 당일06UTC 사이클 → 15시부터
    # "23:00" KST → 당일12UTC 사이클 → 21시부터
    #
    # UPSERT로 나중 사이클이 같은 시간대를 덮어쓰므로,
    # 앞쪽 사이클은 빈 시간대를 채우는 역할, 뒤쪽이 최신 예보로 갱신하는 역할
    NCM_CYCLE_KST    = ["03:00", "11:00", "17:00"]  # KIMG NCM (기존 불변)
    VILAGE_CYCLE_KST = ["01:00", "11:00", "17:00"]  # VilageFcst (D-1 23:00 커버 포함)

    logger.info(f"KMA 예보 수집: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt

    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        target_d = current_dt.date()

        # ── KIMG NCM (기존 사이클 그대로) ──
        try:
            if target_d < today:
                logger.info(f"[{target_date}] NCM 과거 소급 ({len(NCM_CYCLE_KST)} 사이클)")
                for time_str in NCM_CYCLE_KST:
                    kma_data = _fetch_kma_concurrent(target_date, as_of_kst=f"{target_date} {time_str}")
                    if not kma_data.empty:
                        db.save_forecast(kma_data, auto_add_capacity=True)
                logger.info(f"[{target_date}] NCM 과거 소급 완료")
            elif target_d == today:
                logger.info(f"[{target_date}] NCM 오늘 멀티사이클")
                for time_str in NCM_CYCLE_KST:
                    ct = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)
                    if ct > now_kst:
                        continue
                    kma_data = _fetch_kma_concurrent(target_date, as_of_kst=f"{target_date} {time_str}")
                    if not kma_data.empty:
                        db.save_forecast(kma_data, auto_add_capacity=True)
                kma_data = _fetch_kma_concurrent(target_date)
                if not kma_data.empty:
                    db.save_forecast(kma_data, auto_add_capacity=True)
                    logger.info(f"[{target_date}] NCM 오늘 완료")
                else:
                    logger.warning(f"[{target_date}] NCM 최신 사이클 없음")
            else:
                kma_data = _fetch_kma_concurrent(target_date)
                if not kma_data.empty:
                    db.save_forecast(kma_data, auto_add_capacity=True)
                    logger.info(f"[{target_date}] NCM {len(kma_data)}행 저장")
                else:
                    logger.warning(f"[{target_date}] NCM 데이터 없음")
        except Exception as e:
            logger.error(f"[{target_date}] KMA NCM API 실패: {e}")

        # ── VilageFcst 북쪽 풍속 (별도 사이클, D-3~D+2 범위 내에서만 동작) ──
        try:
            if target_d < today:
                for time_str in VILAGE_CYCLE_KST:
                    v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                    if not v.empty:
                        db.save_forecast(v, auto_add_capacity=False)
            elif target_d == today:
                for time_str in VILAGE_CYCLE_KST:
                    ct = datetime.strptime(f"{target_date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=KST)
                    if ct > now_kst:
                        continue
                    v = fetch_kma_vilage_wind(KMA_KEY, target_date, as_of_kst=f"{target_date} {time_str}")
                    if not v.empty:
                        db.save_forecast(v, auto_add_capacity=False)
                v = fetch_kma_vilage_wind(KMA_KEY, target_date)
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
            else:
                v = fetch_kma_vilage_wind(KMA_KEY, target_date)
                if not v.empty:
                    db.save_forecast(v, auto_add_capacity=False)
        except Exception as e:
            logger.warning(f"[{target_date}] VilageFcst 수집 실패 (KIMG 사용): {e}")

        current_dt += timedelta(days=1)
        
    db.close()
    logger.info("KMA 예보 수집 완료")

def daily_forecast_kpx(start_date, end_date):
    """ KPX 예보 데이터만 단독 수집 """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    logger.info(f"KPX 단독 예보 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt
    
    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        try:
            kpx_forecast = fetch_kpx_future(target_date, KPX_KEY)
            if not kpx_forecast.empty:
                db.save_forecast(kpx_forecast, auto_add_capacity=True)
                logger.info(f"[{target_date}] KPX 단독 저장 완료")
            else:
                logger.warning(f"[{target_date}] KPX 데이터 없음")
        except Exception as e:
            logger.error(f"[{target_date}] KPX API 실패: {e}")
            
        current_dt += timedelta(days=1)
    db.close()


def _merge_south_north(ncm_south, ncm_north):
    """남쪽/북쪽 NCM 데이터 병합 헬퍼"""
    if not ncm_south.empty and not ncm_north.empty:
        return pd.concat([ncm_south, ncm_north], axis=1)
    elif not ncm_south.empty:
        logger.warning("북쪽 NCM 없음, 남쪽만 사용")
        return ncm_south
    return pd.DataFrame()


def _fetch_kma_concurrent(target_date, as_of_kst=None):
    """KIMG 남쪽(전체기상) + 북쪽(풍속) NCM 동시 수집 후 병합.
    VilageFcst는 별도 사이클로 파이프라인에서 독립 호출."""
    warmup_kma_session(KMA_KEY)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_south = ex.submit(fetch_kma_future_ncm,
                            33.3284, 126.8366, KMA_KEY, target_date, as_of_kst)
        f_north = ex.submit(fetch_kma_future_ncm_wind,
                            KMA_KEY, target_date, as_of_kst)
        ncm_south = f_south.result()
        ncm_north = f_north.result()
    return _merge_south_north(ncm_south, ncm_north)


# ============================================================================
# 유틸리티: 모델 입력 데이터 준비
# ============================================================================
def prepare_model_input(df):
    """
    모델 입력에 필요한 파생 변수(시간 주기성, 태양광/대기 일사량 등) 생성
    """
    if df.empty:
        return df
    
    df = df.copy()
    if df.index.name == 'timestamp':
        df = df.reset_index()
    
    # timestamp 파싱 (다양한 포맷에 대응하도록 errors='coerce' 사용)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # 시간 피처
    df['Hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
    df['Year_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.dayofyear / 365)
    df['Year_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.dayofyear / 365)
    
    # 1. Extra Radiation 및 태양 고도각
    lat, lon = 33.3284, 126.8366
    times = pd.DatetimeIndex(df['timestamp_dt']).tz_localize('Asia/Seoul')

    df['Solar_Elevation'] = pvlib.solarposition.get_solarposition(times, lat, lon)['elevation'].values

    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    df['Extra_Radiation'] = dni_extra * np.sin(np.radians(df['Solar_Elevation']))
    df['Extra_Radiation'] = df['Extra_Radiation'].clip(lower=0)

    # 2. Solar Elevation 스케일링

    min_el = -78.45367895445581
    max_el = 78.675691856549
    
    # 0으로 나누는 오류 방지
    if max_el != min_el:
        df['Solar_Elevation_scaled'] = (df['Solar_Elevation'] - min_el) / (max_el - min_el)
    else:
        df['Solar_Elevation_scaled'] = 0.0
    
    # smp_gap
    if 'smp_jeju' in df.columns and 'smp_land' in df.columns:
        df['smp_gap'] = df['smp_jeju'] - df['smp_land']
        
    # solar_damping (당일 주간 누적 강수량 기반 일사 감쇄 계수)
    df_temp = df.set_index('timestamp_dt')
    daily_rain = df_temp.groupby(df_temp.index.date)['rainfall'].transform(
        lambda x: x.between_time('06:00', '20:00').sum()
    )
    df['solar_damping'] = np.exp(-0.163 * daily_rain.clip(upper=10).values)
    
    df = df.set_index('timestamp')
    df = df.drop(columns=['timestamp_dt', 'Solar_Elevation'], errors='ignore')
    
    return df


def run_model_prediction(target_date, db, assets):
    solar_model, wind_model, scalers, metadata, device = assets 
    
    # ── seq_len 분리 ──
    seq_len_solar = metadata['SEQ_LEN_SOLAR']  # 336
    seq_len_wind = metadata['SEQ_LEN_WIND']    # 72
    pred_len = metadata['PRED_LEN']             # 24
    
    WIND_SPD_CAP = 20.0
    CUTOFF_WIND_SPD = 25.0
    
    seq_len_max = max(seq_len_solar, seq_len_wind)
    total_len = seq_len_max + pred_len  # 336 + 24 = 360
    
    features_solar = metadata['features_solar']
    features_wind = metadata['features_wind']
    
    future_features_solar = [col for col in features_solar if 'Utilization' not in col]
    future_features_wind = [col for col in features_wind if 'Utilization' not in col]
    
    # DB 데이터 조회
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = target_dt - timedelta(hours=seq_len_max)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = (target_dt.replace(hour=23)).strftime("%Y-%m-%d %H:%M:%S")
    target_date_str = target_dt.strftime("%Y-%m-%d 00:00:00")
    
    # [변경] past=ASOS, future=forecast 명확 분리
    df = db.get_model_input(start_str, end_str, target_date_str)

    # [추가] 롤링 예측 지원 (+2일, +3일)
    if 'est_Solar_Utilization' in df.columns and 'Solar_Utilization' in df.columns:
        df['Solar_Utilization'] = df['Solar_Utilization'].fillna(df['est_Solar_Utilization'])
    if 'est_Wind_Utilization' in df.columns and 'Wind_Utilization' in df.columns:
        df['Wind_Utilization'] = df['Wind_Utilization'].fillna(df['est_Wind_Utilization'])
    
    north_null_count = df['wind_spd_north'].isnull().sum()
    input_info = {
        "total_rows": len(df),
        "expected_rows": total_len,
        "missing_values": 0,
        "past_hours_found": 0,
        "future_hours_found": 0
    }
    
    if df.empty or len(df) < total_len:
        return False, f"[{target_date}] 데이터 길이가 부족합니다. (필요: {total_len}, 현재: {len(df)})", input_info
        
    df = prepare_model_input(df)
    
    # ── 풍력 파생 피처 생성 ──
    df['wind_spd_sq'] = df['wind_spd'] ** 2
    df['wind_spd_cu'] = df['wind_spd'] ** 3
    
    # 1. 과거/미래 분리
    past_df = df.iloc[:seq_len_max]
    future_df = df.iloc[seq_len_max:total_len]
    
    # 2. 결측치 검사
    north_mapping = {
        'wind_spd': 'wind_spd_north',
        'wd_sin': 'wd_sin_north',
        'wd_cos': 'wd_cos_north'
    }
    check_features_wind = [north_mapping.get(col, col) for col in future_features_wind
                           if col not in ['wind_spd_sq', 'wind_spd_cu', 'wind_zone']]
    
    used_features = list(set(future_features_solar + check_features_wind))
    target_cols = ['Solar_Utilization', 'Wind_Utilization']
    
    past_missing = past_df[used_features + target_cols].isnull().sum().sum()
    future_missing = future_df[used_features].isnull().sum().sum()
    real_missing_cnt = int(past_missing + future_missing)
    
    input_info["missing_values"] = real_missing_cnt
    input_info["past_hours_found"] = len(past_df)
    input_info["future_hours_found"] = len(future_df)
    
    # [변경] 결측 에러 메시지 개선
    if real_missing_cnt > 0:
        past_util_missing = past_df[['Solar_Utilization', 'Wind_Utilization']].isnull().sum().sum()
        if past_util_missing > 0:
            return False, (
                f"[{target_date}] 과거 구간에 예측값이 없는 날짜가 있습니다. "
                f"이전 날짜부터 순서대로 예측을 먼저 실행해주세요."
            ), input_info
        return False, (
            f"모델 입력 데이터에 {real_missing_cnt}개의 실제 결측치가 존재합니다. "
            f"[Option A]에서 데이터를 점검하세요."
        ), input_info
        
    df = df.fillna(0).infer_objects(copy=False)
    
    
    # ── fillna 이후 파생 피처 재생성 (안전장치) ──
    df['wind_spd_sq'] = df['wind_spd'] ** 2
    df['wind_spd_cu'] = df['wind_spd'] ** 3
    
    # ==========================================
    # 태양광 / 풍력 독립 스케일링
    # ==========================================
    scaler_solar = scalers['solar']
    scaler_wind = scalers['wind']
    
    # 태양광
    df_solar = df.copy()
    df_solar[future_features_solar] = scaler_solar.transform(df_solar[future_features_solar])
    
    # 풍력 (북쪽 데이터 덮어쓰기 후 파생변수 재계산)
    # VilageFcst(_v) 우선, 없으면 KIMG(_north) 폴백
    df_wind = df.copy()
    if ('wind_spd_north_v' in df_wind.columns
            and df_wind['wind_spd_north_v'].notna().any()):
        df_wind['wind_spd'] = df_wind['wind_spd_north_v']
        df_wind['wd_sin']   = df_wind['wd_sin_north_v']
        df_wind['wd_cos']   = df_wind['wd_cos_north_v']
    elif 'wind_spd_north' in df_wind.columns:
        df_wind['wind_spd'] = df_wind['wind_spd_north']
        df_wind['wd_sin']   = df_wind['wd_sin_north']
        df_wind['wd_cos']   = df_wind['wd_cos_north']            
    # ── wind_zone 생성 (클리핑 전 원본 기준) ──
    conditions = [
        df_wind['wind_spd'] < 15,
        (df_wind['wind_spd'] >= 15) & (df_wind['wind_spd'] < 20),
        (df_wind['wind_spd'] >= 20) & (df_wind['wind_spd'] < CUTOFF_WIND_SPD),
        df_wind['wind_spd'] >= CUTOFF_WIND_SPD
    ]
    values = [0.0, 1.0, 0.5, 0.0]
    df_wind['wind_zone'] = np.select(conditions, values, default=0.0)
    
    # ── 풍속 클리핑 + 파생피처 재계산 ──
    df_wind['wind_spd'] = df_wind['wind_spd'].clip(upper=WIND_SPD_CAP)
    df_wind['wind_spd_sq'] = df_wind['wind_spd'] ** 2
    df_wind['wind_spd_cu'] = df_wind['wind_spd'] ** 3
        
    df_wind[future_features_wind] = scaler_wind.transform(df_wind[future_features_wind])
    
    # ==========================================
    # 배치 생성 함수
    # ==========================================
    def create_batch_from_scaled(df_target, seq_len, future_features_list, target_col):
        """seq_len에 맞춰 과거 구간을 잘라서 배치 생성"""
        start_idx = seq_len_max - seq_len
        
        past_df = df_target.iloc[start_idx : seq_len_max]
        future_df = df_target.iloc[seq_len_max : total_len]
        
        past_numeric = past_df[future_features_list].values
        past_y = past_df[[target_col]].values 
        future_numeric = future_df[future_features_list].values
        
        batch = {
            'past_numeric': torch.FloatTensor(past_numeric).unsqueeze(0),
            'past_y': torch.FloatTensor(past_y).unsqueeze(0),            
            'future_numeric': torch.FloatTensor(future_numeric).unsqueeze(0)
        }
        return batch

    # ==========================================
    # 모델 추론
    # ==========================================
    try:
        # 태양광 (seq_len=336)
        solar_batch = create_batch_from_scaled(df_solar, seq_len_solar, future_features_solar, 'Solar_Utilization')
        with torch.no_grad():
            pred_solar = solar_model(solar_batch, device=device).squeeze().cpu().numpy()
            
        # 풍력 (seq_len=72)
        wind_batch = create_batch_from_scaled(df_wind, seq_len_wind, future_features_wind, 'Wind_Utilization')
        with torch.no_grad():
            pred_wind = wind_model(wind_batch, device=device).squeeze().cpu().numpy()
            
    except Exception as e:
        return False, f"추론 중 에러 발생: {e}", input_info
        
    # 클리핑
    pred_solar = np.clip(pred_solar, a_min=0.0, a_max=1.0)
    pred_wind = np.clip(pred_wind, a_min=0.0, a_max=1.0)

    # ── 태양광 후처리: sigmoid 기반 일사량 스케일링 ──
    # P75 of daytime solar_rad → smooth scale via sigmoid
    # scale = floor + (1 - floor) * sigmoid(k * (P75 - mid))
    # Parameters tuned from 822 days of 2024+ historical data
    SOLAR_SIGMOID_MID = 1.45    # P75 midpoint (fitted from Feb-Apr 2026 model results)
    SOLAR_SIGMOID_K = 5.5       # steepness (softened from 12.6 for robustness)
    SOLAR_SIGMOID_FLOOR = 0.33  # minimum scale (model already adjusts for clouds)
    # 기존 1.32 2.25 0.17

    raw_solar_rad = df['solar_rad'].iloc[seq_len_max:total_len].values
    daytime_rad = raw_solar_rad[raw_solar_rad > 0]
    daily_p75_rad = float(np.percentile(daytime_rad, 75)) if len(daytime_rad) > 0 else 0.0
    daily_max_rad = float(raw_solar_rad.max())

    sigmoid_val = 1.0 / (1.0 + np.exp(-SOLAR_SIGMOID_K * (daily_p75_rad - SOLAR_SIGMOID_MID)))
    daily_scale = SOLAR_SIGMOID_FLOOR + (1.0 - SOLAR_SIGMOID_FLOOR) * sigmoid_val
    daily_scale = min(daily_scale, 1.0)

    pred_solar = pred_solar * daily_scale
    solar_scale_pct = round(float(daily_scale) * 100, 1)

    logger.info(
        f"[{target_date}] 태양광 후처리: "
        f"P75={daily_p75_rad:.2f} MJ/m² → 스케일 {solar_scale_pct:.1f}%"
    )

    input_info["solar_postprocess"] = True
    input_info["solar_scale_pct"] = solar_scale_pct
    input_info["solar_daily_max_rad"] = daily_max_rad
    input_info["solar_daily_p75_rad"] = daily_p75_rad

    # ── cut-off 후처리: 원본 풍속 25m/s 이상이면 이용률 0 ──
    if 'wind_spd_north' in df.columns:
        raw_wind_spd = df['wind_spd_north'].iloc[seq_len_max:total_len].values
    else:
        raw_wind_spd = df['wind_spd'].iloc[seq_len_max:total_len].values
    pred_wind[raw_wind_spd >= CUTOFF_WIND_SPD] = 0.0

    # 결과 DB 저장
    target_timestamps = df.index[seq_len_max:total_len]
    pred_df = pd.DataFrame({
        'timestamp': target_timestamps,
        'est_Solar_Utilization': pred_solar,
        'est_Wind_Utilization': pred_wind
    })

    updated_rows = db.update_forecast_predictions(pred_df)
    return True, f"[Success] [{target_date}] 태양광/풍력 예측 완료 및 {updated_rows}행 저장 성공!", input_info


