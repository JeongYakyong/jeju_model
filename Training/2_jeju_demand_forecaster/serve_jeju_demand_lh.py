"""2-B. 제주 수요 D+1~D+7 장지평 서빙 — DB 전용 (직접 다지평 LGBM, 2-A).

================================================================================
무엇을 하나
================================================================================
input_data_jeju.db 한 곳에서 읽고 쓴다.
  - 입력 : historical (실측 real_demand_jeju + 3지점 기상),
           forecast   (기상예보 west/east/south, day_type)
           data/jeju_ppa_btm_capacity_mw.csv (월별 BTM/PPA 용량)
  - 모델 : model/models/lgbm_jeju_demand_direct.txt  (2-A, h=1..168, quantile α=0.60)
  - 출력 : forecast.jeju_est_demand_lh  ← origin 다음 D+1~D+n 수요 예측 UPSERT
           (★ 기존 배포 D+1 컬럼 jeju_est_demand_new 은 건드리지 않음 — 별도 운영)

직접 다지평이라 **원하는 지평만큼의 예보만 있으면 된다**:
  24h 예보 → D+1, 168h 예보 → D+7. lag168·rec 는 과거 실측에서 와서 사슬이 없다.
  과거 실측 수요는 원점 직전 ~168h(7일) 필요(lag168·rec168).

설계(2-A 학습과 동일)
================================================================================
  origin O = 지정일 23:00. 타깃 = O+1h .. O+(days*24)h.
  피처 22개 = h, lag168, rec24, rec168, 기상4(기온·습도·일사·풍속, 3지점평균/일사2지점),
    구름4(total/midlow_cloud west·south raw, h≤48만), cap_btmppa_mw,
    흐린날피처(solar_deficit·solar_ramp, h≤48만), 달력(hour/dow/month sin·cos), day_type.
  기상 = forecast 예보 우선, 없으면 (월,시) 기후값 폴백(QM 미적용 — 2-0c 결정).
    구름·흐린날피처는 h≤48 만 사용(그 외 NaN, LGBM 네이티브 처리).
  forecast 매핑: temp_c<-temp / humidity<-reh / solar_rad<-radiation / wind_spd<-wind_spd_10m
                 구름은 forecast total_cloud_*/midlow_cloud_* 그대로(west·south).

공개 API
================================================================================
    predict_demand_to_db(origin_date, days_ahead=7, write=True)
    backfill_demand_to_db(start, end, days_ahead=7)
"""
from __future__ import annotations
import os, sqlite3
import numpy as np, pandas as pd
import lightgbm as lgb

HERE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(
    HERE, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
MODEL   = os.path.join(HERE, 'model', 'models', 'lgbm_jeju_demand_direct.txt')
CAP_CSV = os.path.join(HERE, 'data', 'jeju_ppa_btm_capacity_mw.csv')

ST       = ['west', 'east', 'south']      # 기온/습도/풍속 3지점
SOLAR_ST = ['west', 'south']              # 일사 2지점
WX       = ['temp_c', 'humidity', 'solar_rad', 'wind_spd']
CLOUD    = ['total_cloud_west', 'total_cloud_south', 'midlow_cloud_west', 'midlow_cloud_south']
OUT_COL  = 'jeju_est_demand_lh'
CLOUD_MAX_H = 48                          # 구름·흐린날피처는 h<=48 만
# forecast 테이블 매핑 (모델기상 -> forecast 접두사 / 사용 지점)
FORE = {'temp_c': ('temp', ST), 'humidity': ('reh', ST),
        'solar_rad': ('radiation', SOLAR_ST), 'wind_spd': ('wind_spd_10m', ST)}
FEAT = ['h', 'lag168', 'rec24', 'rec168', 'temp_c', 'humidity', 'solar_rad', 'wind_spd',
        'total_cloud_west', 'total_cloud_south', 'midlow_cloud_west', 'midlow_cloud_south',
        'cap_btmppa_mw', 'solar_deficit', 'solar_ramp',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day_type']
DAYTYPE_CATS = ['holiday', 'weekday', 'weekend']   # 2-A astype('category') 정렬과 동일


def _conn():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# 1. 실측 수요 + 실측 일사(램프/원점값용) 시계열 — 연속 시간축, 0/결측 시간보간
# =============================================================================
def load_series():
    sel = ['timestamp', 'real_demand_jeju'] + [f'solar_rad_{s}' for s in SOLAR_ST]
    with _conn() as con:
        d = pd.read_sql(f"SELECT {', '.join(sel)} FROM historical", con, parse_dates=['timestamp'])
    d = d.sort_values('timestamp')
    idx = pd.date_range(d['timestamp'].min(), d['timestamp'].max(), freq='h')
    d = d.set_index('timestamp').reindex(idx)
    dem = d['real_demand_jeju'].replace(0, np.nan).interpolate(method='time')
    solar = d[[f'solar_rad_{s}' for s in SOLAR_ST]].mean(axis=1).interpolate(method='time')
    dem.index.name = 'timestamp'; solar.index.name = 'timestamp'
    return dem, solar


# =============================================================================
# 2. (월,시) 기후값 — 먼 지평 기상 폴백 (모델 기상 + 구름)
# =============================================================================
def build_climatology() -> dict:
    sel = ['timestamp']
    for s in ST:
        sel += [f'temp_c_{s}', f'humidity_{s}', f'wind_spd_{s}']
    for s in SOLAR_ST:
        sel += [f'solar_rad_{s}']
    sel += CLOUD
    with _conn() as con:
        df = pd.read_sql(f"SELECT {', '.join(sel)} FROM historical", con, parse_dates=['timestamp'])
    df['temp_c']   = df[[f'temp_c_{s}' for s in ST]].mean(axis=1)
    df['humidity'] = df[[f'humidity_{s}' for s in ST]].mean(axis=1)
    df['wind_spd'] = df[[f'wind_spd_{s}' for s in ST]].mean(axis=1)
    df['solar_rad']= df[[f'solar_rad_{s}' for s in SOLAR_ST]].mean(axis=1)
    df['month'] = df.timestamp.dt.month; df['hour'] = df.timestamp.dt.hour
    clim = {w: df.groupby(['month', 'hour'])[w].mean() for w in WX + CLOUD}
    return clim


# =============================================================================
# 3. BTM/PPA 용량 (월별) — 결정값, 누락월은 최신값 캐리포워드
# =============================================================================
def load_capacity() -> pd.DataFrame:
    cap = pd.read_csv(CAP_CSV).sort_values(['year', 'month']).reset_index(drop=True)
    return cap


def _cap_for(ts: pd.Timestamp, cap: pd.DataFrame) -> float:
    row = cap[(cap.year == ts.year) & (cap.month == ts.month)]
    if len(row):
        return float(row.cap_btmppa_mw.iloc[0])
    return float(cap.cap_btmppa_mw.iloc[-1])   # 미래월: 최신값 캐리포워드


# =============================================================================
# 4. 타깃 기상 조립 (forecast 우선 + 기후값 폴백) + day_type
# =============================================================================
def load_target_weather(targets: pd.DatetimeIndex, clim: dict) -> pd.DataFrame:
    tmin = targets.min().strftime('%Y-%m-%d %H:%M:%S')
    tmax = targets.max().strftime('%Y-%m-%d %H:%M:%S')
    fcols = ['timestamp', 'day_type']
    for w in WX:
        pre, sts = FORE[w]; fcols += [f'{pre}_{s}' for s in sts]
    fcols += CLOUD
    with _conn() as con:
        cols_sql = ', '.join(f'"{c}"' for c in fcols)
        fc = pd.read_sql(
            f"SELECT {cols_sql} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            con, params=(tmin, tmax), parse_dates=['timestamp'])
    fc = fc.set_index('timestamp')
    out = pd.DataFrame(index=targets)
    # 기상 4종: forecast 평균 우선, 없으면 (월,시) 기후값
    for w in WX:
        pre, sts = FORE[w]; cc = [f'{pre}_{s}' for s in sts]
        fmean = (fc[cc].apply(pd.to_numeric, errors='coerce').mean(axis=1).reindex(targets)
                 if len(fc) else pd.Series(np.nan, index=targets))
        # D+5.5(135h) 이후 forecast 는 3h 행만 존재(KIMG 1h 해상도 한계) -- 그 사이
        # 구멍은 양옆 실예보의 시간 보간이 기후값보다 정확하므로 먼저 메운다.
        # 신뢰성 한계: limit=3 (앵커 간격 4h 까지만, 그 이상 구멍은 보간 안 함).
        # limit_area='inside' 로 예보 범위 밖 외삽 금지.
        fmean = fmean.interpolate(method='time', limit=3, limit_area='inside')
        cl = pd.Series([clim[w].get((t.month, t.hour), np.nan) for t in targets], index=targets)
        out[w] = fmean.where(fmean.notna(), cl)
        out[w + '_src'] = np.where(fmean.notna(), 'forecast', 'climatology')
    # 구름 4종 (raw west/south): forecast 우선(3h 구멍은 위와 같이 보간), 없으면 기후값
    for c in CLOUD:
        fv = (pd.to_numeric(fc[c], errors='coerce').reindex(targets)
              if c in fc.columns else pd.Series(np.nan, index=targets))
        fv = fv.interpolate(method='time', limit=3, limit_area='inside')
        cl = pd.Series([clim[c].get((t.month, t.hour), np.nan) for t in targets], index=targets)
        out[c] = fv.where(fv.notna(), cl)
    # day_type: forecast 우선, 없으면 dow 로 추정
    dt = fc['day_type'].reindex(targets) if 'day_type' in fc else pd.Series(index=targets, dtype=object)
    dt = dt.where(dt.notna(),
                  pd.Series(np.where(targets.dayofweek >= 5, 'weekend', 'weekday'), index=targets))
    out['day_type'] = dt.values
    return out


# =============================================================================
# 5. 핵심 — origin 다음 D+1..D+n 직접 다지평 예측
# =============================================================================
def _resolve_origin(dem: pd.Series, origin_date: str | None) -> pd.Timestamp:
    if origin_date is not None:
        return pd.Timestamp(origin_date).normalize() + pd.Timedelta(hours=23)
    last = dem.dropna().index.max()
    O = last.normalize() + pd.Timedelta(hours=23)
    if O > last:
        O = O - pd.Timedelta(days=1)
    return O


def predict_demand_to_db(origin_date: str | None = None, days_ahead: int = 7,
                         write: bool = True, verbose: bool = True) -> pd.DataFrame:
    if not (1 <= days_ahead <= 7):
        raise ValueError('days_ahead 는 1~7 (모델 학습 지평 1..168h)')
    dem, solar_act = load_series()
    O = _resolve_origin(dem, origin_date)

    win = dem.loc[O - pd.Timedelta(hours=167): O]
    if len(win) < 168 or win.isna().any():
        raise ValueError(f'[history] 원점 윈도우({O-pd.Timedelta(hours=167)}~{O}) '
                         f'부족/NaN — 보유 {len(win)}행 (실측 적재 확인)')
    rec24  = float(dem.loc[O - pd.Timedelta(hours=23): O].mean())
    rec168 = float(win.mean())

    H = np.arange(1, days_ahead * 24 + 1)
    targets = pd.DatetimeIndex([O + pd.Timedelta(hours=int(h)) for h in H])
    clim = build_climatology(); cap = load_capacity()
    wx = load_target_weather(targets, clim)

    df = pd.DataFrame(index=targets)
    df['h'] = H
    df['lag168'] = dem.reindex(targets - pd.Timedelta(hours=168)).values
    df['rec24'] = rec24; df['rec168'] = rec168
    for w in WX: df[w] = wx[w].values
    for c in CLOUD: df[c] = np.where(H <= CLOUD_MAX_H, wx[c].values, np.nan)
    df['cap_btmppa_mw'] = [_cap_for(t, cap) for t in targets]
    # 흐린날 피처 (h<=48만): solar_deficit = 1 - solar/clim_solar, solar_ramp = |Δsolar|
    clim_solar = clim['solar_rad']
    cs = np.array([clim_solar.get((t.month, t.hour), np.nan) for t in targets], float)
    solar_t = df['solar_rad'].values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        deficit = np.where(cs > 5, np.clip(1 - solar_t / cs, -0.5, 1.5), 0.0)
    # ramp: 직전시각 일사 대비. h=1 의 직전 = 원점 O 실측 일사.
    solar_prev = np.empty_like(solar_t)
    solar_prev[0] = float(solar_act.reindex([O]).iloc[0]) if O in solar_act.index else np.nan
    solar_prev[1:] = solar_t[:-1]
    ramp = np.abs(solar_t - solar_prev)
    df['solar_deficit'] = np.where(H <= CLOUD_MAX_H, deficit, np.nan)
    df['solar_ramp']    = np.where(H <= CLOUD_MAX_H, ramp, np.nan)
    hr = targets.hour.values; dw = targets.dayofweek.values; mo = targets.month.values
    df['hour_sin'] = np.sin(2*np.pi*hr/24); df['hour_cos'] = np.cos(2*np.pi*hr/24)
    df['dow_sin']  = np.sin(2*np.pi*dw/7);  df['dow_cos']  = np.cos(2*np.pi*dw/7)
    df['month_sin']= np.sin(2*np.pi*mo/12); df['month_cos']= np.cos(2*np.pi*mo/12)
    df['day_type'] = pd.Categorical(wx['day_type'].values, categories=DAYTYPE_CATS)

    booster = lgb.Booster(model_file=MODEL)
    df[OUT_COL] = booster.predict(df[FEAT]).round(1)
    df['dayahead'] = ((df['h'] - 1) // 24 + 1).astype(int)
    df['weather_src'] = wx['temp_c_src'].values
    out = df.reset_index().rename(columns={'index': 'timestamp'})[
        ['timestamp', 'h', 'dayahead', OUT_COL, 'weather_src', 'day_type']]

    if write:
        rows = [(t.strftime('%Y-%m-%d %H:%M:%S'), float(v))
                for t, v in zip(out['timestamp'], out[OUT_COL])]
        with _conn() as con:
            cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
            if OUT_COL not in cols:
                con.execute(f'ALTER TABLE forecast ADD COLUMN "{OUT_COL}" REAL')
            con.executemany(
                f'INSERT INTO forecast ("timestamp","{OUT_COL}") VALUES (?,?) '
                f'ON CONFLICT("timestamp") DO UPDATE SET "{OUT_COL}"=excluded."{OUT_COL}"', rows)
            con.commit()
        if verbose:
            print(f'[DB] forecast.{OUT_COL} ← origin {O:%Y-%m-%d %H:%M} '
                  f'D+1..D+{days_ahead} ({len(rows)}행) UPSERT')
    if verbose:
        nfc = int((out.weather_src == 'forecast').sum())
        print(f'origin={O:%Y-%m-%d %H:%M}  기상: forecast {nfc}h / climatology {len(out)-nfc}h')
        print(out.head(24).to_string(index=False))
    return out


# =============================================================================
# 6. 백필/평가 — 과거 origin 들에 대해 예측 후 실측과 MAPE (낮시간 분리)
# =============================================================================
def backfill_demand_to_db(start: str, end: str, days_ahead: int = 7,
                          write: bool = False, verbose: bool = True) -> pd.DataFrame:
    dem, _ = load_series()
    origins = pd.date_range(pd.Timestamp(start).normalize() + pd.Timedelta(hours=23),
                            pd.Timestamp(end).normalize() + pd.Timedelta(hours=23), freq='D')
    allrows = []
    for O in origins:
        win = dem.loc[O - pd.Timedelta(hours=167): O]
        if len(win) < 168 or win.isna().any():
            continue
        try:
            o = predict_demand_to_db(O.strftime('%Y-%m-%d'), days_ahead, write=write, verbose=False)
            o['actual'] = dem.reindex(pd.DatetimeIndex(o['timestamp'])).values
            o['hour'] = pd.DatetimeIndex(o['timestamp']).hour
            allrows.append(o)
        except Exception:
            continue
    if not allrows:
        print('예측 가능한 origin 없음'); return pd.DataFrame()
    res = pd.concat(allrows, ignore_index=True)
    if verbose:
        m = res.dropna(subset=['actual']); m = m[m.actual > 0]
        day = m[(m.hour >= 8) & (m.hour <= 16)]
        print(f'[backfill] origins {len(allrows)}  예측행 {len(res)}  실측대조 {len(m)}')
        def mp(g): return float(np.mean(np.abs(g.actual - g[OUT_COL]) / g.actual) * 100)
        for dn, g in m.groupby('dayahead'):
            gd = day[day.dayahead == dn]
            extra = f' | 낮(08-16) {mp(gd):.2f}% (n={len(gd)})' if len(gd) else ''
            print(f'  D+{dn}: 전시간 MAPE {mp(g):.2f}%  (n={len(g)}){extra}')
    return res


# =============================================================================
# 7. CLI
# =============================================================================
if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 수요 D+1~D+7 장지평 서빙 (직접 다지평 LGBM, 2-A)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict', help='origin 다음 D+1..D+n 예측 → DB')
    pp.add_argument('date', nargs='?', default=None, help='origin 날짜 YYYY-MM-DD (생략시 최신)')
    pp.add_argument('--days', type=int, default=7, help='며칠 앞까지 (1~7, 기본 7)')
    pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill', help='과거 구간 예측·MAPE 평가(낮시간 분리)')
    bf.add_argument('start'); bf.add_argument('end')
    bf.add_argument('--days', type=int, default=7)
    bf.add_argument('--write', action='store_true')
    a = p.parse_args()
    if a.cmd == 'predict':
        predict_demand_to_db(a.date, a.days, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_demand_to_db(a.start, a.end, a.days, write=a.write)
