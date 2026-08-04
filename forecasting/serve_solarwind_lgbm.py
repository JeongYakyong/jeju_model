"""제주 Solar/Wind 이용률 → net_load 장지평 서빙 (LGBM, 순수기상 horizon-무관).

3단계 비교(2026-06-08) 결론의 LGBM 쪽 서빙 본체. 하이브리드의 장지평(및 폴백) 담당.
- util = f(기상, 시각, 계절)뿐 → 지평 의존 피처 없음 → 단일 모델 1개(채널당)가 D+1·D+2·D+3·D+7 전부 서빙.
- 각 대상일의 forecast 기상을 넣어 예측. forecast 없으면 (월,시) 기후값 폴백(2-B·5-B와 동일).
- net_load = 수요 − solar_gen − wind_gen (수요=forecast 테이블, gen=util×capacity).
- 피처/clearsky 평년은 학습(3cmp-A)과 동일하게 **메인 DB historical** 의 학습창으로 재현.
  학습창은 feat_meta.json 의 `train` 이 알려주므로, 재학습으로 창이 바뀌어도 자동으로 따라간다
  (2026-07-30: 얼린 CSV + 하드코딩 ≤2024 에서 옮겨왔다. 구 메타면 ≤2024 로 폴백).

피처(§0.6 확정): SOLAR = solar_rad·total_cloud·midlow_cloud·solar_damping(west·south)
  + clearsky_ratio(west·south) + hour sin/cos + month sin/cos
  WIND = wind_spd·wind_zone(west·east) + 풍향(west) + hour sin/cos + year sin/cos
ramp/vol·forecast전용변수는 서빙(forecast)에서 역효과 확인되어 미사용(후처리는 별도).

출력 컬럼(forecast 테이블, _lgbm 접미사 — PatchTST D+1 출력 est_*_jeju 와 분리):
  est_solar_util_jeju_lgbm, est_wind_util_jeju_lgbm, est_solar_gen_jeju_lgbm,
  est_wind_gen_jeju_lgbm, est_net_load_jeju_lgbm

API:  predict_lgbm_to_db(origin, horizons=(1,2,3,7))  /  backfill_lgbm_to_db(start,end)
"""
from __future__ import annotations
import os, sys, json, sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

DB_PATH = P.DB_JEJU
LGBM = P.DIR_MODELS_SOLARWIND_LGBM
# 원자료는 DB historical 에서 읽는다 (구 P.REF_SOLARWIND_RAW CSV 는 2026-06-01 에서 멈춰 있다)

SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'
JEJU_HORIZONS = (1, 2, 3, 7)
DEMAND_COLS = ['jeju_est_demand_lh', 'jeju_est_demand_new']   # 우선순위(장지평 → D+1)
WIND_SPD_CAP, CUTOFF = 20.0, 25.0

OUT = dict(su='est_solar_util_jeju_lgbm', wu='est_wind_util_jeju_lgbm',
           sg='est_solar_gen_jeju_lgbm', wg='est_wind_gen_jeju_lgbm', nl='est_net_load_jeju_lgbm')
OUT_COLS = list(OUT.values())

# forecast(서빙) → 캐노니컬(학습) 컬럼 매핑
FORE_MAP = {'radiation_west': 'solar_rad_west', 'radiation_south': 'solar_rad_south',
            'total_cloud_west': 'total_cloud_west', 'total_cloud_south': 'total_cloud_south',
            'midlow_cloud_west': 'midlow_cloud_west', 'midlow_cloud_south': 'midlow_cloud_south',
            'rainfall_west': 'rainfall_west', 'rainfall_south': 'rainfall_south',
            'wind_spd_10m_west': 'wind_spd_west', 'wind_spd_10m_east': 'wind_spd_east',
            'wd_sin_10m_west': 'wd_sin_west', 'wd_cos_10m_west': 'wd_cos_west'}
# (2026-06-17) east 풍향 추가 시도 → forecast 기준 악화(D+1 wind 0.125→0.135)로 복귀.
# west/east 풍향이 거의 중복이라 east는 예보오차만 더함(실측≠예보 교훈). west 단독 유지.
CANON = list(dict.fromkeys(FORE_MAP.values()))   # 캐노니컬 raw 컬럼(중복 제거)


# =============================================================================
# 피처 빌드 (학습 3cmp-A 와 동일)
# =============================================================================
def _damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    return np.exp(-0.163 * daily.clip(upper=10))


def _wind_zone(raw):
    cond = [raw < 15, (raw >= 15) & (raw < 20), (raw >= 20) & (raw < CUTOFF), raw >= CUTOFF]
    return np.select(cond, [0.0, 1.0, 0.5, 0.0], default=0.0)


def build_features(df, clim=None):
    df = df.copy()
    df['hour_sin'] = np.sin(2*np.pi*df.index.hour/24); df['hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
    df['month_sin'] = np.sin(2*np.pi*df.index.month/12); df['month_cos'] = np.cos(2*np.pi*df.index.month/12)
    df['year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365); df['year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)
    for st in ['west', 'south']:
        df[f'solar_damping_{st}'] = _damping(df, st)
    for st in ['west', 'east']:
        df[f'wind_zone_{st}'] = _wind_zone(df[f'wind_spd_{st}'])
        df[f'wind_spd_{st}'] = df[f'wind_spd_{st}'].clip(upper=WIND_SPD_CAP)
    if clim is None:
        clim = {st: df.groupby([df.index.month, df.index.hour])[f'solar_rad_{st}'].quantile(0.90)
                for st in ['west', 'south']}
    for st in ['west', 'south']:
        key = list(zip(df.index.month, df.index.hour))
        cs = clim[st].reindex(key).values
        ratio = np.where(cs > 0.05, df[f'solar_rad_{st}'].values / cs, 0.0)
        df[f'clearsky_ratio_{st}'] = np.clip(ratio, 0, 1.5)
    return df, clim


# =============================================================================
# 자산 로드 (메모이즈): 모델 + clearsky 평년(train) + 기상 기후값 폴백
# =============================================================================
_A = None


def _raw_from_db():
    """학습 원자료(구 solarwind_raw_jeju.csv)와 같은 컬럼을 메인 DB historical 에서 읽는다."""
    con = sqlite3.connect(DB_PATH)
    try:
        raw = pd.read_sql(f'SELECT timestamp, {", ".join(CANON)} FROM historical ORDER BY timestamp',
                          con, parse_dates=['timestamp'])
    finally:
        con.close()
    return raw.set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')


def _train_end(meta):
    """학습창 끝 시각. feat_meta.json 의 `train`(예 '<=2026-01-31 23:00')을 따른다.

    구 메타에는 이 필드가 없다 — 그 시절 학습창이 train ≤2024 였으므로 그대로 폴백한다.
    이렇게 두면 재학습으로 학습창이 바뀌어도 서빙 clearsky 평년이 저절로 따라간다.
    """
    written = str(meta.get('train', '')).lstrip('<=').strip()
    return written or '2024-12-31 23:00'


def load_assets(force=False):
    global _A
    if _A is not None and not force:
        return _A
    meta = json.load(open(os.path.join(LGBM, 'feat_meta.json'), encoding='utf-8'))
    m_solar = lgb.Booster(model_file=os.path.join(LGBM, 'lgbm_solar_util.txt'))
    m_wind  = lgb.Booster(model_file=os.path.join(LGBM, 'lgbm_wind_util.txt'))
    # clearsky 평년 = 학습 train 구간으로 재현 (학습창은 메타가 알려준다)
    raw = _raw_from_db()
    _, clim = build_features(raw[raw.index <= _train_end(meta)])
    # 기상 기후값(월,시): 캐노니컬 raw 컬럼 평균 (forecast 없을 때 폴백)
    wx_clim = raw[CANON].groupby([raw.index.month, raw.index.hour]).mean()
    _A = (m_solar, m_wind, meta, clim, wx_clim)
    return _A


def _conn():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# 대상일 기상 조립 (forecast 우선 · 없으면 기후값 폴백)
# =============================================================================
def _day_weather(con, day, wx_clim):
    idx = pd.date_range(day, periods=24, freq='h')
    sel = ', '.join(f'"{c}"' for c in ['timestamp'] + list(FORE_MAP))
    # D+5.5(135h) 이후 forecast 는 3h 행만 존재(KIMG 1h 해상도 한계).  4h 이내
    # 구멍은 양옆 실예보의 시간 보간이 기후값보다 정확하므로 기후값 폴백 전에
    # 먼저 메운다 (limit=3 = 앵커 간격 4h 까지, 신뢰성 한계.  limit_area='inside'
    # 로 외삽 금지).  경계 시간도 보간되도록 ±3h 확장 조회 후 대상일로 트림.
    ext = pd.date_range(idx[0] - pd.Timedelta(hours=3), idx[-1] + pd.Timedelta(hours=3), freq='h')
    f = pd.read_sql(f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp', con,
                    params=(ext[0].strftime('%Y-%m-%d %H:%M:%S'), ext[-1].strftime('%Y-%m-%d %H:%M:%S')),
                    parse_dates=['timestamp']).set_index('timestamp')
    f = f.apply(pd.to_numeric, errors='coerce').rename(columns=FORE_MAP).reindex(ext)
    f = f.interpolate(method='time', limit=3, limit_area='inside').reindex(idx)
    src = 'forecast'
    need = f[CANON]
    if len(f) < 24 or need.isna().any().any():
        # 결측 위치를 (월,시) 기후값으로 채움
        fill = pd.DataFrame({c: [wx_clim.loc[(t.month, t.hour), c] for t in idx] for c in CANON}, index=idx)
        for c in CANON:
            f[c] = f[c].fillna(fill[c]) if c in f else fill[c]
        src = 'forecast+clim' if need.notna().any().any() else 'clim'
    return f[CANON], src


def _latest_capacity(con, day, gen_col, cap_col):
    start = (pd.Timestamp(day) - pd.Timedelta(hours=720)).strftime('%Y-%m-%d %H:%M:%S')
    end = (pd.Timestamp(day) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_sql(f'SELECT "{cap_col}", "{gen_col}" FROM historical WHERE timestamp BETWEEN ? AND ?',
                     con, params=(start, end))
    cap = pd.to_numeric(df[cap_col], errors='coerce').dropna()
    if len(cap):
        return float(cap.iloc[-1])
    gen = pd.to_numeric(df[gen_col], errors='coerce')
    return float(gen.max()) if gen.notna().any() else None


def _demand(con, idx):
    cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
    for dc in DEMAND_COLS:
        if dc not in cols:
            continue
        d = pd.read_sql(f'SELECT timestamp, "{dc}" FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
                        con, params=(idx[0].strftime('%Y-%m-%d %H:%M:%S'), idx[-1].strftime('%Y-%m-%d %H:%M:%S')),
                        parse_dates=['timestamp']).set_index('timestamp')
        s = pd.to_numeric(d[dc], errors='coerce').reindex(idx)
        if s.notna().any():
            return s.values, dc
    return np.full(len(idx), np.nan), None


# =============================================================================
# 단일 대상일 예측
# =============================================================================
def _predict_day(con, day, assets):
    m_solar, m_wind, meta, clim, wx_clim = assets
    idx = pd.date_range(day, periods=24, freq='h')
    wx, src = _day_weather(con, day, wx_clim)
    feat, _ = build_features(wx, clim=clim)
    su = np.clip(m_solar.predict(feat[meta['SOLAR_FINAL']]), 0, 1)
    wu = np.clip(m_wind.predict(feat[meta['WIND_FINAL']]), 0, 1)
    scap = _latest_capacity(con, day, 'real_solar_gen_jeju', 'real_solar_capacity_jeju')
    wcap = _latest_capacity(con, day, 'real_wind_gen_jeju', 'real_wind_capacity_jeju')
    if scap is None or wcap is None:
        raise ValueError(f'[{day.date()}] capacity 추정 불가')
    sg, wg = su * scap, wu * wcap
    dem, dem_src = _demand(con, idx)
    nl = dem - sg - wg
    out = pd.DataFrame({'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                        OUT['su']: su.round(4), OUT['wu']: wu.round(4),
                        OUT['sg']: sg.round(3), OUT['wg']: wg.round(3),
                        OUT['nl']: np.round(nl, 3)})
    return out, src, dem_src


def _upsert(con, out):
    cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
    for c in OUT_COLS:
        if c not in cols:
            con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
    setc = ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS)
    colc = ', '.join(f'"{c}"' for c in ['timestamp'] + OUT_COLS)
    ph = ', '.join(['?'] * (1 + len(OUT_COLS)))
    rows = [tuple([r['timestamp']] + [None if pd.isna(r[c]) else float(r[c]) for c in OUT_COLS])
            for _, r in out.iterrows()]
    con.executemany(f'INSERT INTO forecast ({colc}) VALUES ({ph}) '
                    f'ON CONFLICT("timestamp") DO UPDATE SET {setc}', rows)


# =============================================================================
# 공개 API
# =============================================================================
def predict_lgbm_to_db(origin, horizons=JEJU_HORIZONS, write=True, verbose=True) -> pd.DataFrame:
    """origin(발행일) 기준 D+h(h in horizons) 대상일별 util/gen/net_load → forecast UPSERT."""
    assets = load_assets()
    o = pd.Timestamp(origin).normalize()
    outs = []
    with _conn() as con:
        for h in horizons:
            day = o + pd.Timedelta(days=h)
            try:
                out, src, dsrc = _predict_day(con, day, assets)
            except Exception as e:
                if verbose: print(f'  skip D+{h} ({day.date()}): {str(e)[:60]}')
                continue
            out.insert(1, 'horizon', h); out['weather_src'] = src; out['demand_src'] = dsrc
            outs.append(out)
            if write:
                _upsert(con, out.drop(columns=['horizon', 'weather_src', 'demand_src']))
            if verbose:
                nl = out[OUT['nl']]
                print(f'  D+{h} {day.date()} | wx={src} dem={dsrc} | '
                      f"net_load {('NaN' if nl.isna().all() else f'{nl.mean():.0f}MW')}")
        if write:
            con.commit()
    res = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    if verbose and len(res):
        print(f'[DB] forecast ← origin {o.date()} | {res["horizon"].nunique()}지평 UPSERT' if write else '[no-write]')
    return res


def backfill_lgbm_to_db(start, end, horizons=JEJU_HORIZONS, verbose=True) -> pd.DataFrame:
    """[start,end] 의 각 발행일에 대해 D+h 예측·UPSERT."""
    assets = load_assets()
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')
    done = 0
    with _conn() as con:
        for o in days:
            for h in horizons:
                day = o + pd.Timedelta(days=h)
                try:
                    out, *_ = _predict_day(con, day, assets)
                    _upsert(con, out)
                    done += 1
                except Exception:
                    pass
        con.commit()
    if verbose:
        print(f'[backfill] {days[0].date()}~{days[-1].date()} | {len(days)}발행일 × {len(horizons)}지평, {done}건 기록')
    return pd.DataFrame()


if __name__ == '__main__':
    import sys, argparse
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    p = argparse.ArgumentParser(description='제주 solar/wind LGBM 장지평 서빙(D+1,2,3,7)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('origin')
    pp.add_argument('--days', default='1,2,3,7'); pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    bf.add_argument('--days', default='1,2,3,7')
    a = p.parse_args()
    hz = tuple(int(x) for x in a.days.split(','))
    if a.cmd == 'predict':
        predict_lgbm_to_db(a.origin, horizons=hz, write=not a.no_write)
    else:
        backfill_lgbm_to_db(a.start, a.end, horizons=hz)
