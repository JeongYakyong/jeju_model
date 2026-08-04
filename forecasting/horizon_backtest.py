# -*- coding: utf-8 -*-
"""제주 실예보 지평 백테스트 — forecast_horizon 으로 2(수요)·3(신재생→net_load) 체인 정직 재검증.

육지 `07_land_gas_forecaster/training/build_horizon_backtest.py` 의 제주판.

배경
----
제주를 새 DB 구조(forecast_horizon=기상 아카이브 / est_horizon_jeju=서빙결과 / forecast 폐기)로
옮기기 전에, 현행 2·3단계 모델을 base 별 **실제 지평 예보**로 처음 정직하게 재측정한다.
육지 G-16 과 동일한 사고: 과거의 (월,시) 기후값 프록시가 가렸던 지평 열화를 실예보로 드러낸다.

★ 하드 규칙 (육지와 동일): **기후값 폴백 금지.** 예보가 진짜 없는 시각은 기후값으로 채우지
   않고 평가에서 뺀다. 허용 보정은 forecast 앵커 사이 시간 보간(≤4h, 외삽 금지)까지(=실예보의
   재표집). forecast_horizon 의 D+6~7 은 3h 해상도지만 이 보간으로 1h 복원된다.

방법 (현행 서빙 코드 무수정 재사용)
----
- 스크래치 DB(휘발성): 본 historical 전체 복사 + 그 base 의 forecast_horizon 슬라이스를 1h
  보간(≤4h)해 `forecast` 로 주입. day_type 은 공휴일 달력(postprocess)으로 부여. 본 DB 의
  `forecast` 테이블은 읽지도 쓰지도 않는다(폐기 방향).
- 수요(2-B): serve_demand._conn 을 스크래치로 몽키패치 → predict_demand_to_db 실행.
  반환 weather_src=='climatology' 행은 드롭(기후값 폴백 차단).
- 신재생(3 하이브리드): serve_solarwind._predict_day(scratch_con, ...) 직접 호출(con 인자
  지원). solar=PatchTST(가능시)/LGBM 폴백, wind=LGBM. net_load 는 **우리 수요 예측**으로 재계산
  (dem_pred − solar_gen − wind_gen). 예보 결측 시각은 coverage 마스크로 드롭(LGBM 폴백 기후값 차단).

모드
----
- forecast : base 별 실제 지평 예보(정직 측정).
- oracle   : 같은 base 의 **실측 기상**(historical) 을 forecast 양식으로 주입 → 입력 완벽 시
             모델 바닥선. 실예보−ORACLE 격차 = 예보 품질의 대가(육지 진단 핵심).

산출: forecasting/horizon_backtest_jeju.parquet  (mode 컬럼으로 forecast/oracle 구분)
사용: python forecasting/horizon_backtest.py [--mode forecast|oracle|both] [--limit N] [--horizons 1,2,3,7]
(원본: forecastmodel/03_jeju_solarwind_forecaster/training/build_horizon_backtest_jeju.py)
"""
from __future__ import annotations
import os, sys, sqlite3, argparse, tempfile, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

DB = P.DB_JEJU

# 함께 쓰는 모듈 — 원본은 importlib 동적 로드(_load)였으나 일반 import 로 전환 (기능 동일).
# 이 모듈의 build_scratch·set_scratch_forecast 는 운영 체인(serve_chain.py)도 재사용한다.
from forecasting import serve_demand                        # 수요 서빙
from forecasting import serve_solarwind                        # 신재생 서빙(하이브리드)
from collectors import postprocess                          # 공휴일 달력 등 후처리

ST = ['west', 'east', 'south']
SOLAR_ST = ['west', 'south']
# 서빙이 forecast 에서 읽는 컬럼 → 실측(historical) 컬럼 매핑(ORACLE 주입용)
ACT2FORE = {}
for s in ST:
    ACT2FORE[f'temp_{s}']         = f'temp_c_{s}'
    ACT2FORE[f'reh_{s}']          = f'humidity_{s}'
    ACT2FORE[f'wind_spd_10m_{s}'] = f'wind_spd_{s}'
    ACT2FORE[f'wd_sin_10m_{s}']   = f'wd_sin_{s}'
    ACT2FORE[f'wd_cos_10m_{s}']   = f'wd_cos_{s}'
    ACT2FORE[f'total_cloud_{s}']  = f'total_cloud_{s}'
    ACT2FORE[f'midlow_cloud_{s}'] = f'midlow_cloud_{s}'
    ACT2FORE[f'rainfall_{s}']     = f'rainfall_{s}'
for s in SOLAR_ST:
    ACT2FORE[f'radiation_{s}'] = f'solar_rad_{s}'
# coverage 판정용 sentinel 변수(atomic forecast: 하나 있으면 그 base 행 전체 있음)
COVER_VARS = ['temp_west', 'temp_south', 'radiation_south', 'wind_spd_10m_west']
# 실측(평가 타깃) 컬럼
ACT_COLS = ['real_demand_jeju', 'real_net_load_jeju',
            'real_solar_utilization_jeju', 'real_wind_utilization_jeju',
            'real_solar_gen_jeju', 'real_wind_gen_jeju']
HORIZONS_DEFAULT = [1, 2, 3, 4, 5]   # 운영 지평 D+5 축소(2026-07-17)와 일치


def _S(t):
    return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')


# ── 스크래치 DB: historical 전체 1회 복사 ──
def build_scratch(scratch_path):
    if os.path.exists(scratch_path):
        os.remove(scratch_path)
    with sqlite3.connect(DB) as con:
        hist = pd.read_sql('SELECT * FROM historical', con)
    sc = sqlite3.connect(scratch_path)
    hist.to_sql('historical', sc, if_exists='replace', index=False)
    sc.execute('CREATE INDEX IF NOT EXISTS ix_h ON historical(timestamp)')
    sc.commit()
    return sc


def _hourly_interp(df, limit=3):
    """timestamp 인덱스 wide → 1h 재색인 + ≤limit h 시간보간(외삽 금지)."""
    df = df.sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq='h')
    df = df.apply(pd.to_numeric, errors='coerce').reindex(idx)
    df = df.interpolate(method='time', limit=limit, limit_area='inside')
    df.index.name = 'timestamp'
    return df


def set_scratch_forecast(sc, base, postprocess_mod, mode, pad_from_prev=False):
    """그 base 의 forecast(=실예보) 또는 oracle(=실측기상)을 스크래치 `forecast` 로 주입.

    postprocess_mod = collectors.postprocess 모듈 (day_type 공휴일 달력 부여에 사용).
    pad_from_prev=True (18z 당일예보 체인 전용) 면 base 행이 없는 시각을 직전 발표들의
    freshest 행으로 패딩한다 — 18z 예보는 당일 04시부터라 당일 00~03시가 비는데,
    태양광 서빙(_build_solar_direct)은 대상일 24h 전부를 요구하고 수요 서빙은 결측을
    기후값으로 채워버리므로, 12z 뼈대 값으로 메꾼다 (사용자 확정 2026-07-18: 12z 우선,
    그것도 없으면 시간보간 limit=4 폴백).  기본 False — 12z·백테스트 경로 불변.
    """
    O = pd.Timestamp(base).normalize() + pd.Timedelta(hours=23)
    if mode == 'forecast':
        with sqlite3.connect(DB) as con:
            f = pd.read_sql('SELECT * FROM forecast_horizon WHERE base=? ORDER BY timestamp',
                            con, params=(base,), parse_dates=['timestamp'])
        f = f.drop(columns=[c for c in ('base', 'horizon_d') if c in f.columns]).set_index('timestamp')
        if pad_from_prev and not f.empty:
            # 같은 창(대상일 자정 ~ 창 끝)에서 직전 발표들의 freshest 행을 읽어
            # base 행이 없는 timestamp 만 채운다 (base 행 우선 combine_first).
            w0 = pd.Timestamp(base).normalize().strftime('%Y-%m-%d %H:%M:%S')
            w1 = f.index.max().strftime('%Y-%m-%d %H:%M:%S')
            with sqlite3.connect(DB) as con:
                prev = pd.read_sql(
                    'SELECT f.* FROM forecast_horizon f '
                    'JOIN (SELECT timestamp, MAX(base) mb FROM forecast_horizon '
                    '      WHERE base < ? AND timestamp BETWEEN ? AND ? GROUP BY timestamp) m '
                    'ON f.timestamp = m.timestamp AND f.base = m.mb',
                    con, params=(base, w0, w1), parse_dates=['timestamp'])
            if prev.empty:
                print(f'  [WARN] pad_from_prev: 직전 발표 행 없음 — 시간보간(limit=4)만으로 진행')
            else:
                prev = prev.drop(columns=[c for c in ('base', 'horizon_d') if c in prev.columns]) \
                           .set_index('timestamp')
                n_pad = len(prev.index.difference(f.index))
                f = f.combine_first(prev)
                print(f'  [pad] 18z 스크래치 패딩: 직전 발표 행 {n_pad}개 보충 ({w0[:16]}~)')
        f = _hourly_interp(f, limit=4 if pad_from_prev else 3)
    else:  # oracle — 실측 기상을 forecast 컬럼명으로
        tmin = _S(O + pd.Timedelta(hours=1)); tmax = _S(O + pd.Timedelta(hours=24 * 7))
        cols = ['timestamp'] + sorted(set(ACT2FORE.values()))
        with sqlite3.connect(DB) as con:
            h = pd.read_sql(
                f"SELECT {', '.join(chr(34)+c+chr(34) for c in cols)} FROM historical "
                f"WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                con, params=(tmin, tmax), parse_dates=['timestamp']).set_index('timestamp')
        f = pd.DataFrame(index=h.index)
        for fore_c, act_c in ACT2FORE.items():
            if act_c in h.columns:
                f[fore_c] = pd.to_numeric(h[act_c], errors='coerce')
        # tcog 등 실측에 없는 예보전용 변수는 NaN(서빙 후처리가 무보정 통과 — ORACLE=입력완벽 바닥선)
        for s in ST:
            f[f'tcog_{s}'] = np.nan
    # day_type 부착(공휴일 달력)
    f = f.reset_index()
    f['day_type'] = _daytype(f['timestamp'], postprocess_mod)
    f.to_sql('forecast', sc, if_exists='replace', index=False)
    sc.commit()


def _daytype(ts: pd.Series, postprocess_mod) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    if len(idx) == 0:
        return np.array([], dtype=object)
    tmp = pd.DataFrame({'_x': np.zeros(len(idx))}, index=idx)  # 더미컬럼(0컬럼=empty 회피)
    tmp = postprocess_mod.add_day_type(tmp)
    return tmp['day_type'].values


def _coverage(sc, targets):
    """주입된 forecast 에서 sentinel 변수가 모두 non-NaN 인 시각 = 실예보 가용(기후값 아님)."""
    sel = ', '.join(f'"{c}"' for c in ['timestamp'] + COVER_VARS)
    fc = pd.read_sql(f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
                     sc, params=(_S(targets.min()), _S(targets.max())),
                     parse_dates=['timestamp']).set_index('timestamp')
    fc = fc.apply(pd.to_numeric, errors='coerce').reindex(targets)
    return fc.notna().all(axis=1)


def build(modes, horizons, limit=None):
    scratch_path = os.path.join(tempfile.gettempdir(), 'jeju_horizon_bt.db')
    sc = build_scratch(scratch_path)
    # 수요 서빙이 스크래치를 읽도록 _conn 몽키패치(매 호출 새 연결)
    serve_demand._conn = lambda: sqlite3.connect(scratch_path)
    assets3 = serve_solarwind._assets()

    # 실측 시계열(평가 타깃)
    with sqlite3.connect(DB) as con:
        act = pd.read_sql(f"SELECT timestamp, {', '.join(ACT_COLS)} FROM historical",
                          con, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    act = act.apply(pd.to_numeric, errors='coerce')

    with sqlite3.connect(DB) as con:
        # 12z(21:00 발표)만 — 18z(03:00 당일예보) base 는 origin 산식(base일 23시)이 달라
        # 여기 D+1 백테스트에 섞이면 origin 을 하루 늦게 잡는다.  18z 검증은 UI 검증
        # 페이지(발표 필터)가 담당 (2026-07-18 18z 도입).
        bases = [r[0] for r in con.execute(
            "SELECT DISTINCT base FROM forecast_horizon "
            "WHERE substr(base, 12) = '21:00:00' ORDER BY base").fetchall()]
    if limit:
        bases = bases[::max(1, len(bases) // limit)][:limit]

    rows = []; t0 = time.time()
    for bi, base in enumerate(bases, 1):
        O = pd.Timestamp(base).normalize() + pd.Timedelta(hours=23)
        for mode in modes:
            set_scratch_forecast(sc, base, postprocess, mode)
            # 수요: 한 번에 D+1..D+7
            try:
                dem = serve_demand.predict_demand_to_db(base[:10], days_ahead=max(horizons),
                                              write=False, verbose=False)
            except Exception as e:
                if limit:
                    print(f'  [skip dem] {base[:10]} {mode}: {str(e)[:70]}')
                continue
            dem = dem.copy()
            dem['timestamp'] = pd.to_datetime(dem['timestamp'])
            dem = dem.set_index('timestamp')
            dem_keep = dem[dem['weather_src'] == 'forecast']
            dem_pred = pd.to_numeric(dem_keep['jeju_est_demand_lh'], errors='coerce')

            for n in horizons:
                d0 = O.normalize() + pd.Timedelta(days=n)
                day_idx = pd.date_range(d0, periods=24, freq='h')
                cov = _coverage(sc, day_idx)
                try:
                    out, ssrc, wsrc, _ = serve_solarwind._predict_day(sc, O.normalize(), n, assets3)
                except Exception as e:
                    if limit:
                        print(f'  [skip sw] {base[:10]} {mode} D+{n}: {str(e)[:70]}')
                    continue
                out = out.copy()
                out['timestamp'] = pd.to_datetime(out['timestamp'])
                out = out.set_index('timestamp').reindex(day_idx)
                su = pd.to_numeric(out[serve_solarwind.OUT['su']], errors='coerce')
                wu = pd.to_numeric(out[serve_solarwind.OUT['wu']], errors='coerce')
                sg = pd.to_numeric(out[serve_solarwind.OUT['sg']], errors='coerce')
                wg = pd.to_numeric(out[serve_solarwind.OUT['wg']], errors='coerce')
                dp = dem_pred.reindex(day_idx)
                nl = dp - sg - wg
                keep = cov & dp.notna() & su.notna()
                if not keep.any():
                    continue
                ti = day_idx[keep.values]
                rows.append(pd.DataFrame({
                    'mode': mode, 'base': base, 'timestamp': ti, 'horizon': n,
                    'est_demand': dp[keep].values,
                    'est_solar_util': su[keep].values, 'est_wind_util': wu[keep].values,
                    'est_solar_gen': sg[keep].values, 'est_wind_gen': wg[keep].values,
                    'est_net_load': nl[keep].values,
                    'real_demand': act['real_demand_jeju'].reindex(ti).values,
                    'real_solar_util': act['real_solar_utilization_jeju'].reindex(ti).values,
                    'real_wind_util': act['real_wind_utilization_jeju'].reindex(ti).values,
                    'real_solar_gen': act['real_solar_gen_jeju'].reindex(ti).values,
                    'real_wind_gen': act['real_wind_gen_jeju'].reindex(ti).values,
                    'real_net_load': act['real_net_load_jeju'].reindex(ti).values,
                    'solar_src': ssrc, 'wind_src': wsrc,
                    'hour': ti.hour, 'day_type': dem_keep['day_type'].reindex(ti).values,
                }))
        if bi % 20 == 0 or bi == len(bases):
            print(f'  base {bi}/{len(bases)} ({base[:10]})  누적 {len(rows)} blocks  {time.time()-t0:.0f}s')
    sc.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _report(res):
    for mode, rm in res.groupby('mode'):
        print(f'\n=== mode={mode} ===')
        for n, g in rm.groupby('horizon'):
            dd = g.dropna(subset=['real_demand']); dd = dd[dd.real_demand > 0]
            dmape = float(np.mean(np.abs(dd.real_demand - dd.est_demand) / dd.real_demand) * 100)
            dbias = float(np.mean((dd.est_demand - dd.real_demand) / dd.real_demand) * 100)
            day = dd[(dd.hour >= 8) & (dd.hour <= 16)]
            dmape_day = float(np.mean(np.abs(day.real_demand - day.est_demand) / day.real_demand) * 100) if len(day) else float('nan')
            # solar util nMAE(낮), net_load MAPE
            sd = g[(g.hour >= 8) & (g.hour <= 16)].dropna(subset=['real_solar_util'])
            su_nmae = float(np.mean(np.abs(sd.real_solar_util - sd.est_solar_util))) if len(sd) else float('nan')
            wd = g.dropna(subset=['real_wind_util'])
            wu_nmae = float(np.mean(np.abs(wd.real_wind_util - wd.est_wind_util))) if len(wd) else float('nan')
            nn = g.dropna(subset=['real_net_load'])
            nl_mae = float(np.mean(np.abs(nn.real_net_load - nn.est_net_load))) if len(nn) else float('nan')
            # 제주 net_load 는 한낮 0 부근까지 가 MAPE 분모 불안정 → 평균수요 정규화 nMAE
            nl_nmae = (nl_mae / float(dd.real_demand.mean()) * 100) if len(nn) and len(dd) else float('nan')
            print(f'  D+{n}: 수요 MAPE {dmape:.2f}% (낮 {dmape_day:.2f}%) bias {dbias:+.2f}% | '
                  f'solar nMAE(낮) {su_nmae:.3f} | wind nMAE {wu_nmae:.3f} | '
                  f'net_load MAE {nl_mae:.1f}MW (nMAE {nl_nmae:.2f}%) (n={len(g)})')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='forecast', choices=['forecast', 'oracle', 'both'])
    ap.add_argument('--limit', type=int, default=None, help='base 표본수(스모크)')
    ap.add_argument('--horizons', default='1,2,3,4,5')
    ap.add_argument('--out', default=os.path.join(HERE, 'horizon_backtest_jeju.parquet'))
    a = ap.parse_args()
    modes = ['forecast', 'oracle'] if a.mode == 'both' else [a.mode]
    hz = [int(x) for x in a.horizons.split(',')]
    res = build(modes, hz, limit=a.limit)
    if not len(res):
        print('결과 없음'); sys.exit(1)
    print(f'\n총 {len(res)}행  base {res.base.nunique()}  기간 {res.timestamp.min()} ~ {res.timestamp.max()}')
    _report(res)
    if not a.limit:
        res.to_parquet(a.out); print('\nsaved', a.out)
    else:
        print('\n(스모크 — 저장 생략)')
