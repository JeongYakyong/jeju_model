# -*- coding: utf-8 -*-
"""serve_chain_jeju_new.py — 제주 서빙 체인 운영 러너 2→3 → est_horizon_jeju.

육지 `7. land_gas_forecaster/serve_chain_land_new.py` 의 제주판.  cron 한 줄 = 한 역할:
  ① 기상예보 → forecast_horizon  = core/collect_forecast_new.py (--region jeju)
  ② 실측      → historical       = core/collect_data_jeju*.py
  ③ 서빙 체인 → est_horizon_jeju = **이 파일** (2→3)

제주 서빙을 레거시 `forecast` 테이블에서 떼어내 forecast_horizon(기상 입력) → est_horizon_jeju
(예측 출력)으로 이전한다(사용자 결정, 육지와 대칭).  현행 검증 서빙 코드를 **무수정 재사용**:
  - 2단계 수요: serve_jeju_demand_lh.predict_demand_to_db (_conn 을 스크래치로 몽키패치)
  - 3단계 신재생: serve_solarwind_hybrid._predict_day(scratch_con, ...) (야간 0 마스크 포함)
  - net_load = 우리 수요예측 − solar_gen − wind_gen
진단 빌더 `training/build_horizon_backtest_jeju.py` 의 스크래치/주입 헬퍼(build_scratch·
set_scratch_forecast·_load)를 그대로 재사용한다(파일 무수정).  **`forecast` 테이블은 읽지도 쓰지도
않는다.**  SMP(4)는 이번 범위 밖.

운영 러너 ↔ 백테스트 차이: 대상 base = forecast_horizon 최신 1건(또는 --base/--backfill N), 미래
타깃 예측, 실측 대조·드롭 없음(예보 가용 시각 전부 적재), day_type 은 공휴일 달력(set_scratch_forecast).

사용
    python "3. jeju_solarwind_forecaster/serve_chain_jeju_new.py"                # 최신 base 1건
    python "3. jeju_solarwind_forecaster/serve_chain_jeju_new.py" --base 2026-06-16
    python "3. jeju_solarwind_forecaster/serve_chain_jeju_new.py" --backfill 7   # 최근 7 base
    python "3. jeju_solarwind_forecaster/serve_chain_jeju_new.py" --no-write     # 산출만
"""
from __future__ import annotations
import os, sys, sqlite3, importlib.util, tempfile, time, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
DB = os.path.join(ROOT, '1. data_fetcher_and_db', 'data', 'input_data_jeju.db')
BHTJ_PATH = os.path.join(HERE, 'training', 'build_horizon_backtest_jeju.py')

HZ = tuple(range(1, 8))   # D+1..D+7 (제주 forecast_horizon 윈도우)
EST_COLS = ['est_demand_jeju', 'est_solar_util_jeju', 'est_wind_util_jeju',
            'est_solar_gen_jeju', 'est_wind_gen_jeju', 'est_net_load_jeju']


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m)
    return m


# 진단 빌더의 스크래치/주입/모듈로더 재사용 (파일 무수정)
bhtj = _imp('build_horizon_backtest_jeju', BHTJ_PATH)
pp = _imp('pp_jeju', bhtj.PP_PATH)
S2 = _imp('serve_jeju_demand_lh', bhtj.S2_PATH)
S3 = _imp('serve_solarwind_hybrid', bhtj.S3_PATH)


def _S(t):
    return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')


def list_bases() -> list[str]:
    with sqlite3.connect(DB) as con:
        return [r[0] for r in con.execute(
            'SELECT DISTINCT base FROM forecast_horizon ORDER BY base').fetchall()]


def pick_bases(arg_base, backfill) -> list[str]:
    bases = list_bases()
    if not bases:
        return []
    if arg_base:
        hit = [b for b in bases if b[:10] == arg_base[:10]]
        if not hit:
            raise SystemExit(f'forecast_horizon 에 base {arg_base} 없음 (최신={bases[-1]})')
        return hit
    if backfill:
        return bases[-backfill:]
    return [bases[-1]]


def build_base(base: str, sc, assets3) -> pd.DataFrame:
    """그 base 의 2→3 풀체인 → est 컬럼 (미래 타깃)."""
    O = pd.Timestamp(base).normalize() + pd.Timedelta(hours=23)
    bhtj.set_scratch_forecast(sc, base, pp, 'forecast')   # 기상=forecast_horizon, day_type=공휴일달력

    # 2단계 수요 (D+1..D+7 한 번에) — 스크래치 forecast 에서 읽음(_conn 몽키패치됨)
    try:
        dem = S2.predict_demand_to_db(base[:10], days_ahead=max(HZ), write=False, verbose=False)
    except Exception as e:
        print(f'  [skip] base {base[:10]} 수요 산출 실패: {str(e)[:70]}')
        return pd.DataFrame()
    dem = dem.copy(); dem['timestamp'] = pd.to_datetime(dem['timestamp'])
    dem_pred = pd.to_numeric(dem.set_index('timestamp')['jeju_est_demand_lh'], errors='coerce')

    rows = []
    for n in HZ:
        d0 = O.normalize() + pd.Timedelta(days=n)
        day_idx = pd.date_range(d0, periods=24, freq='h')
        try:
            out, ssrc, wsrc, _ = S3._predict_day(sc, O.normalize(), n, assets3)
        except Exception:
            continue
        out = out.copy(); out['timestamp'] = pd.to_datetime(out['timestamp'])
        out = out.set_index('timestamp').reindex(day_idx)
        su = pd.to_numeric(out[S3.OUT['su']], errors='coerce')
        wu = pd.to_numeric(out[S3.OUT['wu']], errors='coerce')
        sg = pd.to_numeric(out[S3.OUT['sg']], errors='coerce')
        wg = pd.to_numeric(out[S3.OUT['wg']], errors='coerce')
        dp = dem_pred.reindex(day_idx)
        keep = dp.notna() & su.notna()
        if not keep.any():
            continue
        ti = day_idx[keep.values]
        rows.append(pd.DataFrame({
            'base': base, 'timestamp': ti, 'horizon_d': n,
            'est_demand_jeju': dp[keep].values,
            'est_solar_util_jeju': su[keep].values, 'est_wind_util_jeju': wu[keep].values,
            'est_solar_gen_jeju': sg[keep].values, 'est_wind_gen_jeju': wg[keep].values,
            'est_net_load_jeju': (dp[keep] - sg[keep] - wg[keep]).values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def upsert_est(r: pd.DataFrame, db_path: str) -> int:
    if r.empty:
        return 0
    def _v(x):
        return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else float(x)
    data = [(_S(row.timestamp), str(row.base), int(row.horizon_d),
             *[_v(getattr(row, c)) for c in EST_COLS])
            for row in r.itertuples(index=False) if np.isfinite(row.est_demand_jeju)]
    if not data:
        return 0
    set_clause = ', '.join(f'{c}=excluded.{c}' for c in (['horizon_d'] + EST_COLS))
    col_list = ', '.join(['timestamp', 'base', 'horizon_d'] + EST_COLS)
    ph = ', '.join('?' * (3 + len(EST_COLS)))
    with sqlite3.connect(db_path) as con:
        con.execute('CREATE TABLE IF NOT EXISTS est_horizon_jeju ('
                    'timestamp TEXT, base TEXT, horizon_d INT, PRIMARY KEY(base, timestamp))')
        cols = [c[1] for c in con.execute('PRAGMA table_info(est_horizon_jeju)')]
        for c in EST_COLS:
            if c not in cols:
                con.execute(f'ALTER TABLE est_horizon_jeju ADD COLUMN "{c}" REAL')
        con.executemany(
            f'INSERT INTO est_horizon_jeju ({col_list}) VALUES ({ph}) '
            f'ON CONFLICT(base, timestamp) DO UPDATE SET {set_clause}', data)
        con.commit()
    return len(data)


def main():
    ap = argparse.ArgumentParser(description='제주 서빙 체인 운영 러너 2→3 → est_horizon_jeju')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--base', default=None, help='특정 base YYYY-MM-DD (기본: forecast_horizon 최신)')
    g.add_argument('--backfill', type=int, default=None, help='최근 N개 base')
    ap.add_argument('--no-write', action='store_true', help='산출만 — 적재 생략')
    a = ap.parse_args()

    bases = pick_bases(a.base, a.backfill)
    if not bases:
        raise SystemExit('forecast_horizon 비어있음 — 먼저 기상예보 수집 필요')
    print(f'[serve_chain_jeju_new] 대상 base {len(bases)}개: {bases[0][:10]} ~ {bases[-1][:10]}')

    scratch_path = os.path.join(tempfile.gettempdir(), 'serve_chain_jeju.db')
    sc = bhtj.build_scratch(scratch_path)
    S2._conn = lambda: sqlite3.connect(scratch_path)   # 수요 서빙이 스크래치를 읽도록
    assets3 = S3._assets()

    t0 = time.time(); total = 0
    for bi, base in enumerate(bases, 1):
        r = build_base(base, sc, assets3)
        if r.empty:
            print(f'  base {bi}/{len(bases)} ({base[:10]}) — 산출 없음'); continue
        print(f'  base {bi}/{len(bases)} ({base[:10]})  {len(r)}행 D+{r.horizon_d.min()}~{r.horizon_d.max()}'
              f'  수요 {r.est_demand_jeju.mean():.0f}MW  net_load {r.est_net_load_jeju.mean():.0f}MW  {time.time()-t0:.0f}s')
        if not a.no_write:
            total += upsert_est(r, DB)
    sc.close()

    if a.no_write:
        print('\n(--no-write: 적재 생략)')
    else:
        with sqlite3.connect(DB) as con:
            tot = con.execute('SELECT COUNT(*) FROM est_horizon_jeju').fetchone()[0]
            rng = con.execute('SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT base) '
                              'FROM est_horizon_jeju').fetchone()
        print(f'\nest_horizon_jeju UPSERT {total}행  (전체 {tot}행, base {rng[2]}개, {rng[0]} ~ {rng[1]})')
    print(f'[serve_chain_jeju_new] done in {(time.time()-t0)/60:.1f}m')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
