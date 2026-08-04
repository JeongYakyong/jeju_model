# -*- coding: utf-8 -*-
"""serve_chain.py — 제주 서빙 체인 운영 러너 수요→신재생 → est_horizon_jeju.

(원본: forecastmodel/03_jeju_solarwind_forecaster/serve_chain_jeju_new.py)
cron 한 줄 = 한 역할:
  ① 기상예보 → forecast_horizon  = collectors/collect_forecast.py (--region jeju)
  ② 실측      → historical       = collectors/collect_historical.py
  ③ 서빙 체인 → est_horizon_jeju = **이 파일** (수요→신재생)

제주 서빙을 레거시 `forecast` 테이블에서 떼어내 forecast_horizon(기상 입력) → est_horizon_jeju
(예측 출력)으로 이전한다(사용자 결정).  검증된 서빙 코드를 로직 무수정 재사용:
  - 수요: serve_demand.predict_demand_to_db (_conn 을 스크래치로 몽키패치)
  - 신재생: serve_solarwind._predict_day(scratch_con, ...) (야간 0 마스크 포함)
  - net_load = 우리 수요예측 − solar_gen − wind_gen
진단 빌더 `horizon_backtest.py` 의 스크래치/주입 헬퍼(build_scratch·set_scratch_forecast)를
그대로 재사용한다.  **`forecast` 테이블은 읽지도 쓰지도 않는다.**  SMP 는 serve_smp.py 담당.

운영 러너 ↔ 백테스트 차이: 대상 base = forecast_horizon 최신 1건(또는 --base/--backfill N), 미래
타깃 예측, 실측 대조·드롭 없음(예보 가용 시각 전부 적재), day_type 은 공휴일 달력(set_scratch_forecast).

사용
    python forecasting/serve_chain.py                # 최신 base 1건
    python forecasting/serve_chain.py --base 2026-06-16
    python forecasting/serve_chain.py --backfill 7   # 최근 7 base
    python forecasting/serve_chain.py --no-write     # 산출만
"""
from __future__ import annotations
import os, sys, sqlite3, tempfile, time, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

DB = P.DB_JEJU

HZ = tuple(range(1, 6))   # 12z: 모델지평 n = horizon_d 1..5 (사용자 확정 2026-07-17 — 수집 --days 5 와 짝)
# 18z(당일예보) 는 모델지평 n = horizon_d + 1.  서빙 모델(수요 LGBM·태양광 PatchTST)이
# 전부 "origin=전일 23시, n=1 이 익일" 구조로 학습돼 있어, 18z base(당일 03시 KST)는
# origin 을 하루 전 23시로 두면 n=1 target 이 '당일' = horizon_d 0 이 된다 — 재학습 불필요
# (basetime 확정 설계 2026-07-17 + KIMR/KIMG 18z 실증 2026-07-18, jejumodel.md).
HZ_18Z = (1, 2, 3)        # 18z: n 1..3 -> horizon_d 0..2 (KIMR 18z lead 72h 와 짝)
EST_COLS = ['est_demand_jeju', 'est_solar_util_jeju', 'est_wind_util_jeju',
            'est_solar_gen_jeju', 'est_wind_gen_jeju', 'est_net_load_jeju']
ORIGIN_HOUR = 23   # origin 시각 (12z=발행일 23시, 18z=전일 23시 — 날짜만 다름)

BASE_HOUR_12Z = '21:00:00'   # base 문자열의 시각부: 12z 발표 = KST 21시
BASE_HOUR_18Z = '03:00:00'   # 18z 발표 = KST 03시 (당일예보)


def base_mode(base: str) -> str:
    """base 문자열의 시각부로 발표 구분: '18z'(03:00, 당일예보) / '12z'(그 외)."""
    return '18z' if str(base)[11:] == BASE_HOUR_18Z else '12z'


# 함께 쓰는 모듈 — 원본은 번호 폴더 제약으로 importlib 동적 로드였으나, 한 패키지로
# 합치면서 일반 import 로 전환했다 (기능 동일).
from forecasting import horizon_backtest as backtest       # 스크래치 DB 빌더·예보 주입 헬퍼
from forecasting import serve_demand                        # 수요 서빙 (predict_demand_to_db)
from forecasting import serve_solarwind                        # 신재생 서빙 (_predict_day)
from collectors import postprocess                          # 공휴일 달력 등 후처리


def _S(t):
    return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')


def list_bases() -> list[str]:
    with sqlite3.connect(DB) as con:
        return [r[0] for r in con.execute(
            'SELECT DISTINCT base FROM forecast_horizon ORDER BY base').fetchall()]


def pick_bases(arg_base, backfill, utc: int | None = None) -> list[str]:
    """대상 base 선택.  utc(12/18) 를 주면 그 발표만 필터 — run_pipeline 이 수집 직후
    자기 base 만 체인하도록 명시 전달한다 (12z/18z 혼재 시 '최신 1건' 모호성 제거).

    --base 는 'YYYY-MM-DD'(그 날짜의 발표 전부 — 12z/18z 둘 다면 둘 다 실행) 또는
    'YYYY-MM-DD HH:MM:SS'(정확히 1건) 를 받는다.
    """
    bases = list_bases()
    if utc is not None:
        suffix = BASE_HOUR_18Z if utc == 18 else BASE_HOUR_12Z
        bases = [b for b in bases if b[11:] == suffix]
    if not bases:
        return []
    if arg_base:
        if len(arg_base) > 10:
            hit = [b for b in bases if b == arg_base]
        else:
            hit = [b for b in bases if b[:10] == arg_base[:10]]
        if not hit:
            raise SystemExit(f'forecast_horizon 에 base {arg_base} 없음 (최신={bases[-1]})')
        return hit
    if backfill:
        return bases[-backfill:]
    return [bases[-1]]


def _predict_horizons_jeju(base, origin_ts, sc, assets3, demand_pred,
                           hz=HZ, hd_offset=0, cut_ts=None) -> pd.DataFrame:
    """모델지평 n(hz) 각각 3단계 신재생을 산출해 수요예측과 합쳐 est 행들을 만든다.

    net_load = 수요예측 − 시장 태양광 − 시장 풍력. 예보 결손·수요 결측 시각은 그 지평에서 제외.
    est 의 horizon_d = n − hd_offset (12z: offset 0 → n 그대로, 18z: offset 1 → 0..2).
    cut_ts 가 있으면 그 이전 시각은 저장하지 않는다 (18z 당일예보 = base 03시부터 — 확정 설계).
    (sc = 스크래치 임시 DB 연결, demand_pred = 2단계 수요예측 시계열.)
    """
    rows = []
    for n in hz:
        target_day = origin_ts.normalize() + pd.Timedelta(days=n)
        target_idx = pd.date_range(target_day, periods=24, freq='h')
        try:
            out, solar_src, wind_src, _ = serve_solarwind._predict_day(sc, origin_ts.normalize(), n, assets3)
        except Exception:
            continue
        out = out.copy(); out['timestamp'] = pd.to_datetime(out['timestamp'])
        out = out.set_index('timestamp').reindex(target_idx)
        solar_util = pd.to_numeric(out[serve_solarwind.OUT['su']], errors='coerce')
        wind_util = pd.to_numeric(out[serve_solarwind.OUT['wu']], errors='coerce')
        solar_gen = pd.to_numeric(out[serve_solarwind.OUT['sg']], errors='coerce')
        wind_gen = pd.to_numeric(out[serve_solarwind.OUT['wg']], errors='coerce')
        demand_day = demand_pred.reindex(target_idx)
        valid_mask = demand_day.notna() & solar_util.notna()
        if cut_ts is not None:
            valid_mask &= (target_idx >= cut_ts)
        if not valid_mask.any():
            continue
        valid_idx = target_idx[valid_mask.values]
        rows.append(pd.DataFrame({
            'base': base, 'timestamp': valid_idx, 'horizon_d': n - hd_offset,
            'est_demand_jeju': demand_day[valid_mask].values,
            'est_solar_util_jeju': solar_util[valid_mask].values, 'est_wind_util_jeju': wind_util[valid_mask].values,
            'est_solar_gen_jeju': solar_gen[valid_mask].values, 'est_wind_gen_jeju': wind_gen[valid_mask].values,
            'est_net_load_jeju': (demand_day[valid_mask] - solar_gen[valid_mask] - wind_gen[valid_mask]).values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_base(base: str, sc, assets3) -> pd.DataFrame:
    """그 base 의 2→3 풀체인 → est 컬럼 (미래 타깃).

    ① 2단계 수요(전 지평 한 번에) → ② 지평별 3단계 신재생 + net_load 조립.
    18z base(당일 03시 발표)는 origin 을 전일 23시로 두고 n=hd+1 로 매핑 —
    학습된 D+1 태스크 그대로에 더 신선한 18z 기상만 공급한다 (모듈 상단 주석 참조).
    """
    m18 = base_mode(base) == '18z'
    origin_day = pd.Timestamp(base).normalize() - pd.Timedelta(days=1) if m18 \
        else pd.Timestamp(base).normalize()
    origin_ts = origin_day + pd.Timedelta(hours=ORIGIN_HOUR)
    hz = HZ_18Z if m18 else HZ
    # 기상=forecast_horizon → 스크래치 주입.  18z 는 base 행이 당일 04시부터라 00~03시가
    # 비므로 직전 발표(12z 뼈대) 행으로 패딩한다 (pad_from_prev — 사용자 확정 2026-07-18).
    backtest.set_scratch_forecast(sc, base, postprocess, 'forecast', pad_from_prev=m18)

    # ── 2단계 수요 (전 지평 한 번에) — 스크래치 forecast 에서 읽음(_conn 몽키패치됨) ──
    try:
        demand_df = serve_demand.predict_demand_to_db(
            origin_day.strftime('%Y-%m-%d'), days_ahead=max(hz), write=False, verbose=False)
    except Exception as e:
        print(f'  [skip] base {base[:10]} 수요 산출 실패: {str(e)[:70]}')
        return pd.DataFrame()
    demand_df = demand_df.copy(); demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])
    demand_pred = pd.to_numeric(demand_df.set_index('timestamp')['jeju_est_demand_lh'], errors='coerce')

    # ── 지평별 3단계 신재생 + net_load ──
    return _predict_horizons_jeju(base, origin_ts, sc, assets3, demand_pred,
                                  hz=hz, hd_offset=1 if m18 else 0,
                                  cut_ts=pd.Timestamp(base) if m18 else None)


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


def _check_12z_backbone(base18: str) -> bool:
    """18z 체인 전 뼈대(전일 12z base) 존재 확인 — 없으면 경고 (12z 수집 실패 알림).

    데이터 대체는 추가 장치 없음: freshest-wins 조회가 12z 부재 시각을 18z 또는
    전날 12z(hd+1)로 자동 커버한다 (사용자 확정 2026-07-18).
    """
    prev12 = (pd.Timestamp(base18).normalize() - pd.Timedelta(days=1)
              + pd.Timedelta(hours=21)).strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DB) as con:
        n = con.execute('SELECT COUNT(*) FROM forecast_horizon WHERE base=?',
                        (prev12,)).fetchone()[0]
    if n == 0:
        print(f'  [WARN] 뼈대 12z base({prev12}) 부재 — 어젯밤 12z 수집 실패 여부 확인 필요 '
              f'(freshest-wins 가 18z/전날 12z 로 자동 대체하나 4~5일후 지평이 낡아진다)')
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description='제주 서빙 체인 운영 러너 2→3 → est_horizon_jeju')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--base', default=None,
                   help="특정 base 'YYYY-MM-DD'(그 날짜 발표 전부) 또는 'YYYY-MM-DD HH:MM:SS'(1건)")
    g.add_argument('--backfill', type=int, default=None, help='최근 N개 base')
    ap.add_argument('--utc', type=int, choices=[12, 18], default=None,
                    help='base 발표 필터: 12=21:00 발표만, 18=03:00 발표(당일예보)만 — '
                         'cron 이 수집 직후 자기 base 만 체인하도록 명시 전달')
    ap.add_argument('--no-write', action='store_true', help='산출만 — 적재 생략')
    a = ap.parse_args()

    bases = pick_bases(a.base, a.backfill, a.utc)
    if not bases:
        raise SystemExit('forecast_horizon 비어있음 (또는 --utc 필터에 걸리는 base 없음) '
                         '— 먼저 기상예보 수집 필요')
    print(f'[serve_chain] 대상 base {len(bases)}개: {bases[0]} ~ {bases[-1]}')

    rc = 0
    if a.utc == 18 and not _check_12z_backbone(bases[-1]):
        rc = 1   # 경고 exit code — cron 알림용 (체인 자체는 계속 진행)

    scratch_path = os.path.join(tempfile.gettempdir(), 'serve_chain_jeju.db')
    sc = backtest.build_scratch(scratch_path)
    serve_demand._conn = lambda: sqlite3.connect(scratch_path)   # 수요 서빙이 스크래치를 읽도록
    assets3 = serve_solarwind._assets()

    t0 = time.time(); total = 0
    for bi, base in enumerate(bases, 1):
        r = build_base(base, sc, assets3)
        if r.empty:
            print(f'  base {bi}/{len(bases)} ({base}) — 산출 없음'); rc = max(rc, 1); continue
        print(f'  base {bi}/{len(bases)} ({base} {base_mode(base)})  {len(r)}행 '
              f'hd {r.horizon_d.min()}~{r.horizon_d.max()}'
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
    print(f'[serve_chain] done in {(time.time()-t0)/60:.1f}m')
    if rc:
        sys.exit(rc)   # 경고 exit code (뼈대 12z 부재 / 산출 없음) — cron 알림용


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
