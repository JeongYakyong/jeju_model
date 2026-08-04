# -*- coding: utf-8 -*-
"""serve_smp.py — 제주 SMP 서빙 러너 → est_smp_horizon_jeju.

(원본: forecastmodel/04_jeju_smp_forecaster/serve_smp_horizon_jeju.py)
수요·신재생 체인(`serve_chain.py`→`est_horizon_jeju`)과 대칭으로, SMP 를 레거시
`forecast` 테이블에서 떼어내 **`est_horizon_jeju`(예측 입력) → `est_smp_horizon_jeju`(예측 출력)**
로 재배선한다.  잠긴 SMP 모델·피처(2026-06-05 사용자 확정)는 그대로 두고, 입출력 테이블만
스크래치 DB + monkeypatch 로 갈아끼운다(serve_chain 패턴, 잠긴 파일 무수정).

체인:
  est_horizon_jeju(base·horizon_d 1~3) ─rename→ 스크래치 `forecast`(+ historical DA)
      ↓ smp_d1.predict_smp_to_db   (D+1: est_smp_jeju·smp_neg_proba_jeju·smp_danger_jeju)
      ↓ smp_d2.predict_d2_to_db    (D+2: est_smp_jeju_d2·smp_neg_proba_d2·smp_danger_d2)
  → 핵심 3컬럼만 수확 → est_smp_horizon_jeju(base·timestamp·horizon_d·est_smp·smp_neg_proba·smp_danger)

핵심만(사용자 확정): 가격선 est_smp · 음수확률 smp_neg_proba · 경보 smp_danger.  깊이(p10/50/90)·
캘리브레이션·주야 위험레이어·D+2 고확신경보는 적재하지 않는다(깊이/위험 파이프라인 미실행).

재배선 두 지점(파일 무수정, 이 러너에서 monkeypatch):
  ① DB_PATH(smp_features·smp_d1·smp_d2) → 스크래치 경로.
  ② smp_d2.load_forecast → with_target=False 강제.  원본 `_build_serving`은
     load_forecast(with_target=True) inner-join(rt)으로 미래(rt 없는) D+2 행을 떨어뜨려
     backfill 전용이 된다 → 미래 서빙을 위해 타깃 조인을 끈다(예측만, 평가 리포트 미사용).

스크래치 `forecast` 에 horizon_d 1~3 을 함께 적재하는 이유: lead(shift-1/-2)·D+2 전일대비
변화(d_net_load=shift24)가 D+1↔D+2↔D+3 경계에서 NaN 없이 계산되도록(수확은 D+1·D+2 만).

사용
    python forecasting/serve_smp.py                # 최신 base 1건
    python forecasting/serve_smp.py --base 2026-06-14
    python forecasting/serve_smp.py --backfill 7   # 최근 7 base
    python forecasting/serve_smp.py --no-write     # 산출만
"""
from __future__ import annotations
import os, sys, sqlite3, tempfile, time, argparse, warnings, gc
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

DB = P.DB_JEJU

HZ = (1, 2)            # 출력 지평: D+1·D+2 (사용자 확정 = D+2까지만)
HZ_SCRATCH = (1, 2, 3)  # 스크래치 적재 지평(D+3 은 D+2 lead/shift 경계용, 수확 안 함)

# est_horizon_jeju(예측 입력) → 스크래치 forecast 컬럼명(잠긴 SMP 코드가 기대하는 이름)
EH_TO_FC = {
    'est_net_load_jeju': 'est_net_load_jeju',          # net_load (이름 동일)
    'est_demand_jeju': 'jeju_est_demand_new',          # est_demand (1단계 출력, 옛 이름)
    'est_solar_util_jeju': 'est_solar_utilization_jeju',  # D+2 깊이 룩업 키(수확 안 하나 주입)
}

# 수확: 스크래치 forecast 산출 컬럼 → 통합 출력 컬럼 (D+1·D+2 동일 이름으로 horizon_d 구분)
HARVEST = {
    1: {'est_smp_jeju': 'est_smp', 'smp_neg_proba_jeju': 'smp_neg_proba',
        'smp_danger_jeju': 'smp_danger'},
    2: {'est_smp_jeju_d2': 'est_smp', 'smp_neg_proba_d2': 'smp_neg_proba',
        'smp_danger_d2': 'smp_danger'},
}
OUT_COLS = ['est_smp', 'smp_neg_proba', 'smp_danger']


def _S(t):
    return pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')


def _safe_remove(path: str):
    """잠긴 파이프라인의 with-블록이 닫지 않은 스크래치 연결을 gc 로 회수 후 삭제(Windows 파일잠김)."""
    if not os.path.exists(path):
        return
    for _ in range(5):
        try:
            os.remove(path); return
        except PermissionError:
            gc.collect(); time.sleep(0.2)
    os.remove(path)  # 마지막 시도 — 실패 시 예외 전파


def list_bases() -> list[str]:
    # 12z(21:00 발표) 전용 — 18z(03:00 당일예보) base 는 hd=0 이 21행(03시~)이라
    # smp_d1 의 '대상일 24행' 검증과 맞지 않고, SMP 는 12z 풀 파이프라인의
    # 몫이다 (basetime 확정 설계 2026-07-17, 18z cron 은 수집+체인만).
    with sqlite3.connect(DB) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT base FROM est_horizon_jeju "
            "WHERE substr(base, 12) = '21:00:00' ORDER BY base").fetchall()]


def pick_bases(arg_base, backfill) -> list[str]:
    bases = list_bases()
    if not bases:
        return []
    if arg_base:
        hit = [b for b in bases if b[:10] == arg_base[:10]]
        if not hit:
            raise SystemExit(f'est_horizon_jeju 에 base {arg_base} 없음 (최신={bases[-1]})')
        return hit
    if backfill:
        return bases[-backfill:]
    return [bases[-1]]


def _read_eh(base: str) -> pd.DataFrame:
    """그 base 의 est_horizon_jeju (horizon_d 1~3) → 스크래치 forecast 프레임."""
    sel = 'timestamp, horizon_d, ' + ', '.join(EH_TO_FC)
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(
            f'SELECT {sel} FROM est_horizon_jeju WHERE base=? '
            f'AND horizon_d IN ({",".join("?" * len(HZ_SCRATCH))}) ORDER BY timestamp',
            con, params=(base, *HZ_SCRATCH), parse_dates=['timestamp'])
    return df.rename(columns=EH_TO_FC)


def build_scratch(base: str, scratch_path: str) -> dict | None:
    """그 base 의 스크래치 DB 구성 → {date_d1, date_d2, ts_d1, ts_d2} (없으면 None).

    historical 전체 복사(DA lag·rt) + est_horizon_jeju 1~3 을 forecast 형태로 적재
    (smp_jeju_da 는 historical 에서 timestamp join — D+1 은 발표값, D+2/3 은 NaN→예측이 덮음)."""
    eh = _read_eh(base)
    if eh.empty:
        return None
    blocks = {n: eh[eh.horizon_d == n].copy() for n in HZ_SCRATCH if (eh.horizon_d == n).any()}
    if 1 not in blocks or 2 not in blocks:
        return None  # D+1·D+2 둘 다 있어야 서빙

    source_con = sqlite3.connect(DB)
    da = pd.read_sql('SELECT timestamp, smp_jeju_da FROM historical ORDER BY timestamp',
                     source_con, parse_dates=['timestamp'])
    _safe_remove(scratch_path)
    scratch_con = sqlite3.connect(scratch_path)
    # historical 통째 복사 (DA 전구간 lag + rt 타깃 join 용)
    source_con.backup(scratch_con, name='main', pages=0)  # 전체 복제 후 forecast 만 교체
    source_con.close()
    # 기존 forecast(있으면) 비우고 스크래치 입력으로 재구성
    scratch_con.execute('DROP TABLE IF EXISTS forecast')

    forecast_df = eh.drop(columns=['horizon_d']).copy()
    forecast_df = forecast_df.merge(da, on='timestamp', how='left')        # smp_jeju_da 부착
    forecast_df['timestamp'] = forecast_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    forecast_df.to_sql('forecast', scratch_con, index=False, if_exists='replace')
    # 파이프라인 _upsert 의 ON CONFLICT(timestamp) 충족용 UNIQUE 제약
    scratch_con.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_fc_ts ON forecast(timestamp)')
    scratch_con.commit(); scratch_con.close()

    return {
        'date_d1': blocks[1]['timestamp'].dt.normalize().min().strftime('%Y-%m-%d'),
        'date_d2': blocks[2]['timestamp'].dt.normalize().min().strftime('%Y-%m-%d'),
        'ts_d1': set(blocks[1]['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')),
        'ts_d2': set(blocks[2]['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')),
    }


def _patch_db_path(mods, scratch_path):
    from forecasting import smp_features
    for m in mods + [smp_features]:
        if hasattr(m, 'DB_PATH'):
            m.DB_PATH = scratch_path


def _harvest_smp(forecast_df: pd.DataFrame, meta: dict, base: str) -> list:
    """스크래치 forecast 산출에서 D+1·D+2 각각의 핵심 3컬럼만 골라 통합 프레임 행들로 만든다.

    지평별로 그 지평의 타깃 시각(meta['ts_dN'])만 남기고, 잠긴 코드가 쓴 컬럼명(HARVEST 키)을
    통합 출력 이름(est_smp·smp_neg_proba·smp_danger)으로 바꾼다.
    """
    rows = []
    for n in HZ:
        keep_ts = meta[f'ts_d{n}']
        src_cols = HARVEST[n]
        have = [c for c in src_cols if c in forecast_df.columns]
        if not have:
            continue
        sub = forecast_df.loc[forecast_df.index.isin(keep_ts), have].rename(columns=src_cols)
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={'index': 'timestamp'})
        sub['base'] = base; sub['horizon_d'] = n
        rows.append(sub)
    return rows


def serve_base(base: str, pipeline_d1, pipeline_d2, scratch_path: str) -> pd.DataFrame:
    """그 base 의 D+1·D+2 SMP 핵심 3컬럼 → 통합 프레임.

    ① 스크래치 DB 구성 → ② 잠긴 파이프라인(pipeline_d1 D+1·pipeline_d2 D+2)으로 예측(스크래치 forecast 에 산출
    컬럼을 덧씀) → ③ 핵심 3컬럼 수확.  pipeline_d1/pipeline_d2 의 DB_PATH 는 스크래치로 몽키패치돼 있다.
    """
    meta = build_scratch(base, scratch_path)
    if meta is None:
        print(f'  [skip] base {base[:10]} — est_horizon_jeju D+1·D+2 부족')
        return pd.DataFrame()

    _patch_db_path([pipeline_d1, pipeline_d2], scratch_path)
    try:
        pipeline_d1.predict_smp_to_db(meta['date_d1'], write=True, verbose=False)   # D+1
    except Exception as e:
        print(f'  [skip-D1] base {base[:10]}: {str(e)[:70]}')
    try:
        pipeline_d2.predict_d2_to_db(meta['date_d2'], write=True, verbose=False)    # D+2
    except Exception as e:
        print(f'  [skip-D2] base {base[:10]}: {str(e)[:70]}')

    scratch_con = sqlite3.connect(scratch_path)
    try:
        forecast_df = pd.read_sql('SELECT * FROM forecast', scratch_con, parse_dates=False)
    finally:
        scratch_con.close()   # with-블록은 연결을 닫지 않음(파일 잠김 방지 위해 명시 close)
    forecast_df = forecast_df.set_index('timestamp')
    rows = _harvest_smp(forecast_df, meta, base)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def upsert_est(r: pd.DataFrame, db_path: str) -> int:
    if r.empty:
        return 0

    def _v(x):
        if x is None:
            return None
        try:
            f = float(x)
            return None if not np.isfinite(f) else f
        except (TypeError, ValueError):
            return None
    data = []
    for row in r.itertuples(index=False):
        vals = [getattr(row, c, None) for c in OUT_COLS]
        if all(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in vals):
            continue
        data.append((_S(row.timestamp), str(row.base), int(row.horizon_d),
                     _v(vals[0]), _v(vals[1]),
                     None if _v(vals[2]) is None else int(round(_v(vals[2])))))
    if not data:
        return 0
    with sqlite3.connect(db_path) as con:
        con.execute('CREATE TABLE IF NOT EXISTS est_smp_horizon_jeju ('
                    'timestamp TEXT, base TEXT, horizon_d INT, '
                    'est_smp REAL, smp_neg_proba REAL, smp_danger INT, '
                    'PRIMARY KEY(base, timestamp))')
        con.executemany(
            'INSERT INTO est_smp_horizon_jeju '
            '(timestamp, base, horizon_d, est_smp, smp_neg_proba, smp_danger) '
            'VALUES (?,?,?,?,?,?) ON CONFLICT(base, timestamp) DO UPDATE SET '
            'horizon_d=excluded.horizon_d, est_smp=excluded.est_smp, '
            'smp_neg_proba=excluded.smp_neg_proba, smp_danger=excluded.smp_danger', data)
        con.commit()
    return len(data)


def main():
    ap = argparse.ArgumentParser(description='제주 SMP 서빙 러너 D+1·D+2 → est_smp_horizon_jeju')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--base', default=None, help='특정 base YYYY-MM-DD (기본: est_horizon_jeju 최신)')
    g.add_argument('--backfill', type=int, default=None, help='최근 N개 base')
    ap.add_argument('--no-write', action='store_true', help='산출만 — 적재 생략')
    a = ap.parse_args()

    bases = pick_bases(a.base, a.backfill)
    if not bases:
        raise SystemExit('est_horizon_jeju 비어있음 — 먼저 2·3단계 서빙(serve_chain_jeju_new) 필요')
    print(f'[serve_smp] 대상 base {len(bases)}개: {bases[0][:10]} ~ {bases[-1][:10]}')

    # 잠긴 SMP 파이프라인 — 부트스트랩이 루트를 sys.path 에 올려 둬 일반 import 가능
    from forecasting import smp_d1 as pipeline_d1
    from forecasting import smp_d2 as pipeline_d2
    from forecasting import smp_features
    # ② D+2 미래 서빙 = 타깃 조인 끔 (rt 없는 미래행 보존)
    pipeline_d2.load_forecast = lambda with_target=True: smp_features.load_forecast(with_target=False)

    scratch_path = os.path.join(tempfile.gettempdir(), 'serve_smp_horizon_jeju.db')
    t0 = time.time(); total = 0
    for bi, base in enumerate(bases, 1):
        r = serve_base(base, pipeline_d1, pipeline_d2, scratch_path)
        if r.empty:
            print(f'  base {bi}/{len(bases)} ({base[:10]}) — 산출 없음'); continue
        d1 = r[r.horizon_d == 1]; d2 = r[r.horizon_d == 2]
        print(f'  base {bi}/{len(bases)} ({base[:10]})  D+1 {len(d1)}h(경보 {int(d1.smp_danger.sum())}h '
              f'est_smp {d1.est_smp.mean():.1f})  D+2 {len(d2)}h(경보 {int(d2.smp_danger.sum())}h '
              f'est_smp {d2.est_smp.mean():.1f})  {time.time()-t0:.0f}s')
        if not a.no_write:
            total += upsert_est(r, DB)

    if a.no_write:
        print('\n(--no-write: 적재 생략)')
    else:
        with sqlite3.connect(DB) as con:
            tot = con.execute('SELECT COUNT(*) FROM est_smp_horizon_jeju').fetchone()[0]
            rng = con.execute('SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT base) '
                              'FROM est_smp_horizon_jeju').fetchone()
        print(f'\nest_smp_horizon_jeju UPSERT {total}행  (전체 {tot}행, base {rng[2]}개, {rng[0]} ~ {rng[1]})')
    print(f'[serve_smp] done in {(time.time()-t0)/60:.1f}m')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
