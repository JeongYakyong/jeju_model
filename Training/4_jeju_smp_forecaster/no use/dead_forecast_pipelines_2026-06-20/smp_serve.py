"""4단계 제주 SMP — D+1·D+2 통합 서빙 오케스트레이터 (단일 진입점).

D-2 운영: 매일 23:00 예측, 대상 48h = 앞24h(D+1) + 뒤24h(D+2).
이 파일 하나로 두 구간 산출물을 forecast 테이블에 일괄 채운다. 개별 파이프라인을
import만 해서 호출(로직 중복 없음) — 각 모델/컬럼은 각 파이프라인이 그대로 책임.

────────────────────────────────────────────────────────────────────────
구성 (전부 input_data_jeju.db forecast 테이블 UPSERT)
  ■ D+1 (앞24h, DA 발표됨 → DA 그대로 가격선)
      smp_db_pipeline       est_smp_jeju · smp_neg_proba_jeju · smp_danger_jeju
      smp_depth_pipeline    smp_neg_depth_p10/p50/p90
      smp_softest_pipeline  smp_neg_proba_cal_jeju · smp_rt_soft_est ·
                            smp_danger_day_jeju · smp_danger_night_jeju
  ■ D+2 (뒤24h, DA 미발표 → 예측 DA + D+1 음수검지기 오버레이)
      smp_d2_pipeline       est_smp_jeju_d2 · smp_neg_proba_d2 ·
                            smp_danger_d2(균형) · smp_danger_d2_hi(고확신) ·
                            smp_neg_depth_d2_p10/p50/p90

제주 SMP는 제주 데이터만 사용(육지 SMP 연계 영구 배제).
────────────────────────────────────────────────────────────────────────
공개 API
    serve_day(date, scope='both')         # 단일일 D+1·D+2 서빙
    serve_range(start, end, scope='both') # 구간 일괄(백필)
CLI
    python smp_serve.py day 2026-03-19
    python smp_serve.py range 2025-12-13 2026-05-28
    python smp_serve.py range 2025-12-13 2026-05-28 --scope d2   # D+2만
"""
from __future__ import annotations

# D+1 (앞24h) — 기존 A안/Phase2/Phase4 서빙
from smp_db_pipeline import predict_smp_to_db, backfill_smp_to_db
from smp_depth_pipeline import predict_depth_to_db, backfill_depth_to_db
from smp_softest_pipeline import predict_softest_to_db, backfill_softest_to_db
# D+2 (뒤24h) — 예측 DA + 오버레이
from smp_d2_pipeline import predict_d2_to_db, run as backfill_d2_to_db

SCOPES = ('both', 'd1', 'd2')


def _run_d1_day(date, write):
    predict_smp_to_db(date, write=write, verbose=False)      # 가격선+경보
    predict_depth_to_db(date, write=write, verbose=False)    # 깊이 overlay
    predict_softest_to_db(date, write=write, verbose=False)  # 위험 레이어


def _run_d1_range(start, end, write):
    # 개별 backfill 함수는 write 전제(no-write 미지원) → write=True일 때만 호출
    backfill_smp_to_db(start, end, verbose=False)
    backfill_depth_to_db(start, end, verbose=False)
    backfill_softest_to_db(start, end, verbose=False)


def serve_day(date: str, scope: str = 'both', write: bool = True, verbose: bool = True):
    """date(YYYY-MM-DD)의 D+1·D+2 산출물을 forecast 테이블에 UPSERT."""
    _check_scope(scope)
    if scope in ('both', 'd1'):
        if not write:
            raise ValueError('D+1 단계는 no-write 미지원 — scope=d2로 분리하거나 write=True')
        _run_d1_day(date, write)
        if verbose:
            print(f'[D+1] {date} 가격선·경보·깊이·위험레이어 UPSERT')
    if scope in ('both', 'd2'):
        predict_d2_to_db(date, write=write, verbose=verbose)
    if verbose:
        print(f'[serve_day] {date} (scope={scope}) 완료')


def serve_range(start: str, end: str, scope: str = 'both', write: bool = True, verbose: bool = True):
    """[start,end] 구간 D+1·D+2 일괄 백필."""
    _check_scope(scope)
    if scope in ('both', 'd1'):
        if not write:
            raise ValueError('D+1 백필은 no-write 미지원 — scope=d2로 분리하거나 write=True')
        _run_d1_range(start, end, write)
        if verbose:
            print(f'[D+1] {start}~{end} 백필 완료')
    if scope in ('both', 'd2'):
        backfill_d2_to_db(start, end, write=write, verbose=verbose)
    if verbose:
        print(f'[serve_range] {start}~{end} (scope={scope}) 완료')


def _check_scope(scope):
    if scope not in SCOPES:
        raise ValueError(f'scope must be one of {SCOPES}, got {scope!r}')


if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 SMP D+1·D+2 통합 서빙 (DB 전용)')
    p.add_argument('--scope', choices=SCOPES, default='both', help='both(기본)/d1/d2')
    p.add_argument('--no-write', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)
    dd = sub.add_parser('day', help='단일일 서빙'); dd.add_argument('date')
    rr = sub.add_parser('range', help='구간 백필'); rr.add_argument('start'); rr.add_argument('end')
    a = p.parse_args()
    if a.cmd == 'day':
        serve_day(a.date, scope=a.scope, write=not a.no_write)
    elif a.cmd == 'range':
        serve_range(a.start, a.end, scope=a.scope, write=not a.no_write)
