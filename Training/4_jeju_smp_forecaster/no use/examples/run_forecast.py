"""net_load_forecaster CLI demo / 명령줄 데모

Subcommands:
    predict DATE   예측 결과 받기 (필요시 데이터 자동 수집)
                   Get prediction result (auto-fetches data if missing)

    fetch   DATE   데이터만 수집 (예측은 안 함)
                   Fetch KPX/KMA data only (no prediction)

Examples:
    python -m examples.run_forecast predict 2026-05-22
    python -m examples.run_forecast fetch   2026-05-22
    python -m examples.run_forecast predict 2026-05-22 --no-fetch
    python -m examples.run_forecast fetch   2026-05-22 --kind forecast
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta

# Force UTF-8 stdout so Korean text renders on Windows PowerShell consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from net_load_forecaster import (
    DB_PATH,
    JejuEnergyDB,
    compute_net_load_for_date,
    fetch_data,
    predict,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

HISTORY_DAYS = 14


def _history_window(target_date: str) -> tuple[str, str]:
    """For a target date, return the historical window (past 14 days)."""
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (target_dt - timedelta(days=HISTORY_DAYS + 1)).strftime("%Y-%m-%d")
    end   = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: predict
# ─────────────────────────────────────────────────────────────────────────────
def cmd_predict(args: argparse.Namespace) -> int:
    target = args.date

    if not args.no_fetch:
        h_start, h_end = _history_window(target)
        print(f"\n[1/3] 실측 데이터 수집 / Fetching historical  ({h_start} ~ {h_end}) ...")
        fetch_data(h_start, h_end, kind='historical')

        print(f"\n[2/3] 예보 데이터 수집 / Fetching forecast    ({target}) ...")
        fetch_data(target, target, kind='forecast')
    else:
        print("\n[1-2/3] --no-fetch: DB에 이미 데이터가 있다고 가정")

    print(f"\n[3/3] 모델 추론 / Running inference for {target} ...")
    pred_df = predict(target)
    print(pred_df.head())

    print(f"\n--- Net load for {target} ---")
    db = JejuEnergyDB(str(DB_PATH))
    try:
        net_load_df = compute_net_load_for_date(target, db)
        print(net_load_df.round(2))
        print(f"\nPeak net_load: {net_load_df['net_load_mw'].max():.1f} MW")
        print(f"Min  net_load: {net_load_df['net_load_mw'].min():.1f} MW")
    finally:
        db.close()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: fetch
# ─────────────────────────────────────────────────────────────────────────────
def cmd_fetch(args: argparse.Namespace) -> int:
    target = args.date
    kind   = args.kind

    if kind == 'historical':
        h_start, h_end = _history_window(target)
        print(f"\n실측 데이터 수집 / Fetching historical  ({h_start} ~ {h_end}) ...")
        fetch_data(h_start, h_end, kind='historical')

    elif kind == 'forecast':
        print(f"\n예보 데이터 수집 / Fetching forecast    ({target}) ...")
        fetch_data(target, target, kind='forecast')

    else:  # 'all' (default)
        h_start, h_end = _history_window(target)
        print(f"\n[1/2] 실측 데이터 수집 / Fetching historical  ({h_start} ~ {h_end}) ...")
        fetch_data(h_start, h_end, kind='historical')
        print(f"\n[2/2] 예보 데이터 수집 / Fetching forecast    ({target}) ...")
        fetch_data(target, target, kind='forecast')

    print("\n수집 완료 / Done. (predict 명령으로 예측 실행 가능 / Run `predict` next)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run_forecast',
        description=(
            "Jeju net-load forecaster CLI / 제주 순부하 예측 CLI\n\n"
            "Two subcommands:\n"
            "  predict DATE   예측 결과 받기 (필요시 데이터 자동 수집)\n"
            "                 Get prediction (auto-fetches missing data)\n"
            "  fetch   DATE   데이터만 수집 (예측 없음)\n"
            "                 Fetch KPX/KMA data only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples / 예시:\n"
            "  python -m examples.run_forecast predict 2026-05-22\n"
            "  python -m examples.run_forecast fetch   2026-05-22\n"
            "  python -m examples.run_forecast fetch   2026-05-22 --kind forecast\n"
            "  python -m examples.run_forecast predict 2026-05-22 --no-fetch\n"
        ),
    )
    sub = parser.add_subparsers(dest='command', required=True, metavar='{predict,fetch}')

    # predict
    p_predict = sub.add_parser(
        'predict',
        help='예측 실행 (필요시 데이터 자동 수집) / Run prediction',
        description='Run the PatchTST model for DATE and print net_load.',
    )
    p_predict.add_argument('date', help='YYYY-MM-DD')
    p_predict.add_argument(
        '--no-fetch', action='store_true',
        help='데이터 수집을 건너뜀 (DB에 이미 있다고 가정) / Skip data collection',
    )
    p_predict.set_defaults(func=cmd_predict)

    # fetch
    p_fetch = sub.add_parser(
        'fetch',
        help='데이터만 수집 / Fetch data only',
        description='Collect KPX/KMA data into the SQLite DB for DATE.',
    )
    p_fetch.add_argument('date', help='YYYY-MM-DD')
    p_fetch.add_argument(
        '--kind', choices=['all', 'historical', 'forecast'], default='all',
        help=(
            "수집 종류 / what to fetch:\n"
            "  all         = 과거 14일 실측 + DATE 예보 (default)\n"
            "                past 14d actuals + DATE forecast\n"
            "  historical  = 과거 14일 실측만 / actuals only\n"
            "  forecast    = DATE 예보만 / forecast only"
        ),
    )
    p_fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
