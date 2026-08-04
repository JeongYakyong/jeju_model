"""smp_forecaster CLI demo / 명령줄 데모

Subcommands:
    train          모델 학습 (DB → BANK → 학습 → joblib 저장)
                   Train SMP model from DB and save artifact

    predict DATE   해당 날짜의 24h SMP 예측 출력
                   Print 24h SMP prediction for DATE

    ingest         RT SMP CSV를 DB로 적재 (선택)
                   Ingest realtime SMP CSV into DB (optional)

Examples:
    python -m examples.run_smp_forecast ingest
    python -m examples.run_smp_forecast train
    python -m examples.run_smp_forecast predict 2026-05-14
"""
from __future__ import annotations

import argparse
import logging
import sys

# Windows PowerShell에서 한글이 깨지지 않도록 UTF-8 강제.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from smp_forecaster import ingest_rt_smp, predict_smp, train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: train
# ─────────────────────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> int:
    summary = train()
    print("\n=== 학습 완료 / Training done ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: predict
# ─────────────────────────────────────────────────────────────────────────────
def cmd_predict(args: argparse.Namespace) -> int:
    target = args.date
    print(f"\n[*] {target} SMP 예측 / Predicting SMP for {target}")
    df = predict_smp(target)
    print(df.round(3).to_string())
    print(f"\nPeak smp_pred: {df['smp_pred'].max():.2f}")
    print(f"Min  smp_pred: {df['smp_pred'].min():.2f}")
    n_danger = int(df['danger'].sum())
    if n_danger:
        print(f"위험띠 시간수 / Danger hours: {n_danger}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: ingest
# ─────────────────────────────────────────────────────────────────────────────
def cmd_ingest(args: argparse.Namespace) -> int:
    n = ingest_rt_smp(csv_path=args.csv)
    print(f"\n[+] RT SMP {n:,}행을 DB에 적재 / ingested {n:,} rows")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run_smp_forecast',
        description=(
            "Jeju SMP forecaster CLI (Stage 2) / 제주 SMP 예측 CLI\n\n"
            "Subcommands:\n"
            "  train          모델 학습 / Train model\n"
            "  predict DATE   24h SMP 예측 / Predict 24h SMP for DATE\n"
            "  ingest         RT SMP CSV → DB (선택) / Ingest CSV (optional)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples / 예시:\n"
            "  python -m examples.run_smp_forecast ingest\n"
            "  python -m examples.run_smp_forecast train\n"
            "  python -m examples.run_smp_forecast predict 2026-05-14\n"
        ),
    )
    sub = parser.add_subparsers(
        dest='command', required=True, metavar='{train,predict,ingest}',
    )

    p_train = sub.add_parser(
        'train',
        help='모델 학습 / Train SMP model',
        description='DB에서 데이터를 읽어 학습 후 models/smp_model.pkl 저장.',
    )
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser(
        'predict',
        help='24h SMP 예측 / Predict 24h SMP',
        description='forecast_data에서 DATE 예보를 읽고 24h SMP 예측을 출력.',
    )
    p_predict.add_argument('date', help='YYYY-MM-DD')
    p_predict.set_defaults(func=cmd_predict)

    p_ingest = sub.add_parser(
        'ingest',
        help='RT SMP CSV를 DB로 / Ingest RT SMP CSV',
        description='realtime_smp_24-26.csv 를 시간별로 정리해 DB에 UPSERT.',
    )
    p_ingest.add_argument(
        '--csv', default=None,
        help='CSV 경로 (기본: config.RT_SMP_CSV)',
    )
    p_ingest.set_defaults(func=cmd_ingest)

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
