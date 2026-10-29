# -*- coding: utf-8 -*-
"""AI 브리핑 일괄 생성·점검 — 화면 버튼과 같은 로직(brief_jeju)을 Streamlit 없이 돌린다.

    python pages/gen_briefs.py                       # 매일 cron: 오늘·내일 브리핑 갱신(덮어쓰기)
    python pages/gen_briefs.py --days 3              # 오늘부터 3일
    python pages/gen_briefs.py --start 2025-12-20 --end 2026-10-29 --missing-only   # backfill
    python pages/gen_briefs.py --start 2025-12-20 --end 2026-10-29 --check          # 무결성 점검(API 0회)

- 임계값은 화면 기본값(chart_warn._WARN_DEFAULTS)을 쓴다 — 화면에서 바꾼 임계값은 그 세션에만 적용.
- 실패 응답("Generation failed" 등)은 저장하지 않는다(brief_jeju.is_valid_brief).
- 예측(est_net_load_jeju)이 없는 날은 API 를 부르지 않고 건너뛴다.
- 범위는 최신 날짜부터 처리한다 — 할당량이 중간에 떨어져도 최근 날짜가 먼저 채워진다.
- 할당량 초과(429/RESOURCE_EXHAUSTED)가 재시도 후에도 계속되면 남은 날짜는 멈춘다.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit.logger

streamlit.logger.set_log_level("error")   # Streamlit 밖 실행 시 'missing ScriptRunContext' 경고 소음 제거

from pages import common as C
from pages import brief_jeju
from pages import brief_store
from pages.chart_warn import _WARN_DEFAULTS

MAX_ATTEMPTS = 3          # 날짜 하나당 API 시도 횟수
QUOTA_WAIT_SECONDS = 60   # 429 를 받으면 이만큼 쉬고 다시 시도 (분당 한도 회복 대기)
ERROR_WAIT_SECONDS = 10   # 그 밖의 실패 후 재시도 대기


def needs_brief(saved: dict | None) -> str | None:
    """저장된 브리핑을 새로 만들어야 하는 이유 — 괜찮으면 None."""
    if saved is None:
        return "없음"
    if not brief_jeju.is_valid_brief(saved.get("brief_text")):
        return "실패 문구"
    if not C.show_smp() and "SMP" in (saved.get("fact_text") or ""):
        return "SMP 포함(구 형식)"
    return None


def has_forecast(fact_df: pd.DataFrame) -> bool:
    return "est_net_load_jeju" in fact_df.columns and fact_df["est_net_load_jeju"].notna().any()


def generate_one(day: pd.Timestamp) -> tuple[str, str | None]:
    """한 날짜 생성·저장 → (결과, 상세). 결과 = 'saved' / 'no_forecast' / 'failed' / 'quota'."""
    fact_df = brief_jeju.build_fact_frame(day)
    if not has_forecast(fact_df):
        return "no_forecast", None
    text = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text, facts = brief_jeju.generate_energy_narrative(
            fact_df, _WARN_DEFAULTS["warn_low"], _WARN_DEFAULTS["warn_high"])
        if brief_jeju.is_valid_brief(text):
            brief_store.save(day.strftime("%Y-%m-%d"), days=1, kind="daily",
                             brief_text=text, fact_text=facts, model=brief_jeju.GEMINI_MODEL)
            return "saved", None
        is_quota = "429" in text or "RESOURCE_EXHAUSTED" in text
        if attempt < MAX_ATTEMPTS:
            time.sleep(QUOTA_WAIT_SECONDS if is_quota else ERROR_WAIT_SECONDS)
    detail = (text or "(빈 응답)")[:200]
    return ("quota" if is_quota else "failed"), detail


def run_check(days: list[pd.Timestamp]) -> int:
    """무결성 점검 — 날짜마다 저장 상태를 보고 문제 있는 날을 나열한다(API 호출 없음)."""
    problems = {}
    for day in days:
        reason = needs_brief(brief_store.load(day.strftime("%Y-%m-%d"), days=1, kind="daily"))
        if reason:
            problems.setdefault(reason, []).append(day.strftime("%Y-%m-%d"))
    ok_count = len(days) - sum(len(v) for v in problems.values())
    print(f"[check] {days[-1]:%Y-%m-%d} ~ {days[0]:%Y-%m-%d}  {len(days)}일 중 정상 {ok_count}일")
    for reason, dates in problems.items():
        print(f"  {reason}: {len(dates)}일  {', '.join(sorted(dates))}")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description="제주 AI 브리핑 일괄 생성·점검")
    parser.add_argument("--start", help="시작일 YYYY-MM-DD (없으면 오늘)")
    parser.add_argument("--end", help="종료일 YYYY-MM-DD (없으면 시작일 + --days - 1)")
    parser.add_argument("--days", type=int, default=2, help="--end 가 없을 때 시작일부터 며칠 (기본 2 = 오늘·내일)")
    parser.add_argument("--missing-only", action="store_true",
                        help="이미 정상 저장된 날은 건너뜀 (없음·실패 문구·SMP 포함 구 형식만 생성)")
    parser.add_argument("--check", action="store_true", help="무결성 점검만 (API 호출 없음)")
    parser.add_argument("--limit", type=int, default=None, help="이번 실행의 최대 생성 건수 (할당량 보호)")
    parser.add_argument("--sleep", type=float, default=4.0, help="생성 사이 대기 초 (분당 한도 보호, 기본 4)")
    args = parser.parse_args()

    start = pd.Timestamp(args.start) if args.start else pd.Timestamp.now().normalize()
    end = pd.Timestamp(args.end) if args.end else start + pd.Timedelta(days=args.days - 1)
    days = list(pd.date_range(start, end, freq="D"))[::-1]   # 최신 날짜부터
    if not days:
        raise SystemExit(f"빈 범위: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")

    if args.check:
        sys.exit(run_check(days))

    if args.missing_only:
        days = [d for d in days
                if needs_brief(brief_store.load(d.strftime("%Y-%m-%d"), days=1, kind="daily"))]
    if args.limit is not None:
        days = days[:args.limit]
    print(f"[gen_briefs] 대상 {len(days)}일 ({brief_jeju.GEMINI_MODEL}, SMP {'포함' if C.show_smp() else '제외'})")

    counts = {"saved": 0, "no_forecast": 0, "failed": 0, "quota": 0}
    for i, day in enumerate(days, 1):
        t0 = time.time()
        result, detail = generate_one(day)
        counts[result] += 1
        print(f"  [{i}/{len(days)}] {day:%Y-%m-%d}  {result}  ({time.time() - t0:.1f}s)"
              + (f"  {detail}" if detail else ""), flush=True)
        if result == "quota":
            print("[gen_briefs] 할당량 초과가 계속되어 중단합니다 — 남은 날짜는 다음 실행에서 채워집니다.")
            break
        if result == "saved" and i < len(days):
            time.sleep(args.sleep)

    print(f"[gen_briefs] 완료 — 저장 {counts['saved']} · 예측없음 {counts['no_forecast']} · "
          f"실패 {counts['failed']} · 할당량중단 {counts['quota']}")
    sys.exit(1 if counts["failed"] or counts["quota"] else 0)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
