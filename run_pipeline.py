# -*- coding: utf-8 -*-
"""run_pipeline.py — 수집→예측 전체 파이프라인 단일 진입점 (cron·관리자 메뉴 공용).

역할
----
①실측 수집 → ②기상예보 수집 → ③예측 체인(수요+신재생) → ④SMP 예측 을 순서대로 실행한다.
  - 리눅스 서버 crontab 이 이 파일을 그대로 호출한다 (등록 예시는 README 참고).
  - streamlit 관리자 메뉴의 "▶ 오늘 예측" 원클릭도 같은 단계 정의(PIPELINE_STEPS)를 공유한다
    → 파이프라인이 바뀌면 이 파일 하나만 고치면 된다.

실행 정책
----
한 단계가 실패해도 나머지 단계는 계속 시도한다(수집 일부 실패가 예측 전체를 막지 않도록).
단, 하나라도 실패하면 종료코드 1 — cron 의 실패 알림(메일/로그)이 걸리게 한다.
모든 출력은 stdout + logs/pipeline_YYYYMMDD_HHMMSS.log 에 동시 기록된다.

사용
----
    python run_pipeline.py                     # 12z 풀 (①→④, cron 00:20 KST)
    python run_pipeline.py --steps light18     # 18z 당일예보 라이트 (①→②′→②′-2→③′, cron ~08:00 KST)
    python run_pipeline.py --steps collect     # 수집만 (①②)
    python run_pipeline.py --steps predict     # 예측만 (③④)
    python run_pipeline.py --steps historical,smp   # 단계 이름 나열도 가능
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

LOG_DIR = ROOT / "logs"

# 파이프라인 단계 정의 — (키, 표시 이름, 스크립트 경로, 인자).
# 관리자 메뉴(page_main.render_admin)가 이 목록을 import 해 같은 단계를 실행한다.
PIPELINE_STEPS = [
    # 실측 = KPX 수급·RT SMP + ASOS (최근 3일).  KPX 하루전(*_da)만 today+2 까지
    # 받아 화면의 SMP 검증 참조선·수요 비교선에 내일치가 들어간다 (2026-07-21).
    ("historical", "① 실측 수집 (최근 3일, *_da 는 +2일)", P.COLLECT_HISTORICAL,
     ["--historical-days", "3"]),
    # 지평 5일 (사용자 확정 2026-07-17): KIMR 이 5일까지 1h 전량 커버.
    # KIMG 는 서빙 입력 유지용 — 서빙 운량·일사(total/midlow_cloud_*, radiation_*)는
    # KIMG(NE57) 분포로 학습돼 있어 재훈련 전에는 KIMG 단독 공급을 유지한다.
    # (2026-07-31 실측으로 재확인: KIMR 일사·운량은 KIMG 와 다른 모델이고 열세다.)
    # met 은 2026-08-04 부터 KIMR std NC (구 GRIB 경로 폐기).  3h 결손은 수집 단계에서
    # 보간한다(연속 2개까지, 외삽 금지) — 아카이브와 같은 postprocess.fill_short_gaps.
    ("forecast", "② 기상예보 수집 (전일 밤 12z, 익일~5일후)", P.COLLECT_FORECAST,
     ["--region", "jeju", "--days", "5"]),
    # KIMR/KIMG 소스 분리 기상 수집 (forecast_kimr NC / forecast_kimg → 메인 DB).
    # 2026-07-30 격리 아카이브(weather_kim.db)를 메인 DB 로 흡수.  재학습이 이 두
    # 테이블에서 서빙 입력을 새로 만든다.  3h 결손 보간은 ② 와 공용(fill_short_gaps).
    ("weather", "②-2 KIMR/KIMG 소스 분리 수집 (12z 5일 1h → forecast_kimr/kimg)",
     P.COLLECT_ARCHIVE, []),
    ("chain", "③ 예측 체인 (12z → est_horizon_jeju)", P.SERVE_CHAIN, ["--utc", "12"]),
    ("smp", "④ SMP 예측 (12z 전용 → est_smp_horizon_jeju)", P.SERVE_SMP, []),
    # ── 18z 라이트 (당일예보, cron ~08:00 KST — basetime 확정 설계 2026-07-17) ──
    # 18z(당일 03시 KST 발표)도 12z 와 동일한 KIMR met + KIMG 일사·운량 병합으로 수집
    # (2026-07-18 사용자 결정: 재학습 대비 NE57 아카이브 연속성).  지평 = 당일~모레
    # (horizon_d 0~2, KIMR 18z lead 72h).  3~5일후는 12z 가 freshest-wins 로 담당.
    ("forecast18", "②′ 기상예보 수집 (당일 새벽 18z, 당일~모레)", P.COLLECT_FORECAST,
     ["--region", "jeju", "--days", "3", "--utc", "18"]),
    ("weather18", "②′-2 KIMR/KIMG 소스 분리 수집 (18z 3일 1h → forecast_kimr/kimg)",
     P.COLLECT_ARCHIVE, ["--utc", "18", "--days", "3"]),
    ("chain18", "③′ 예측 체인 (18z 당일예보 → est_horizon_jeju)", P.SERVE_CHAIN,
     ["--utc", "18"]),
]
STEP_GROUPS = {
    # all = 12z 풀 파이프라인 (명시 고정 — 18z 단계는 light18 그룹 전용)
    "all": ["historical", "forecast", "weather", "chain", "smp"],
    "collect": ["historical", "forecast", "weather"],
    "predict": ["chain", "smp"],
    # 18z 라이트: historical 선행 필수 — 수요 서빙의 과거창 168h(전일 23시까지 실측)와
    # 태양광 서빙의 전일 이용률 실측이 있어야 당일예보(hd=0)가 나온다.
    "light18": ["historical", "forecast18", "weather18", "chain18"],
}
STEP_TIMEOUT_SECONDS = 3600   # 단계당 상한 — 예보 수집(KIMG 3지점)이 가장 오래 걸린다(~3분)


def _resolve_steps(arg: str) -> list[str]:
    """--steps 인자를 단계 키 목록으로 — 그룹 이름(all/collect/predict)과 키 나열 둘 다 허용."""
    keys = [k for k, *_ in PIPELINE_STEPS]
    chosen: list[str] = []
    for token in arg.split(","):
        token = token.strip()
        if token in STEP_GROUPS:
            chosen += [k for k in STEP_GROUPS[token] if k not in chosen]
        elif token in keys:
            if token not in chosen:
                chosen.append(token)
        else:
            raise SystemExit(f"알 수 없는 단계 '{token}' — 사용 가능: {keys + list(STEP_GROUPS)}")
    return [k for k in keys if k in chosen]   # 실행 순서는 항상 정의 순서


def run_step(script, args, log_file, timeout: int = STEP_TIMEOUT_SECONDS) -> int:
    """한 단계를 현재 인터프리터로 실행 — 출력을 실시간으로 stdout·로그에 동시 기록."""
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    start = time.time()
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
        if time.time() - start > timeout:
            proc.kill()
            print(f"[timeout] {timeout}s 초과 — 단계 중단")
            log_file.write(f"[timeout] {timeout}s 초과 — 단계 중단\n")
            return 124
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="제주 수집→예측 파이프라인 (cron·관리자 공용)")
    parser.add_argument("--steps", default="all",
                        help="실행 단계 — all(12z 풀) / collect / predict / light18(18z 당일예보) "
                             "/ 단계키 나열 (historical,forecast,weather,chain,smp,"
                             "forecast18,weather18,chain18)")
    parsed = parser.parse_args()
    selected = _resolve_steps(parsed.steps)

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    results: list[tuple[str, int, float]] = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        def echo(msg: str):
            print(msg)
            log_file.write(msg + "\n")

        echo(f"[run_pipeline] 시작 {datetime.now():%Y-%m-%d %H:%M:%S}  단계: {selected}")
        for key, label, script, args in PIPELINE_STEPS:
            if key not in selected:
                continue
            echo(f"\n{'=' * 60}\n{label}\n  $ {Path(script).name} {' '.join(args)}\n{'=' * 60}")
            t0 = time.time()
            rc = run_step(script, args, log_file)
            elapsed = time.time() - t0
            results.append((key, rc, elapsed))
            echo(f"→ {label}: {'성공' if rc == 0 else f'실패(rc={rc})'}  ({elapsed:.0f}s)")

        echo(f"\n{'=' * 60}\n[run_pipeline] 요약")
        for key, rc, elapsed in results:
            echo(f"  {key:10s} {'OK ' if rc == 0 else 'FAIL'}  rc={rc}  {elapsed:.0f}s")
        failed = [k for k, rc, _ in results if rc != 0]
        echo(f"[run_pipeline] 종료 {datetime.now():%Y-%m-%d %H:%M:%S}  "
             f"{'전체 성공' if not failed else '실패 단계: ' + ','.join(failed)}")
        echo(f"로그: {log_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
