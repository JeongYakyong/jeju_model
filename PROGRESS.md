# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-10-29

## 완료

- AI 브리핑: `pages/gen_briefs.py`(backfill `--missing-only` / 점검 `--check` / 매일 갱신). light18(08:00) 끝 `brief` 단계가 오늘·내일 갱신. 실패 응답은 저장 안 함.
- SMP 화면·브리핑 숨김 — 관리자 메뉴 토글(`ui_settings.json`, 기본 숨김). SMP 예측 생성은 유지.
- 서버 반영(커밋 33e6e92, `git pull`), 서버 DB 로 AppTest 4메뉴 렌더 확인.
- 브리핑 21일(10-10~10-30) 생성.

## 현재 상태

| | |
|---|---|
| 브리핑 | 정상 21일 / 남은 294일(없음 293 + 08-26 구 형식) |
| 할당량 | 서버 키 Free Tier — gemini-3-flash **하루 20회**, 16:00 KST 리셋 |
| 서버 cron | 16:10 KST `gen_briefs --missing-only --limit 15` 추가(최신 날짜부터) → 약 11-18 완료 |

## 다음 할 일

1. 서버에서 `sudo systemctl restart jeju_model.service` (자동 모드에서 거부돼 못 함).
2. 며칠 뒤 `python pages/gen_briefs.py --start 2025-12-20 --end <오늘> --check` 로 진행 확인, `logs/cron.log` 의 `[gen_briefs]` 줄 확인.
3. backfill 완료 후 16:10 cron 줄은 지워도 된다(남겨도 호출 0회).
4. 이전 이월: 풍력 새 구성(54f3930, 서버 반영됨)의 정오·강풍 편향 실측 확인.

## 주의사항

- 10-30 08:00 brief 는 할당량 리셋 전이라 실패(저장 안 됨) — 정상.
- 수동 브리핑 생성을 많이 하면 다음 날 08:00 갱신이 실패할 수 있다.
- project2026 제주 브리핑은 나중에 jeju_model `ai_briefings.db` 를 읽게 할 것(별도 생성 금지).
- 일사 배율 보정은 철회됨 — 다시 제안하지 말 것.
