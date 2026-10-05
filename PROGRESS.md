# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-10-05

## 다음: **운영 관찰 (서버 적용 완료 2026-10-05 13:50)**

1. 오늘 밤 00:20 로그에서 `jma` 단계 대기 시간·실행(12 UTC 이어야 함) 확인, 01:10/01:40 동기화 반영 확인.
2. 며칠 쌓이면 새 모델 실운영 성능 확인(맑은날·흐린날), 필요 시 `fit_solar_scale.py --check`.
3. `historical` 단계가 KPX 0행이어도 "성공"으로 끝나는 문제 — 경고 처리 검토(SMP 는 우선순위 낮음).

## 현재 상태

| | |
|---|---|
| 서버 | d2db12f 배포. 새 태양광 모델(MSE·JMA 운량·피처 6개), D+1~3 PatchTST / D+4·5 LGBM |
| 서버 DB | 로컬 백필 병합(12z 143·18z 40 base, SMP 143 base, forecast_jma 144 실행) — 로컬과 합계 일치 |
| 서버 백업 | `data/input_data_jeju.db.bak_20261005_before_jma`, `~/crontab_backup_20261005.txt` |
| 로컬 백업 | git + `nouse/` (db_backups, training_outputs/model_backup_20261005_before_jma) |

## 주의사항

- SMP 는 사용자가 중요하지 않다고 함(2026-10-05) — 수요·태양광·풍력 서빙만 챙긴다.
- ★KPX `*_da`·실시간 SMP API 가 10-02 이후 0건(응답은 OK) — 10-01 이후 base SMP 산출 불가.
  `smp_backfill`(05:00) 이 5일 안에 재개되면 자동 복구, 더 길면 수동 백필.
- ★서버 SMP 는 09-17 이후 멈춰 있었음(00:20 D+2 부족 → 05:00 백필이 SMP 미재생성). `backfill5` 에 `smp_backfill` 추가로 해결(배포 필요).
- Model_api_added `sync_forecast` 에 01:10·01:40 추가(서버 crontab, 00:40·08:20 유지) — JMA 대기로 체인이 늦어도 받게. 원본 `~/crontab_backup_20261005.txt`.
- 남은 맑은날 과소(JMA 도 맑다고 본 시각 −0.07)는 운량 외 원인 — 추적 보류.
- `solar_scale` 은 옛 모델 기준 값 그대로. 운영 자료가 쌓이면 `fit_solar_scale.py --check`.
- 2026-05-12 이전 예측 기록은 옛 모델 값(JMA 자료 없음).
