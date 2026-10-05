# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-10-05

## 다음: **서버 적용 (D단계, 사용자 단계별 확인)**

로컬 적용·백필 완료. 남은 것: 커밋 → (사용자 push) → 서버 DB 백업 → `git pull` →
패키지 scp → `deploy_jma_merge.py merge` → 검증 → Model_api_added 동기화 시각 조정 검토.

## 현재 상태

| | |
|---|---|
| 로컬 서빙 모델 | 새 태양광 PatchTST(MSE 손실, 미래 운량=JMA, 피처 6개). D+1 가중치를 D2·D3 재사용, D+4·5 LGBM |
| 로컬 DB | 서버 10-05 사본 + `forecast_jma`(144 실행) + 백필(12z 143·18z 40 base, SMP 143 base) |
| 배포 패키지 | `data/jma_deploy_20261005.db`(5.8MB) — 서버 사본 병합 시험 결과 로컬과 0행 차이 |
| 서버 | 아직 옛 모델·옛 코드. `forecast_jma` 없음 |
| 백업 | 로컬 DB·옛 가중치 = git + `nouse/` (db_backups, training_outputs/model_backup_20261005_before_jma) |

## 주의사항

- ★KPX `*_da`·실시간 SMP API 가 10-02 이후 0건(응답은 OK) — 10-01 이후 base SMP 산출 불가.
  `smp_backfill`(05:00) 이 5일 안에 재개되면 자동 복구, 더 길면 수동 백필.
- ★서버 SMP 는 09-17 이후 멈춰 있었음(00:20 D+2 부족 → 05:00 백필이 SMP 미재생성). `backfill5` 에 `smp_backfill` 추가로 해결(배포 필요).
- Model_api_added `sync_forecast` 가 00:40 — JMA 대기로 체인이 늦으면 전날 예측을 가져갈 수 있음.
- 남은 맑은날 과소(JMA 도 맑다고 본 시각 −0.07)는 운량 외 원인 — 추적 보류.
- `solar_scale` 은 옛 모델 기준 값 그대로. 운영 자료가 쌓이면 `fit_solar_scale.py --check`.
- 2026-05-12 이전 예측 기록은 옛 모델 값(JMA 자료 없음).
