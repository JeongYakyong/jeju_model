# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-10-06

## 완료

- 12z 갱신 3방식(정식 / 06 UTC 가교 D+1 / 08:00 마감 뒤 KMA+LGBM), `est_horizon_jeju.solar_model` 기록 (0e2c202, fbff733).
- cron 재구성: 00:30 정규, 매시 :30 `repair`, 08:00 18z, 23:00 `jma_bridge`, 전부 flock. 05:00 `backfill5` 폐지.
- 서버 배포(2026-10-06 20:35), 첫 repair 성공 — 12z 아카이브 최근 2 base 100% 복구.

## 다음: 실운영 확인

1. 23:00 `jma_bridge` 가 그날 06 UTC 를 받는지, 00:30 체인 로그의 `갱신 방식`이 bridge → (12 UTC 도착 후 repair) fresh 로 바뀌는지.
2. Model_api_added 캡션에 "태양광 모델: … PatchTST(임시) …" 가 떴다 사라지는지 화면 확인.
3. 며칠 뒤 새 태양광 모델 실성능(맑은날·흐린날), 필요 시 `fit_solar_scale.py --check`.

## 현재 상태

| | |
|---|---|
| 서버 | fbff733. D+1~3 PatchTST / D+4·5 LGBM |
| 서버 백업 | `~/crontab_backup_20261006_2017.txt`, `data/input_data_jeju.db.bak_20261005_before_jma` |

## 주의사항

- KMA 06z 는 KIMG 87h·KIMR 72h 라 D+4·5 불가 — 12z 대체로 쓰지 않음. 00:20 시점 KMA 12z D+1 은 최근 20일 매일 있었음.
- 12z 아카이브 과거분(forecast_kimr 09-24~, forecast_kimg 부분 결손)은 복구 안 함.
- SMP 는 중요하지 않음(사용자). backfill 은 "성공"이지만 피처 결측으로 0행, KPX `*_da`·RT SMP API 10-02 이후 0건.
- `historical` 이 KPX 0행이어도 "성공"으로 끝나는 문제 — 경고 처리 검토 보류.
- 18z 체인 로그의 `D+n` 은 모델 번호(horizon_d+1)로 찍힘 — 동작 영향 없음.
