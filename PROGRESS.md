# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-10-09

## 완료

- 태양광: 일사·운량을 JMA 원본으로 학습·서빙, Year 없는 모델 서버 반영(10-09). test MAE 0.088→0.081.
- 풍력(로컬만 변경, 서버 미반영): 2020~ 학습 + 출력제어 의심 시각 제외(`data/refdata/curtailment_excluded_hours.csv`), 입력 QM 시각대별, tcog 가산 제거. 새 모델 `lgbm_wind_util.txt`(트리 243), `wind_qm.json`(5개 시각대).
- 서빙 검증: `serve_chain.py --utc 12 --no-write` 정상(120행).

## 현재 상태

| | |
|---|---|
| 서버 | 태양광 새 모델 / 풍력은 옛 모델·옛 QM·tcog 가산 그대로 |
| 로컬 | 풍력 새 구성(git 미커밋). 옛 모델·QM·tcog 백업: `nouse/wind_backup_20261009/` |

## 다음 할 일

1. 풍력 변경 커밋·push 후 서버 `git pull` (사용자가 Desktop 에서). 모델·wind_qm.json·serve_solarwind.py·CSV 가 한 세트.
2. 서버 반영 후 며칠간 풍력 정오·강풍(west 13m/s↑) 편향 실측 확인. 강풍은 예보가 못 잡아 과소(≈-0.12)가 남는다.
3. 14:30 repair, 00:30 정식 실행 로그에서 LGBM 폴백 없는지 확인.

## 주의사항

- 풍력 평가는 서빙 입력(예보 풍속)으로 할 것. 실측 풍속 입력 수치는 낙관적.
- `wind_zone_east` 피처는 east 풍속이 15를 넘는 적이 거의 없어 사실상 항상 0(죽은 피처). 손대지 않음.
- 풍력 80m 풍속·kma_historical 사용은 사용자가 보류/제외. 태양광 tcog 가산은 그대로 둠.
- 일사 배율 보정은 철회됨 — 다시 제안하지 말 것. 서빙 일사는 JMA 유지.
