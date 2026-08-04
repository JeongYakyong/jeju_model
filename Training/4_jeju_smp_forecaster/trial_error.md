# 4단계 SMP — 시도·결과 압축 로그 (trial_error)

> **용도**: 4단계(SMP)와 4-1(D-2 연장)에서 우리가 시도한 것과 그 결과만 압축. 헤맨 과정은 트림.
> 4단계를 **심플하게 재시작**할 때 "이미 된 것 / 하지 말 것"을 빠르게 참고하기 위한 단일 문서.
> 메모리(smp-model-3class·smp-d2-forecaster·rt-smp-collection)와 PROJECT.md §7의 4단계 로그를 여기로 이관(2026-06-03).
> ※ 4단계가 만족스러운 성능으로 마무리되면 핵심만 PROJECT.md로 옮긴다. 그 전엔 PROJECT.md에 4단계 기록 없음.

---

## 0. 데이터 자산 — **이미 됨, 재수집/재작업 금지**

재시작해도 그대로 쓰는 것들. (이건 "시도"가 아니라 인프라)

- **타깃 데이터** (`input_data_jeju.db` historical):
  - `smp_jeju_rt` = 시간평균 RT SMP (제주 실시간시장). 19,680행, 2024-03-01~2026-05-30.
  - `smp_rt_g1`~`smp_rt_g4` = 15분 4구간 원시 RT SMP (원시 보관, 파생은 사용 시점 계산). 19,656행, ~2026-05-28.
  - `smp_rt_neg_num` = `count(g* < 5)` = 음수권(바닥 0~5 포함) 구간 개수 0~4.
- **입력/앵커 데이터**:
  - `smp_jeju_da` (DA SMP, **2020~** 전 구간), `jeju_est_demand_da` (수요예측 DA, 2020~).
  - `est_net_load_jeju` (3단계 출력, forecast 테이블), `est_solar/wind_utilization_jeju`, `est_solar/wind_gen_jeju`.
- **수집 코드** (1단계 `1. data_fetcher_and_db/core/`):
  - `api_fetchers_jeju.fetch_kpx_jeju_rt_smp(start,end)` + `_fetch_jeju_rt_smp_one_day` (API `getJejuSmpLfd2`, data.go.kr B552115, `KPX_API_KEY`). hour×gugan 피벗. `collect_data_jeju.build_historical` [H4/4] 연결.
  - DB 백업: `data/input_data_jeju.db.bak_20260603`.

---

## 1. RT SMP 특성 — 모델 설계 전제 (데이터 실측 기반)

재시작 시 반드시 숙지. 전부 DB 직접 확인.

- **① 가용성 = RT는 타깃 전용, lag 피처 금지.** `getJejuSmpLfd2`는 매일 23:00 발행이나 KPX 불안정으로 지연 가변(최대 익일 18:00). day-ahead 예측 시점에 직전 RT가 미발행일 수 있음 → RT의 lag/이동평균을 입력피처로 쓰면 누수+결측. **입력 anchor = `smp_jeju_da`.** RT NULL일 때만 서빙/학습 레이어에서 da로 대체(저장은 순수 유지).
- **② 레짐 경계 2024-06-01(시범사업).** RT는 2024-03-01부터만 존재(학습창 하한). 2024-03~05는 다른 레짐(corr(rt,da)≈0.28 vs 이후 0.6~0.85) → 학습창은 **2024-06~** 사용.
- **③ 음수 = "크기"보다 "발생".** floor ≈ −70~−79 (시변, min −79.32) — 행정적으로 막혀 깊은 음수 magnitude 회귀는 의미 약함 → **분류(발생 여부)가 핵심.** 음수띠: 봄 한낮 9–14h(peak 12h 20%), 여름 ≈ 0. 강한 계절성.
- **④ 거동.** mean 113 / std 41 / max 316. plateau 20%(직전시간 동일값). da는 noisy anchor(corr 0.59, rt−da std 33.7).

---

## 2. 4단계 (SMP day-ahead) — 시도→결과

| 시도 | 결과 |
|---|---|
| RT 가격 직접 회귀 / hurdle(음수로 점예측선 덮어쓰기) | **실패.** 헛경보마다 선이 −70으로 튀어 MAE 16~18 폭발. |
| 3-class hurdle | 폐기(중간단계). 자동튜너가 과도기 클래스를 스스로 버림. |
| **A안 = DA 가격선 + 이진 음수경보 오버레이** ✅채택 | 가격선 = `smp_jeju_da` 그대로(비음수는 RT≈DA, MAE 8.3). 그 위에 이진 분류기가 "음수 위험"만 경보. **ROC-AUC 0.971, 치명 recall 0.934, 가격선 MAE 8.3.** θ=0.24 + 연속 2h 지속규칙. |

- **핵심 통찰**: 가격은 DA 무적(점예측 덮어쓰면 MAE 폭발). 음수는 magnitude 회귀 무의미 → **발생 분류**가 정답. 오버레이라 경보를 켜도 가격선 MAE 불변 = MAE↔recall 트레이드오프 소멸.
- **치명(fatal) 정의**: DA≥0 인데 실제 RT<0 (발전사업자가 돈 받을 줄 알았다 무는 경우). 음수 137건 중 136건이 치명 → 총recall≈치명recall.
- **피처(10)**: smp_jeju_da, net_load(=real_demand−real_renew), nl_lead_1/2, solar_util, wind_util, wind_spd_west, solar_rad_south, hour, month.
- **산출물(현행, 유지)**: `train_binary_smp.py`(학습→`models_db/smp_binary.pkl`), `smp_db_pipeline.py`(서빙: `est_smp_jeju`·`smp_neg_proba_jeju`·`smp_danger_jeju` UPSERT), `train_smp_db.py`(공통 로더), `6. report only/compare_smp.py`(보고서).
- 비교: 구 model9(구DB·BANK) recall 0.937과 사실상 동급인데 구조 단순·가격선 깨끗. BANK 노이즈주입은 과적합 우려로 **거부**.

---

## 3. 4-1 (D-2 / 뒤24h 연장) — 시도→결과 (**종결, 死路 다수**)

D-2: 매일 23:00 예측, 대상 48h. 앞24h(D+1)는 DA 발표됨, 뒤24h(D+2)는 DA 없음.

| 시도 | 결과 |
|---|---|
| 앞24h 회귀로 가격선 만들기 (6번 실험) | **DA 못 이김.** 앞24h는 `smp_jeju_da` 그대로 쓰는 게 정답. |
| 뒤24h: 앵커(전일DA `smp_da_lag24`) + 잔차회귀 (RT 타깃) | baseline 전일DA persist 18.68 → **new 16.94** (이김, 진짜 피처가 만든 것: corr(예측잔차,실제잔차) 0.57). |
| **가중평균/일별레벨 앵커 강화 (stage0)** | **死路.** 가중평균 ≈ 단순 일평균(수집 불필요). 레벨 완벽 오라클도 MAE 개선 0(19.44). 병목은 레벨 아니라 **시간형태(shape)**. |
| shape 보강 (solar_util·solar_rad·is_midday 추가, 앵커구조 A/B/C 변경) | **벽.** 추가 피처는 서빙 과적합(16.94→17.09). 앵커구조 바꿔도 16.9~18.4. 입력을 완벽 실측으로 줘도 16.33(−0.6뿐). shape 오라클 11.34와의 5점 격차 = **net_load·solar·기상 무엇으로도 설명 안 되는 RT 시장 고유노이즈(예측 불가).** |
| **DA 타깃 실험** (target = `smp_jeju_da`, 뒤24h D+2 DA 예측) 🟢유망 | DA는 완만(floor 0). baseline 전일DA 14.32 → **new 12.94**(학습창 2020+, 6년). 서빙(예보입력) 13.34. RT(16.9)보다 훨씬 나음. |

- **결론**: RT 시간별 MAE는 **물리 바닥 도달**(16.9). 더 깎기 = 비생산적. RT 추종은 시장노이즈라 한계. 현실적 방향 = **DA 타깃 회귀** 또는 **4단계식 음수경보**(신호 존재).
- **Transformer/PatchTST 제외**: 실질 학습창 24개월·계절 2주기뿐 → 굶주림/과적합. LGBM이 정답.

---

## 4. 재시작 메모 (심플 4단계)

- 검증된 정답 = **A안(DA 가격선 + 음수경보)**. 여기서 출발하면 됨.
- 절대 다시 하지 말 것: RT 가격 직접/hurdle 덮어쓰기, 가중평균/레벨앵커 stage0, 무리한 shape 피처, Transformer, RT lag 피처.
- 데이터는 §0 그대로 있음 — 수집부터 다시 하지 말 것.
- 코드: 4단계 A안(`train_binary_smp.py`·`smp_db_pipeline.py`·`train_smp_db.py`) 유지. 4-1 `v2/`는 삭제됨.
