# smp_step4_report.md — 4단계 SMP 다듬기 결과 보고

> 근거 문서: `smp_step4_instruction.md` / 시도이력: `trial_error.md`
> 재현 스크립트: `smp_phase1_residual.py` (읽기 전용, 원본 DB·A안 코드 미변경)
> 학습창: **2024-06-01 ~ 2026-05-30** (레짐 경계 이후), RT NULL 제외·보간 없음, 17,472행

---

## Phase 1 — 예측 불가 입증 (RT−DA 잔차 분해)

### 1.1 분석 대상
RT 자체가 아니라 **잔차 = `smp_jeju_rt` − `smp_jeju_da`** 를 분석한다.
DA가 가격레벨(trend/seasonal)을 흡수하므로, 잔차를 분해해야 "DA로 설명되는 부분"과
"진짜 시장노이즈"가 갈라진다 — 우리 3층 서사(가격선=DA, 음수만 정량화)와 일치.

- 잔차 통계: mean **−1.74**, std **27.06**, min −285.7, max 249.3
- 분해 그림: `fig_stl_residual.png` (STL, period=24, robust)
  - 구조적 index 결손 24h(0.14%)는 STL 그림 생성용으로만 선형보간, **분산분해 수치에서는 제외**.

### 1.2 분산 분해 표 (잔차 총분산 V = 732.4 = 100%)

| 성분 | 비율 | 의미 |
|---|---:|---|
| (1) 트렌드 | **2.0%** | 저주파 변동 — 거의 없음 |
| (2) 계절(일주기, period=24) | **24.2%** | 한낮 음수 등 시간대 패턴 (설명 가능) |
| (3) 기상으로 설명 | **1.7%** | STL remainder를 A안 기상·물리 연속피처로 회귀, R²=0.023 |
| (4) **순수 노이즈** | **71.9%** | ★어떤 피처로도 설명 안 되는 시장 고유노이즈★ |

- (3)+(4) = remainder 분산비 73.6%. 그 중 기상 회귀가 설명하는 건 2.3%(R²)에 불과.
- **결론: 잔차 분산의 약 72%가 비관측 시장행동에서 오는 예측 불가 노이즈.**

### 1.3 "음수만 설명 가능" 확인 (기상 R² 구간 비교)

| 구간 | n | R²(기상 → 잔차) |
|---|---:|---:|
| 전체 | 17,470 | 0.063 |
| **음수 (rt<0)** | 380 | **0.195** |
| 비음수 (rt≥0) | 17,090 | 0.009 |

→ 기상이 잔차를 설명하는 힘이 **음수구간에서 비음수구간 대비 약 22배**(0.195 vs 0.009).
즉 잔차 중 데이터로 잡히는 신호는 사실상 **"신재생 과잉 → 음수" 구간에만 존재** —
우리가 음수만 정량화하는 설계의 정당성.

### 1.4 요약 (3~5줄, 리포트용)
RT−DA 잔차의 분산은 트렌드 2% · 일주기 계절 24% · 기상설명 2%에 그치고,
**나머지 ~72%가 어떤 피처로도 설명되지 않는 순수 시장노이즈**다.
이는 trial_error의 실측 증거("수요·신재생을 완벽 실측으로 줘도 RT 시간별 MAE 16.3이 천장,
shape 오라클 11.3과의 5점 격차")와 같은 결론을 분산 차원에서 재확인한다.
가격선(level)은 DA가 무적이고, 잔차에서 설명 가능한 신호는 음수구간(기상 R² 0.195)에만
몰려 있으므로 — **RT magnitude 점예측은 포기하고 음수만 확률·강도로 정량화**하는 A안이 정당하다.

---

## Decision Gate (Phase 1 → Phase 2) — 측정값

§3 끝 게이트 기준에 따라 측정만 보고. **P2 방식(A/B) 선택 대기 중.**

### G1. 음수 샘플 수
- **N (rt<0, 학습창) = 380** → 형식 기준 **N ≥ 300 ⇒ P2-A(LGBM quantile) 후보**.
- 단, 아래 G3(깊이 분포 퇴화)을 함께 고려해 최종 선택할 것.

### G2. 지속(run-length) · 심도(neg_num) 분포
- **연속 음수 run-length**: 사건 127건, 평균 **2.99h**, 중앙 3h, max 7h.
  - 분포(시간:건수) — 1h:27, 2h:30, 3h:28, 4h:15, 5h:17, 6h:6, 7h:4.
  - → 음수는 한 번 시작되면 **평균 ~3시간 연속**. 지속은 데이터로 충분히 잡힘.
- **`smp_rt_neg_num`(시간내 15분 음수권 0~4)**: neg_num=4가 **91.3%**(347건), 3이 8.7%(33건).
  - → 음수 시간은 거의 항상 **네 구간 모두 음수권**. 시간내 심도는 사실상 풀(이진에 가까움).

### G3. 음수 깊이(magnitude) 분포 — ★선택에 결정적★
- rt<0 magnitude: mean **−61.4**, P10 −72.3, P50 −70.4, P90 **−24.0**, min −79.3.
- 절반 이상이 행정 바닥(~−70) 근처에 몰려 있고, 상위 10%만 −24~0으로 얕음.
- → 깊이는 **"바닥(~−70) vs 얕은 음수"의 양극** 형태. 연속 magnitude 회귀가 설명할 여지가 좁음
  (trial_error의 "음수 magnitude 회귀 무의미"와 정합).

### 게이트 해석 (사용자 선택 필요)
- **형식 기준**(N=380≥300)만 보면 P2-A(LGBM quantile).
- 그러나 **깊이 분포가 바닥에 퇴화**(P50≈−70, 91%가 neg_num=4)되어,
  적은 양성 샘플로 quantile 3개를 회귀하면 대부분 floor를 예측하는 과적합 위험.
- 정직하고 안정적인 대안 = **P2-B(경험적 조건부 분포)**: (hour×solar) 격자별 깊이 P10/50/90 룩업.
- → **P2-A / P2-B 중 선택을 받은 뒤 Phase 2 착수.**

---

## Phase 2 — 음수 깊이·지속 정량화 (P2-B 경험적 조건부 분포)

**게이트 선택: P2-B.** N=380이 형식상 P2-A를 가리키나, 깊이가 행정바닥(~−70)에 퇴화
(P50≈−70, neg_num 91%가 4)되어 quantile 회귀 시 floor 과적합 위험 → **경험적 분포**가 정직·무과적합.

### 2.1 깊이 룩업표 — (시간대 × solar_util 수준), 음수 발생 조건부 P10/50/90
- 빌더: `training/smp_phase2_depth.py` → `models_weight/smp_depth_lookup.json`
- 격자 근거(데이터): 음수는 9–15시(정오 피크)에 집중, **solar_util↑ → 깊이↓**(corr −0.34;
  deep군 solar 0.71 vs shallow 0.57). solar 임계(터셜) = [0.666, 0.769]. 셀<12 표본은 시간대→전체 폴백.

| 시간대 | solar | n | P10 | P50 | P90 |
|---|---|--:|--:|--:|--:|
| 09–10 | lo | 45 | −73.6 | −69.8 | −22.9 |
| 11–13 | lo | 71 | −75.0 | −69.8 | −17.0 |
| 11–13 | mid | 94 | −72.3 | −71.3 | −67.2 |
| 11–13 | hi | 109 | −72.2 | −71.3 | −58.5 |
| 14–15 | mid | 23 | −72.3 | −69.8 | −25.7 |
| (전체 폴백) | — | 380 | −72.3 | −70.4 | −24.0 |

→ 핵심: 같은 정오대(11–13시)라도 **고일사 P90 −58.5 vs 저일사 P90 −17.0** — solar 수준이 깊이의
상단 꼬리를 가른다. overlay가 "항상 바닥"이 아니라 조건부 정보를 실제로 담음.

### 2.2 지속(duration) — best-effort
- 연속 음수 run-length: 사건 127건, 평균 **2.99h**, P50 3h, P90 5h, max 7h.
- 데모 문구용: "발생 시 예상 지속 ~3h(P90 5h)". (심도 neg_num은 91%가 풀(4)이라 이진에 가까워 부차)

### 2.3 서빙·검증 (DoD)
- 서빙: `smp_depth_pipeline.py` → forecast 테이블에 **`smp_neg_depth_p10/p50/p90`** UPSERT
  (2025-12-13~2026-06-01, 4,104행). 룩업키 = (hour, est_solar_utilization_jeju), 폴백 내장.
- **비음수 가격선 불변 확인**: `est_smp_jeju` = `smp_jeju_da` (max|차이|=0.0), 비음수구간 MAE **8.38**(A안 8.3 유지).
  깊이는 별도 컬럼이라 가격선·경보(P3)에 영향 0.
- **경보 ON 구간 overlay 확인**: smp_danger_jeju=1 인 442시간 전부 깊이값 채워짐(결측 0).
  표시층에서 경보 ON 구간에만 보여줌 → 전 구간 점예측 아님.

### 2.4 최종 출력 형태 (데모/리포트)
> "내일 12–14시 음수 위험 NN%(이진 경보). 발생 시 예상 깊이 P50 −71 / P90 −59원(고일사 기준), 예상 지속 ~3h."

## Phase 3 — A안 이진 경보 (피처 개정, 사용자 지시 2026-06-05)

원래 "변경 없음"이었으나, 사용자 직관 분석으로 분류기 피처를 개정(로직 골격은 동일: DA 가격선 + 이진 음수경보 + 2h 지속).

### 3.1 변경 내용
- **타깃**: `rt < 0` → **`rt < 5`** (5 = 0에 가까운 무가치 SMP ≈ 음수). `NEG_THRESH=5`.
- **제거**: 기상/신재생 4피처(`solar_util`, `wind_util`, `wind_spd_west`, `solar_rad_south`).
  - 근거: 음수 신호가 이미 `net_load`/`smp_jeju_da`에 내포 → 제거 후 ROC-AUC 오히려 ↑.
- **추가**: `est_demand`(+`est_demand_lead_1/2`). train=`real_demand_jeju` / serve=`jeju_est_demand_new`(1단계).
- **검토 후 제거**: `is_midday`/`is_neg_season` 시도했으나 중요도 최하위(hour/month와 중복) → 최종 제외.
- **최종 9피처**: `smp_jeju_da, net_load, nl_lead_1, nl_lead_2, est_demand, est_demand_lead_1, est_demand_lead_2, hour, month`.

### 3.2 성능 (구 vs 신)
| | 구(rt<0, 기상포함 10피처) | 신(rt<5, 9피처) |
|---|---|---|
| ROC-AUC | 0.971 | **VAL 0.976 / TEST 0.973** |
| 운영점 | θ=0.24, 치명recall 0.934 | **θ=0.20**, TEST 치명recall **0.969** / 총recall 0.986 / prec 0.233 |
| 피처중요도 | — | da 113 > net_load계열 165 ≈ est_demand계열 141 > hour 40 / month 37 |

- `est_demand` 추가가 net_load 다음가는 기여(직관 적중). 기상 제거는 성능 손실 없음.

### 3.3 서빙 갱신
- `models_weight/smp_binary.pkl`에 **θ=0.20·neg_thresh=5** 저장. `smp_db_pipeline.py` backfill로 forecast 갱신
  (170일, 경보 **627h**). 가격선 `est_smp_jeju`=DA 불변(max|차이|=0.0), 깊이 overlay 결측 0.
- `train_smp_db.py`/`train_binary_smp.py`만 수정(로더·피처·타깃). `smp_db_pipeline.py`는 FEATURES 동적사용이라 코드 변경 0.

---

## Phase 4 — 의미 강화 레이어: 음수 위험 프로파일 (사용자 지시 2026-06-05)

단순 0/1 경보로는 의사결정 정보가 약함 → 우리가 가진 재료(확률·깊이·지속)를 **한 그림**으로 결합.

### 4.1 이중 운영점 (precision 튜닝)
proba가 0.25–0.26에 몰려 절벽 존재. 보수적 이중 운영점 채택(`THETA={'p25':0.25,'p26':0.26}`):
| θ | TEST 총recall / precision | 성격 |
|---|---|---|
| 0.25 | 0.857 / 0.380 | 경보(균형~precision) |
| 0.26 | 0.490 / 0.673 | 고확신(헛경보 최소, recall 절반) |
- 평가지표는 **recall/precision만**(MAE류 폐기).

### 4.2 확률 보정 (Isotonic) + 위험조정 기대선
- raw proba 미보정(0.26인데 실제 rt<5 빈도 ~0.53). **Isotonic 회귀**로 P_cal 보정
  (`training/smp_calibrate.py` → `models_weight/smp_calibrator.pkl`). 매핑 예: raw 0.268→**P_cal 0.84**.
- **위험조정 기대선**: `E = (1−P_cal)·DA + P_cal·d`, d=E[rt|rt<5]=**−53.2**.
  - RT 점예측 아님 — '위험 무게중심'을 잇는 연속 시나리오선(실제 RT는 이중모드). MAE 평가 안 함.
  - 심각 시간대(DA가 이미 낮음)에서 자연히 **0~−10원**(저DA·고위험일은 −52까지) → "음수 암시" 시각 충족.

### 4.3 시각화 (`smp_risk_profile.py`) — 단순화판(사용자 확정)
요소 5개만(깊이밴드·등급·고확신해치는 버림 — 경보는 구간만):
- **DA 가격선**(메인, solid bold) = est_smp_jeju
- **위험조정 기대선**(서브, dotted) = (1−P_cal)·DA + P_cal·d
- **RT 참고선**(서브2, 실제) = smp_jeju_rt — 경보 적중/헛경보를 한 그림에서 검증(이중모드 확인)
- **주간 경보 zone**(깊이 미정) = θ=0.25 경보 중 주간[8–16]
- **야간 경보 zone**(깊이 미정) = 주간 밖 경보(비물리 시간, 별도색)
- 산출: `fig_risk_profile_<date>.png`. 쇼케이스 2026-03-19(경보구간서 RT 실제 −72 추락 → 경고 적중),
  대비 2026-04-23(RT 98 유지 → 헛경보).

### 4.4 DB 저장 (서빙) — 완료
`smp_softest_pipeline.py` → forecast 테이블에 위험 레이어 UPSERT(2025-12-13~2026-05-31, 4,080행):
| 컬럼 | 의미 |
|---|---|
| `smp_neg_proba_cal_jeju` | 보정 음수확률 P_cal (0~1) — "음수 위험 N%" 근거 |
| `smp_rt_soft_est` | 위험조정 기대선 = (1−P_cal)·DA + P_cal·d (RT 점예측 아님) |
| `smp_danger_day_jeju` | 주간 경보구간 0/1 (θ=0.25+2h, 주간[8–16]) |
| `smp_danger_night_jeju` | 야간 경보구간 0/1 (주간 밖) |
- backfill: 주간경보 334h / 야간경보 2h. 가격선(est_smp_jeju)·RT(smp_jeju_rt)는 기존 컬럼 그대로.
- 쇼케이스(그림)는 이 컬럼들 + est_smp_jeju + smp_jeju_rt 만으로 DB에서 재현 가능.

---

## 부록. 발견·수정된 이슈 (P3 자산 관련)
- `train_smp_db.load_historical()`이 구 컬럼 **`smp_rt_neg_flag`**(현재 `smp_rt_neg_num`로 교체됨)를
  SELECT해 쿼리 단계 에러 → `train_binary_smp.py`(A안 학습)도 실행 불가였음.
  - **수정 완료(사용자 승인)**: SELECT에서 `smp_rt_neg_flag` 제거(미사용 컬럼). 로더 정상 복구(19,776행 로드 확인).
    A안 로직(타깃 rt<0, 피처)은 불변 — 깨진 한 줄만 제거.
  - `no use/`의 구 코드들이 `smp_rt_neg_flag`를 참조하나 비활성이라 무관.

## 데이터 딕셔너리 (Phase 1 사용)
| 컬럼 | 출처 | 용도 |
|---|---|---|
| `smp_jeju_rt` | historical | 타깃(RT SMP 시간평균) |
| `smp_jeju_da` | historical | 앵커(DA SMP), 잔차 기준선 |
| `smp_rt_neg_num` | historical | 시간내 15분 음수권 개수 0~4 (심도) |
| `real_demand_jeju` − `real_renew_gen_jeju` | historical | net_load |
| `real_solar/wind_utilization_jeju` | historical | solar_util / wind_util |
| `wind_spd_west`, `solar_rad_south` | historical | 기상 피처 |
