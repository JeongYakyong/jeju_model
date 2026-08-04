# 3단계 비교 — 3cmp-0 EDA 보고서 (G-9 게이트)

> 목적: PatchTST vs LGBM 비교(흐린날 과대예측 중점, D+1~D+6) 착수 전 관계·시계열·안정성 점검.
> 데이터: `training/solarwind_raw_jeju.csv` (실측 기상 + 실측 이용률, 2020-01 ~ 2026-05, 56,256행).
> 학습창: train ≤2024(43,848) / val 2025(8,760) / test 2026 1~5월(3,648).
> 산출: `fig/3cmp-0_*.png` (5장), `tab/3cmp-0_*.csv` (4종).

## 1. 타깃(이용률) 시계열 구조
- **일중 주기**가 지배적(solar는 정오 피크, 야간 0). 월별로 봄·가을 solar 이용률이 높고 여름 장마·겨울 저각이 낮음.
- 타깃을 **이용률(0~1)로 정규화**하는 설계가 타당함을 재확인(아래 2).

## 2. 용량 표류 vs 이용률 안정성 (covariate shift)
- **설비용량은 크게 표류**: solar 254→405MW, wind 258→364MW (2020→2026).
- 그러나 **이용률은 연도별로 안정**: solar 낮시간(9-16h) 0.37~0.47, wind 0.22~0.28 범위에서 추세 표류 없음.
- → 절대 발전량(MW)이 아니라 **이용률을 타깃**으로 두면 연도 간 함수 표류가 작다. LGBM·PatchTST 모두 이용률 학습이 맞음. (`tab/3cmp-0_capacity_util_drift.csv`, `fig/3cmp-0_capacity_drift.png`)

## 3. 기상 ↔ 이용률 관계 (`fig/3cmp-0_weather_scatter.png`, `tab/3cmp-0_weather_corr.csv`)
**Solar (낮 8-17h 상관)**
| 피처 | 상관 | 피처 | 상관 |
|---|---|---|---|
| solar_rad_west | **+0.878** | total_cloud_west | **−0.544** |
| solar_rad_south | +0.832 | total_cloud_south | −0.513 |
| | | midlow_cloud_south | −0.494 |
| rainfall_west | −0.129 | midlow_cloud_west | −0.485 |

- **일사(solar_rad)가 1순위 구동변수**, 구름이 2순위(음). 강수는 약함(damping으로 흡수 가능).

**Wind (전시간 상관)**
| 피처 | 상관 |
|---|---|
| wind_spd_west | **+0.752** |
| wind_spd_east | +0.517 |
| wind_spd_south | +0.249 (약함) |

- west가 지배, south는 약함 → 현 PatchTST가 wind에서 south를 뺀 설계와 일치.

## 4. 흐린날 과대예측의 표적 구간 (`fig/3cmp-0_regime_diurnal.png`, `tab/3cmp-0_regime_util.csv`)
낮시간 평균 총운량으로 일(day) 단위 분류(≤0.3 맑음 / ≥0.7 흐림):

| regime | solar 이용률 평균 | 표본(시간) | 일수 |
|---|---|---|---|
| sunny | 0.557 | 5,360 | 536일 |
| mixed | 0.453 | 6,990 | — |
| **cloudy** | **0.229** | 11,086 | **1,109일** |

- **흐린날이 맑은날보다 2배 이상 흔함**(1,109 vs 536일). "맑은날이 대부분이라 흐린날을 맑게 예측"이라기보다, **흐린날 이용률(정오 ~0.33)이 맑은날(~0.80)의 절반 이하로 떨어지는 큰 비선형**을 모델이 충분히 끌어내리지 못하는 것이 과대예측의 실체.
- → 비교의 핵심 지표 = **흐린날 부분집합에서의 bias(예측−실측)**. 이 구간을 EDA에서 명시적 평가 대상으로 고정.

## 5. train/test 분포 겹침 (`fig/3cmp-0_split_overlap.png`, `tab/3cmp-0_split_summary.csv`)
- solar_util 평균 train 0.361 / val 0.403 / test 0.409, solar_rad p90 train 2.77 / test 2.98 — **겹침 양호**(test가 근소하게 맑은 쪽, 외삽 위험 낮음).
- wind_util test 평균 0.278로 train 0.244보다 약간 높으나 분포 겹침 정상.

## 6. 자기상관 — 장지평 lag 유효성 (재귀 롤링의 한계 근거)
이용률 자기상관 lag[1, 24, 48, 72, 168]:
- **solar**: 0.94, 0.81, 0.76, 0.75, 0.73 — 일중주기로 지속성 높음(같은 시각 반복).
- **wind**: 0.96, **0.31, 0.11, 0.09, 0.10** — 24h 넘으면 지속성 급붕괴.

→ **wind은 하루 너머의 과거 이용률(past_y)이 거의 무정보**. PatchTST 재귀 롤링이 wind 장지평에서 빠르게 신호를 잃는 구조적 이유. 반대로 **forecast 기상에만 의존하는 LGBM-direct가 장지평에서 유리할 여지**(검증 대상 가설).

## 7. G-9 판정
- 관계 강함(solar_rad +0.88 / cloud −0.54 / wind_spd +0.75), 연도 안정(이용률 정규화), train↔test 겹침 양호 → **G-9 통과**.
- 모델링 표적 확정: ① 흐린날 부분집합 bias ② 장지평(24~144h) 성능 ③ 이용률·net_load 두 레벨.
- **다음 게이트: 최종 피처 선택(§0.6) — 사용자 확정 후 모델 착수.**
