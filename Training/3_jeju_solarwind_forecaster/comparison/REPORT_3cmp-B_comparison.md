# 3단계 비교 — PatchTST vs LGBM 종합 (3cmp-B 실측기상 / 3cmp-C forecast)

> 목적: 3단계 net_load 예측기(PatchTST 기반)의 성능·장지평 가능성 점검 + 흐린날 과대예측 진단.
> 설계(사용자 확정 2026-06-08): 평가 = 이용률 + net_load 둘 다 / 기상 = 실측 우선·forecast 보조 /
> PatchTST 장지평 = 재귀 롤링 / LGBM = 순수기상 horizon-무관 단일모델.
> 피처(§0.6 확정): SOLAR = PatchTST피처 + clearsky_ratio + month, WIND = PatchTST 동일.
> 평가창 = test 2026. 산출 `model/3cmp-A,B,C*.py`, `tab/3cmp-*`, `fig/3cmp-*`.

## 한 줄 결론
**D+1에서 두 모델은 사실상 동급이다. 그러나 ① PatchTST는 재귀 롤링이라 지평이 늘수록 열화(이용률 bias·MAE 악화)하는 반면 LGBM은 24~144h 평평하고, ② 실서빙(forecast 기상)에서는 LGBM이 net_load·wind에서 더 견고하다. 그리고 ③ 사용자가 본 "흐린날 과대예측"은 모델 탓이 아니라 forecast 기상 오차 탓이며, LGBM으로 바꿔도 해소되지 않는다(오히려 근소하게 더 심함).** → **장지평(D+1~D+6)·실서빙 견고성 측면에서 LGBM-direct가 더 나은 후보**, PatchTST는 D+1 단기에서 강점.

---

## A. 실측기상(perfect) — 모델 자체 성능 (3cmp-B, 발행일 145일)
실측 기상을 양쪽에 동일 투입 → 예보오차를 제거하고 **알고리즘 자체**를 본다.

### A-1. solar 이용률 (낮 8-17h), 지평별
| D+h | PatchTST MAE | PatchTST bias | LGBM MAE | LGBM bias |
|---|---|---|---|---|
| 1 | 0.0664 | −0.0162 | 0.0667 | +0.0031 |
| 3 | 0.0666 | −0.0179 | 0.0659 | +0.0023 |
| 6 | 0.0684 | **−0.0211** | 0.0662 | +0.0013 |

- MAE는 사실상 동률(~0.066). **PatchTST는 음(−)의 bias가 지평과 함께 커진다**(재귀 롤링 누적). LGBM은 거의 무편향·평평.

### A-2. ★ 흐린날/맑은날 × 지평 (핵심 진단)
| regime | D+h | PatchTST bias | LGBM bias |
|---|---|---|---|
| sunny | 1 | −0.029 | +0.002 |
| sunny | 6 | **−0.038** | −0.000 |
| cloudy | 1 | −0.004 | +0.009 |
| cloudy | 6 | −0.007 | +0.007 |

- **실측기상에서는 흐린날 과대예측이 없다**(PatchTST 오히려 −, LGBM +0.007~0.009로 미미). PatchTST의 실제 약점은 **맑은날 과소예측**(−0.03~−0.04, 지평 따라 악화).

### A-3. wind 이용률 / net_load
- wind: PatchTST MAE 0.0865→0.092(D+1→D+2 재귀 열화 후 정체), bias +0.013~0.017. LGBM MAE 0.086→0.084(평평), bias −0.016.
- net_load(demand 고정, 평균 526MW): PatchTST nMAE **6.60%→6.91%**(MAE 34.7→36.3), LGBM **7.17%→7.06%**(평평). → **실측기상 D+1은 PatchTST가 근소 우위, D+6에서 격차 0.15%p로 수렴.**

`fig/3cmp-B_horizon_compare.png`

---

## B. forecast 기상(실서빙 D+1) — 사용자 관찰의 무대 (3cmp-C, 171일)
forecast 기상 가용구간(2025-12-13~2026-06-01)만, lead=D+1(다지평 아카이브 없음). regime은 ★실측 구름으로 분류.

### B-1. ★ solar 이용률 regime별 — 흐린날 과대예측 재현
| regime | PatchTST bias | LGBM bias | (ablation) bias |
|---|---|---|---|
| sunny | −0.065 | −0.034 | −0.048 |
| **cloudy** | **+0.054** | **+0.078** | +0.057 |
| ALL | −0.005 | +0.020 | +0.001 |

- **흐린날 과대예측이 두 모델 모두에서 나타난다.** 실측기상(A-2)에선 없던 것이 forecast에선 +0.05~0.08로 커짐 → **원인은 모델이 아니라 forecast 기상 오차**(예보가 구름을 못 맞혀 맑게 들어옴; 2단계서 확인한 구름 예보 event-skill 한계와 동일).
- LGBM이 PatchTST보다 흐린날 과대가 더 큼(+0.078 vs +0.054). 단 ablation(PatchTST 동일피처)은 +0.057로 PatchTST와 거의 동일 → **추가한 clearsky_ratio가 틀린 forecast 일사를 증폭**한 부분(피처 trade-off, 정직 보고). 즉 **LGBM 전환은 흐린날 문제를 풀지 못한다.**

### B-2. wind / net_load (forecast D+1)
| 모델 | wind bias | wind MAE | net_load MAE | net_load nMAE% |
|---|---|---|---|---|
| PatchTST | +0.122 | 0.151 | 59.9 | **11.33** |
| LGBM | +0.071 | 0.132 | 56.1 | **10.62** |

- forecast 풍속이 고편향(현 설계가 south를 뺀 이유와 같은 결) → **PatchTST가 이를 크게 증폭**(wind bias +0.12). LGBM은 절반 수준.
- **net_load 실서빙은 LGBM 우위(10.62% vs 11.33%)** — 실측기상(A)과 정반대. LGBM이 forecast 오차에 더 견고.

`fig/3cmp-C_forecast_regime.png`

---

## C. 종합 해석
| 관점 | PatchTST | LGBM-direct |
|---|---|---|
| D+1 실측기상 net_load | **6.6%**(우위) | 7.2% |
| D+6 실측기상 net_load | 6.9% | 7.1%(수렴) |
| 지평 안정성(24~144h) | 재귀 롤링 → 열화 | **평평**(horizon-무관) |
| 실서빙(forecast) net_load | 11.3% | **10.6%**(우위) |
| 실서빙 wind 견고성 | 예보편차 증폭(+0.12) | **+0.07**(절반) |
| 흐린날 과대예측 | forecast 탓(공통) | forecast 탓(공통, 근소 더 심) |

- **흐린날 과대예측(점 5·6)**: 모델 선택 문제가 아니라 **forecast 기상 품질 문제**. 두 모델 모두 발생, LGBM도 해소 못 함. 해결 경로는 모델 교체가 아니라 ① 구름·일사 예보 보정(QM/event-skill) ② 비대칭 손실·흐린날 특화(2단계 2-0c와 동형) ③ 후처리 음의 보정.
- **장지평 가능성(점 7)**: LGBM-direct는 D+1~D+6에서 품질이 평평하고 forecast에 견고 → **장지평 확장의 더 나은 골격**. PatchTST는 D+1 단기 정밀도가 강점이나 재귀로 길어질수록 불리.

## D. 권고(미결정 — 사용자 판단 사항)
1. **현 PatchTST 서빙은 D+1 용도로 유지**(단기 강점). 
2. **장지평(D+2~D+6)용으로 LGBM-direct 추가**를 권고(2·5단계와 동일한 "PatchTST D+1 + LGBM 다지평" 이원 구조). 
3. 흐린날 과대예측은 **forecast 보정 과제로 별도 분리**(모델 교체로 풀리지 않음).
→ 6단계(land_net_load) 골격은 이 결론을 그대로 이식: **전국은 LGBM-direct 단일로 시작**(land 5·7과 일관, PatchTST 불필요 가능성 높음).
