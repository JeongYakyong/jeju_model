# 2-0 EDA 요약 — 제주 수요 예측(장지평) (G-9)

## 데이터
- 타깃 real_demand_jeju: 2020-01 ~ 2026-06-05, 56,352행.
  NaN=10(시간보간), 0값=5.
- 베이스라인 jeju_est_demand_da(KPX 하루전): 연도별 base MAPE 표 참조(tab/2-0_year_summary.csv).
- 학습창: train ≤2025-02 / val ~2026-03 / test ~2026-05 (기존 제주 2단계와 동일).

## 시계열 구조
- 일/주/계절 주기 뚜렷(그림 2-0_seasonality). 자기상관 lag24=+0.894, lag168=+0.822.
- 기온↔수요 비선형(U/V자, 그림 2-0_temp_demand). 3지점 평균기온↔수요 상관 -0.004.
- rec24↔수요 +0.692, rec168↔수요 +0.648 → 레벨 신호 유효.

## 서빙 제약 (예보에 있는 변수만)
- 서빙 가능(O): temp_c, humidity, solar_rad, wind_spd (제주 3지점 평균/일사 2지점).
  → land 와 달리 **습도 서빙 가능** → 사용자 확정대로 기상 4종 유지.

## train↔test 분포
- 2026 test 가 train 분포 안에 겹침(표 참조) → 정직한 검증 가능.

## 확정 피처(15개, 사용자 확정 — Decision Gate)
- h, lag168, rec24, rec168, 기상4(temp_c/humidity/solar_rad/wind_spd),
  hour_sin/cos, dow_sin/cos, month_sin/cos, day_type.
- 구조: LGBM 단독·풀드 직접 다지평 1~168h(D+1~D+7), origin 23:00. (lag24·과거 land의 h 외 제거 결정 반영: lag24 제외, h 유지)
- 기존 PatchTST+LGBM D+1 파이프라인은 그대로 유지. D+1(24h)에서 정확도 비교 예정.

→ 결론: 주간 lag·기상·달력 가정 모두 데이터에 근거 있음. **G-9 통과**, 2-A 학습 진행.
