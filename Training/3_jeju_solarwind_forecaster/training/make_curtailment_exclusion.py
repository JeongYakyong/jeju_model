# -*- coding: utf-8 -*-
"""풍력 학습에서 뺄 '출력제어(curtailment) 의심 시각' 목록을 만든다 — 일회성 스크립트.

★ 평소 재학습 때는 이 스크립트를 돌리지 않는다. 결과물(data/refdata/curtailment_excluded_hours.csv)을
  train_wind_lgbm.py 가 읽어서 해당 시각만 빼고 학습한다. 이 파일은 "그 CSV 를 어떻게 만들었는지"
  의 기록이다. 옛 구간(2025-01 이전) 자료는 바뀌지 않으므로 CSV 도 다시 만들 필요가 없다.

왜 필요한가
  2020 ~ 2024-06 에는 출력제어가 잦았다(2023 약 150건 → 2025 0건, 특히 봄·가을 한낮). 옛 구간을
  그대로 학습하면 모델이 "한낮에는 같은 풍속인데 발전이 낮다"를 배워 정책 변경 이후에도 한낮을
  낮게 예측한다(2026-10-05 정오 사례). 출력제어 시각에 직접 꼬리표가 없어서 아래 방식으로 추정한다.

판별 방식
  1) 정책 변경 이후(2025-01-01 ~ 2025-09-19) 자료만으로 기준 LGBM 을 학습한다 — 출력제어가 거의
     없는 세상에서 "이 풍속·시각이면 이 정도 나온다"를 배운 모델이다. 입력 피처·파라미터는 본 모델과 같다.
  2) 이 기준 모델로 옛 구간(2025-01 이전)을 예측하고, 아래 두 조건을 모두 만족하면 출력제어로 본다.
       · 실제 이용률이 기준 예측보다 RESIDUAL_THRESHOLD(0.15) 이상 낮다
       · 기준 예측이 MIN_REFERENCE_PREDICTION(0.2) 을 넘는다 (원래 발전이 많았어야 하는 시각만 본다)
  3) 서부 풍속 CUT_OUT_PROTECT_WIND(20 m/s) 이상은 판별에서 제외(보호)한다. 태풍 때 터빈이 안전 정지한
     "강풍 정지(cut-out)"는 출력제어가 아니라 풍력의 물리적 특성이라 모델이 배워야 한다.
  ※ 2025-01 이후 구간은 판별하지 않는다 (출력제어가 거의 없다고 보고 전부 학습에 쓴다).

임계값을 정한 근거 (2026-10-09 비교, 서빙 입력인 예보 풍속 기준 평가)
  · -0.10 : 12% 를 지워 오히려 과대예측(+0.008)으로 치우쳐 기각
  · -0.15 : 옛 구간의 약 5% 제거. 한낮(10~15시) 편향 -0.018 → -0.000, 강풍(13m/s 이상) 편향 -0.132 → -0.119
  · 단순 규칙(풍속 10 이상 & 이용률 0.2 미만) : 0.9% 만 잡혀 효과 없음 → 기각
  같은 방식을 2025-01~09 자체에 적용하면 2.1% 가 걸린다(순수 잡음 수준). 옛 구간 5.2% 중 실제 출력제어는
  약 3%p 로 추정한다. 설비 점검·고장도 일부 섞일 수 있다.

주의: 이 스크립트를 다시 돌리면 CSV 가 덮어써진다. 임계값을 바꿀 때만 돌리고, 바꿨다면 train_wind_lgbm.py 로
재학습한 뒤 예보 입력 기준(블록 교차검증)으로 다시 평가할 것.

실행: python Training/3_jeju_solarwind_forecaster/training/make_curtailment_exclusion.py
"""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P

_spec = importlib.util.spec_from_file_location(
    'cmpA', os.path.join(ROOT, 'Training', '3_jeju_solarwind_forecaster', 'comparison', 'model', '3cmp-A_lgbm_solarwind.py'))
cmpA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cmpA)

# Training/**/*.csv 는 gitignore 대상이라, git 이 추적하는 data/refdata/ 에 둔다 (다른 곳에서 재학습해도 따라온다)
OUT_CSV = os.path.join(P.REFDATA, 'curtailment_excluded_hours.csv')
REFERENCE_START, REFERENCE_END = '2025-01-01', '2025-09-19 23:00'   # 기준 모델 학습 구간(정책 변경 이후)
JUDGE_END = '2024-12-31 23:00'                                      # 이 시각까지만 판별한다
RESIDUAL_THRESHOLD = 0.15
MIN_REFERENCE_PREDICTION = 0.2
CUT_OUT_PROTECT_WIND = 20.0
FEATURES = list(cmpA.WIND_FINAL)
WIND_UTIL = cmpA.WU
PARAMS = dict(objective='regression_l1', n_estimators=300, learning_rate=0.03, num_leaves=63,
              min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1, random_state=0)


def main():
    raw = cmpA.load_raw()
    features, _ = cmpA.build_features(raw)
    features = features.dropna(subset=FEATURES + [WIND_UTIL])

    reference_data = features[(features.index >= REFERENCE_START) & (features.index <= REFERENCE_END)]
    reference_model = lgb.LGBMRegressor(**PARAMS).fit(reference_data[FEATURES], reference_data[WIND_UTIL])

    judged = features[features.index <= JUDGE_END]
    reference_prediction = np.clip(reference_model.predict(judged[FEATURES]), 0, 1)
    shortfall = judged[WIND_UTIL].values - reference_prediction          # 음수일수록 기준보다 낮게 나옴
    is_curtailed = (shortfall < -RESIDUAL_THRESHOLD) & (reference_prediction > MIN_REFERENCE_PREDICTION)
    is_cut_out = judged['wind_spd_west'].values >= CUT_OUT_PROTECT_WIND   # 강풍 정지는 지우지 않는다
    excluded = judged[is_curtailed & ~is_cut_out]

    result = pd.DataFrame({'timestamp': excluded.index,
                           'actual_utilization': excluded[WIND_UTIL].values.round(4),
                           'reference_prediction': reference_prediction[is_curtailed & ~is_cut_out].round(4),
                           'wind_spd_west': excluded['wind_spd_west'].values})
    result.to_csv(OUT_CSV, index=False)
    print(f'판별 대상 {len(judged)}행 중 제외 {len(result)}행 ({len(result) / len(judged) * 100:.1f}%)'
          f'  | cut-out 보호로 남긴 행 {int((is_curtailed & is_cut_out).sum())}')
    print('연도별 제외:', result.groupby(result.timestamp.dt.year).size().to_dict())
    print('saved', OUT_CSV)


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    main()
