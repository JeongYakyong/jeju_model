# -*- coding: utf-8 -*-
"""풍력 LGBM 재학습 스크립트 — 언제든 이 파일만 실행하면 재학습된다.

실행:   python Training/3_jeju_solarwind_forecaster/training/train_wind_lgbm.py            # 학습 + 저장
        python Training/3_jeju_solarwind_forecaster/training/train_wind_lgbm.py --no-save  # 학습·검증만, 저장 안 함

════════════════════════════════════════════════════════════════════════════
 재학습할 때 알아둘 것 (요약)
════════════════════════════════════════════════════════════════════════════
 1. 입력 자료: DB historical (ASOS 실측 풍속·풍향 + 풍력 실측 이용률). 서버에서 수집이 계속 쌓으므로
    그냥 이 스크립트를 다시 돌리면 최신 실측까지 학습한다. 따로 손볼 것은 없다.
 2. 학습에서 빼는 시각: data/refdata/curtailment_excluded_hours.csv (옛 출력제어 의심 시각, 2,168행).
    이 목록은 2025-01 이전 구간에서 한 번 만든 고정 파일이다(만든 방법: make_curtailment_exclusion.py).
    ★ 재학습 때 다시 만들 필요가 없다. 2025-01 이후 새 자료는 출력제어가 거의 없다고 보고 전부 쓴다.
 3. 트리 수(n_estimators): 아래 VALIDATION 구간으로 early stopping 해서 정하고, 최종 모델은 그 트리 수를
    고정한 채 전 구간으로 다시 학습한다. 코드가 알아서 한다.
 4. 풍속 입력 보정(QM)은 별개다: models/solarwind_lgbm/wind_qm.json. 모델을 다시 학습해도 QM 은 그대로 유효하다
    (QM 은 "예보 풍속 → 실측 풍속" 변환일 뿐 모델과 무관). 예보 자료가 쌓였을 때만
    fit_wind_qm.py 로 갱신하면 된다.
 5. 저장 위치: models/solarwind_lgbm/lgbm_wind_util.txt (덮어씀). 덮어쓰기 전에 옛 파일을 백업해 둘 것.
    배포는 서버에서 git pull.

════════════════════════════════════════════════════════════════════════════
 왜 이렇게 학습하는가 (2026-10-09 결정, 상세 근거는 DECISIONS.md)
════════════════════════════════════════════════════════════════════════════
 · 문제: 2020 ~ 2024-06 에는 풍력 출력제어(curtailment)가 잦았다(특히 봄·가을 한낮). 그대로 학습하면
   모델이 "한낮에는 같은 풍속인데 발전이 낮다"를 배워 한낮을 낮게 예측한다(2026-10-05 정오 사례).
 · 대안 비교 (서빙 입력인 예보 풍속으로 평가, 예보 구간 2025-12-20~ 을 2개월씩 5블록으로 나눈 교차검증):
     ① 2020~ 전체       : 한낮 편향 -0.018 / 강풍(west 13m/s 이상) 편향 -0.132
     ② 2020~ 출력제어 시각 제외 : 한낮 -0.000 / 강풍 -0.119   ← 채택(이 스크립트)
     ④ 2024-07~ 만      : 한낮 -0.011 / 강풍 -0.186   (정책 변경 이후만 쓰면 강풍 자료가 적어 약함)
   전체 MAE 는 모델 간 차이가 0.002 이하라 의미가 없고, 한낮·강풍 편향이 선택 기준이었다.
   (이용률 0.1 이 약 40~50MW 이므로 편향 0.02~0.05 도 가스 기동 판단에 무시할 수 없다.)
 · 강풍 정지(cut-out) 자료는 원래 거의 없다(west 25m/s 이상 47시간). 그래서 짧은 기간만 쓰면 강풍 자료가
   모자라 약해진다 → 옛 구간을 버리지 않고 쓰되 출력제어 의심 시각만 뺀다.

════════════════════════════════════════════════════════════════════════════
 구간 (TRAIN / VALIDATION / 최종)
════════════════════════════════════════════════════════════════════════════
   TRAIN       2020-01-01 ~ 2025-09-19   (제외 CSV 의 시각은 뺀다)
   VALIDATION  2025-09-20 ~ 2025-12-19   early stopping 으로 트리 수만 정한다
   최종 모델   2020-01-01 ~ 최신 실측 전체 (제외 CSV 시각 제외), 트리 수 고정
 * TEST 구간은 두지 않는다. 풍력 모델 성능은 "서빙 입력(예보 풍속)" 으로 재야 의미가 있는데 예보 자료는
   2025-12-20 이후뿐이라, 평가를 위해 학습 자료를 떼어 내면 안 된다. 대신 예보 구간을 2개월씩 나누고
   한 블록을 빼고 학습해 그 블록을 평가하는 블록 교차검증으로 확인했다(위 표). 모델을 크게 바꿀 때는
   같은 방식으로 다시 평가할 것.
 * 정책 변경 시점(2024-06) 이후 구간을 VALIDATION 으로 쓰는 이유: 트리 수를 새 정책에 맞는 쪽으로 고르기 위해.

산출: models/solarwind_lgbm/lgbm_wind_util.txt, feat_meta.json 의 wind_train 항목
"""
from __future__ import annotations
import os, sys, json, importlib.util, argparse
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P

# 피처 빌더·피처 목록은 기존 3cmp-A 와 같은 것을 쓴다 (서빙도 같은 build_features 를 쓰므로 학습=서빙 정합).
# 피처를 바꾸려면 3cmp-A 의 WIND_FINAL 과 서빙(forecasting/serve_solarwind_lgbm.py)을 함께 바꿔야 한다.
_spec = importlib.util.spec_from_file_location(
    'cmpA', os.path.join(ROOT, 'Training', '3_jeju_solarwind_forecaster', 'comparison', 'model', '3cmp-A_lgbm_solarwind.py'))
cmpA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cmpA)

TRAIN_START = '2020-01-01'
TRAIN_END = '2025-09-19 23:00'
VALIDATION_START, VALIDATION_END = '2025-09-20', '2025-12-19 23:00'
EXCLUDED_HOURS_CSV = os.path.join(P.REFDATA, 'curtailment_excluded_hours.csv')   # 옛 출력제어 의심 시각 (고정 파일)
WIND_UTIL = cmpA.WU                      # 타깃: real_wind_utilization_jeju (0~1 이용률)
FEATURES = list(cmpA.WIND_FINAL)
PARAMS = dict(objective='regression_l1', n_estimators=1200, learning_rate=0.03, num_leaves=63,
              min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1, random_state=0)


def main(save: bool):
    raw = cmpA.load_raw()
    features, _ = cmpA.build_features(raw)
    features = features.dropna(subset=FEATURES + [WIND_UTIL])
    features = features[features.index >= TRAIN_START]

    # 옛 출력제어 의심 시각을 뺀다 (2025-01 이전 구간에만 해당하는 고정 목록)
    excluded_hours = pd.to_datetime(pd.read_csv(EXCLUDED_HOURS_CSV)['timestamp'])
    features = features.drop(features.index.intersection(excluded_hours))
    print(f'제외 시각 {len(excluded_hours)}개 반영 → 사용 가능 {len(features)}행, 최신 {features.index.max()}')

    train = features[features.index <= TRAIN_END]
    validation = features[(features.index >= VALIDATION_START) & (features.index <= VALIDATION_END)]
    print(f'train {len(train)}행 / validation {len(validation)}행')

    # 1단계: validation 으로 early stopping — 여기서는 "트리를 몇 개 쓸지"만 정한다
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(train[FEATURES], train[WIND_UTIL], eval_set=[(validation[FEATURES], validation[WIND_UTIL])],
              callbacks=[lgb.early_stopping(60, verbose=False)])
    best_iteration = int(model.best_iteration_)
    # 트리 수에 따른 validation MAE — 값이 평평하면 트리 수 선택에 민감하지 않다는 뜻 (재학습 때 이상 징후 확인용)
    curve = {n: round(float(np.abs(np.clip(model.predict(validation[FEATURES], num_iteration=n), 0, 1)
                                   - validation[WIND_UTIL].values).mean()), 4)
             for n in sorted({50, 100, best_iteration, best_iteration + 30, model.booster_.num_trees()})}
    print('트리 수별 validation MAE:', curve, '| best_iteration =', best_iteration)

    # 2단계: 트리 수를 고정하고 최신 실측까지 전체로 다시 학습 → 이것이 배포 모델
    final_model = lgb.LGBMRegressor(**dict(PARAMS, n_estimators=best_iteration))
    final_model.fit(features[FEATURES], features[WIND_UTIL])
    if not save:
        print('(--no-save: 저장 안 함)'); return

    out_path = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'lgbm_wind_util.txt')
    final_model.booster_.save_model(out_path)
    meta_path = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'feat_meta.json')
    meta = json.load(open(meta_path, encoding='utf-8'))
    meta['wind_train'] = {
        'train': f'{TRAIN_START}..{TRAIN_END} (출력제어 의심 시각 제외)', 'validation': f'{VALIDATION_START}..{VALIDATION_END}',
        'deploy_refit': f'{TRAIN_START}..{features.index.max()} 전체 재적합, n_estimators={best_iteration}',
        'excluded_hours': f'data/refdata/curtailment_excluded_hours.csv ({len(excluded_hours)}행)',
        'note': '2026-10-09 재학습. 입력 풍속 보정은 wind_qm.json(시각대별 QM). 근거는 이 스크립트 상단 주석.'}
    json.dump(meta, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved', out_path)


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-save', action='store_true')
    main(save=not parser.parse_args().no_save)
