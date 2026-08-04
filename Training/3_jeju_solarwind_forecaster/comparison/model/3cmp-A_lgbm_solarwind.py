"""3cmp-A — solar/wind 이용률 LGBM 학습 (순수기상 horizon-무관 단일모델).

피처 확정(사용자 §0.6, 2026-06-08):
  SOLAR(final): solar_rad·total_cloud·midlow_cloud·solar_damping(west·south)
              + clearsky_rad_ratio(west·south, 흐린날 명시) + hour sin/cos + month sin/cos
  SOLAR(ablation=PatchTST 동일): 위에서 clearsky_ratio·month 제거
  WIND(final): wind_spd·wind_zone(west·east) + 풍향(west sin/cos) + hour sin/cos + year sin/cos

학습창(2026-07-30 재학습, 사용자 확정): train ≤2026-01 / val 2026-02~05 / test 2026-06~07.
  (구: train ≤2024 / val 2025 / test 2026 — 연 단위 분할)
순수기상 horizon-무관: 이용률=f(기상,시각,계절)만 → h 미사용(D+1~D+5 동일 적용).
산출: model/lgbm_solar_util.txt, lgbm_solar_util_ablation.txt, lgbm_wind_util.txt,
      model/clearsky_clim.csv (clearsky 평년, train 기준), model/feat_meta.json.
공개 함수 build_features() 는 비교 하니스(3cmp-B)에서 재사용.

2026-07-30 재학습:
  - 데이터 소스를 얼린 CSV(solarwind_raw_jeju.csv, ~2026-06-01) 에서 **메인 DB historical**
    로 바꿨다. 컬럼 25개가 그대로 있고 DB 가 2개월 더 최신이다.
  - 배포본은 val 로 잡은 best_iteration 을 고정해 **train+val+test 전체 재적합**.
  - ★clearsky 평년(clim)은 train 구간에서 만든다. 학습창이 바뀌면 평년도 바뀌므로
    서빙(forecasting/serve_solarwind_lgbm._assets)이 재현하는 학습창도 함께 맞춰야 한다.
"""
import os, sys, json, sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
CMP  = os.path.normpath(os.path.join(HERE, '..'))
# 저장소 경로 SSOT — project_paths.py 가 있는 폴더가 루트다
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P
DB = P.DB_JEJU
SU, WU = 'real_solar_utilization_jeju', 'real_wind_utilization_jeju'

# 학습창 (타깃 시각 기준) — demand 2-A 와 같은 경계를 쓴다
TRAIN_END = '2026-01-31 23:00'
VAL_BEG, VAL_END = '2026-02-01', '2026-05-31 23:00'
TEST_BEG, TEST_END = '2026-06-01', '2026-07-31 23:00'

# CSV 원자료와 같은 컬럼을 DB historical 에서 그대로 읽는다
RAW_COLS = ['timestamp', 'solar_rad_west', 'solar_rad_south',
            'total_cloud_west', 'total_cloud_south', 'midlow_cloud_west', 'midlow_cloud_south',
            'rainfall_west', 'rainfall_south',
            'wind_spd_west', 'wind_spd_east', 'wind_spd_south',
            'wd_sin_west', 'wd_cos_west', 'wd_sin_east', 'wd_cos_east', 'wd_sin_south', 'wd_cos_south',
            SU, WU, 'real_solar_gen_jeju', 'real_wind_gen_jeju',
            'real_solar_capacity_jeju', 'real_wind_capacity_jeju']


def load_raw():
    """메인 DB historical → 학습 원자료 프레임 (구 solarwind_raw_jeju.csv 와 동일 스키마)."""
    con = sqlite3.connect(DB)
    df = pd.read_sql(f'SELECT {", ".join(RAW_COLS)} FROM historical ORDER BY timestamp',
                     con, parse_dates=['timestamp'])
    con.close()
    df = df.set_index('timestamp').sort_index().apply(pd.to_numeric, errors='coerce')
    print(f'[load] DB historical {df.shape}  {df.index.min()} ~ {df.index.max()}')
    return df

# (2026-06-17) 사용자 피처 확정: solar_damping·clearsky_ratio 는 south 단독(west 제거).
# 일사·구름은 west+south 유지. PatchTST(주력)는 미재학습이라 이 변경은 LGBM 폴백에만 반영.
SOLAR_FINAL = ['solar_rad_west', 'solar_rad_south', 'total_cloud_west', 'total_cloud_south',
               'midlow_cloud_west', 'midlow_cloud_south', 'solar_damping_south',
               'clearsky_ratio_south',
               'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
SOLAR_ABLATION = ['solar_rad_west', 'solar_rad_south', 'total_cloud_west', 'total_cloud_south',
                  'midlow_cloud_west', 'midlow_cloud_south', 'solar_damping_west', 'solar_damping_south',
                  'hour_sin', 'hour_cos']
# 2026-06-17: east 풍향(wd_sin/cos_east) 추가 1회 실험 → forecast 백테스트서 악화(wind D+1
# 0.125→0.135, west/east 풍향 중복=예보오차만 더함) → 미채택, 풍향 west 단독 유지.
# 실험 스크립트 training/exp_wind_east_dir.py(기록 보존).
# 사용자 피처 확정: wind_zone 은 east 단독(wind_zone_west 제거).
WIND_FINAL = ['wind_spd_west', 'wind_spd_east', 'wind_zone_east',
              'wd_sin_west', 'wd_cos_west', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']

WIND_SPD_CAP, CUTOFF = 20.0, 25.0


def _wind_zone(raw):
    cond = [raw < 15, (raw >= 15) & (raw < 20), (raw >= 20) & (raw < CUTOFF), raw >= CUTOFF]
    return np.select(cond, [0.0, 1.0, 0.5, 0.0], default=0.0)


def _damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    return np.exp(-0.163 * daily.clip(upper=10))


def build_features(df, clim=None):
    """raw CSV/DB 프레임 → 피처 컬럼 추가. clim=None이면 train으로 clearsky 평년 산출·반환."""
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    for st in ['west', 'south']:
        df[f'solar_damping_{st}'] = _damping(df, st)
    for st in ['west', 'east']:
        df[f'wind_zone_{st}'] = _wind_zone(df[f'wind_spd_{st}'])
        df[f'wind_spd_{st}'] = df[f'wind_spd_{st}'].clip(upper=WIND_SPD_CAP)
    # clearsky_ratio: (month,hour)별 train rad 90분위 평년 대비
    if clim is None:
        clim = {}
        for st in ['west', 'south']:
            g = df.groupby([df.index.month, df.index.hour])[f'solar_rad_{st}'].quantile(0.90)
            clim[st] = g
    for st in ['west', 'south']:
        key = list(zip(df.index.month, df.index.hour))
        cs = clim[st].reindex(key).values
        ratio = np.where(cs > 0.05, df[f'solar_rad_{st}'].values / cs, 0.0)
        df[f'clearsky_ratio_{st}'] = np.clip(ratio, 0, 1.5)
    return df, clim


def main():
    df = load_raw()
    tr = df[df.index <= TRAIN_END]
    feat_tr, clim = build_features(tr)
    feat_all, _ = build_features(df, clim=clim)
    feat_all['split'] = np.where(feat_all.index <= TRAIN_END, 'train',
                        np.where(feat_all.index <= VAL_END, 'val',
                        np.where(feat_all.index <= TEST_END, 'test', 'after')))
    print('  split 행수:', feat_all.split.value_counts().to_dict())

    # clearsky 평년 저장
    cs_df = pd.concat({st: clim[st] for st in clim}, axis=1)
    cs_df.columns = [f'clearsky90_{c}' for c in cs_df.columns]
    cs_df.to_csv(os.path.join(HERE, 'clearsky_clim.csv'))

    params = dict(objective='regression_l1', n_estimators=1200, learning_rate=0.03,
                  num_leaves=63, min_child_samples=80, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

    def fit(feats, target, name):
        """검증본(train 만)으로 best_iteration 을 잡고, 배포본은 전 구간 재적합해 저장한다.

        반환은 (검증본, 배포본). 성능표는 test 를 안 본 검증본으로 내고,
        디스크에 남는 것은 최신 데이터까지 학습한 배포본이다.
        """
        tr_m = feat_all[feat_all.split == 'train']
        va_m = feat_all[feat_all.split == 'val']
        m = lgb.LGBMRegressor(**params)
        m.fit(tr_m[feats], tr_m[target], eval_set=[(va_m[feats], va_m[target])],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        best = int(m.best_iteration_)

        full = feat_all[feat_all.split.isin(['train', 'val', 'test'])]
        deploy_params = dict(params, n_estimators=best)
        m_deploy = lgb.LGBMRegressor(**deploy_params)
        m_deploy.fit(full[feats], full[target])
        m_deploy.booster_.save_model(os.path.join(HERE, name))

        imp = pd.Series(m.booster_.feature_importance('gain'), index=feats).sort_values(ascending=False)
        print(f'\n[{name}] best_iter={best}  재적합 {len(tr_m)}행 -> {len(full)}행  중요도(gain) top:')
        print((imp / imp.sum()).round(3).head(8).to_string())
        return m, m_deploy

    print('=' * 60)
    print(f'LGBM 학습 (train ≤{TRAIN_END} / val {VAL_BEG}~{VAL_END}, early stop)')
    print(f'배포본은 test({TEST_BEG}~{TEST_END}) 까지 포함해 재적합')
    m_solar, m_solar_deploy = fit(SOLAR_FINAL, SU, 'lgbm_solar_util.txt')
    m_solar_abl, _ = fit(SOLAR_ABLATION, SU, 'lgbm_solar_util_ablation.txt')
    m_wind, m_wind_deploy = fit(WIND_FINAL, WU, 'lgbm_wind_util.txt')

    json.dump({'SOLAR_FINAL': SOLAR_FINAL, 'SOLAR_ABLATION': SOLAR_ABLATION,
               'WIND_FINAL': WIND_FINAL, 'target_solar': SU, 'target_wind': WU,
               'train': f'<={TRAIN_END}', 'val': f'{VAL_BEG}..{VAL_END}',
               'test': f'{TEST_BEG}..{TEST_END}',
               'deploy_refit': 'train+val+test 전체 재적합 (n_estimators=best_iteration 고정)',
               'source': 'main DB historical (2026-07-30 재학습; 구 solarwind_raw_jeju.csv 대체)',
               'clearsky_clim': 'train 구간 (month,hour) rad 90분위 — 서빙이 같은 학습창으로 재현해야 함'},
              open(os.path.join(HERE, 'feat_meta.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    # ---- test 이용률 평가 (perfect weather, horizon-무관) ----
    te = feat_all[feat_all.split == 'test'].copy()
    te['pred_solar'] = np.clip(m_solar.predict(te[SOLAR_FINAL]), 0, 1)
    te['pred_solar_abl'] = np.clip(m_solar_abl.predict(te[SOLAR_ABLATION]), 0, 1)
    te['pred_wind'] = np.clip(m_wind.predict(te[WIND_FINAL]), 0, 1)

    # 낮시간(태양광) + 흐림 regime
    day = te[(te.index.hour >= 8) & (te.index.hour <= 17)].copy()
    day_cloud = day.groupby(day.index.date)['total_cloud_west'].mean()
    cloudy = set(day_cloud[day_cloud >= 0.7].index); sunny = set(day_cloud[day_cloud <= 0.3].index)
    day['regime'] = np.where([d in cloudy for d in day.index.date], 'cloudy',
                    np.where([d in sunny for d in day.index.date], 'sunny', 'mixed'))

    def util_metrics(frame, pred, true):
        e = frame[pred] - frame[true]
        return dict(MAE=round(e.abs().mean(), 4), bias=round(e.mean(), 4),
                    n=len(frame))

    print('\n' + '=' * 60)
    print(f'[SOLAR 이용률 test {TEST_BEG}~{TEST_END}, 낮 8-17h] LGBM(final) vs LGBM(ablation)')
    rows = []
    for r in ['sunny', 'mixed', 'cloudy', 'ALL']:
        sub = day if r == 'ALL' else day[day.regime == r]
        rows.append(dict(regime=r, **{f'final_{k}': v for k, v in util_metrics(sub, 'pred_solar', SU).items()},
                         abl_bias=round((sub['pred_solar_abl'] - sub[SU]).mean(), 4),
                         abl_MAE=round((sub['pred_solar_abl'] - sub[SU]).abs().mean(), 4)))
    sm = pd.DataFrame(rows); print(sm.to_string(index=False))
    sm.to_csv(os.path.join(CMP, 'tab', '3cmp-A_solar_util_test.csv'), index=False)

    print(f'\n[WIND 이용률 test {TEST_BEG}~{TEST_END}, 전시간]')
    print(util_metrics(te, 'pred_wind', WU))
    print('\n학습·평가 완료.')


if __name__ == '__main__':
    main()
