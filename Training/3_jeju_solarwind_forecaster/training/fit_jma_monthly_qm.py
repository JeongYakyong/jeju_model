"""학습 "미래" 입력용 예보-유사 시계열 생성 — 월별 QM (2026-10-02, fit_jma_kimg_qm.py 대체).

사용자 확정 (2026-10-02)
================================================================================
  - **운량은 JMA 만 쓴다. KIMG 운량은 쓰지 않는다** (KIMG 전운량이 과하게 높아 부적절).
    서빙도 JMA 운량 예보를 받게 된다 → 학습 운량은 KIMG 분포로 바꾸지 않는다.
      2020-01 ~ 2022-07-02 : JMA 분석치(historical) → JMA D+1 예보 분포로 월별 QM
      2022-07-02 ~         : JMA D+1 예보(previous_day1) 원본
  - **일사·강수는 KIMG 유지** — 과거 구간은 JMA → KIMG(12z D+1) 월별 QM.
      2020-01 ~ 2022-07-02 : JMA 분석치 → KIMG
      2022-07-02 ~ 2025-12-19 : JMA D+1 예보 → KIMG
    (2025-12-20 ~ 은 export_solarwind_csv.py 가 실제 KIMG 를 붙인다)
  - 중하층운량은 쓰지 않는다 (JMA previous run 에 층별운량이 없음).
  - QM 은 **월별**. 전체 한 벌로 맞추면 계절 차이가 뭉개진다.

월별 표본이 모자란 달
  KIMG 겹침 구간이 2025-12-20 ~ 2026-09-30 뿐이라 10~11월이 없고 12월은 12일이다.
  그 달은 앞뒤 달로 창을 넓혀(±1, ±2 …) 최소 MIN_DAYS 일을 채운다. 넓힌 달은 보고서에 남긴다.

결측: 원천이 NaN 이면 NaN 그대로 둔다 (DATA_GUIDE 6절 — 다른 자료로 채우지 않는다, 강수는 보간 금지).

출력 (data/refdata/meteo_data/processed/, 원본 CSV 는 건드리지 않는다)
  jma_monthly_qm.json            월별 매핑표 + 월별 표본수·창 폭·편향 보고
  jma_future_{west,south}.csv    time_kst, total_cloud, total_cloud_d2, radiation, rainfall, *_source

    python Training/3_jeju_solarwind_forecaster/training/fit_jma_monthly_qm.py
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

# 저장소 루트 = project_paths.py 가 있는 상위 폴더
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P   # noqa: E402

RAW_DIR = os.path.join(ROOT, 'data', 'refdata', 'meteo_data')
OUT_DIR = os.path.join(RAW_DIR, 'processed')
STATIONS = ['west', 'south']          # solar 모델 지점만
MJ_PER_W = 0.0036                     # W/m^2 -> MJ/m^2/h (DATA_GUIDE 5절)
N_QUANTILES = 199
MIN_DAYS = 20                         # 월별 매핑에 필요한 최소 일수 — 모자라면 앞뒤 달로 창을 넓힌다
RAIN_HOURLY_CAP = 100.0               # KIMG 강수 sentinel(~1310.7mm) 방지

KIMG_START = '2025-12-20'             # forecast_horizon 실존 시작
KIMG_END = '2026-09-30 23:00'
QM_OUTPUT_END = '2025-12-19 23:00'    # 일사·강수 QM 값은 여기까지만 쓴다 (이후는 실제 KIMG)


# ── 원천 읽기 ────────────────────────────────────────────────────────────────
def load_jma_analysis(st):
    raw = pd.read_csv(os.path.join(RAW_DIR, f'jma_historical_{st}.csv'), parse_dates=['time_kst'])
    return pd.DataFrame({
        'total_cloud': pd.to_numeric(raw['cloud_cover'], errors='coerce') / 100.0,
        'radiation': pd.to_numeric(raw['shortwave_radiation'], errors='coerce') * MJ_PER_W,
        'rainfall': pd.to_numeric(raw['precipitation'], errors='coerce'),
    }).set_index(raw['time_kst']).sort_index()


def load_jma_forecast(st):
    raw = pd.read_csv(os.path.join(RAW_DIR, f'jma_previous_runs_{st}.csv'), parse_dates=['time_kst'])
    return pd.DataFrame({
        'total_cloud': pd.to_numeric(raw['cloud_cover_previous_day1'], errors='coerce') / 100.0,
        'total_cloud_d2': pd.to_numeric(raw['cloud_cover_previous_day2'], errors='coerce') / 100.0,
        'radiation': pd.to_numeric(raw['shortwave_radiation_previous_day1'], errors='coerce') * MJ_PER_W,
        'rainfall': pd.to_numeric(raw['precipitation_previous_day1'], errors='coerce'),
    }).set_index(raw['time_kst']).sort_index()


def load_kimg_d1():
    """KIMG 12z 발표 D+1 (서빙 조건과 같은 행만)."""
    con = sqlite3.connect(P.DB_JEJU)
    cols = ', '.join(f'radiation_{st}, rainfall_{st}' for st in STATIONS)
    kimg = pd.read_sql(
        f"SELECT timestamp, {cols} FROM forecast_horizon "
        f"WHERE substr(base, 12) = '21:00:00' AND horizon_d = 1 ORDER BY timestamp",
        con, parse_dates=['timestamp']).set_index('timestamp')
    con.close()
    kimg = kimg.apply(pd.to_numeric, errors='coerce')
    for st in STATIONS:
        kimg[f'rainfall_{st}'] = kimg[f'rainfall_{st}'].clip(upper=RAIN_HOURLY_CAP)
    return kimg


# ── 월별 QM ──────────────────────────────────────────────────────────────────
def month_distance(months, target_month):
    gap = np.abs(np.asarray(months) - target_month)
    return np.minimum(gap, 12 - gap)


def collapse_ties(source_q, target_q):
    """분위수표에서 source 값이 겹치는 구간을 한 점으로 접는다.

    강수처럼 0 이 대부분이면 source_q 앞쪽이 전부 0 이라 np.interp 가 0 을 양수로 바꿔 버린다
    (실측: 겨울 강수 편향 +0.1~0.2mm/h 로 악화). 맨 아래 겹침 = 최솟값(0 은 0 으로),
    맨 위 겹침 = 최댓값(운량 1.0 은 1.0 으로), 중간 겹침 = 평균.
    """
    unique_source = np.unique(source_q)
    collapsed_target = []
    for value in unique_source:
        tied_target = target_q[source_q == value]
        if value == source_q[0]:
            collapsed_target.append(tied_target.min())
        elif value == source_q[-1]:
            collapsed_target.append(tied_target.max())
        else:
            collapsed_target.append(tied_target.mean())
    return unique_source, np.array(collapsed_target)


def fit_monthly(source, target, clip):
    """source -> target 월별 분위수 매핑표. 두 Series 는 같은 index(시각)로 맞춰져 있어야 한다."""
    pair = pd.DataFrame({'source': source, 'target': target}).dropna()
    months = pair.index.month.values
    quantile_levels = np.linspace(0.005, 0.995, N_QUANTILES)
    tables = {}
    for month in range(1, 13):
        for window in range(0, 7):
            chosen = pair[month_distance(months, month) <= window]
            if chosen.index.normalize().nunique() >= MIN_DAYS:
                break
        source_q, target_q = collapse_ties(np.quantile(chosen['source'], quantile_levels),
                                           np.quantile(chosen['target'], quantile_levels))
        mapped = np.clip(np.interp(chosen['source'], source_q, target_q), *clip)
        in_month = chosen.index.month == month
        tables[month] = {
            'source_q': source_q.round(5).tolist(), 'target_q': target_q.round(5).tolist(),
            'window_months': window, 'n_days': int(chosen.index.normalize().nunique()),
            'n_days_this_month': int(chosen[in_month].index.normalize().nunique()),
            'bias_before': float((chosen['source'] - chosen['target']).mean()),
            'bias_after': float((mapped - chosen['target']).mean()),
        }
    return tables


def apply_monthly(series, tables, clip):
    out = pd.Series(np.nan, index=series.index)
    for month in range(1, 13):
        rows = (series.index.month == month) & series.notna().values
        table = tables[month]
        out[rows] = np.clip(np.interp(series[rows], table['source_q'], table['target_q']), *clip)
    return out


def print_tables(label, tables):
    widened = {m: t['window_months'] for m, t in tables.items() if t['window_months'] > 0}
    bias = ' '.join(f"{m}:{t['bias_before']:+.3f}→{t['bias_after']:+.3f}" for m, t in tables.items())
    print(f'  [{label}] 월별 편향 {bias}')
    if widened:
        print(f'  [{label}] ⚠창을 넓힌 달(±개월): {widened}')


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kimg = load_kimg_d1()
    payload = {'_doc': '월별 QM — 운량: JMA 분석치→JMA D+1 예보 / 일사·강수: JMA→KIMG 12z D+1',
               'min_days': MIN_DAYS, 'stations': {}}

    for st in STATIONS:
        analysis = load_jma_analysis(st)
        forecast = load_jma_forecast(st)
        forecast_start = forecast['total_cloud'].first_valid_index()
        print(f'\n=== {st} === JMA D+1 예보 시작 {forecast_start}')
        station_tables = {}

        # 1) 운량: 분석치 -> JMA D+1 예보 (겹침 = 예보 시작 이후 전 구간, 4년+)
        cloud_tables = fit_monthly(analysis['total_cloud'], forecast['total_cloud'].reindex(analysis.index),
                                   clip=(0.0, 1.0))
        print_tables('운량 분석치→JMA예보', cloud_tables)
        station_tables['total_cloud_analysis_to_jma_forecast'] = cloud_tables

        # 2) 일사·강수: JMA(분석치 / D+1 예보 각각) -> KIMG 12z D+1 (겹침 = KIMG 구간)
        kimg_window = slice(KIMG_START, KIMG_END)
        for var, clip in [('radiation', (0.0, None)), ('rainfall', (0.0, None))]:
            target = kimg[f'{var}_{st}'][kimg_window]
            for source_name, source in [('analysis', analysis), ('forecast', forecast)]:
                tables = fit_monthly(source[var][kimg_window].reindex(target.index), target,
                                     clip=(clip[0], np.inf if clip[1] is None else clip[1]))
                print_tables(f'{var} JMA{source_name}→KIMG', tables)
                station_tables[f'{var}_{source_name}_to_kimg'] = tables
        payload['stations'][st] = station_tables

        # 3) 출력 시계열 — 2020-01 ~ 2026-09 전체 시각
        index = analysis.index
        forecast_full = forecast.reindex(index)
        before_forecast = index < forecast_start
        out = pd.DataFrame(index=index)

        cloud_from_analysis = apply_monthly(analysis['total_cloud'], cloud_tables, (0.0, 1.0))
        out['total_cloud'] = np.where(before_forecast, cloud_from_analysis, forecast_full['total_cloud'])
        out['total_cloud_source'] = np.where(before_forecast, 'jma_analysis_qm', 'jma_forecast_d1')
        out['total_cloud_d2'] = forecast_full['total_cloud_d2']          # 평가 ② 전용 (원본)

        for var in ['radiation', 'rainfall']:
            from_analysis = apply_monthly(analysis[var], station_tables[f'{var}_analysis_to_kimg'], (0.0, np.inf))
            from_forecast = apply_monthly(forecast_full[var], station_tables[f'{var}_forecast_to_kimg'], (0.0, np.inf))
            out[var] = np.where(before_forecast, from_analysis, from_forecast)
            out[f'{var}_source'] = np.where(before_forecast, 'jma_analysis_qm_kimg', 'jma_forecast_d1_qm_kimg')
            out.loc[out.index > pd.Timestamp(QM_OUTPUT_END), var] = np.nan   # 이후는 실제 KIMG 를 쓴다

        path = os.path.join(OUT_DIR, f'jma_future_{st}.csv')
        out.rename_axis('time_kst').reset_index().to_csv(path, index=False)
        print(f'  결측률: ' + ', '.join(f'{c} {out[c].isna().mean():.4f}'
                                       for c in ['total_cloud', 'radiation', 'rainfall']))
        print(f'  -> {os.path.normpath(path)} ({len(out)}행)')

    with open(os.path.join(OUT_DIR, 'jma_monthly_qm.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print('\nsaved jma_monthly_qm.json')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
