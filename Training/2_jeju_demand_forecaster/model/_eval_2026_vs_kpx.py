# -*- coding: utf-8 -*-
"""2026 월별·계절별 비교 — 제주 수요 모델(2-A 구성) D+1 vs KPX 하루전 예측.

보고서용 백테스트. 배포본은 2026-07 까지 재적합돼 있어 2026 을 그대로 평가하면 학습 구간을
다시 맞추는 셈이 된다 → 같은 구성(피처 22개·quantile α=0.60·낮가중2·하이퍼파라미터)을
2025 년까지 데이터로만 다시 학습해 2026 전 구간을 학습에 안 쓴 데이터로 평가한다.

  학습 절차 (배포본과 같은 방식: val 로 iteration 결정 → 전체 재적합)
    train ≤ 2025-08-31 / val 2025-09-01 ~ 2025-12-31 → best_iteration
    최종 모델 = ≤ 2025-12-31 전체를 best_iteration 고정으로 재적합
    학습 기상은 실측(ASOS) — 배포본과 동일

  평가 (2026-01-01 ~ DB 마지막 날)
    원점 = 전날 23:00, D+1 = 대상일 00~23시 (h=1..24)
    기상 = 전날 12z(21:00) 발표 forecast_horizon horizon_d=1 — 실서빙과 같은 입력,
           예보가 빈 시각은 (월,시) 기후값 폴백(serve_demand 와 동일)
    비교 대상 = historical.jeju_est_demand_da (KPX 하루전 예측)
    두 예측과 실측이 모두 있는 시각만 집계 (공통 표본)

  계절: 겨울 12~2월 / 봄 3~5월 / 여름 6~8월 / 가을 9~11월

사용:
    python _eval_2026_vs_kpx.py                  # 메인 DB
    python _eval_2026_vs_kpx.py --db <경로>      # 서버 DB 복사본 등
"""
import os, sys, sqlite3, argparse
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt, matplotlib as mpl
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'project_paths.py')):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import project_paths as P

TAB = os.path.join(HERE, 'tab'); os.makedirs(TAB, exist_ok=True)
FIG = os.path.join(HERE, 'fig'); os.makedirs(FIG, exist_ok=True)

TRAIN_END = '2025-08-31 23:00'
VAL_BEG, VAL_END = '2025-09-01', '2025-12-31 23:00'
FINAL_TRAIN_END = '2025-12-31 23:00'
TEST_BEG = '2026-01-01'

HMAX = 120
STATIONS = ['west', 'east', 'south']; SOLAR_ST = ['west', 'south']
WX = ['temp_c', 'humidity', 'solar_rad', 'wind_spd']
CLOUD = ['total_cloud_west', 'total_cloud_south', 'midlow_cloud_west', 'midlow_cloud_south']
FEAT = ['h', 'lag168', 'rec24', 'rec168', 'temp_c', 'humidity', 'solar_rad', 'wind_spd',
        'total_cloud_west', 'total_cloud_south', 'midlow_cloud_west', 'midlow_cloud_south',
        'cap_btmppa_mw', 'solar_deficit', 'solar_ramp',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day_type']
CAT = ['day_type']
DAYTYPE_CATS = ['holiday', 'weekday', 'weekend']
PARAMS = dict(objective='quantile', alpha=0.60, metric='quantile', learning_rate=0.024, num_leaves=244,
              min_data_in_leaf=76, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
              lambda_l2=0.1, verbosity=-1, random_state=42)
DAY_WEIGHT = 2.0
SEASON_OF_MONTH = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄',
                   6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을', 11: '가을'}
SEASON_ORDER = ['겨울', '봄', '여름', '가을']
# 계측 오류로 보이는 실측 — 앞뒤 시각 750~830MW 인데 혼자 1916MW. 결측 처리(보간·채점 제외)
BAD_ACTUAL_HOURS = ['2026-04-07 19:00']


# ── 1. 데이터 ─────────────────────────────────────────────────────────────
def load_hourly(db_path):
    pull = ['timestamp', 'real_demand_jeju', 'jeju_est_demand_da', 'day_type']
    for st in STATIONS:
        pull += [f'temp_c_{st}', f'humidity_{st}', f'wind_spd_{st}']
    pull += [f'solar_rad_{st}' for st in SOLAR_ST] + CLOUD
    with sqlite3.connect(db_path) as con:
        raw = pd.read_sql(f"SELECT {', '.join(pull)} FROM historical", con, parse_dates=['timestamp'])
        forecast = pd.read_sql(
            "SELECT timestamp, base, temp_west, temp_east, temp_south, reh_west, reh_east, reh_south, "
            "wind_spd_10m_west, wind_spd_10m_east, wind_spd_10m_south, radiation_west, radiation_south, "
            "total_cloud_west, total_cloud_south, midlow_cloud_west, midlow_cloud_south "
            "FROM forecast_horizon WHERE substr(base,12)='21:00:00' AND horizon_d=1",
            con, parse_dates=['timestamp', 'base'])
    raw = raw.sort_values('timestamp')
    raw['temp_c'] = raw[[f'temp_c_{s}' for s in STATIONS]].mean(axis=1)
    raw['humidity'] = raw[[f'humidity_{s}' for s in STATIONS]].mean(axis=1)
    raw['wind_spd'] = raw[[f'wind_spd_{s}' for s in STATIONS]].mean(axis=1)
    raw['solar_rad'] = raw[[f'solar_rad_{s}' for s in SOLAR_ST]].mean(axis=1)

    full_idx = pd.date_range(raw.timestamp.min(), raw.timestamp.max(), freq='h')
    hourly = raw.set_index('timestamp').reindex(full_idx); hourly.index.name = 'timestamp'
    # 평가용 원본 실측(보간 전) — 0/결측 시각은 채점에서 뺀다
    hourly['actual_raw'] = hourly['real_demand_jeju'].where(hourly['real_demand_jeju'] > 0)
    hourly.loc[pd.to_datetime(BAD_ACTUAL_HOURS), 'actual_raw'] = np.nan
    hourly['real_demand_jeju'] = hourly['actual_raw'].interpolate(method='time')
    for w in WX + CLOUD:
        hourly[w] = hourly[w].interpolate(method='time')
    hourly['day_type'] = hourly['day_type'].ffill().bfill()

    cap = pd.read_csv(P.REF_JEJU_PPA_BTM)
    cap_key = pd.MultiIndex.from_arrays([hourly.index.year, hourly.index.month])
    hourly['cap_btmppa_mw'] = (cap.set_index(['year', 'month'])['cap_btmppa_mw']
                               .reindex(cap_key).values)
    hourly['cap_btmppa_mw'] = hourly['cap_btmppa_mw'].ffill()   # 미래월 캐리포워드(서빙과 동일)
    return hourly, forecast


# ── 2. 직접 다지평 샘플 (origin × h) ─────────────────────────────────────
def build_samples(hourly, weather, clim_solar, origin_positions, horizons):
    demand = hourly.real_demand_jeju.values.astype(float)
    rec24 = pd.Series(demand).rolling(24, min_periods=24).mean().values
    rec168 = pd.Series(demand).rolling(168, min_periods=168).mean().values
    target_pos = (origin_positions[:, None] + horizons[None, :]).ravel()
    horizon_col = np.broadcast_to(horizons, (len(origin_positions), len(horizons))).ravel()
    n_h = len(horizons)
    index = hourly.index

    solar = weather['solar_rad']
    with np.errstate(divide='ignore', invalid='ignore'):
        solar_deficit = np.where(clim_solar > 5, np.clip(1 - solar / clim_solar, -0.5, 1.5), 0.0)
    solar_ramp = np.abs(np.diff(solar, prepend=np.nan))

    samples = pd.DataFrame({
        'y': demand[target_pos],
        'actual_raw': hourly.actual_raw.values[target_pos],
        'kpx_da': hourly.jeju_est_demand_da.values[target_pos],
        'h': horizon_col.astype(np.int16),
        'lag168': demand[target_pos - 168],
        'rec24': np.repeat(rec24[origin_positions], n_h),
        'rec168': np.repeat(rec168[origin_positions], n_h),
        'cap_btmppa_mw': hourly.cap_btmppa_mw.values[target_pos],
        'day_type': hourly.day_type.values.astype(object)[target_pos],
        'timestamp': index.values[target_pos],
    })
    for w in WX:
        samples[w] = weather[w][target_pos]
    within_48h = samples.h <= 48
    for c in CLOUD:
        samples[c] = np.where(within_48h, weather[c][target_pos], np.nan)
    samples['solar_deficit'] = np.where(within_48h, solar_deficit[target_pos], np.nan)
    samples['solar_ramp'] = np.where(within_48h, solar_ramp[target_pos], np.nan)
    ts = pd.DatetimeIndex(samples.timestamp)
    samples['hour'] = ts.hour
    samples['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24); samples['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
    samples['dow_sin'] = np.sin(2 * np.pi * ts.dayofweek / 7); samples['dow_cos'] = np.cos(2 * np.pi * ts.dayofweek / 7)
    samples['month_sin'] = np.sin(2 * np.pi * ts.month / 12); samples['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
    samples['day_type'] = pd.Categorical(samples.day_type, categories=DAYTYPE_CATS)
    return samples


def forecast_weather(hourly, forecast, climatology):
    """시각별 D+1 예보(전날 12z 발표) 기상 배열. 빈 시각은 3h 이내 보간 → (월,시) 기후값."""
    fc = forecast.set_index('timestamp').sort_index()
    fc = fc[~fc.index.duplicated(keep='last')]
    columns = {
        'temp_c': [f'temp_{s}' for s in STATIONS],
        'humidity': [f'reh_{s}' for s in STATIONS],
        'wind_spd': [f'wind_spd_10m_{s}' for s in STATIONS],
        'solar_rad': [f'radiation_{s}' for s in SOLAR_ST],
    }
    weather, source = {}, None
    month, hour = hourly.index.month, hourly.index.hour
    for name in WX + CLOUD:
        cols = columns.get(name, [name])
        value = fc[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).reindex(hourly.index)
        value = value.interpolate(method='time', limit=3, limit_area='inside')
        if name == 'temp_c':
            source = np.where(value.notna(), 'forecast', 'climatology')
        clim = climatology[name].reindex(pd.MultiIndex.from_arrays([month, hour])).values
        weather[name] = np.where(value.notna(), value.values, clim)
    return weather, source


# ── 3. 지표 ───────────────────────────────────────────────────────────────
def mape(actual, pred): return float(np.mean(np.abs(actual - pred) / actual) * 100)
def mae(actual, pred): return float(np.mean(np.abs(actual - pred)))
def bias(actual, pred): return float(np.mean(pred - actual))


def summarize(group):
    actual = group.actual_raw.values
    daytime = group[(group.hour >= 8) & (group.hour <= 16)]
    daily = group.groupby(group.timestamp.dt.date).apply(
        lambda g: pd.Series({'model': mape(g.actual_raw, g.pred), 'kpx': mape(g.actual_raw, g.kpx_da)}))
    return pd.Series({
        '시간수': len(group), '일수': len(daily),
        '모델_MAPE': mape(actual, group.pred.values), 'KPX_MAPE': mape(actual, group.kpx_da.values),
        '모델_MAE': mae(actual, group.pred.values), 'KPX_MAE': mae(actual, group.kpx_da.values),
        '모델_편향': bias(actual, group.pred.values), 'KPX_편향': bias(actual, group.kpx_da.values),
        '모델_낮MAPE': mape(daytime.actual_raw, daytime.pred), 'KPX_낮MAPE': mape(daytime.actual_raw, daytime.kpx_da),
        '모델승률_일': float((daily.model < daily.kpx).mean() * 100),
        '평균실측': float(actual.mean()),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=P.DB_JEJU)
    args = parser.parse_args()
    print('DB:', args.db)
    hourly, forecast = load_hourly(args.db)
    index = hourly.index
    n = len(index)
    month, hour = index.month, index.hour

    # 일사 평년·기후값은 학습 구간(≤2025-12)만으로 — 2026 정보 차단
    train_mask = index <= FINAL_TRAIN_END
    solar_train = pd.Series(np.where(train_mask, hourly.solar_rad.values, np.nan), index=index)
    clim_solar_by_mh = solar_train.groupby([month, hour]).mean()
    clim_solar = clim_solar_by_mh.reindex(pd.MultiIndex.from_arrays([month, hour])).values
    train_part = hourly[train_mask]
    climatology = {w: train_part.groupby([train_part.index.month, train_part.index.hour])[w].mean()
                   for w in WX + CLOUD}

    # 학습 샘플: 실측 기상, h=1..120, 타깃이 최종 학습창(≤2025-12) 안
    actual_weather = {w: hourly[w].values.astype(float) for w in WX + CLOUD}
    positions = np.arange(n)
    last_train_pos = int(np.searchsorted(index, pd.Timestamp(FINAL_TRAIN_END)))
    train_origins = np.where((hour == 23) & (positions >= 167) & (positions + HMAX <= last_train_pos))[0]
    train_samples = build_samples(hourly, actual_weather, clim_solar, train_origins, np.arange(1, HMAX + 1))
    train_samples = train_samples[train_samples.y.notna() & train_samples.lag168.notna()].reset_index(drop=True)
    weights_all = np.where((train_samples.hour >= 8) & (train_samples.hour <= 16), DAY_WEIGHT, 1.0)

    ts = train_samples.timestamp
    fit_mask = (ts <= TRAIN_END).values
    val_mask = ((ts >= VAL_BEG) & (ts <= VAL_END)).values
    dtrain = lgb.Dataset(train_samples.loc[fit_mask, FEAT], train_samples.y[fit_mask],
                         weight=weights_all[fit_mask], categorical_feature=CAT)
    dval = lgb.Dataset(train_samples.loc[val_mask, FEAT], train_samples.y[val_mask],
                       weight=weights_all[val_mask], categorical_feature=CAT, reference=dtrain)
    model = lgb.train(PARAMS, dtrain, num_boost_round=4000, valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    best_iteration = int(model.best_iteration)
    print(f'best_iteration={best_iteration} (배포본 569)  최종 재적합 행 {len(train_samples)}')
    dfull = lgb.Dataset(train_samples[FEAT], train_samples.y, weight=weights_all, categorical_feature=CAT)
    final_model = lgb.train(PARAMS, dfull, num_boost_round=best_iteration)

    # 평가 샘플: 2026 D+1, 예보 기상
    weather_fc, weather_source = forecast_weather(hourly, forecast, climatology)
    first_test_pos = int(np.searchsorted(index, pd.Timestamp(TEST_BEG)))
    test_origins = np.where((hour == 23) & (positions >= first_test_pos - 1) & (positions + 24 <= n - 1))[0]
    test = build_samples(hourly, weather_fc, clim_solar, test_origins, np.arange(1, 25))
    test['weather_source'] = weather_source[(test_origins[:, None] + np.arange(1, 25)[None, :]).ravel()]
    test['pred'] = final_model.predict(test[FEAT])
    n_all = len(test)
    test = test[test.actual_raw.notna() & test.kpx_da.notna() & (test.kpx_da > 0)].reset_index(drop=True)
    test['timestamp'] = pd.to_datetime(test.timestamp)
    test['월'] = test.timestamp.dt.strftime('%Y-%m')
    test['계절'] = test.timestamp.dt.month.map(SEASON_OF_MONTH)
    print(f'평가 {test.timestamp.min():%Y-%m-%d} ~ {test.timestamp.max():%Y-%m-%d}  '
          f'공통표본 {len(test)}/{n_all}시간  기후값 폴백 {int((test.weather_source != "forecast").sum())}시간')

    monthly = test.groupby('월').apply(summarize)
    seasonal = test.groupby('계절').apply(summarize).reindex(SEASON_ORDER)
    seasonal.insert(0, '포함월', [', '.join(sorted({m[-2:] for m in test[test.계절 == s].월})) for s in SEASON_ORDER])
    overall = summarize(test).to_frame('2026 전체').T
    for table in (monthly, seasonal, overall):
        table['MAPE개선_%p'] = table.KPX_MAPE - table.모델_MAPE
        table['MAPE개선율_%'] = table['MAPE개선_%p'] / table.KPX_MAPE * 100

    pd.set_option('display.width', 250)
    cols = ['일수', '모델_MAPE', 'KPX_MAPE', 'MAPE개선_%p', '모델_MAE', 'KPX_MAE',
            '모델_편향', 'KPX_편향', '모델_낮MAPE', 'KPX_낮MAPE', '모델승률_일']
    print('\n=== 월별 ===');   print(monthly[cols].round(2).to_string())
    print('\n=== 계절별 ==='); print(seasonal[['포함월'] + cols].round(2).to_string())
    print('\n=== 전체 ===');   print(overall[cols].round(2).to_string())

    monthly.round(3).to_csv(os.path.join(TAB, '2026_vs_kpx_monthly.csv'), encoding='utf-8-sig')
    seasonal.round(3).to_csv(os.path.join(TAB, '2026_vs_kpx_seasonal.csv'), encoding='utf-8-sig')
    overall.round(3).to_csv(os.path.join(TAB, '2026_vs_kpx_overall.csv'), encoding='utf-8-sig')
    test[['timestamp', 'actual_raw', 'pred', 'kpx_da', 'weather_source']].to_csv(
        os.path.join(TAB, '2026_vs_kpx_hourly.csv'), index=False, encoding='utf-8-sig')
    plot_monthly(monthly)


# ── 4. 그림 — 월별 MAPE (계절 구획) ─────────────────────────────────────
def plot_monthly(monthly):
    # 보고서(본문 13pt) 삽입용 — 본문 폭(~16cm)에 줄여 넣어도 글자가 본문과 비슷하게 보이도록
    # 그림을 작게 잡고 글자는 크고 굵게.
    mpl.rcParams['font.family'] = 'Malgun Gothic'; mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.weight'] = 'bold'
    color_model, color_kpx = '#2a78d6', '#898781'   # pages/common.py 의 demand / reference 색
    text_primary, text_muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    months = list(monthly.index)
    x = np.arange(len(months)); width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars_kpx = ax.bar(x - width / 2 - 0.01, monthly.KPX_MAPE, width, color=color_kpx, label='전력거래소 예측모델')
    bars_model = ax.bar(x + width / 2 + 0.01, monthly.모델_MAPE, width, color=color_model, label='신규 예측모델')
    for bars in (bars_kpx, bars_model):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_muted)
    # 계절 구획선·이름
    seasons = [SEASON_OF_MONTH[int(m[-2:])] for m in months]
    top = max(monthly.KPX_MAPE.max(), monthly.모델_MAPE.max()) * 1.22
    start = 0
    for i in range(1, len(months) + 1):
        if i == len(months) or seasons[i] != seasons[start]:
            if i < len(months):
                ax.axvline(i - 0.5, color=grid, linewidth=1, zorder=0)
            ax.text((start + i - 1) / 2, top * 0.97, seasons[start], ha='center', va='top',
                    fontsize=14, fontweight='bold', color=text_primary)
            start = i
    labels = [f"{int(m[-2:])}월" for m in months]
    if months and int(months[-1][-2:]) == 10:
        labels[-1] = labels[-1] + '\n(부분)'
    ax.set_xticks(x); ax.set_xticklabels(labels, color=text_primary, fontsize=13, fontweight='bold')
    ax.set_ylim(0, top); ax.set_ylabel('MAPE (%)', color=text_primary, fontsize=13, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold', color=text_primary)
    ax.yaxis.grid(True, color=grid, linewidth=0.8); ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(grid); ax.tick_params(length=0)
    ax.set_title('2026년 월별 전력 수요예측 오차비교(하루 전)', loc='left',
                 color=text_primary, fontsize=15, fontweight='bold', pad=36)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              prop={'size': 13, 'weight': 'bold'}, handlelength=1.2, columnspacing=1.6)
    fig.tight_layout()
    out = os.path.join(FIG, '2026_vs_kpx_monthly_mape.png')
    fig.savefig(out, facecolor='white'); plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
