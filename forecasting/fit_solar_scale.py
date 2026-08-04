# -*- coding: utf-8 -*-
"""태양광 일 스케일링 파라미터 적합·점검 — 흐린날 과대예측 억제 후처리.

무엇을 왜
================================================================================
예보는 흐린날 일사를 크게 과대예측한다(실측 대비 bias +0.41, 2026-03~06 실측).
실측 기상으로 학습하는 모델은 이 편향을 못 고친다 — 2026-07-30 재학습이 흐린날에서
개선 0% 였던 이유다. 그래서 **모델 출력**을 하루 단위로 눌러 준다:

    scale = min(floor + (1-floor) * sigmoid(k * (dstat - mid)), 1.0)
    보정 이용률 = 예측 이용률 * scale

dstat = 그날 예보 일사(**south 단독**)의 낮시간 **60분위**. 시점별이 아니라 **하루 집계**로
그날 성격을 재는 게 핵심이다 — 같은 예보값이 흐린날에도 맑은날에도 나오므로 시점별
분위수매핑(QM)으로는 안 잡힌다(2026-07-31 실측: QM 은 흐림을 19~39% 만 줄이면서 맑음을
6~12% 악화시켰다). min(...,1) 이라 **낮추기만 하고 절대 키우지 않는다**.

2026-08-04 개정 — 홀드아웃 재검증에서 드러난 두 가지를 고쳤다
================================================================================
① **적합 목표를 편향 0 이 아니라 절반으로.** 흐림 편향의 크기 자체가 시간에 따라
   줄어들어서(적합창 +0.060 vs 홀드아웃 +0.026), 적합창에서 0 까지 누르는 강도를
   그대로 쓰면 이후 구간에서 **과보정**된다. 홀드아웃 실측:
       목표 0    -> 흐림 −0.0287, MAE 대가 +0.0112 (7.7%)
       목표 50%  -> 흐림 −0.0025, MAE 대가 +0.0038 (2.6%)  ← 채택
② **자유도를 걷어냈다.** 지점 조합(south/west/평균/min) 4종이 홀드아웃에서 사실상
   동일했고(MAE 변화 +0.0102~+0.0121), 지평별 floor 는 D+1~D+5 가 같은 값으로 수렴해
   정보가 없었다. → 지표는 south 단독 P60 고정, 파라미터는 **지평 공통 한 벌**.
   (구 버전이 "지표를 잘 골라서 정확도와 위험을 다 잡았다"고 한 것은 그 검증창이
    이미 지표 선택에 쓰여 생긴 착시였다.)
③ **K 도 탐색하지 않는다**(K_FIXED=3.5). 한 번 격자에 넣어 봤더니 적합창 노이즈를
   쫓아 K=12(급경사)를 골랐는데, 같은 적합 목표에서 K 만 바꿔 재면 분할 4개 전부에서
   K=3.5 가 이겼다(MAE 대가 +0.0041~0.0057 vs +0.0068~0.0089).
   → 남는 자유도는 **mid·floor 둘뿐**이다. 자유도를 줄이는 것이 여기서는 정규화다.

★한계: 이 보정은 **낮추기만** 하므로, 예보가 오히려 과소예측하는 구간에서는 손해다.
  2026-07 이 그랬다(흐림 편향 −0.049 — 12~6월 7개월 중 유일한 음수). 파라미터로는
  못 고치는 구조적 한계이니 `--check` 로 주기적으로 편향 부호를 볼 것.

사용
================================================================================
    python forecasting/fit_solar_scale.py --check           # 재적합 없이 점검만 (빠름)
    python forecasting/fit_solar_scale.py                   # 전 구간(9개월)로 재적합
    python forecasting/fit_solar_scale.py --months 4        # 최근 4개월만
    python forecasting/fit_solar_scale.py --no-write        # dry-run

재적합은 **보정을 끈 상태로 과거 예측을 다시 만들어** 적합한다(이중 적용 방지).
base 당 1~2초라 전 구간(~220 base)이면 5분쯤 걸린다.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

DB_PATH = P.DB_JEJU
OUT_JSON = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'solar_scale.json')
SU = 'real_solar_utilization_jeju'
GRID_MID = [0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.1, 1.3, 1.6, 2.0]
GRID_FLOOR = np.arange(0.10, 0.96, 0.05)
# ★K 는 **탐색하지 않고 고정**한다 (2026-08-04 실측).  K 를 격자에 넣었더니 적합창
#   노이즈를 쫓아 K=12(급경사)를 골랐는데, 같은 적합 목표에서 K 만 바꿔 비교하면
#   **분할 4개 전부에서 K=3.5 가 이겼다** — 홀드아웃 MAE 대가 +0.0041~+0.0057 vs
#   K=12 의 +0.0068~+0.0089.  급경사는 임계 근처에서 스케일이 튀어 견고성도 나쁘다.
#   (구 버전이 K 를 5.5 로 고정한 판단 자체는 옳았고, 값만 3.5 로 낮춘다.)
K_FIXED = 3.5
DAY_LO, DAY_HI = 8, 17            # 평가 시간대(낮)
CLOUDY_TH, CLEAR_TH = 0.7, 0.3    # 일평균 실측 운량 기준 흐림/맑음

# ★적합 목표는 "흐림 편향을 0 으로" 가 아니라 **이 비율까지만 줄이기** 다 (2026-08-04 개정).
#   편향의 크기 자체가 시간에 따라 줄어들기 때문에, 적합창에서 0 까지 누르는 강도를 그대로
#   쓰면 이후 구간에서 **과보정**된다.  홀드아웃 실측(적합 2025-12~2026-05 / 검증 06~07):
#     목표 0    -> 홀드아웃 흐림 −0.0287, MAE 대가 +0.0112 (7.7%)
#     목표 50%  -> 홀드아웃 흐림 −0.0025, MAE 대가 +0.0038 (2.6%)   ← 채택
BIAS_TARGET_FRAC = 0.5

# 판정지표 — **south 단독** 낮 P60 (2026-08-04 단순화).
#   지점 조합(south/west/평균/min)을 홀드아웃에서 재평가하니 넷이 사실상 동일했다
#   (MAE 변화 +0.0102~+0.0121).  조합을 전부 탐색해 고르던 것이 지난번 과적합의 원인이라
#   자유도를 없애고 한 지점으로 고정한다.  단일 컬럼이라 reduce 는 min/mean 이 같다.
STAT_COLS = ['radiation_south']
STAT_REDUCE = 'mean'
STAT_Q = 0.60


def scale_of(p75, mid, k, floor):
    return np.minimum(floor + (1.0 - floor) / (1.0 + np.exp(-k * (p75 - mid))), 1.0)


def build_unscaled_predictions(months):
    """보정을 끈 상태로 과거 base 들의 태양광 예측을 다시 만든다."""
    from forecasting import serve_chain as SC
    from forecasting import serve_demand, serve_solarwind
    from forecasting import horizon_backtest as backtest

    serve_solarwind.APPLY_SOLAR_SCALE = False      # ★ 이중 적용 방지
    # 메모이즈 캐시는 **None** 으로 비운다.  `{}` 로 두면 _solar_scale_cfg 가
    # "이미 로드됨"으로 보고 빈 dict 을 그대로 돌려주고, 호출부의
    # `cfg, params = _solar_scale_cfg()` 가 언패킹에서 터진다 → build_base 가 매 base
    # 예외를 내고 아래 `except Exception: continue` 가 그것을 삼켜 **적합이 조용히
    # 0행으로 끝난다** (2026-08-04 발견).  None 이면 APPLY_SOLAR_SCALE=False 를 보고
    # (None, {}) 를 새로 만들어 보정이 정상적으로 꺼진다.
    serve_solarwind._SSCALE = None

    cutoff = (pd.Timestamp.now().normalize() - pd.DateOffset(months=months)).strftime('%Y-%m-%d')
    con = sqlite3.connect(DB_PATH)
    bases = [r[0] for r in con.execute(
        """SELECT DISTINCT base FROM forecast_horizon
           WHERE substr(base,12)='21:00:00' AND base>=? ORDER BY base""", (cutoff,)).fetchall()]
    con.close()
    print(f'[fit] 보정 끈 상태로 예측 재생성 — base {len(bases)}개 ({cutoff} 이후)')

    scratch = os.path.join(tempfile.gettempdir(), 'fit_solar_scale.db')
    sc = backtest.build_scratch(scratch)
    serve_demand._conn = lambda: sqlite3.connect(scratch)
    assets = serve_solarwind._assets()
    rows, t0 = [], time.time()
    for i, base in enumerate(bases, 1):
        try:
            r = SC.build_base(base, sc, assets)
        except Exception:
            continue
        if not r.empty:
            rows.append(r[['timestamp', 'base', 'horizon_d', 'est_solar_util_jeju']])
        if i % 30 == 0:
            print(f'   {i}/{len(bases)}  {time.time()-t0:.0f}s')
    sc.close()
    if not rows:
        raise SystemExit('[ERR] 예측을 만들지 못했다')
    return pd.concat(rows, ignore_index=True)


def attach_context(pred):
    """예측에 예보 P75·실측 이용률·regime 을 붙인다."""
    pred = pred.copy()
    pred['timestamp'] = pd.to_datetime(pred['timestamp'])
    pred['base'] = pred['base'].astype(str)
    con = sqlite3.connect(DB_PATH)
    fc = pd.read_sql(f"""SELECT timestamp, base, horizon_d, {', '.join(STAT_COLS)}
                         FROM forecast_horizon WHERE substr(base,12)='21:00:00'""",
                     con, parse_dates=['timestamp'])
    act = pd.read_sql(f'SELECT timestamp, {SU} su, total_cloud_west tc FROM historical',
                      con, parse_dates=['timestamp'])
    con.close()
    fc['base'] = fc['base'].astype(str)
    m = (pred.merge(fc, on=['timestamp', 'base', 'horizon_d'], how='inner')
             .merge(act, on='timestamp', how='left')
             .dropna(subset=['su', 'est_solar_util_jeju']))
    rad_cols = m[STAT_COLS]
    m['rad'] = rad_cols.min(axis=1) if STAT_REDUCE == 'min' else rad_cols.mean(axis=1)
    m['date'] = m.timestamp.dt.normalize()
    stat = (m[m.rad > 0].groupby(['base', 'horizon_d', 'date'])['rad']
            .quantile(STAT_Q).rename('dstat').reset_index())
    m = m.merge(stat, on=['base', 'horizon_d', 'date'], how='left')
    m = m[(m.timestamp.dt.hour >= DAY_LO) & (m.timestamp.dt.hour <= DAY_HI)].dropna(subset=['dstat'])
    daily_cloud = act.set_index('timestamp').tc.resample('D').mean()
    m['regime'] = np.where(m.date.map(daily_cloud) >= CLOUDY_TH, '흐림',
                  np.where(m.date.map(daily_cloud) <= CLEAR_TH, '맑음', '보통'))
    return m.rename(columns={'est_solar_util_jeju': 'u'})


def metrics(frame, pred_vals):
    err = pred_vals - frame.su.values
    out = {'MAE': float(np.mean(np.abs(err))), '과대율': float((err > 0).mean() * 100)}
    for r in ['흐림', '보통', '맑음']:
        msk = (frame.regime == r).values
        out[r] = float(np.mean(err[msk])) if msk.any() else float('nan')
    return out


def search(frame):
    """적합창 흐림 편향을 **목표치까지만** 줄이는 (mid, k, floor) 한 벌을 고른다.

    두 가지가 2026-08-04 에 바뀌었다:
    ① 목표가 0 이 아니라 `BIAS_TARGET_FRAC`(=0.5)배다. 0 목표는 이후 구간에서
       과보정을 낳는 것이 홀드아웃으로 확인됐다(상단 상수 주석 참고).
    ② **지평별로 나누지 않는다.** 지평별 자유 탐색은 D+1~D+5 가 같은 값으로 수렴해
       정보가 없으면서 자유도만 늘린다(같은 실측). 한 벌을 전 지평에 쓴다.
    ③ K 는 탐색하지 않는다(K_FIXED). 격자에 넣으면 적합창 노이즈를 쫓아 급경사를
       고르는데, 분할 4개 전부에서 홀드아웃이 더 나빴다. 남는 자유도는 mid·floor 둘뿐.

    맑음 훼손·MAE 악화 벌점은 그대로 둔다 — 정확도를 위험과 맞바꾸지 않기 위한 가드다.
    """
    base = metrics(frame, frame.u.values)
    target = BIAS_TARGET_FRAC * base['흐림']
    best = None
    for mid in GRID_MID:
        for fl in GRID_FLOOR:
            adj = np.clip(frame.u.values * scale_of(frame.dstat.values, mid, K_FIXED, fl), 0, 1)
            mt = metrics(frame, adj)
            clear_hurt = max(0.0, abs(mt['맑음']) - abs(base['맑음']))
            mae_hurt = max(0.0, mt['MAE'] - base['MAE'])
            score = abs(mt['흐림'] - target) + 2.0 * clear_hurt + 1.5 * mae_hurt
            if best is None or score < best[0]:
                best = (score, float(mid), float(fl))
    return best[1], K_FIXED, best[2]


def do_check():
    """재적합 없이: 표가 언제 적합됐는지 + 현재 서빙 결과에서 흐림 편향이 남아 있는지."""
    if not os.path.exists(OUT_JSON):
        print(f'[check] 파라미터 파일 없음: {OUT_JSON}')
        return 1
    meta = json.load(open(OUT_JSON, encoding='utf-8'))
    print(f"[check] 적합일 {meta.get('fitted_at')} / 적합창 {meta.get('fit_window')}")
    print(f"        대상 모델 {meta.get('fitted_for')}")
    try:
        age = (datetime.now() - datetime.strptime(meta['fitted_at'], '%Y-%m-%d')).days
        print(f'        적합 후 {age}일 경과')
        if age > 120:
            print('        ⚠ 계절이 한 바퀴 돌았다 — 재적합 권장')
    except Exception:
        age = None

    # 실제 서빙 결과(est_horizon_jeju)에서 흐림 편향이 남아 있는지 본다.
    # ⚠해석 주의: est_horizon_jeju 행은 **그때그때 cron 이 만든 값**이라, 파라미터를
    #   새로 배포한 날 이전 구간은 **옛 파라미터(또는 무보정)** 로 만들어진 값이다.
    #   즉 배포 직후에는 이 수치가 "지금 파라미터의 성능"이 아니다 — 배포일 이후로
    #   충분히 쌓인 뒤에 봐야 한다. (2026-08-04 확인: 7/31 배포 전 구간이 대부분이라
    #   여기 흐림 +0.034 가 사실상 무보정 편향이었다.)
    con = sqlite3.connect(DB_PATH)
    est = pd.read_sql("""SELECT timestamp, horizon_d, est_solar_util_jeju u
                         FROM est_horizon_jeju WHERE substr(base,12)='21:00:00'
                           AND horizon_d BETWEEN 1 AND 5""", con, parse_dates=['timestamp'])
    act = pd.read_sql(f'SELECT timestamp, {SU} su, total_cloud_west tc FROM historical',
                      con, parse_dates=['timestamp'])
    con.close()
    m = est.merge(act, on='timestamp', how='left').dropna(subset=['su', 'u'])
    m = m[(m.timestamp.dt.hour >= DAY_LO) & (m.timestamp.dt.hour <= DAY_HI)]
    fit_end = str(meta.get('fit_window', '')).split('..')[-1][:10]
    fresh = m[m.timestamp > fit_end] if fit_end else m
    if len(fresh) < 300:
        print(f'        적합창 이후 서빙 표본 {len(fresh)}행 — 판단하기 이르다')
        return 0
    daily_cloud = act.set_index('timestamp').tc.resample('D').mean()
    fresh = fresh.copy()
    fresh['regime'] = np.where(fresh.timestamp.dt.normalize().map(daily_cloud) >= CLOUDY_TH, '흐림',
                      np.where(fresh.timestamp.dt.normalize().map(daily_cloud) <= CLEAR_TH, '맑음', '보통'))
    mt = metrics(fresh, fresh.u.values)
    print(f'\n적합창 이후 실제 서빙({fit_end} 초과, {len(fresh)}행) 편향:')
    print(f"  흐림 {mt['흐림']:+.4f}   보통 {mt['보통']:+.4f}   맑음 {mt['맑음']:+.4f}   "
          f"MAE {mt['MAE']:.4f}   과대율 {mt['과대율']:.1f}%")
    if mt['흐림'] > 0.04:
        print('  ⚠ 흐림 편향이 다시 커졌다(>0.04) → 재적합 권장')
    elif mt['흐림'] < -0.04:
        print('  ⚠ 과보정 상태다(<-0.04) → 재적합 권장')
    else:
        print('  흐림 편향이 허용 범위(±0.04) 안이다 — 재적합 불필요')
    return 0


def main():
    ap = argparse.ArgumentParser(description='태양광 일 스케일링 파라미터 적합·점검')
    ap.add_argument('--check', action='store_true', help='재적합 없이 드리프트 점검만')
    ap.add_argument('--months', type=int, default=9,
                    help='최근 N개월로 적합 (기본 9 = forecast_horizon 전 구간). '
                         '적합 기간은 길수록 좋다 — 계절이 한 바퀴 들어와야 한다')
    ap.add_argument('--no-write', action='store_true', help='dry-run')
    a = ap.parse_args()

    if a.check:
        raise SystemExit(do_check())

    m = attach_context(build_unscaled_predictions(a.months))
    # 마지막 30% 를 홀드아웃으로 떼어 **일반화 성능**을 확인한다.
    # 파라미터는 적합창에서만 고르고, 아래 표는 그 파라미터를 홀드아웃에 그대로 적용한 결과다.
    cut = m.timestamp.quantile(0.7)
    fit_part, val_part = m[m.timestamp <= cut], m[m.timestamp > cut]
    print(f'\n적합 {len(fit_part)}행 (~{cut:%Y-%m-%d}) / 홀드아웃 {len(val_part)}행')

    mid, k, fl = search(fit_part)          # 지평 공통 한 벌
    fb = metrics(fit_part, fit_part.u.values)
    print(f'\n적합 결과 (지평 공통): mid={mid} k={k} floor={fl}')
    print(f'  적합창 흐림 편향 {fb["흐림"]:+.4f} → 목표 {BIAS_TARGET_FRAC*fb["흐림"]:+.4f} '
          f'(0 이 아니라 절반까지만 — 과보정 방지)')
    # 서빙(_solar_scale_cfg)이 지평별 dict 을 기대하므로 같은 값을 전 지평에 채운다.
    params = {int(h): {'mid': round(mid, 2), 'k': round(k, 2), 'floor': round(fl, 2)}
              for h in sorted(m.horizon_d.unique())}

    rows = []
    for h in sorted(m.horizon_d.unique()):
        v = val_part[val_part.horizon_d == h]
        if v.empty:
            continue
        vb = metrics(v, v.u.values)
        va = metrics(v, np.clip(v.u.values * scale_of(v.dstat.values, mid, k, fl), 0, 1))
        rows.append({'지평': f'D+{h}', 'n': len(v),
                     '홀드흐림_전': round(vb['흐림'], 4), '홀드흐림_후': round(va['흐림'], 4),
                     '홀드MAE_전': round(vb['MAE'], 4), '홀드MAE_후': round(va['MAE'], 4)})
    print('\n지평별 (홀드아웃 구간 기준 — 파라미터는 적합창에서만 골랐다)')
    print(pd.DataFrame(rows).to_string(index=False))

    adj = np.clip(val_part.u.values * scale_of(val_part.dstat.values, mid, k, fl), 0, 1)
    vb, va = metrics(val_part, val_part.u.values), metrics(val_part, adj)
    print(f"\n홀드아웃 전체  무보정 흐림 {vb['흐림']:+.4f} 맑음 {vb['맑음']:+.4f} "
          f"MAE {vb['MAE']:.4f} 과대율 {vb['과대율']:.1f}%")
    print(f"               보정후 흐림 {va['흐림']:+.4f} 맑음 {va['맑음']:+.4f} "
          f"MAE {va['MAE']:.4f} 과대율 {va['과대율']:.1f}%")
    print(f"               MAE 대가 {va['MAE']-vb['MAE']:+.4f} "
          f"({(va['MAE']/vb['MAE']-1)*100:+.1f}%) / 과대율 감소 {vb['과대율']-va['과대율']:.1f}%p")

    if a.no_write:
        print('\n(--no-write: 저장 생략)')
        return

    old = json.load(open(OUT_JSON, encoding='utf-8')) if os.path.exists(OUT_JSON) else {}
    # 구 스키마 잔재 제거 (old.update 는 안 건드린 키를 남기므로 명시적으로 지운다).
    for stale in ('k_fixed', 'k_note', 'mid_note', 'floor_note', 'result', 'val_window'):
        old.pop(stale, None)
    old.update({
        'fitted_at': datetime.now().strftime('%Y-%m-%d'),
        'fit_window': f'{fit_part.timestamp.min():%Y-%m-%d}..{cut:%Y-%m-%d}',
        'holdout_window': f'{val_part.timestamp.min():%Y-%m-%d}..{val_part.timestamp.max():%Y-%m-%d}',
        'statistic': {'columns': STAT_COLS, 'reduce': STAT_REDUCE, 'quantile': STAT_Q,
                      '_note': 'south 단독 낮 P60 고정(2026-08-04). 지점 조합을 탐색하던 것이 '
                               '과적합 원인이었고, 홀드아웃에서 조합 4종이 사실상 동일했다.'},
        'bias_target_frac': BIAS_TARGET_FRAC,
        'params': {str(h): p for h, p in params.items()},
        'params_note': '지평 공통 한 벌을 전 지평에 복제한다(서빙이 지평별 dict 을 기대). '
                       '지평별 자유 탐색은 D+1~D+5 가 같은 값으로 수렴해 정보가 없었다.',
        'validation': {
            '_note': '홀드아웃 = 적합창 이후 30%. 파라미터는 적합창에서만 골랐다.',
            '무보정': {'흐림_bias': round(vb['흐림'], 4), '맑음_bias': round(vb['맑음'], 4),
                       'MAE': round(vb['MAE'], 4), '과대율_pct': round(vb['과대율'], 1)},
            '이_파라미터': {'흐림_bias': round(va['흐림'], 4), '맑음_bias': round(va['맑음'], 4),
                            'MAE': round(va['MAE'], 4), '과대율_pct': round(va['과대율'], 1)}},
    })
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print(f'\n[OK] 저장 {OUT_JSON}')
    print('     서빙 확인: python forecasting/serve_chain.py --utc 12 --no-write')


if __name__ == '__main__':
    main()
