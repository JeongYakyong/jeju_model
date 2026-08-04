"""제주 Solar/Wind → net_load 통합 하이브리드 서빙 (3cmp-F).

채널 분리 구성(2026-06-08, 사용자 확정 — 3cmp-G 결과):
  - SOLAR = PatchTST direct (D+1 기존 + D+2~D+7 신규). D+8 이상은 LGBM 폴백.
  - WIND  = LGBM 전 지평(D+1~). PatchTST wind는 forecast 풍속오차 증폭으로 미사용.
  - net_load = 수요(forecast) − solar_gen − wind_gen.
단일 진입점으로 D+1~D+7(제주)을 forecast 테이블에 UPSERT.

solar PatchTST direct: 발행 origin(23:00)까지의 과거(historical) + 대상일 forecast 기상.
  학습 offset((n-1)*24h)이 origin↔target 간격을 메우므로 누수 없음(재귀 아님).
wind/capacity/demand/기상 폴백은 serve_solarwind_lgbm(LGBM) 자산 재사용.

출력(forecast, _lh 접미사 = 하이브리드 공식 다지평 출력. D+1 PatchTST est_*_jeju, LGBM est_*_jeju_lgbm 과 분리):
  est_solar_util_jeju_lh, est_wind_util_jeju_lh, est_solar_gen_jeju_lh,
  est_wind_gen_jeju_lh, est_net_load_jeju_lh

API: predict_hybrid_to_db(origin, horizons=(1..7)) / backfill_hybrid_to_db(start,end)
(원본: forecastmodel/03_jeju_solarwind_forecaster/serve_solarwind_hybrid.py)
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd, torch
import pvlib

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import project_paths as P   # 저장소 안의 모든 경로는 여기 한곳에 모아 둔다

from forecasting import patchtst                             # PatchTST 모델·스케일러·메타·헬퍼
from forecasting import serve_solarwind_lgbm as lgbm_serve   # LGBM wind/capacity/demand/폴백

PKL = P.DIR_MODELS_SOLARWIND_PATCHTST_HORIZON   # solar direct D+2~D+7 .pth 폴더
SOLAR_PT_HORIZONS = [2, 3, 4, 5]            # direct 학습 지평 (D+1은 patchtst.py 가 로드)
# 2026-07-30 재학습: solar D+1~D+5 를 새 스케일러로 다시 학습 → 지평을 운영 상한(5일)에 맞췄다.
# ★D+6·D+7 .pth 는 남아 있어도 **옛 스케일러 기준**이라 새 D+1~D+5 와 섞으면 안 된다.
#   예보 수집도 --days 5 라 D+6/7 은 서빙 입력이 기후값 폴백뿐이었다.
APPLY_TCOG = True                            # 대류일(tcog>0) 후처리 보정 토글(3cmp-3)
TCOG_JSON = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'tcog_postproc.json')
JEJU_HORIZONS = (1, 2, 3, 4, 5, 6, 7)

# 야간 0 마스크(사용자 확정 2026-06-17): pvlib 태양 고도 < 5° 면 태양광 강제 0.
# 모델은 해질녘/밤에 가짜 이용률(겨울 18h ~0.15, 최대 92MW)을 흘림 — 천문 일출일몰로 차단.
JEJU_LAT, JEJU_LON, SOLAR_ELEV_MIN = 33.38, 126.55, 5.0   # 제주 남/서 태양광권역 대표 좌표

# 풍력 입력 풍속 QM 보정(사용자 확정 2026-06-23): NWP 예보 풍속 → 학습 분포(실측) 분위수 매핑.
# 풍력 LGBM 은 실측 풍속으로 학습했으나 서빙은 NWP 를 먹어 분포 불일치(특히 east +1.4m/s 과대)
# → 이용률 +7.5%p 과대예측. QM 으로 입력을 학습 분포에 정합. 풍력 입력만 보정(태양광 무관).
# 검증(전 기간 OOF, 단지평): nMAE 13.57→11.27%·bias +7.5→+1.4%p. 강풍(실측≥12)은 NWP 한계로
# 보정 못 함(가스 관점 보수적). 적합=training/fit_wind_qm.py, 상세=training/REPORT_wind_qm.md.
APPLY_WIND_QM = True
WIND_QM_JSON = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'wind_qm.json')
_WQM = None

# 태양광 일 스케일링 후처리(2026-07-31 이식 — 구 파이프라인에 있던 방식을 되살렸다).
# 예보는 흐린날 일사를 크게 과대예측한다(실측 대비 bias +0.41). 실측으로 학습하는 모델은
# 이걸 못 고쳐 흐린날 이용률을 과대예측하고, 그건 **net_load 과소예측 = 발전 준비 부족**이라
# 업무상 가장 위험한 방향이다. 시점별 QM 으로는 못 잡는다 — 같은 예보값이 흐린날에도
# 맑은날에도 나오기 때문이다. 그래서 **하루 단위**(그날 예보 일사 P75)로 그날 성격을 재고
# sigmoid 스케일을 곱한다. min(scale,1) 이라 **낮추기만 하고 절대 키우지 않는다**.
# 검증(2026-05-16~06-30): 흐림 bias 0.067→0.013, 맑음 bias 불변, 과대율 51.3→41.6%.
# 파라미터는 지평별이고 solar_scale.json 이 SSOT — 재적합은 forecasting/fit_solar_scale.py.
APPLY_SOLAR_SCALE = True
SOLAR_SCALE_JSON = os.path.join(P.DIR_MODELS_SOLARWIND_LGBM, 'solar_scale.json')
_SSCALE = None


def _wind_qm():
    """QM 보정표 로드(메모이즈). {station: (fc_q, obs_q)}. 없거나 끄면 {}."""
    global _WQM
    if _WQM is not None:
        return _WQM
    if not (APPLY_WIND_QM and os.path.exists(WIND_QM_JSON)):
        _WQM = {}; return _WQM
    d = json.load(open(WIND_QM_JSON, encoding='utf-8'))
    _WQM = {st: (np.asarray(v['fc_q'], float), np.asarray(v['obs_q'], float))
            for st, v in d['stations'].items()}
    return _WQM


def _apply_wind_qm(wx):
    """서빙 풍속(예보)을 실측 분포로 분위수 매핑(단조). NaN(기후값 폴백)은 통과.
    build_features 직전에 호출 — wind_zone_east 도 보정된 east 에서 파생됨."""
    qm = _wind_qm()
    if not qm:
        return wx
    wx = wx.copy()
    for st in ('west', 'east'):
        col = f'wind_spd_{st}'
        if col in wx.columns and st in qm:
            fc_q, obs_q = qm[st]
            v = pd.to_numeric(wx[col], errors='coerce').values.astype(float)
            m = np.isfinite(v)
            v[m] = np.clip(np.interp(v[m], fc_q, obs_q), 0, None)
            wx[col] = v
    return wx


def _solar_scale_cfg():
    """일 스케일링 설정 로드(메모이즈). (판정지표 정의, {horizon:(mid,k,floor)}). 끄면 (None,{})."""
    global _SSCALE
    if _SSCALE is not None:
        return _SSCALE
    if not (APPLY_SOLAR_SCALE and os.path.exists(SOLAR_SCALE_JSON)):
        _SSCALE = (None, {})
        return _SSCALE
    d = json.load(open(SOLAR_SCALE_JSON, encoding='utf-8'))
    stat = d.get('statistic') or {}
    cfg = {'columns': stat.get('columns', ['radiation_west', 'radiation_south']),
           'reduce': stat.get('reduce', 'min'),
           'quantile': float(stat.get('quantile', 0.60))}
    params = {int(h): (float(v['mid']), float(v['k']), float(v['floor']))
              for h, v in d.get('params', {}).items()}
    _SSCALE = (cfg, params)
    return _SSCALE


def _apply_solar_daily_scale(con, idx, su, n):
    """그날 예보 일사로 sigmoid 스케일을 만들어 태양광 이용률을 낮춘다.

    시점별이 아니라 **하루 단위**로 판단하는 게 요점이다 — 개별 시각 예보는 틀려도 하루
    분포는 그날이 흐린지를 훨씬 잘 담는다. 시점별 QM 이 실패한 이유가 여기 있다(같은
    예보값이 흐린날에도 맑은날에도 나온다).

    판정지표는 solar_scale.json 이 정한다 — 기본은 west·south 중 **어두운 쪽**(min)의
    낮시간 **60분위**. 지점 조합·분위수를 전부 탐색해 고른 값이라 코드에 박지 않는다.
    scale 은 min(...,1) 이라 **절대 키우지 않고**, floor 아래로도 내려가지 않는다.
    반환 (보정된 su, 적용한 scale 또는 None).
    """
    cfg, params = _solar_scale_cfg()
    if not cfg or n not in params:
        return su, None
    cols = [c for c in cfg['columns']]
    try:
        sel = ', '.join(f'"{c}"' for c in ['timestamp'] + cols)
        t = pd.read_sql(
            f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp', con,
            params=(idx[0].strftime('%Y-%m-%d %H:%M:%S'), idx[-1].strftime('%Y-%m-%d %H:%M:%S')),
            parse_dates=['timestamp']).set_index('timestamp')
    except Exception:
        return su, None
    rad_cols = t[cols].apply(pd.to_numeric, errors='coerce')
    rad = rad_cols.min(axis=1) if cfg['reduce'] == 'min' else rad_cols.mean(axis=1)
    daytime = rad[rad > 0].values
    if daytime.size == 0:
        return su, None
    stat = float(np.quantile(daytime, cfg['quantile']))
    mid, k, floor = params[n]
    scale = float(min(floor + (1.0 - floor) / (1.0 + np.exp(-k * (stat - mid))), 1.0))
    return np.clip(su * scale, 0, 1), scale


def _daylight_mask(idx) -> np.ndarray:
    """태양 고도 >= SOLAR_ELEV_MIN(5°) 인 시각 = 낮(True). pvlib 기준, 입력은 KST 가정."""
    t = pd.DatetimeIndex(idx)
    t = t.tz_localize('Asia/Seoul') if t.tz is None else t.tz_convert('Asia/Seoul')
    elev = pvlib.solarposition.get_solarposition(t, JEJU_LAT, JEJU_LON)['apparent_elevation'].values
    return elev >= SOLAR_ELEV_MIN
PL = 24

OUT = dict(su='est_solar_util_jeju_lh', wu='est_wind_util_jeju_lh', sg='est_solar_gen_jeju_lh',
           wg='est_wind_gen_jeju_lh', nl='est_net_load_jeju_lh')
OUT_COLS = list(OUT.values())

_HA = None


def _assets():
    """PatchTST solar(D+1~D+6) + LGBM 자산."""
    global _HA
    if _HA is not None:
        return _HA
    solar1, _wind1, sc_solar, _scw, md, device = patchtst.load_assets()
    solar_models = {1: solar1}
    for n in SOLAR_PT_HORIZONS:
        p = os.path.join(PKL, f'best_patchtst_solar_model_D{n}.pth')
        if not os.path.exists(p):
            continue
        m = patchtst.PatchTST_Weather_Model(num_features=len(md['features_solar']),
                                      seq_len=md['SEQ_LEN_SOLAR'], pred_len=PL, **patchtst.SOLAR_HP).to(device)
        m.load_state_dict(torch.load(p, map_location=device)); m.eval()
        solar_models[n] = m
    lgbm_assets = lgbm_serve.load_assets()   # (m_solar, m_wind, meta, clim, wx_clim)
    betas = json.load(open(TCOG_JSON, encoding='utf-8')) if (APPLY_TCOG and os.path.exists(TCOG_JSON)) else None
    _HA = (solar_models, sc_solar, md, device, lgbm_assets, betas)
    return _HA


def _apply_tcog(con, idx, su, wu, betas):
    """대류일 후처리: corrected = clip(pred + beta*tcog_station, 0,1). tcog 없으면 무보정.
    지점은 잔차적합으로 선택(3cmp-3): solar=tcog_south, wind=tcog_east(west는 모델 주피처라 잉여)."""
    if betas is None:
        return su, wu, False
    s_st = betas.get('solar_tcog', 'south'); w_st = betas.get('wind_tcog', 'east')
    sel = ', '.join(f'"{c}"' for c in ['timestamp', f'tcog_{s_st}', f'tcog_{w_st}'])
    try:
        t = pd.read_sql(f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp', con,
                        params=(idx[0].strftime('%Y-%m-%d %H:%M:%S'), idx[-1].strftime('%Y-%m-%d %H:%M:%S')),
                        parse_dates=['timestamp']).set_index('timestamp').apply(pd.to_numeric, errors='coerce').reindex(idx)
    except Exception:
        return su, wu, False
    tcs = t[f'tcog_{s_st}'].fillna(0).clip(lower=0).values
    tcw = t[f'tcog_{w_st}'].fillna(0).clip(lower=0).values
    su2 = np.clip(su + betas['solar_beta'] * tcs, 0, 1)
    wu2 = np.clip(wu + betas['wind_beta'] * tcw, 0, 1)
    applied = bool((tcs > 0).any() or (tcw > 0).any())
    return su2, wu2, applied


# =============================================================================
# SOLAR PatchTST direct — 발행 origin 기준 D+n 대상일 24h
# =============================================================================
def _build_solar_direct(con, origin, n, seq_len):
    d = pd.Timestamp(origin).normalize() + pd.Timedelta(days=n)   # 대상일 00:00
    offset = (n - 1) * 24
    first = d - pd.Timedelta(hours=offset + seq_len)
    last_past = d - pd.Timedelta(hours=offset + 1)                # = origin 23:00
    fut_end = d + pd.Timedelta(hours=PL - 1)
    s = lambda t: t.strftime('%Y-%m-%d %H:%M:%S')

    hist_cols, fore_map = [], {}
    for st in patchtst.SOLAR_STATIONS:
        hist_cols += [f'solar_rad_{st}', f'total_cloud_{st}', f'midlow_cloud_{st}', f'rainfall_{st}']
        fore_map[f'radiation_{st}'] = f'solar_rad_{st}'
        fore_map[f'total_cloud_{st}'] = f'total_cloud_{st}'
        fore_map[f'midlow_cloud_{st}'] = f'midlow_cloud_{st}'
        fore_map[f'rainfall_{st}'] = f'rainfall_{st}'
    util_col = 'real_solar_utilization_jeju'

    past = patchtst._read_hist(con, s(first), s(last_past), hist_cols + [util_col])
    fore = patchtst._read_fore(con, s(d), s(fut_end), list(fore_map))
    fore = fore.apply(pd.to_numeric, errors='coerce').rename(columns=fore_map)
    if len(past) < seq_len or past[util_col].isna().any():
        raise ValueError(f'past 부족/NaN ({len(past)}/{seq_len})')
    if len(fore) != PL or fore[hist_cols].isna().any().any():
        raise ValueError(f'forecast {d.date()} {PL}행/결측')

    combined = pd.concat([past[hist_cols], fore[hist_cols]]).sort_index()
    combined = combined.interpolate(limit=3).ffill().bfill()
    patchtst._add_time_feats(combined)
    for st in patchtst.SOLAR_STATIONS:
        patchtst._add_solar_damping(combined, st)
    past_idx = combined.index[combined.index <= last_past]
    fut_idx = combined.index[combined.index >= d]
    return combined, past_idx, fut_idx, past[util_col]


def _solar_util(con, origin, n, assets):
    """D+n solar 이용률 24h. PatchTST(가능시) 또는 LGBM 폴백. → (util, src)."""
    solar_models, sc_solar, md, device, lgbm_assets, _betas = assets
    target_day = pd.Timestamp(origin).normalize() + pd.Timedelta(days=n)
    if n in solar_models:
        try:
            combined, past_idx, fut_idx, past_util = _build_solar_direct(con, origin, n, md['SEQ_LEN_SOLAR'])
            util = patchtst._infer(solar_models[n], sc_solar, combined, past_idx, fut_idx, past_util,
                             md['future_features_solar'], 'Solar_Utilization', md['SEQ_LEN_SOLAR'], device)
            return util, 'patchtst'
        except Exception:
            pass
    # 폴백: LGBM solar (forecast 기상 → build_features)
    m_solar, _mw, meta, clim, wx_clim = lgbm_assets
    wx, _src = lgbm_serve._day_weather(con, target_day, wx_clim)
    feat, _ = lgbm_serve.build_features(wx, clim=clim)
    return np.clip(m_solar.predict(feat[meta['SOLAR_FINAL']]), 0, 1), 'lgbm'


def _wind_util(con, origin, n, assets):
    """D+n wind 이용률 24h — LGBM."""
    _sm, _scs, _md, _dev, lgbm_assets, _betas = assets
    _ms, m_wind, meta, clim, wx_clim = lgbm_assets
    d = pd.Timestamp(origin).normalize() + pd.Timedelta(days=n)
    wx, src = lgbm_serve._day_weather(con, d, wx_clim)
    wx = _apply_wind_qm(wx)   # NWP 풍속 → 실측 분포 QM 보정(풍력 입력만)
    feat, _ = lgbm_serve.build_features(wx, clim=clim)
    return np.clip(m_wind.predict(feat[meta['WIND_FINAL']]), 0, 1), src


def _predict_day(con, origin, n, assets):
    """지평 n(D+n)의 24시간 태양광·풍력 이용률·발전·net_load 를 산출한다.

    태양광=PatchTST(폴백 LGBM)·풍력=LGBM → 대류일(tcog) 후처리 → **일 스케일링**(흐린날
    과대예측 억제) → 야간 0 마스크(태양고도<5°) → 용량 곱해 발전량, net_load=수요−태양광−풍력.
    반환 (out, 태양광소스, 풍력소스, 수요소스).
    """
    target_day = pd.Timestamp(origin).normalize() + pd.Timedelta(days=n)
    idx = pd.date_range(target_day, periods=PL, freq='h')
    solar_util, solar_src = _solar_util(con, origin, n, assets)
    wind_util, wind_src = _wind_util(con, origin, n, assets)
    solar_util, wind_util, tcog_on = _apply_tcog(con, idx, solar_util, wind_util, assets[5])
    if tcog_on:
        solar_src += '+tcog'; wind_src += '+tcog'
    # 일 스케일링은 tcog 다음·야간마스크 앞이다. tcog 가 더한 뒤의 최종 이용률을 낮춰야
    # 흐린날 과대분이 실제로 빠지고, 야간은 어차피 0 이 되므로 순서상 앞에 와야 한다.
    solar_util, day_scale = _apply_solar_daily_scale(con, idx, solar_util, n)
    if day_scale is not None and day_scale < 1.0:
        solar_src += f'+scale{day_scale:.2f}'
    solar_util = np.where(_daylight_mask(idx), solar_util, 0.0)   # 야간(태양고도<5°) 태양광 강제 0
    solar_cap = lgbm_serve._latest_capacity(con, target_day, 'real_solar_gen_jeju', 'real_solar_capacity_jeju')
    wind_cap = lgbm_serve._latest_capacity(con, target_day, 'real_wind_gen_jeju', 'real_wind_capacity_jeju')
    if solar_cap is None or wind_cap is None:
        raise ValueError('capacity 추정 불가')
    solar_gen, wind_gen = solar_util * solar_cap, wind_util * wind_cap
    demand, demand_src = lgbm_serve._demand(con, idx)
    net_load = demand - solar_gen - wind_gen
    out = pd.DataFrame({'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                        OUT['su']: solar_util.round(4), OUT['wu']: wind_util.round(4),
                        OUT['sg']: solar_gen.round(3), OUT['wg']: wind_gen.round(3), OUT['nl']: np.round(net_load, 3)})
    return out, solar_src, wind_src, demand_src


def _upsert(con, out):
    cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
    for c in OUT_COLS:
        if c not in cols:
            con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
    setc = ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS)
    colc = ', '.join(f'"{c}"' for c in ['timestamp'] + OUT_COLS)
    ph = ', '.join(['?'] * (1 + len(OUT_COLS)))
    rows = [tuple([r['timestamp']] + [None if pd.isna(r[c]) else float(r[c]) for c in OUT_COLS])
            for _, r in out.iterrows()]
    con.executemany(f'INSERT INTO forecast ({colc}) VALUES ({ph}) '
                    f'ON CONFLICT("timestamp") DO UPDATE SET {setc}', rows)


def predict_hybrid_to_db(origin, horizons=JEJU_HORIZONS, write=True, verbose=True) -> pd.DataFrame:
    assets = _assets()
    o = pd.Timestamp(origin).normalize()
    outs = []
    with patchtst._conn() as con:
        for n in horizons:
            try:
                out, ssrc, wsrc, dsrc = _predict_day(con, o, n, assets)
            except Exception as e:
                if verbose: print(f'  skip D+{n}: {str(e)[:60]}')
                continue
            o2 = out.copy(); o2.insert(1, 'horizon', n)
            o2['solar_src'] = ssrc; o2['wind_src'] = wsrc; outs.append(o2)
            if write:
                _upsert(con, out)
            if verbose:
                nl = out[OUT['nl']]
                print(f'  D+{n} {(o+pd.Timedelta(days=n)).date()} | solar={ssrc} wind={wsrc} dem={dsrc} | '
                      f"net_load {('NaN' if nl.isna().all() else f'{nl.mean():.0f}MW')}")
        if write:
            con.commit()
    if verbose:
        print(f'[DB] forecast ← origin {o.date()} | {len(outs)}지평 ({"write" if write else "no-write"})')
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def backfill_hybrid_to_db(start, end, horizons=JEJU_HORIZONS, verbose=True):
    assets = _assets()
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')
    done = 0
    with patchtst._conn() as con:
        for o in days:
            for n in horizons:
                try:
                    out, *_ = _predict_day(con, o, n, assets)
                    _upsert(con, out); done += 1
                except Exception:
                    pass
        con.commit()
    if verbose:
        print(f'[backfill] {days[0].date()}~{days[-1].date()} | {len(days)}발행일×{len(horizons)}지평, {done}건')


if __name__ == '__main__':
    import argparse
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    p = argparse.ArgumentParser(description='제주 solar/wind 하이브리드 서빙(solar=PatchTST, wind=LGBM)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('origin')
    pp.add_argument('--days', default='1,2,3,4,5,6,7'); pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    bf.add_argument('--days', default='1,2,3,4,5,6,7')
    a = p.parse_args()
    hz = tuple(int(x) for x in a.days.split(','))
    if a.cmd == 'predict':
        predict_hybrid_to_db(a.origin, horizons=hz, write=not a.no_write)
    else:
        backfill_hybrid_to_db(a.start, a.end, horizons=hz)
