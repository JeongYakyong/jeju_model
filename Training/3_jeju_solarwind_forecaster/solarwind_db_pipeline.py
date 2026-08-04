"""제주 D+1 Solar/Wind 이용률 → net_load 예측 — DB 전용 파이프라인 (신버전).

================================================================================
구버전 / 신버전 구분  (반드시 헷갈리지 말 것)
================================================================================
  구버전: net_load_forecaster/  패키지 (옛 DB jeju_energy.db, 단일지점 가중치 models/).
          predict()/run_model_prediction() — 손대지 않는다.
  신버전: 이 파일 + solarwind_models/  (새 DB input_data_jeju.db, 3지점 가중치).
          수요의 demand_db_pipeline.py 와 같은 패턴. 완전 독립.

================================================================================
무엇을 하나
================================================================================
input_data_jeju.db 한 곳에서 모두 읽고 쓴다.
  - 과거(past) : historical  — 3지점 관측 기상 + 실측 이용률(past_y)
  - 미래(future): forecast    — D+1 24h 3지점 예보 기상
  - 수요       : forecast.jeju_est_demand_new  (1단계 산출) → net_load 계산용
  - 출력       : forecast 에 아래 컬럼 UPSERT
        est_solar_utilization, est_wind_utilization   (0~1)
        est_solar_gen, est_wind_gen                   (MW)
        est_net_load                                  (MW = 수요 - solar - wind)

학습(3지점, 평균 X)과 동일하게 지점별 피처를 "별도 채널"로 만든다.
train(historical) ↔ serve(forecast) 컬럼 매핑:
  solar_rad_{st} ← solar_rad_{st}    / radiation_{st}      (east 일사 없음 → west,south)
  total_cloud_{st}, midlow_cloud_{st}, rainfall_{st}  동명
  wind_spd_{st}  ← wind_spd_{st}     / wind_spd_10m_{st}   (west,east,south)
  wd_sin/cos_{st}← wd_sin/cos_{st}   / wd_sin/cos_10m_{st}

================================================================================
공개 API
================================================================================
    predict_solarwind_to_db(date)       # D+1 24h 예측 → forecast UPSERT
    backfill_solarwind_to_db(start,end) # 구간 백필
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# =============================================================================
# 0. 경로 / 상수
# =============================================================================
HERE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(
    HERE, '..', '1. data_fetcher_and_db', 'data', 'input_data_jeju.db'))
MODELS  = os.path.join(HERE, 'solarwind_models')   # ← 신버전 3지점 가중치

PRED_LEN = 24
DEMAND_COL = 'jeju_est_demand_new'   # 1단계 수요 예측 (net_load 입력)

# 출력 컬럼 (신버전; _jeju 접미사로 권역 명시 + 구버전 est_Solar_Utilization 과 구분)
OUT_SOLAR_UTIL = 'est_solar_utilization_jeju'
OUT_WIND_UTIL  = 'est_wind_utilization_jeju'
OUT_SOLAR_GEN  = 'est_solar_gen_jeju'
OUT_WIND_GEN   = 'est_wind_gen_jeju'
OUT_NET_LOAD   = 'est_net_load_jeju'
OUT_COLS = [OUT_SOLAR_UTIL, OUT_WIND_UTIL, OUT_SOLAR_GEN, OUT_WIND_GEN, OUT_NET_LOAD]

# 학습과 동일한 아키텍처 하이퍼파라미터 (metadata 에 없는 차원)
SOLAR_HP = dict(patch_len=24, stride=12, d_model=256, num_heads=4,
                num_layers=3, d_ff=1024, dropout=0.2)
WIND_HP  = dict(patch_len=12, stride=6,  d_model=128, num_heads=4,
                num_layers=2, d_ff=256,  dropout=0.3)

# 캐노니컬(학습) 컬럼 ← serve(forecast) 매핑.  past 는 동명(=historical).
SOLAR_STATIONS = ['west', 'south']
WIND_STATIONS  = ['west', 'east']   # south 제외(약상관+예보 과대편차로 serve 악화). 학습과 동일.
WIND_DIR_STATION = 'west'   # 풍향 대표 지점(지점 간 유사 → 공유 1쌍). 학습과 동일.

WIND_SPD_CAP = 20.0
CUTOFF_WIND_SPD = 25.0


def _conn():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# 1. 모델 정의 (production architecture 와 동일 — inst_norm 없음 + weather_bypass)
#    가중치(strict) 호환을 위해 파라미터 이름/구성을 그대로 둔다.
# =============================================================================
class _PatchWeatherAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(key_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, fut_patch, past_patches, transformer_out):
        Q = self.W_Q(fut_patch).unsqueeze(1)
        K = self.W_K(past_patches)
        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        attn = F.softmax(score, dim=-1)
        return torch.bmm(attn, transformer_out).squeeze(1), attn


class PatchTST_Weather_Model(nn.Module):
    weather_attn: _PatchWeatherAttention

    def __init__(self, num_features, seq_len, pred_len=24, patch_len=24, stride=12,
                 d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.2):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.seq_len, self.pred_len = seq_len, pred_len
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len * num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                         dim_feedforward=d_ff, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.num_weather_feats = num_features - 1
        fut_flat = pred_len * self.num_weather_feats
        w_patch = patch_len * self.num_weather_feats
        self.weather_attn = _PatchWeatherAttention(fut_flat, w_patch, d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model + fut_flat, 256), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(256, pred_len))
        self.weather_bypass = nn.Linear(fut_flat, pred_len)

    def forward(self, batch, device='cpu'):
        p_num = batch['past_numeric'].to(device)
        p_y   = batch['past_y'].to(device)
        f_num = batch['future_numeric'].to(device)
        B = p_num.shape[0]
        x_past = torch.cat([p_num, p_y], dim=-1)
        xp = x_past.unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        enc_out = self.transformer_encoder(self.dropout(self.patch_embedding(xp) + self.pos_embedding))
        fut_flat = f_num.reshape(B, -1)
        xw = x_past[..., :-1].unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        context, _ = self.weather_attn(fut_flat, xw, enc_out)
        return self.regressor(torch.cat([context, fut_flat], dim=1)) + self.weather_bypass(fut_flat)


# =============================================================================
# 2. 자산 로드 (메모이즈) — 신버전 solarwind_models/
# =============================================================================
_ASSETS = None


def load_assets(force=False):
    """(solar_model, wind_model, scaler_solar, scaler_wind, metadata, device)."""
    global _ASSETS
    if _ASSETS is not None and not force:
        return _ASSETS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    md = joblib.load(os.path.join(MODELS, 'metadata.pkl'))
    sc_solar = joblib.load(os.path.join(MODELS, 'MinMax_scaler_solar.pkl'))
    sc_wind  = joblib.load(os.path.join(MODELS, 'MinMax_scaler_wind.pkl'))

    solar = PatchTST_Weather_Model(
        num_features=len(md['features_solar']), seq_len=md['SEQ_LEN_SOLAR'],
        pred_len=md['PRED_LEN'], **SOLAR_HP).to(device)
    solar.load_state_dict(torch.load(
        os.path.join(MODELS, 'best_patchtst_solar_model.pth'), map_location=device))
    solar.eval()

    wind = PatchTST_Weather_Model(
        num_features=len(md['features_wind']), seq_len=md['SEQ_LEN_WIND'],
        pred_len=md['PRED_LEN'], **WIND_HP).to(device)
    wind.load_state_dict(torch.load(
        os.path.join(MODELS, 'best_patchtst_wind_model.pth'), map_location=device))
    wind.eval()

    _ASSETS = (solar, wind, sc_solar, sc_wind, md, device)
    return _ASSETS


# =============================================================================
# 3. 입력 윈도우 조립 (past=historical + future=forecast → 캐노니컬 피처)
# =============================================================================
def _read_hist(con, start, end, cols):
    sel = ', '.join(f'"{c}"' for c in ['timestamp'] + cols)
    df = pd.read_sql(f'SELECT {sel} FROM historical WHERE timestamp BETWEEN ? AND ? '
                     f'ORDER BY timestamp', con, params=(start, end),
                     parse_dates=['timestamp']).set_index('timestamp')
    return df.apply(pd.to_numeric, errors='coerce')


def _read_fore(con, start, end, cols):
    sel = ', '.join(f'"{c}"' for c in ['timestamp'] + cols)
    df = pd.read_sql(f'SELECT {sel} FROM forecast WHERE timestamp BETWEEN ? AND ? '
                     f'ORDER BY timestamp', con, params=(start, end),
                     parse_dates=['timestamp']).set_index('timestamp')
    return df


def _add_time_feats(df):
    df['Hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
    df['Hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
    df['Year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
    df['Year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)
    return df


def _add_solar_damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    df[f'solar_damping_{st}'] = np.exp(-0.163 * daily.clip(upper=10))


def _add_wind_feats(df, st):
    # 학습 축소안과 동일: 지점별 wind_spd(clip) + wind_zone 만. sq/cu 제거.
    raw = df[f'wind_spd_{st}']
    cond = [raw < 15, (raw >= 15) & (raw < 20),
            (raw >= 20) & (raw < CUTOFF_WIND_SPD), raw >= CUTOFF_WIND_SPD]
    df[f'wind_zone_{st}'] = np.select(cond, [0.0, 1.0, 0.5, 0.0], default=0.0)
    df[f'wind_spd_{st}'] = raw.clip(upper=WIND_SPD_CAP)


def _build_canonical(con, tgt, seq_len, kind):
    """past(historical)+future(forecast) 합쳐 캐노니컬 raw 컬럼 프레임 반환.

    kind: 'solar' | 'wind'.  반환: (combined_df, past_idx, fut_idx, util_series)
    """
    first = tgt - pd.Timedelta(hours=seq_len)
    last_past = tgt - pd.Timedelta(hours=1)
    fut_end = tgt + pd.Timedelta(hours=PRED_LEN - 1)
    s = lambda t: t.strftime('%Y-%m-%d %H:%M:%S')

    if kind == 'solar':
        stations = SOLAR_STATIONS
        hist_cols, fore_map = [], {}
        for st in stations:
            hist_cols += [f'solar_rad_{st}', f'total_cloud_{st}',
                          f'midlow_cloud_{st}', f'rainfall_{st}']
            fore_map[f'radiation_{st}']    = f'solar_rad_{st}'
            fore_map[f'total_cloud_{st}']  = f'total_cloud_{st}'
            fore_map[f'midlow_cloud_{st}'] = f'midlow_cloud_{st}'
            fore_map[f'rainfall_{st}']     = f'rainfall_{st}'
        util_col = 'real_solar_utilization_jeju'
    else:
        stations = WIND_STATIONS
        hist_cols, fore_map = [], {}
        for st in stations:                       # 지점별 풍속만
            hist_cols += [f'wind_spd_{st}']
            fore_map[f'wind_spd_10m_{st}'] = f'wind_spd_{st}'
        d = WIND_DIR_STATION                      # 공유 풍향(대표 1지점)
        hist_cols += [f'wd_sin_{d}', f'wd_cos_{d}']
        fore_map[f'wd_sin_10m_{d}'] = f'wd_sin_{d}'
        fore_map[f'wd_cos_10m_{d}'] = f'wd_cos_{d}'
        util_col = 'real_wind_utilization_jeju'

    # past: historical (canonical = 동명) + 이용률
    past = _read_hist(con, s(first), s(last_past), hist_cols + [util_col])
    # future: forecast → 캐노니컬 이름으로 rename
    fore = _read_fore(con, s(tgt), s(fut_end), list(fore_map.keys()))
    fore = fore.apply(pd.to_numeric, errors='coerce').rename(columns=fore_map)

    if len(past) < seq_len or past[util_col].isna().any():
        raise ValueError(f'[{kind}] past 윈도우 {first}~{last_past} 부족/NaN '
                         f'(보유 {len(past)}/{seq_len})')
    if len(fore) != PRED_LEN:
        raise ValueError(f'[{kind}] forecast {tgt.date()} {PRED_LEN}행 필요 — 발견 {len(fore)}')

    canon = [c for c in hist_cols]
    combined = pd.concat([past[canon], fore[canon]]).sort_index()
    # 짧은 결측 보간 (학습과 동일: interpolate(limit=3) + 양끝 채움)
    combined = combined.interpolate(limit=3).ffill().bfill()

    # 파생
    _add_time_feats(combined)
    if kind == 'solar':
        for st in stations:
            _add_solar_damping(combined, st)
    else:
        for st in stations:
            _add_wind_feats(combined, st)
        combined['wd_sin'] = combined[f'wd_sin_{WIND_DIR_STATION}']   # 공유 풍향
        combined['wd_cos'] = combined[f'wd_cos_{WIND_DIR_STATION}']

    past_idx = combined.index[combined.index <= last_past]
    fut_idx  = combined.index[combined.index >= tgt]
    return combined, past_idx, fut_idx, past[util_col]


def _latest_capacity(con, tgt, gen_col, cap_col):
    """tgt 직전 720h 의 capacity(최신 유효값) 또는 발전량 rolling max 로 추정."""
    start = (tgt - pd.Timedelta(hours=720)).strftime('%Y-%m-%d %H:%M:%S')
    end   = (tgt - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_sql(f'SELECT "{cap_col}", "{gen_col}" FROM historical '
                     f'WHERE timestamp BETWEEN ? AND ?', con, params=(start, end))
    cap = pd.to_numeric(df[cap_col], errors='coerce').dropna()
    if len(cap):
        return float(cap.iloc[-1])
    gen = pd.to_numeric(df[gen_col], errors='coerce')
    return float(gen.max()) if gen.notna().any() else None


# =============================================================================
# 4. 단일 날짜 추론
# =============================================================================
@torch.no_grad()
def _infer(model, scaler, combined, past_idx, fut_idx, util_series,
           future_features, target_col, seq_len, device):
    feat = combined.copy()
    feat[future_features] = scaler.transform(feat[future_features])
    past_numeric = feat.loc[past_idx, future_features].values[-seq_len:]
    future_numeric = feat.loc[fut_idx, future_features].values[:PRED_LEN]
    past_y = util_series.reindex(past_idx).values[-seq_len:].reshape(-1, 1)
    batch = {
        'past_numeric':   torch.FloatTensor(past_numeric).unsqueeze(0),
        'past_y':         torch.FloatTensor(past_y).unsqueeze(0),
        'future_numeric': torch.FloatTensor(future_numeric).unsqueeze(0),
    }
    pred = model(batch, device=device).squeeze(0).cpu().numpy()
    return np.clip(pred, 0.0, 1.0)


def _predict_core(con, date, assets):
    solar, wind, sc_solar, sc_wind, md, device = assets
    tgt = pd.Timestamp(date).normalize()

    ff_solar = md['future_features_solar']
    ff_wind  = md['future_features_wind']

    c_s, p_s, f_s, u_s = _build_canonical(con, tgt, md['SEQ_LEN_SOLAR'], 'solar')
    solar_util = _infer(solar, sc_solar, c_s, p_s, f_s, u_s,
                        ff_solar, 'Solar_Utilization', md['SEQ_LEN_SOLAR'], device)

    c_w, p_w, f_w, u_w = _build_canonical(con, tgt, md['SEQ_LEN_WIND'], 'wind')
    wind_util = _infer(wind, sc_wind, c_w, p_w, f_w, u_w,
                       ff_wind, 'Wind_Utilization', md['SEQ_LEN_WIND'], device)

    solar_cap = _latest_capacity(con, tgt, 'real_solar_gen_jeju', 'real_solar_capacity_jeju')
    wind_cap  = _latest_capacity(con, tgt, 'real_wind_gen_jeju',  'real_wind_capacity_jeju')
    if solar_cap is None or wind_cap is None:
        raise ValueError(f'[{date}] capacity 추정 불가 (solar={solar_cap}, wind={wind_cap})')

    idx = pd.date_range(tgt, periods=PRED_LEN, freq='h')
    solar_gen = solar_util * solar_cap
    wind_gen  = wind_util * wind_cap

    # 수요(1단계) 읽어 net_load 계산 (없으면 NaN)
    dem = pd.read_sql(f'SELECT timestamp, "{DEMAND_COL}" FROM forecast '
                      f'WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp', con,
                      params=(idx[0].strftime('%Y-%m-%d %H:%M:%S'),
                              idx[-1].strftime('%Y-%m-%d %H:%M:%S')),
                      parse_dates=['timestamp']).set_index('timestamp')
    demand = pd.to_numeric(dem[DEMAND_COL], errors='coerce').reindex(idx).values \
        if DEMAND_COL in dem.columns else np.full(PRED_LEN, np.nan)
    net_load = demand - solar_gen - wind_gen

    out = pd.DataFrame({
        'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
        OUT_SOLAR_UTIL: solar_util.round(4),
        OUT_WIND_UTIL:  wind_util.round(4),
        OUT_SOLAR_GEN:  solar_gen.round(3),
        OUT_WIND_GEN:   wind_gen.round(3),
        OUT_NET_LOAD:   np.round(net_load, 3),
    })
    return out


def _upsert(con, out):
    cols = [c[1] for c in con.execute('PRAGMA table_info(forecast)')]
    for c in OUT_COLS:
        if c not in cols:
            con.execute(f'ALTER TABLE forecast ADD COLUMN "{c}" REAL')
    set_sql = ', '.join(f'"{c}"=excluded."{c}"' for c in OUT_COLS)
    col_sql = ', '.join(f'"{c}"' for c in ['timestamp'] + OUT_COLS)
    ph = ', '.join(['?'] * (1 + len(OUT_COLS)))
    rows = [tuple([r['timestamp']] + [None if pd.isna(r[c]) else float(r[c]) for c in OUT_COLS])
            for _, r in out.iterrows()]
    con.executemany(
        f'INSERT INTO forecast ({col_sql}) VALUES ({ph}) '
        f'ON CONFLICT("timestamp") DO UPDATE SET {set_sql}', rows)


def predict_solarwind_to_db(date, write=True, verbose=True) -> pd.DataFrame:
    """D+1 24h solar/wind 이용률·MW·net_load 예측 → forecast UPSERT."""
    assets = load_assets()
    with _conn() as con:
        out = _predict_core(con, date, assets)
        if write:
            _upsert(con, out)
            con.commit()
            if verbose:
                print(f'[DB] forecast ← {date} 24행 UPSERT ({", ".join(OUT_COLS)})')
    if verbose:
        print(out.to_string(index=False))
    return out


# =============================================================================
# 5. 구간 백필
# =============================================================================
def backfill_solarwind_to_db(start, end, verbose=True) -> pd.DataFrame:
    """[start,end] 의 forecast 보유 날짜(24행) 전부에 solar/wind/net_load 채움."""
    assets = load_assets()
    with _conn() as con:
        fdates = pd.read_sql(
            "SELECT substr(timestamp,1,10) d, COUNT(*) n FROM forecast "
            "WHERE timestamp BETWEEN ? AND ? GROUP BY d HAVING n=24 ORDER BY d",
            con, params=(f'{start} 00:00:00', f'{end} 23:00:00'))['d'].tolist()

        done, skipped = [], []
        for date in fdates:
            try:
                out = _predict_core(con, date, assets)
                _upsert(con, out)
                done.append(date)
            except Exception as e:
                skipped.append((date, str(e)[:70]))
        con.commit()

    if verbose:
        print(f'[backfill] 완료 {len(done)}일 / 건너뜀 {len(skipped)}일')
        if done:
            print(f'  범위: {done[0]} ~ {done[-1]}')
        for d, why in skipped[:10]:
            print(f'  skip {d}: {why}')
    return pd.DataFrame(skipped, columns=['date', 'reason'])


# =============================================================================
# 6. CLI
# =============================================================================
if __name__ == '__main__':
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='제주 solar/wind D+1 예측 (DB 전용, 신버전)')
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('predict'); pp.add_argument('date')
    pp.add_argument('--no-write', action='store_true')
    bf = sub.add_parser('backfill'); bf.add_argument('start'); bf.add_argument('end')
    a = p.parse_args()

    if a.cmd == 'predict':
        predict_solarwind_to_db(a.date, write=not a.no_write)
    elif a.cmd == 'backfill':
        backfill_solarwind_to_db(a.start, a.end)
