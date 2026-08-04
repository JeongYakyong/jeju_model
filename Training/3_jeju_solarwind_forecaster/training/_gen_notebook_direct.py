"""train_solarwind_direct_d2d3_colab.ipynb 생성기 (하이브리드용 direct 지평 D+2/D+3).

배경(2026-06-08, 3단계 비교 결론): 하이브리드 서빙 = PatchTST D+1~D+3(실사용 핵심) +
LGBM-direct D+4~D+6(시연). D+1은 기존 가중치 사용. 여기서는 **D+2/D+3 direct PatchTST**를
추가 학습한다(재귀 롤링 누적오차 회피).

D+1 학습(train_solarwind_3station_colab.ipynb)과 단 하나만 다르다:
  PatchTSTDataset 의 future/target 윈도우를 **offset(D+2=24h, D+3=48h)** 만큼 뒤로 민다.
  (과거 윈도우=origin까지 동일, 아키텍처/피처/손실/스케일러 동일 → 스케일러·metadata 재사용)

산출물: best_patchtst_{solar,wind}_model_D2.pth / _D3.pth (4개).
  → repo `solarwind_models/` 에 추가로 복사(기존 D+1 5파일·스케일러·metadata 불변).

    python "3. jeju_solarwind_forecaster/training/_gen_notebook_direct.py"
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_solarwind_direct_d2d3_colab.ipynb"
CELLS = []
def md(s):  CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# 제주 Solar/Wind 이용률 PatchTST — direct 지평 D+2 / D+3 추가 학습 (하이브리드)

3단계 비교(2026-06-08) 결론에 따른 **하이브리드 서빙**의 PatchTST 쪽 보강.
- 서빙: **D+1~D+3 = PatchTST**(실사용 핵심), **D+4~D+6 = LGBM-direct**(시연).
- D+1 가중치는 기존(`best_patchtst_{solar,wind}_model.pth`) 그대로 사용.
- 이 노트북은 **D+2/D+3 direct** 가중치만 새로 만든다(재귀 롤링 대신 직접 예측 → 누적오차 회피).

**D+1 학습과 차이 = 한 줄:** Dataset 의 future/target 윈도우를 horizon offset 만큼 뒤로 민다
(D+2 = +24h, D+3 = +48h). 과거 윈도우(origin까지)·아키텍처·피처·손실 모두 동일.
→ 스케일러·metadata 는 지평 무관이라 **기존 것 재사용**(이 노트북은 가중치 4개만 저장).

**입력**: `solarwind_raw_jeju.csv` (D+1 학습과 동일). **산출**: `_D2.pth` / `_D3.pth` ×(solar,wind).
""")

code(r"""
import numpy as np, pandas as pd, torch, os
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
from tqdm.auto import tqdm
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE =', DEVICE)
""")

# ── CONFIG (D+1 노트북과 동일 + HORIZONS) ─────────────────────────────────
code(r"""
CSV_PATH = '/content/solarwind_raw_jeju.csv'
OUT_DIR  = '/content/out'
os.makedirs(OUT_DIR, exist_ok=True)

PRED_LEN = 24
# ★ direct 지평: 이름 -> future/target 시작 offset(시간). D+1=0(기존)은 기존 가중치 사용.
#   offset은 반드시 24의 배수(일 경계 정렬)여야 24h 블록이 캘린더 하루와 일치한다.
#   D+2=24, D+3=48, D+4=72, D+5=96, D+6=120.  (108 같은 비배수는 날짜 경계를 가로질러 금지)
#   GPU 켠 김에 전 지평 학습 후, 아래 '지평별 test MAE'를 LGBM과 비교해 PatchTST/LGBM 경계 결정.
HORIZONS = {'D2': 24, 'D3': 48, 'D4': 72, 'D5': 96, 'D6': 120}

TRAIN_END = '2025-03-01'
VAL_END   = '2026-01-01'

SOLAR_STATIONS = ['west', 'south']
WIND_STATIONS  = ['west', 'east']
SOLAR_HP = dict(seq_len=336, patch_len=24, stride=12,
                d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2)
WIND_HP  = dict(seq_len=72,  patch_len=12, stride=6,
                d_model=128, num_heads=4, num_layers=2, d_ff=256,  dropout=0.3)
EPOCHS = 100; BATCH_SIZE = 128; LR = 1e-3
""")

# ── 데이터 로드 + 파생 (D+1 노트북과 동일) ────────────────────────────────
code(r"""
df = pd.read_csv(CSV_PATH); df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
print('rows:', len(df), '| range:', df.index.min(), '->', df.index.max())
df['Hour_sin'] = np.sin(2*np.pi*df.index.hour/24); df['Hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
df['Year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365); df['Year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].interpolate(limit=3); df[num_cols] = df[num_cols].ffill().bfill()

# Solar 피처
def add_solar_damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    df[f'solar_damping_{st}'] = np.exp(-0.163 * daily.clip(upper=10))
for st in SOLAR_STATIONS: add_solar_damping(df, st)
df['Solar_Utilization'] = df['real_solar_utilization_jeju'].clip(0, 1)
future_features_solar = []
for st in SOLAR_STATIONS:
    future_features_solar += [f'solar_rad_{st}', f'total_cloud_{st}', f'midlow_cloud_{st}', f'solar_damping_{st}']
future_features_solar += ['Hour_sin', 'Hour_cos']
features_solar = future_features_solar + ['Solar_Utilization']

# Wind 피처
WIND_SPD_CAP = 20.0; CUTOFF_WIND_SPD = 25.0; DIR_STATION = 'west'
def add_wind_feats(df, st):
    raw = df[f'wind_spd_{st}']
    cond = [raw < 15, (raw >= 15)&(raw < 20), (raw >= 20)&(raw < CUTOFF_WIND_SPD), raw >= CUTOFF_WIND_SPD]
    df[f'wind_zone_{st}'] = np.select(cond, [0.0, 1.0, 0.5, 0.0], default=0.0)
    df[f'wind_spd_{st}'] = raw.clip(upper=WIND_SPD_CAP)
for st in WIND_STATIONS: add_wind_feats(df, st)
df['wd_sin'] = df[f'wd_sin_{DIR_STATION}']; df['wd_cos'] = df[f'wd_cos_{DIR_STATION}']
df['Wind_Utilization'] = df['real_wind_utilization_jeju'].clip(0, 1)
future_features_wind = []
for st in WIND_STATIONS: future_features_wind += [f'wind_spd_{st}', f'wind_zone_{st}']
future_features_wind += ['wd_sin', 'wd_cos', 'Hour_sin', 'Hour_cos', 'Year_sin', 'Year_cos']
features_wind = future_features_wind + ['Wind_Utilization']
print('solar feats', len(future_features_solar), '| wind feats', len(future_features_wind))
""")

# ── ★ Dataset (offset 추가 — D+1과 유일한 차이) ───────────────────────────
code(r"""
class PatchTSTDatasetH(Dataset):
    # future/target 윈도우를 origin 직후가 아니라 +offset 시간 뒤로(직접 지평)
    def __init__(self, data_array, seq_len, pred_len, future_idx, target_idx, offset=0):
        self.data = data_array; self.seq_len = seq_len; self.pred_len = pred_len
        self.future_idx = future_idx; self.target_idx = target_idx; self.offset = offset
    def __len__(self):
        return len(self.data) - self.seq_len - self.offset - self.pred_len + 1
    def __getitem__(self, idx):
        past = self.data[idx: idx + self.seq_len]
        s = idx + self.seq_len + self.offset                 # ★ +offset
        fut = self.data[s: s + self.pred_len]
        return {
            'past_numeric':   torch.FloatTensor(past[:, self.future_idx]),
            'past_y':         torch.FloatTensor(past[:, self.target_idx: self.target_idx + 1]),
            'future_numeric': torch.FloatTensor(fut[:, self.future_idx]),
            'future_y':       torch.FloatTensor(fut[:, self.target_idx]),
        }
""")

# ── Model (D+1과 동일) ────────────────────────────────────────────────────
code(r"""
class Patch_Weather_Attention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(key_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)
    def forward(self, fw, pw, to):
        Q = self.W_Q(fw).unsqueeze(1); K = self.W_K(pw)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor, dim=-1)
        return torch.bmm(attn, to).squeeze(1), attn

class PatchTST_Weather_Model(nn.Module):
    def __init__(self, num_features, seq_len=336, pred_len=24, patch_len=24, stride=12,
                 d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len; self.stride = stride; self.seq_len = seq_len; self.pred_len = pred_len
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len * num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.num_weather_feats = num_features - 1
        fut_flat = pred_len * self.num_weather_feats; w_patch = patch_len * self.num_weather_feats
        self.weather_attn = Patch_Weather_Attention(fut_flat, w_patch, d_model)
        self.regressor = nn.Sequential(nn.Linear(d_model + fut_flat, 256), nn.LeakyReLU(0.1),
                                       nn.Dropout(dropout), nn.Linear(256, pred_len))
        self.weather_bypass = nn.Linear(fut_flat, pred_len)
    def forward(self, batch):
        p_num = batch['past_numeric'].to(DEVICE); p_y = batch['past_y'].to(DEVICE); f_num = batch['future_numeric'].to(DEVICE)
        B = p_num.shape[0]
        x_past = torch.cat([p_num, p_y], dim=-1)
        xp = x_past.unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        enc_out = self.transformer_encoder(self.dropout(self.patch_embedding(xp) + self.pos_embedding))
        fut_flat = f_num.reshape(B, -1)
        xw = x_past[..., :-1].unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        context, _ = self.weather_attn(fut_flat, xw, enc_out)
        return self.regressor(torch.cat([context, fut_flat], dim=1)) + self.weather_bypass(fut_flat)
""")

# ── Loss (D+1과 동일: 흐린날 과대 페널티) ─────────────────────────────────
code(r"""
class DaylightWeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.01, low_util_cutoff=0.25, high_weight=3.0, overpredict_penalty=1.5):
        super().__init__(); self.threshold=threshold; self.low_util_cutoff=low_util_cutoff
        self.high_weight=high_weight; self.overpredict_penalty=overpredict_penalty; self.mse=nn.MSELoss(reduction='none')
    def forward(self, pred, target):
        mask = (target > 0) | (pred > self.threshold)
        if mask.sum() == 0: return torch.tensor(0.0, requires_grad=True, device=pred.device)
        loss_all = self.mse(pred, target); w = torch.ones_like(target)
        cloudy = (target > self.threshold) & (target <= self.low_util_cutoff)
        w[cloudy] = self.high_weight
        w[cloudy & (pred > target)] = self.high_weight * self.overpredict_penalty
        return (loss_all * w)[mask].mean()
""")

# ── split + train (offset 인자 추가) ──────────────────────────────────────
code(r"""
def prepare_split(df, features, future_features, target_col, scaler_range=(0, 1)):
    idx = df.index
    tr = df[idx <= TRAIN_END].copy(); va = df[(idx > TRAIN_END) & (idx <= VAL_END)].copy(); te = df[idx > VAL_END].copy()
    scaler = MinMaxScaler(feature_range=scaler_range)
    tr[future_features] = scaler.fit_transform(tr[future_features])
    va[future_features] = scaler.transform(va[future_features]); te[future_features] = scaler.transform(te[future_features])
    fidx = [features.index(c) for c in future_features]; tidx = features.index(target_col)
    return tr[features].values, va[features].values, te[features].values, scaler, fidx, tidx

def train_model(name, tr_arr, va_arr, fidx, tidx, hp, criterion, save_path, offset, epochs=EPOCHS, patience=15):
    num_features = len(fidx) + 1
    model = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    tr_ds = PatchTSTDatasetH(tr_arr, hp['seq_len'], PRED_LEN, fidx, tidx, offset=offset)
    va_ds = PatchTSTDatasetH(va_arr, hp['seq_len'], PRED_LEN, fidx, tidx, offset=offset)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    best = float('inf'); bad = 0
    print(f'== {name} offset={offset}h | feats={num_features} | train_ds={len(tr_ds)} val_ds={len(va_ds)}')
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for b in tqdm(tr_ld, desc=f'{name} ep{ep}', leave=False):
            opt.zero_grad(); loss = criterion(model(b), b['future_y'].to(DEVICE)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in va_ld: vl += criterion(model(b), b['future_y'].to(DEVICE)).item()
        tl /= len(tr_ld); vl /= len(va_ld); sch.step(vl)
        if vl < best:
            best = vl; bad = 0; torch.save(model.state_dict(), save_path)
        else:
            bad += 1
            if bad >= patience: print(f'  early stop @ ep{ep}'); break
    print(f'== {name} done. best val={best:.5f} -> {save_path}')
    return best
""")

# ── ★ 지평 루프: D2/D3 × (solar, wind) 학습 ───────────────────────────────
code(r"""
s_tr, s_va, s_te, scaler_solar, s_fidx, s_tidx = prepare_split(df, features_solar, future_features_solar, 'Solar_Utilization')
w_tr, w_va, w_te, scaler_wind,  w_fidx, w_tidx = prepare_split(df, features_wind,  future_features_wind,  'Wind_Utilization')

for hname, off in HORIZONS.items():
    print('\n' + '='*60 + f'\nHORIZON {hname} (offset {off}h)')
    train_model(f'SOLAR_{hname}', s_tr, s_va, s_fidx, s_tidx, SOLAR_HP,
                DaylightWeightedMSELoss(0.01, 0.25, 3.0, 1.5),
                f'{OUT_DIR}/best_patchtst_solar_model_{hname}.pth', offset=off)
    train_model(f'WIND_{hname}', w_tr, w_va, w_fidx, w_tidx, WIND_HP,
                nn.MSELoss(), f'{OUT_DIR}/best_patchtst_wind_model_{hname}.pth', offset=off)
""")

# ── 지평별 test MAE ───────────────────────────────────────────────────────
code(r"""
@torch.no_grad()
def eval_mae(path, hp, arr, fidx, tidx, off, num_features):
    m = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **hp).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
    ds = PatchTSTDatasetH(arr, hp['seq_len'], PRED_LEN, fidx, tidx, offset=off)
    ld = DataLoader(ds, batch_size=256, shuffle=False); P, A = [], []
    for b in ld: P.append(m(b).cpu().numpy()); A.append(b['future_y'].numpy())
    return mean_absolute_error(np.concatenate(A).ravel(), np.concatenate(P).ravel())

for hname, off in HORIZONS.items():
    smae = eval_mae(f'{OUT_DIR}/best_patchtst_solar_model_{hname}.pth', SOLAR_HP, s_te, s_fidx, s_tidx, off, len(s_fidx)+1)
    wmae = eval_mae(f'{OUT_DIR}/best_patchtst_wind_model_{hname}.pth', WIND_HP, w_te, w_fidx, w_tidx, off, len(w_fidx)+1)
    print(f'{hname}: TEST util MAE solar={smae:.4f} wind={wmae:.4f}')
""")

# ── 패키징 ────────────────────────────────────────────────────────────────
code(r"""
import shutil
shutil.make_archive('/content/solarwind_direct', 'zip', OUT_DIR)
print('zip -> /content/solarwind_direct.zip (가중치 4개)')
try:
    from google.colab import files; files.download('/content/solarwind_direct.zip')
except Exception: pass
""")

md(r"""
## 산출물 적용
`out/` 의 4개 가중치를 repo `3. jeju_solarwind_forecaster/solarwind_models/` 에 **추가** 복사:
```
best_patchtst_solar_model_D2.pth / _D3.pth
best_patchtst_wind_model_D2.pth  / _D3.pth
```
스케일러(`MinMax_scaler_*.pkl`)·`metadata.pkl` 은 **D+1과 동일하므로 덮어쓰지 않는다**(지평 무관).
하이브리드 서빙 wrapper(다음 단계)가 D+1=기존, D+2/D+3=이 가중치, D+4~D+6=LGBM 으로 묶는다.
""")


def main():
    nb = {
        "cells": [
            {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if k == "code" else {})}
            for k, s in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "| cells:", len(CELLS))


if __name__ == "__main__":
    main()
