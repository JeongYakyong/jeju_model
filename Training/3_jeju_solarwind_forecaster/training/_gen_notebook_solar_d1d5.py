"""train_solar_d1d5_colab.ipynb 생성기 — solar PatchTST D+1~D+5 재학습 (2026-07-30).

왜 새 빌더인가
================================================================================
기존 빌더 둘을 하나로 합치고 운영 조건에 맞춘 것이다.
  - `_gen_notebook.py`        : D+1 solar+wind + 스케일러·metadata 생성
  - `_gen_notebook_direct.py` : D+2~D+6 direct(offset) 학습, 스케일러는 재사용
이번 재학습은 **solar 만 / D+1~D+5 / 한 노트북**이라, 스케일러를 새로 만들면
D+1 도 같은 스케일러로 다시 학습해야 해서(불일치 방지) 둘을 합치는 게 맞다.

사용자 확정 (2026-07-30)
================================================================================
  - **solar 만** 재학습한다. wind 는 LGBM 이 담당하고 이미 재학습·배포했다
    (하이브리드 구성: SOLAR=PatchTST / WIND=LGBM).
  - **지평 D+1~D+5** (offset 0/24/48/72/96h). 운영 지평 5일·예보 수집 --days 5 와 일치.
    구 D+6·D+7 은 만들지 않는다.
  - 학습창 train ≤2026-01 / val 2026-02~05 / test 2026-06~07 (demand·LGBM 과 동일 경계).
  - 입력 CSV 는 `export_solarwind_csv.py` 가 **메인 DB** 에서 뽑은 것을 쓴다.

절대 바꾸면 안 되는 것 (서빙 호환)
================================================================================
서빙은 state_dict 만 읽고 아키텍처는 **코드에 하드코딩**돼 있다:
  forecasting/patchtst.py       SOLAR_HP = patch_len24/stride12/d_model256/heads4/layers3/d_ff1024
                                SEQ_LEN_SOLAR=336, PRED_LEN=24
  forecasting/serve_solarwind.py 가 D+2~ 를 `best_patchtst_solar_model_D{n}.pth` 로 찾는다
→ SOLAR_HP·피처 순서·파일명을 바꾸면 로드가 깨진다. 이 노트북은 그대로 유지한다.

metadata.pkl 은 solar·wind 키를 **함께** 갖는다. wind 를 재학습하지 않아도
서빙(patchtst.load_assets)이 wind 키를 읽으므로 기존과 동일하게 재현해 넣는다.

    python Training/3_jeju_solarwind_forecaster/training/_gen_notebook_solar_d1d5.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_solar_d1d5_colab.ipynb"
CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ── 0. 개요 ───────────────────────────────────────────────────────────────
md(r"""
# 제주 Solar 이용률 PatchTST — D+1~D+5 재학습 (2026-07-30)

**solar 만** 재학습한다. wind 는 LGBM 담당이라 이 노트북에 없다.

## 준비물 (좌측 파일창에 업로드)
- `solarwind_raw_jeju.csv` — 로컬에서 `export_solarwind_csv.py` 로 메인 DB 에서 뽑은 것.
  2020-01-01 ~ 2026-07-31, 57,696행, 약 10MB.

## 런타임
**반드시 GPU 런타임**으로 바꾼다: 런타임 → 런타임 유형 변경 → T4 GPU.
CPU 로 돌리면 몇 시간 걸린다. 아래 첫 셀이 DEVICE 를 찍으니 `cuda` 인지 확인할 것.

## 산출물 (마지막 셀이 zip 으로 묶어 다운로드)
| 파일 | 반입 위치 |
|---|---|
| `best_patchtst_solar_model.pth` (D+1) | `models/solarwind_patchtst/` |
| `MinMax_scaler_solar.pkl`, `metadata.pkl` | `models/solarwind_patchtst/` |
| `best_patchtst_solar_model_D2..D5.pth` | `models/solarwind_patchtst_horizon/` |

> ⚠ **5개 모델은 한 세트다.** 스케일러를 새로 만들기 때문에 D+1~D+5 를 전부 같이
> 반입해야 한다. 일부만 바꾸면 옛 스케일러로 학습된 모델과 섞여 예측이 틀어진다.

## 예상 시간
T4 기준 모델당 10~25분, 5개 합쳐 **1~2시간** 정도다(조기종료라 데이터에 따라 다름).
각 지평이 끝날 때마다 가중치가 저장되므로 중간에 끊겨도 그때까지는 건진다.
""")

# ── 1. import ─────────────────────────────────────────────────────────────
code(r"""
# Colab 기본 제공으로 추가 설치 불필요 (torch/pandas/sklearn/joblib)
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
from tqdm.auto import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE =', DEVICE)
if DEVICE == 'cpu':
    print('!! GPU 런타임이 아니다. 런타임 > 런타임 유형 변경 > T4 GPU 로 바꾸고 다시 실행할 것.')
else:
    print('GPU :', torch.cuda.get_device_name(0))
""")

# ── 2. CONFIG ─────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# CONFIG — 경로 / 학습창 / 지평 / 하이퍼파라미터
# ==========================================================================
CSV_PATH = '/content/solarwind_raw_jeju.csv'   # 업로드한 CSV
OUT_DIR  = '/content/out'
os.makedirs(OUT_DIR, exist_ok=True)

PRED_LEN = 24            # 한 번에 24h 예측

# 학습창 (사용자 확정 2026-07-30 — demand 2-A / solarwind LGBM 과 같은 경계)
TRAIN_END = '2026-01-31 23:00'
VAL_END   = '2026-05-31 23:00'      # val = TRAIN_END 다음 ~ VAL_END
TEST_END  = '2026-07-31 23:00'      # test = VAL_END 다음 ~ TEST_END

# ★ direct 지평: 이름 -> future/target 윈도우를 뒤로 미는 offset(시간).
#   D+1 은 offset 0 (과거 윈도우 바로 다음 24h). offset 은 24의 배수여야 날짜 경계와 맞는다.
#   운영 지평이 5일이라 D+6·D+7 은 만들지 않는다.
HORIZONS = {'D1': 0, 'D2': 24, 'D3': 48, 'D4': 72, 'D5': 96}

SOLAR_STATIONS = ['west', 'south']   # east 는 예보에 일사·구름이 없어 제외
WIND_STATIONS  = ['west', 'east']    # metadata 재현용 (이 노트북은 wind 를 학습하지 않는다)
DIR_STATION    = 'west'

# ── Solar 하이퍼파라미터 — forecasting/patchtst.py SOLAR_HP 와 반드시 동일 ──
SOLAR_HP = dict(seq_len=336, patch_len=24, stride=12,
                d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2)
WIND_SEQ_LEN = 72        # metadata 의 SEQ_LEN_WIND (기존값 유지)

EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 15
""")

# ── 3. 데이터 로드 ────────────────────────────────────────────────────────
code(r"""
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
print('rows:', len(df), '| range:', df.index.min(), '->', df.index.max())

# 시간 파생
df['Hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
df['Hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
df['Year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
df['Year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)

# 짧은 결측 보간 (기존 학습과 동일: limit=3)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].interpolate(limit=3)
df[num_cols] = df[num_cols].ffill().bfill()

# 학습창이 데이터 안에 들어오는지 먼저 확인 (여기서 걸러야 학습 몇 시간 날리지 않는다)
for name, bound in [('TRAIN_END', TRAIN_END), ('VAL_END', VAL_END), ('TEST_END', TEST_END)]:
    assert df.index.min() < pd.Timestamp(bound) <= df.index.max() + pd.Timedelta('1h'), \
        f'{name}={bound} 가 데이터 범위 밖이다 ({df.index.min()} ~ {df.index.max()})'
n_tr = (df.index <= TRAIN_END).sum()
n_va = ((df.index > TRAIN_END) & (df.index <= VAL_END)).sum()
n_te = ((df.index > VAL_END) & (df.index <= TEST_END)).sum()
print(f'train {n_tr}행 / val {n_va}행 / test {n_te}행')
""")

# ── 4. Solar 피처 + metadata 용 wind 피처 ─────────────────────────────────
code(r"""
# ==========================================================================
# Solar 피처 파생 (지점별) — 기존 학습과 동일한 순서를 유지해야 서빙이 맞는다
# ==========================================================================
def add_solar_damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    df[f'solar_damping_{st}'] = np.exp(-0.163 * daily.clip(upper=10))

for st in SOLAR_STATIONS:
    add_solar_damping(df, st)

df['Solar_Utilization'] = df['real_solar_utilization_jeju'].clip(0, 1)

future_features_solar = []
for st in SOLAR_STATIONS:
    future_features_solar += [f'solar_rad_{st}', f'total_cloud_{st}',
                              f'midlow_cloud_{st}', f'solar_damping_{st}']
future_features_solar += ['Hour_sin', 'Hour_cos']
features_solar = future_features_solar + ['Solar_Utilization']
print('solar future_features (%d):' % len(future_features_solar), future_features_solar)

# ── metadata 재현용 wind 피처 (학습하지 않는다) ──
# 서빙 patchtst.load_assets 가 metadata 의 wind 키를 읽으므로 기존과 같은 값을 넣어 둔다.
WIND_SPD_CAP, CUTOFF_WIND_SPD = 20.0, 25.0
future_features_wind = []
for st in WIND_STATIONS:
    future_features_wind += [f'wind_spd_{st}', f'wind_zone_{st}']
future_features_wind += ['wd_sin', 'wd_cos', 'Hour_sin', 'Hour_cos', 'Year_sin', 'Year_cos']
features_wind = future_features_wind + ['Wind_Utilization']
print('wind  future_features (%d, 학습 안 함):' % len(future_features_wind), future_features_wind)
""")

# ── 5. Dataset (offset 지원) ──────────────────────────────────────────────
code(r"""
# ==========================================================================
# Dataset — future/target 윈도우를 offset 만큼 뒤로 민다 (direct 다지평)
#   과거 윈도우는 origin 까지 그대로 → 누수 없음. offset=0 이면 D+1(기존 구조)과 동일.
# ==========================================================================
class PatchTSTDatasetH(Dataset):
    def __init__(self, data_array, seq_len, pred_len, future_idx, target_idx, offset=0):
        self.data = data_array
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.future_idx = future_idx
        self.target_idx = target_idx
        self.offset = offset

    def __len__(self):
        return len(self.data) - self.seq_len - self.offset - self.pred_len + 1

    def __getitem__(self, idx):
        past = self.data[idx: idx + self.seq_len]
        past_numeric = past[:, self.future_idx]
        past_y = past[:, self.target_idx: self.target_idx + 1]
        s = idx + self.seq_len + self.offset          # ★ offset 만큼 뒤로
        fut = self.data[s: s + self.pred_len]
        return {
            'past_numeric':   torch.FloatTensor(past_numeric),
            'past_y':         torch.FloatTensor(past_y),
            'future_numeric': torch.FloatTensor(fut[:, self.future_idx]),
            'future_y':       torch.FloatTensor(fut[:, self.target_idx]),
        }
""")

# ── 6. Model ──────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# PatchTST + Weather Attention — forecasting/patchtst.py 와 동일 구성
#   파라미터 이름/차원이 같아야 서빙이 state_dict 를 그대로 로드한다. 손대지 말 것.
# ==========================================================================
class Patch_Weather_Attention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(key_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, future_weather_patch, past_weather_patches, transformer_output):
        Q = self.W_Q(future_weather_patch).unsqueeze(1)
        K = self.W_K(past_weather_patches)
        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, transformer_output)
        return context.squeeze(1), attn


class PatchTST_Weather_Model(nn.Module):
    def __init__(self, num_features, seq_len=336, pred_len=24, patch_len=24,
                 stride=12, d_model=128, num_heads=4, num_layers=2,
                 d_ff=256, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len * num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.num_weather_feats = num_features - 1
        fut_flat = pred_len * self.num_weather_feats
        w_patch = patch_len * self.num_weather_feats
        self.weather_attn = Patch_Weather_Attention(fut_flat, w_patch, d_model)

        self.regressor = nn.Sequential(
            nn.Linear(d_model + fut_flat, 256), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(256, pred_len))
        self.weather_bypass = nn.Linear(fut_flat, pred_len)

    def forward(self, batch):
        p_num = batch['past_numeric'].to(DEVICE)
        p_y   = batch['past_y'].to(DEVICE)
        f_num = batch['future_numeric'].to(DEVICE)
        B = p_num.shape[0]

        x_past = torch.cat([p_num, p_y], dim=-1)
        x_patches = x_past.unfold(1, self.patch_len, self.stride)
        x_patches = x_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        enc_out = self.patch_embedding(x_patches) + self.pos_embedding
        enc_out = self.transformer_encoder(self.dropout(enc_out))

        fut_flat = f_num.reshape(B, -1)
        x_past_w = x_past[..., :-1]
        w_patches = x_past_w.unfold(1, self.patch_len, self.stride)
        w_patches = w_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)

        context, _ = self.weather_attn(fut_flat, w_patches, enc_out)
        main = self.regressor(torch.cat([context, fut_flat], dim=1))
        return main + self.weather_bypass(fut_flat)
""")

# ── 7. Loss ───────────────────────────────────────────────────────────────
code(r"""
# Solar: 낮(발전구간) + 흐린날 가중 MSE — 기존 학습과 동일
class DaylightWeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.01, low_util_cutoff=0.25,
                 high_weight=3.0, overpredict_penalty=1.5):
        super().__init__()
        self.threshold = threshold
        self.low_util_cutoff = low_util_cutoff
        self.high_weight = high_weight
        self.overpredict_penalty = overpredict_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        mask = (target > 0) | (pred > self.threshold)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        loss_all = self.mse(pred, target)
        w = torch.ones_like(target)
        cloudy = (target > self.threshold) & (target <= self.low_util_cutoff)
        w[cloudy] = self.high_weight
        w[cloudy & (pred > target)] = self.high_weight * self.overpredict_penalty
        return (loss_all * w)[mask].mean()
""")

# ── 8. 분할 + 학습 유틸 ───────────────────────────────────────────────────
code(r"""
def prepare_split(df, features, future_features, target_col):
    # train/val/test 분할 + 스케일러(train 에만 fit). 부호비교라 정렬·중복에 안전.
    idx = df.index
    tr = df[idx <= TRAIN_END].copy()
    va = df[(idx > TRAIN_END) & (idx <= VAL_END)].copy()
    te = df[(idx > VAL_END) & (idx <= TEST_END)].copy()
    for nm, part in [('train', tr), ('val', va), ('test', te)]:
        if len(part) == 0:
            raise ValueError(f'[prepare_split] {nm} 0행! 업로드 CSV 범위를 확인할 것.')
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr[future_features] = scaler.fit_transform(tr[future_features])
    va[future_features] = scaler.transform(va[future_features])
    te[future_features] = scaler.transform(te[future_features])
    fidx = [features.index(c) for c in future_features]
    tidx = features.index(target_col)
    return (tr[features].values, va[features].values, te[features].values,
            scaler, fidx, tidx)


def train_model(name, train_arr, val_arr, fidx, tidx, hp, criterion,
                save_path, offset, epochs=EPOCHS, patience=PATIENCE):
    num_features = len(fidx) + 1
    model = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    tr_ds = PatchTSTDatasetH(train_arr, hp['seq_len'], PRED_LEN, fidx, tidx, offset=offset)
    va_ds = PatchTSTDatasetH(val_arr,   hp['seq_len'], PRED_LEN, fidx, tidx, offset=offset)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best, bad, started = float('inf'), 0, time.time()
    print(f'== train {name} | offset={offset}h feats={num_features} '
          f'| train_ds={len(tr_ds)} val_ds={len(va_ds)}')
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for b in tqdm(tr_ld, desc=f'{name} ep{ep}', leave=False):
            opt.zero_grad()
            loss = criterion(model(b), b['future_y'].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in va_ld:
                vl += criterion(model(b), b['future_y'].to(DEVICE)).item()
        tl /= len(tr_ld); vl /= len(va_ld)
        sch.step(vl)
        print(f'  ep{ep:03d} train={tl:.5f} val={vl:.5f} lr={opt.param_groups[0]["lr"]:.6f}')
        if vl < best:
            best = vl; bad = 0
            torch.save(model.state_dict(), save_path)
            print(f'    * saved (val={best:.5f})')
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop @ ep{ep}'); break
    print(f'== {name} done. best val={best:.5f}  {(time.time()-started)/60:.1f}분 -> {save_path}')
    return model, best


@torch.no_grad()
def eval_mae(model, arr, fidx, tidx, seq_len, offset):
    model.eval()
    ds = PatchTSTDatasetH(arr, seq_len, PRED_LEN, fidx, tidx, offset=offset)
    ld = DataLoader(ds, batch_size=256, shuffle=False)
    P, A = [], []
    for b in ld:
        P.append(model(b).cpu().numpy()); A.append(b['future_y'].numpy())
    return mean_absolute_error(np.concatenate(A).ravel(), np.concatenate(P).ravel())
""")

# ── 9. D+1~D+5 학습 ───────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# solar D+1 ~ D+5 학습 — 스케일러는 지평 무관이라 한 번만 만들어 공유한다
# ==========================================================================
s_tr, s_va, s_te, scaler_solar, s_fidx, s_tidx = prepare_split(
    df, features_solar, future_features_solar, 'Solar_Utilization')
joblib.dump(scaler_solar, f'{OUT_DIR}/MinMax_scaler_solar.pkl')
print('saved MinMax_scaler_solar.pkl  (D+1~D+5 공용)\n')

results = {}
all_started = time.time()
for hname, off in HORIZONS.items():
    # D+1 만 파일명이 다르다 — 서빙이 D+1 을 models/solarwind_patchtst/ 에서 찾는다
    fname = ('best_patchtst_solar_model.pth' if hname == 'D1'
             else f'best_patchtst_solar_model_{hname}.pth')
    print('=' * 70)
    print(f'HORIZON {hname} (offset {off}h) -> {fname}')
    model, best = train_model(
        f'SOLAR_{hname}', s_tr, s_va, s_fidx, s_tidx, SOLAR_HP,
        criterion=DaylightWeightedMSELoss(threshold=0.01, low_util_cutoff=0.25,
                                          high_weight=3.0, overpredict_penalty=1.5),
        save_path=f'{OUT_DIR}/{fname}', offset=off)
    mae = eval_mae(model, s_te, s_fidx, s_tidx, SOLAR_HP['seq_len'], off)
    results[hname] = dict(val_loss=best, test_mae=mae, file=fname)
    print(f'   {hname} test util MAE = {mae:.4f}')

print('\n' + '=' * 70)
print(f'전체 {(time.time()-all_started)/60:.1f}분')
print(pd.DataFrame(results).T.to_string())
""")

# ── 10. metadata + 패키징 ─────────────────────────────────────────────────
code(r"""
# ==========================================================================
# metadata.pkl — 서빙 forecasting/patchtst.load_assets 가 읽는 키 구성
#   solar 키는 이번 학습 값, wind 키는 기존 그대로(재학습 안 했으므로).
# ==========================================================================
metadata = {
    'features_solar':        features_solar,
    'future_features_solar': future_features_solar,
    'features_wind':         features_wind,
    'future_features_wind':  future_features_wind,
    'SEQ_LEN_SOLAR': SOLAR_HP['seq_len'],
    'SEQ_LEN_WIND':  WIND_SEQ_LEN,
    'PRED_LEN':      PRED_LEN,
    'solar_stations': SOLAR_STATIONS,
    'wind_stations':  WIND_STATIONS,
    # 이번 재학습 기록 (서빙은 안 읽지만 추적용)
    'retrained': '2026-07-30 solar D+1~D+5 (wind 미학습 — LGBM 담당)',
    'train': f'<={TRAIN_END}', 'val': f'~{VAL_END}', 'test': f'~{TEST_END}',
    'horizons_solar': HORIZONS,
}
joblib.dump(metadata, f'{OUT_DIR}/metadata.pkl')
print('saved metadata.pkl | solar num_features =', len(features_solar))

import shutil
shutil.make_archive('/content/solar_d1d5', 'zip', OUT_DIR)
print('\n산출물:')
for f in sorted(os.listdir(OUT_DIR)):
    print('  ', f, f'{os.path.getsize(os.path.join(OUT_DIR,f))/1e6:.1f}MB')
print('\nzip -> /content/solar_d1d5.zip')
try:
    from google.colab import files
    files.download('/content/solar_d1d5.zip')
except Exception as e:
    print('자동 다운로드 실패 — 좌측 파일창에서 solar_d1d5.zip 을 직접 내려받을 것:', e)
""")

# ── 11. 반입 안내 ─────────────────────────────────────────────────────────
md(r"""
## 반입 (로컬에서)

`solar_d1d5.zip` 을 풀고 **7개 파일을 한 번에** 옮긴다. 일부만 옮기면 스케일러가 어긋난다.

```
models/solarwind_patchtst/
    best_patchtst_solar_model.pth      (D+1)
    MinMax_scaler_solar.pkl
    metadata.pkl
models/solarwind_patchtst_horizon/
    best_patchtst_solar_model_D2.pth
    best_patchtst_solar_model_D3.pth
    best_patchtst_solar_model_D4.pth
    best_patchtst_solar_model_D5.pth
```

`MinMax_scaler_wind.pkl` 과 `best_patchtst_wind_model.pth` 는 **건드리지 않는다**
(wind 는 재학습하지 않았다).

### 함께 필요한 코드 수정
`forecasting/serve_solarwind.py` 의 `SOLAR_PT_HORIZONS = [2,3,4,5,6,7]` 을 `[2,3,4,5]` 로
줄인다. D+6·D+7 가중치는 이번에 만들지 않았고, 남아 있는 옛 파일은 **옛 스케일러 기준**이라
새 스케일러와 섞이면 안 된다.

### 검증
```bash
python forecasting/serve_chain.py --utc 12 --no-write     # 120행 hd 1~5 나오는지
```
그 다음 재학습 전후 정확도 비교(`est_horizon_jeju` 기준선 대조)를 돌린다.
""")


def main():
    nb = {
        "cells": [
            {"cell_type": k, "metadata": {},
             "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if k == "code" else {})}
            for k, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "| cells:", len(CELLS))


if __name__ == "__main__":
    main()
