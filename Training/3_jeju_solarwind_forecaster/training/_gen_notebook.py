"""train_solarwind_3station_colab.ipynb 생성기.

셀 소스를 리스트로 관리해 ipynb(JSON)를 만든다. 직접 ipynb JSON 을 손으로
편집하면 깨지기 쉬워서 생성기로 둔다. 수정은 이 파일을 고치고 다시 실행.

    python "3. solar_wind_forecaster/training/_gen_notebook.py"
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train_solarwind_3station_colab.ipynb"

CELLS = []  # (kind, source)


def md(s):  CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))


# ── 0. 개요 ───────────────────────────────────────────────────────────────
md(r"""
# 제주 Solar / Wind 이용률 PatchTST 재학습 (3지점, Cross-Attention)

기존 단일지점 모델을 **3지점 입력**으로 재학습한다. 산자부 공모전 3단계
(`solar_wind_forecaster` = net_load forecaster).

**설계 결정 (2026-06-01 게이트):**
- **결합**: 지점별 기상을 *별도 채널로 concat* (평균 X) → 노트북 원본의
  `Patch_Weather_Attention`(미래예보 ↔ 과거패치 temporal cross-attention)을 그대로 재사용.
- **Solar 입력 지점**: `west`(고산) + `south`(남)  — east 는 추론(forecast) 시점에 일사·구름이 없어 제외.
- **Wind 입력 지점**: `west`(고산) + `east`(성산) + `south`(남)  — 풍속/풍향 3지점 모두 가용.
- **Target**: `real_solar_utilization_jeju` / `real_wind_utilization_jeju` (0~1, DB 기성, 제주 계통 단일 이용률).

**입력**: `solarwind_raw_jeju.csv` (로컬 `export_solarwind_csv.py` 로 DB 에서 추출).
**산출물(이 노트북이 저장)**: `best_patchtst_solar_model.pth`, `best_patchtst_wind_model.pth`,
`MinMax_scaler_solar.pkl`, `MinMax_scaler_wind.pkl`, `metadata.pkl`
→ repo 의 `3. solar_wind_forecaster/models/` 에 그대로 덮어쓰면 기존
`net_load_forecaster` 로더가 새 차원으로 바로 로드(아키텍처 동일).

> 첫 학습은 이 CSV(historical 관측치)로만 진행. 운영(serve) 시 미래 피처는
> DB `forecast` 테이블에서 받아온다 — 그 연결은 학습 검증 후 별도 단계.
""")

# ── 1. import ─────────────────────────────────────────────────────────────
code(r"""
# Colab 기본 제공으로 추가 설치 불필요 (torch/pandas/sklearn/joblib)
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
""")

# ── 2. CONFIG ─────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# CONFIG  — 경로/분할/하이퍼파라미터
# ==========================================================================
# Colab: 드라이브 마운트 후 CSV 경로 지정, 또는 좌측 파일창에 직접 업로드.
# from google.colab import drive; drive.mount('/content/drive')
CSV_PATH = '/content/solarwind_raw_jeju.csv'   # ← 업로드한 CSV 경로
OUT_DIR  = '/content/out'                       # 산출물 저장 폴더

import os
os.makedirs(OUT_DIR, exist_ok=True)

PRED_LEN = 24            # +24h 예측 (두 모델 공통)

# 시간 분할 (데이터: 2020-01-01 ~ 2026-06-01)
TRAIN_END = '2025-03-01'         # train: 시작 ~ TRAIN_END
VAL_END   = '2026-01-01'         # val:   TRAIN_END+1h ~ VAL_END
# test:  VAL_END+1h ~ 끝

# 지점
SOLAR_STATIONS = ['west', 'south']
# wind: south(남)는 태양광 지점 — 시스템풍력과 약상관(0.25)+예보풍속 과대편차(+3.94)로
#       serve 악화 → 제외. 실제 터빈단지 west(고산)+east(성산)만.
WIND_STATIONS  = ['west', 'east']

# ── Solar 하이퍼파라미터 (기존 production 과 동일 차원 = 로더 호환) ──
SOLAR_HP = dict(seq_len=336, patch_len=24, stride=12,
                d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2)
# ── Wind 하이퍼파라미터 ──
WIND_HP  = dict(seq_len=72,  patch_len=12, stride=6,
                d_model=128, num_heads=4, num_layers=2, d_ff=256,  dropout=0.3)

EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
""")

# ── 3. 데이터 로드 + 시간/보간 ────────────────────────────────────────────
code(r"""
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
print('rows:', len(df), '| range:', df.index.min(), '->', df.index.max())

# 시간 파생 (지점 공통)
df['Hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
df['Hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
df['Year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
df['Year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)

# 짧은 결측 보간 (원 노트북과 동일: limit=3)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].interpolate(limit=3)
df[num_cols] = df[num_cols].ffill().bfill()   # 양끝 잔여 결측 방지
""")

# ── 4. Solar 피처 파생 ────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# Solar 피처 파생 (지점별)
# ==========================================================================
# solar_damping: 당일 주간(06~20h) 누적 강수량 -> 감쇠계수 exp(-0.163*rain)
def add_solar_damping(df, st):
    daily = df.groupby(df.index.date)[f'rainfall_{st}'].transform(
        lambda x: x.between_time('06:00', '20:00').sum())
    df[f'solar_damping_{st}'] = np.exp(-0.163 * daily.clip(upper=10))

for st in SOLAR_STATIONS:
    add_solar_damping(df, st)

df['Solar_Utilization'] = df['real_solar_utilization_jeju'].clip(0, 1)

# 미래(예보)로 쓸 피처 = 지점별 기상 + 시간. 타겟(Utilization)은 누수 방지로 제외.
future_features_solar = []
for st in SOLAR_STATIONS:
    future_features_solar += [f'solar_rad_{st}', f'total_cloud_{st}',
                              f'midlow_cloud_{st}', f'solar_damping_{st}']
future_features_solar += ['Hour_sin', 'Hour_cos']
features_solar = future_features_solar + ['Solar_Utilization']
print('solar future_features (%d):' % len(future_features_solar), future_features_solar)
""")

# ── 5. Wind 피처 파생 ─────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# Wind 피처 파생 (지점별)
# ==========================================================================
# 피처 축소(2026-06-01): serve(예보기상) 악화 대응.
#  - wind_spd_sq / wind_spd_cu 삭제: 예보 풍속오차를 제곱·세제곱으로 증폭(특히 cubic)
#    → 관측 학습엔 도움되나 예보 추론에서 오차 폭주. 제거.
#  - 풍향(wd_sin/cos)은 지점 간 거의 동일 → 지점별 6채널은 중복 노이즈.
#    대표 1지점(DIR_STATION)만 공유 사용.
#  - wind_spd(지점별, 속도는 지점차가 유효신호) + wind_zone(컷아웃, 0~1이라 증폭없음) 유지.
WIND_SPD_CAP = 20.0
CUTOFF_WIND_SPD = 25.0
DIR_STATION = 'west'     # 풍향 대표 지점(고산, 주 풍력단지)

def add_wind_feats(df, st):
    raw = df[f'wind_spd_{st}']
    cond = [raw < 15, (raw >= 15) & (raw < 20),
            (raw >= 20) & (raw < CUTOFF_WIND_SPD), raw >= CUTOFF_WIND_SPD]
    df[f'wind_zone_{st}'] = np.select(cond, [0.0, 1.0, 0.5, 0.0], default=0.0)
    df[f'wind_spd_{st}'] = raw.clip(upper=WIND_SPD_CAP)

for st in WIND_STATIONS:
    add_wind_feats(df, st)

# 공유 풍향 (대표 지점 1곳)
df['wd_sin'] = df[f'wd_sin_{DIR_STATION}']
df['wd_cos'] = df[f'wd_cos_{DIR_STATION}']

df['Wind_Utilization'] = df['real_wind_utilization_jeju'].clip(0, 1)

future_features_wind = []
for st in WIND_STATIONS:
    future_features_wind += [f'wind_spd_{st}', f'wind_zone_{st}']
future_features_wind += ['wd_sin', 'wd_cos', 'Hour_sin', 'Hour_cos', 'Year_sin', 'Year_cos']
features_wind = future_features_wind + ['Wind_Utilization']
print('wind future_features (%d):' % len(future_features_wind), future_features_wind)
""")

# ── 6. Dataset ────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# Dataset (past_y 포함) — 원 노트북과 동일
# ==========================================================================
class PatchTSTDataset(Dataset):
    def __init__(self, data_array, seq_len, pred_len, future_idx, target_idx):
        self.data = data_array
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.future_idx = future_idx
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        past = self.data[idx: idx + self.seq_len]
        past_numeric = past[:, self.future_idx]
        past_y = past[:, self.target_idx: self.target_idx + 1]
        fut = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        future_numeric = fut[:, self.future_idx]
        future_y = fut[:, self.target_idx]
        return {
            'past_numeric':   torch.FloatTensor(past_numeric),
            'past_y':         torch.FloatTensor(past_y),
            'future_numeric': torch.FloatTensor(future_numeric),
            'future_y':       torch.FloatTensor(future_y),
        }
""")

# ── 7. Model (production architecture 와 동일: inst_norm 없음 + weather_bypass) ──
code(r"""
# ==========================================================================
# PatchTST + Weather Attention
#  - net_load_forecaster/architecture.py 와 동일한 모듈 구성(파라미터 이름)
#    -> 학습 가중치가 기존 로더에 그대로 로드됨.
#  - 핵심: future_weather 가 과거 패치(transformer 출력)에 cross-attention.
#    지점을 별도 채널로 concat 했으므로 다지점 정보가 자연히 들어감.
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

# ── 8. Loss ──────────────────────────────────────────────────────────────
code(r"""
# Solar: 낮(발전구간) + 흐린날 가중 MSE (원 노트북 채택안)
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

# ── 9. 학습 유틸 ──────────────────────────────────────────────────────────
code(r"""
def prepare_split(df, features, future_features, target_col, scaler_range=(0, 1)):
    # 경계 마스킹(견고): 라벨슬라이스/.iloc 대신 부호비교 — 정렬·중복에 안전.
    idx = df.index
    tr = df[idx <= TRAIN_END].copy()
    va = df[(idx > TRAIN_END) & (idx <= VAL_END)].copy()
    te = df[idx > VAL_END].copy()
    for nm, part in [('train', tr), ('val', va), ('test', te)]:
        if len(part) == 0:
            raise ValueError(
                f'[prepare_split] {nm} split 0행! df range '
                f'{idx.min()}~{idx.max()} vs TRAIN_END={TRAIN_END}, VAL_END={VAL_END}. '
                f'→ 업로드한 CSV가 전체(2020~2026)인지, 셀3 "rows/range" 출력 확인.')
    scaler = MinMaxScaler(feature_range=scaler_range)
    tr[future_features] = scaler.fit_transform(tr[future_features])
    va[future_features] = scaler.transform(va[future_features])
    te[future_features] = scaler.transform(te[future_features])
    fidx = [features.index(c) for c in future_features]
    tidx = features.index(target_col)
    return (tr[features].values, va[features].values, te[features].values,
            scaler, fidx, tidx)


def train_model(name, train_arr, val_arr, fidx, tidx, hp, criterion,
                save_path, epochs=EPOCHS, patience=15):
    num_features = len(fidx) + 1
    model = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    tr_ds = PatchTSTDataset(train_arr, hp['seq_len'], PRED_LEN, fidx, tidx)
    va_ds = PatchTSTDataset(val_arr,   hp['seq_len'], PRED_LEN, fidx, tidx)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best = float('inf'); bad = 0
    print(f'== train {name} | feats={num_features} | train_ds={len(tr_ds)} val_ds={len(va_ds)}')
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
    print(f'== {name} done. best val={best:.5f} -> {save_path}')
    return model, best
""")

# ── 10. Solar 학습 ────────────────────────────────────────────────────────
code(r"""
s_tr, s_va, s_te, scaler_solar, s_fidx, s_tidx = prepare_split(
    df, features_solar, future_features_solar, 'Solar_Utilization')

solar_model, solar_best = train_model(
    'SOLAR', s_tr, s_va, s_fidx, s_tidx, SOLAR_HP,
    criterion=DaylightWeightedMSELoss(threshold=0.01, low_util_cutoff=0.25,
                                      high_weight=3.0, overpredict_penalty=1.5),
    save_path=f'{OUT_DIR}/best_patchtst_solar_model.pth')

joblib.dump(scaler_solar, f'{OUT_DIR}/MinMax_scaler_solar.pkl')
""")

# ── 11. Wind 학습 ─────────────────────────────────────────────────────────
code(r"""
w_tr, w_va, w_te, scaler_wind, w_fidx, w_tidx = prepare_split(
    df, features_wind, future_features_wind, 'Wind_Utilization')

wind_model, wind_best = train_model(
    'WIND', w_tr, w_va, w_fidx, w_tidx, WIND_HP,
    criterion=nn.MSELoss(),
    save_path=f'{OUT_DIR}/best_patchtst_wind_model.pth')

joblib.dump(scaler_wind, f'{OUT_DIR}/MinMax_scaler_wind.pkl')
""")

# ── 12. Test MAE (숫자만, 시각화는 가중치 확정 후 별도) ────────────────────
code(r"""
@torch.no_grad()
def eval_mae(model, arr, fidx, tidx, seq_len):
    model.eval()
    ds = PatchTSTDataset(arr, seq_len, PRED_LEN, fidx, tidx)
    ld = DataLoader(ds, batch_size=256, shuffle=False)
    P, A = [], []
    for b in ld:
        P.append(model(b).cpu().numpy()); A.append(b['future_y'].numpy())
    return mean_absolute_error(np.concatenate(A).ravel(), np.concatenate(P).ravel())

solar_mae = eval_mae(solar_model, s_te, s_fidx, s_tidx, SOLAR_HP['seq_len'])
wind_mae  = eval_mae(wind_model,  w_te, w_fidx, w_tidx, WIND_HP['seq_len'])
print(f'TEST util MAE | solar={solar_mae:.4f}  wind={wind_mae:.4f}')
""")

# ── 13. 메타데이터 저장 + 패키징 ──────────────────────────────────────────
code(r"""
# ==========================================================================
# metadata.pkl 저장 (net_load_forecaster/loader.py 가 읽는 키 구성)
# ==========================================================================
metadata = {
    'features_solar':        features_solar,
    'future_features_solar': future_features_solar,
    'features_wind':         features_wind,
    'future_features_wind':  future_features_wind,
    'SEQ_LEN_SOLAR': SOLAR_HP['seq_len'],
    'SEQ_LEN_WIND':  WIND_HP['seq_len'],
    'PRED_LEN':      PRED_LEN,
    # 참고용 (serve 단계에서 forecast 컬럼 매핑에 사용)
    'solar_stations': SOLAR_STATIONS,
    'wind_stations':  WIND_STATIONS,
}
joblib.dump(metadata, f'{OUT_DIR}/metadata.pkl')
print('saved metadata.pkl')
print('  solar num_features =', len(features_solar))
print('  wind  num_features =', len(features_wind))

# 산출물 zip 으로 묶어 다운로드
import shutil
shutil.make_archive('/content/solarwind_models', 'zip', OUT_DIR)
print('zip -> /content/solarwind_models.zip')
try:
    from google.colab import files
    files.download('/content/solarwind_models.zip')
except Exception:
    pass
""")

# ── 14. 배포 안내 ─────────────────────────────────────────────────────────
md(r"""
## 산출물 적용 (학습 후)

`out/` 의 5개 파일을 repo `3. solar_wind_forecaster/models/` 에 덮어쓴다:

```
best_patchtst_solar_model.pth
best_patchtst_wind_model.pth
MinMax_scaler_solar.pkl
MinMax_scaler_wind.pkl
metadata.pkl
```

아키텍처(`architecture.py`)는 동일하고 `loader.py` 가 `metadata` 의
`features_*` 길이로 `num_features` 를 잡으므로 **코드 수정 없이** 새 다지점
가중치가 로드된다.

### 다음 단계 (serve 연결 — 별도 게이트)
운영 추론은 미래 피처를 DB `forecast` 테이블에서 받아야 한다. 학습 피처 ↔
forecast 컬럼 매핑:

| 학습 피처 | historical(학습) | forecast(serve) |
|---|---|---|
| `solar_rad_{st}`   | `solar_rad_{st}`   | `radiation_{st}` |
| `total_cloud_{st}` | `total_cloud_{st}` | `total_cloud_{st}` |
| `midlow_cloud_{st}`| `midlow_cloud_{st}`| `midlow_cloud_{st}` |
| `rainfall_{st}`→damping | `rainfall_{st}` | `rainfall_{st}` |
| `wind_spd_{st}`    | `wind_spd_{st}`    | `wind_spd_10m_{st}` |
| `wd_sin/cos_{st}`  | `wd_sin/cos_{st}`  | `wd_sin/cos_10m_{st}` |

`net_load_forecaster/data_pipeline.py` 의 `prepare_model_input` 을 위 매핑·
다지점 피처에 맞게 갱신하면 `predict(date)` 가 새 모델로 동작. (학습 결과
검증 후 진행)
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
