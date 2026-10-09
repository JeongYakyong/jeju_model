"""train_solar_d1d5_colab.ipynb 생성기 — solar PatchTST 재학습 (2026-10-02 train/serve skew 해소).

2026-10-02 재학습 — 무엇이 바뀌었나
================================================================================
★핵심: **학습의 "미래" 입력을 실측(ASOS)에서 예보로 바꾼다.**
  구 노트북은 과거·미래 슬라이스를 같은 ASOS 열에서 잘랐다(`future_idx` 공용).  그런데
  서빙은 과거=ASOS / 미래=KIMG 예보를 먹는다 — 모델은 "미래 날씨를 완벽히 안다"고 배우고
  서빙에서만 불완전한 예보를 받았다(train/serve skew).  과소예측의 핵심 원인.

  미래 열 = `*_fut` (fit_jma_monthly_qm.py → export_solarwind_csv.py --with-future):
    운량      = **JMA 만** (KIMG 운량은 과대라 부적절 — 사용자 확정). 서빙도 JMA 운량 예보를 받는다.
                ~2022-07-01 JMA 분석치 → JMA D+1 예보 분포 월별 QM / 2022-07-02~ JMA D+1 예보
    일사·강수 = KIMG.  ~2025-12-19 JMA → KIMG 12z D+1 월별 QM / 2025-12-20~ 실제 KIMG 12z D+1
    중하층운량은 피처에서 뺀다 (JMA previous run 에 층별운량 없음).

  1. `PatchTSTDatasetH` 가 과거 블록(ASOS)·미래 블록(`_fut`)을 따로 읽는다.
  2. 스케일러는 **변수별로 과거+미래 값을 모아서** train 구간에서 적합한다 — 서빙이 한
     스케일러로 둘 다 변환하므로 같은 물리량은 같은 범위여야 한다.  열 이름은 기존과 같다.
  3. **피처 단순화: 지점별 [solar_rad, total_cloud(JMA)] + Hour_sin/cos (6개).**
     midlow_cloud(JMA 예보에 없음)·solar_damping·Kt 제거 — 실험(2026-10-02)에서 Kt·강수감쇠를 빼도
     나빠지지 않았고 permutation 비중도 거의 0 이었다.
  4. (2026-10-08 갱신) 일사·강수도 JMA 원본으로 바꾸고 split 을 train ≤2024 / val 2025 / test 2026~ 로 변경.
     Year_sin/cos 는 생성기 옵션 `--year` 로 켠 변형과 끈 변형을 따로 만든다.
     (아래는 10-02 시점 기록) 07-30 구성으로 복원 — Year_sin/cos 없음, train ≤2026-01 / val 2026-02~05 / test 2026-06~.
     08-25 재학습(Year + val 4계절)은 실패했고 두 변경을 같이 해 원인 분리가 안 됐다.
     이번엔 "미래 소스 + 피처 단순화" 만 바꾼다. 최근 표본 가중치(반감기)는 차이가 없어 쓰지 않는다.
  5. **D+1 만 기본 학습** (목표 = D+1 정확도).  D+2~D+5 는 평가 ②로 "D+1 가중치 재사용"을
     먼저 검증하고, 필요하면 `TRAIN_D2_TO_D5=True` 로 offset 학습.
  6. 결측이 낀 윈도우는 **건너뛴다** (구: ffill/bfill 로 채움 — 2026-07-31~08-02 같은 3일
     발표 누락을 가짜 값으로 메웠다).
  7. test 평가를 **best 에폭 가중치**로 한다 (구: 마지막 에폭 모델로 재고 있었다).

평가 셀 (test 구간, 서빙과 같은 00시 시작 윈도우, 낮 시간)
  ① 새 D+1(JMA 운량) vs 현행 서빙 D+1(KIMG 운량·중하층) — 각자 서빙될 입력, 같은 날짜
  ② 새 D+1 가중치를 D+2 에 재사용 vs 현행 D+2 — JMA 운량 예보가 D+2 까지뿐이라 D+3~ 은 평가 불가
  ③ permutation importance (확인용)

절대 바꾸면 안 되는 것 (서빙 호환)
================================================================================
서빙은 state_dict 만 읽고 아키텍처는 **코드에 하드코딩**돼 있다:
  forecasting/patchtst.py       SOLAR_HP = patch_len24/stride12/d_model256/heads4/layers3/d_ff1024
                                SEQ_LEN_SOLAR=336, PRED_LEN=24
  forecasting/serve_solarwind.py 가 D+2~ 를 `best_patchtst_solar_model_D{n}.pth` 로 찾는다
→ SOLAR_HP·파일명을 바꾸면 로드가 깨진다.  피처 수는 metadata 에서 읽으므로 피처 변경은 괜찮다.

metadata.pkl 은 solar·wind 키를 **함께** 갖는다. wind 를 재학습하지 않아도
서빙(patchtst.load_assets)이 wind 키를 읽으므로 기존과 동일하게 재현해 넣는다.

    python Training/3_jeju_solarwind_forecaster/training/_gen_notebook_solar_d1d5.py
"""
import json
import sys
from pathlib import Path

# 2026-10-08: `--year` 를 주면 Year_sin/cos 를 미래 입력에 넣은 변형을 만든다 (없음/있음 두 벌을 같은 split 으로 비교).
USE_YEAR = "--year" in sys.argv
OUT = Path(__file__).resolve().parent / ("train_solar_d1d5_colab_year.ipynb" if USE_YEAR
                                         else "train_solar_d1d5_colab.ipynb")
CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n").replace("__USE_YEAR__", str(USE_YEAR))))


# ── 0. 개요 ───────────────────────────────────────────────────────────────
md(r"""
# 제주 Solar 이용률 PatchTST — 예보 기반 미래 입력 재학습 (2026-10-02)

학습의 **미래 입력을 실측(ASOS) → 예보**로 바꿔 train/serve skew 를 없앤다.
운량·일사·강수 = **전부 JMA 원본**(QM·KIMG 없음 — 2026-10-08, 서빙도 JMA 일사를 받는다).
과거 입력은 기존대로 ASOS. 피처 = 지점별 **일사·운량** + 시간(6개). 목표는 **D+1 정확도**.

## 준비물 (좌측 파일창에 업로드)
- `solarwind_raw_jeju_trainserve_jmaraw.csv` — 로컬에서
  `export_solarwind_csv.py --with-future --jma-raw --out .../solarwind_raw_jeju_trainserve_jmaraw.csv` 로 만든 것.
- (비교용, 선택) 현행 서빙 모델을 `/content/old_model/` 에:
  `best_patchtst_solar_model.pth`, `MinMax_scaler_solar.pkl`, `metadata.pkl`
  (`models/solarwind_patchtst/`), `best_patchtst_solar_model_D2..D5.pth`
  (`models/solarwind_patchtst_horizon/`). 없으면 비교 행만 빠진다.

## 런타임
**반드시 GPU 런타임**: 런타임 → 런타임 유형 변경 → T4 GPU. 첫 셀이 DEVICE 를 찍는다.

## 산출물 (zip 으로 다운로드)
| 파일 | 반입 위치 |
|---|---|
| `best_patchtst_solar_model.pth` (D+1) | `models/solarwind_patchtst/` |
| `MinMax_scaler_solar.pkl`, `metadata.pkl` | `models/solarwind_patchtst/` |
| (`TRAIN_D2_TO_D5=True` 일 때만) `best_patchtst_solar_model_D2..D5.pth` | `models/solarwind_patchtst_horizon/` |

> ⚠ **스케일러·피처 구성이 바뀐다(10 → 6개).** 옛 D2~D5 가중치는 새 metadata 로 로드되지 않아
> 서빙이 멈춘다 — 반입 안내(맨 끝) 참고.
""")

# ── 1. import ─────────────────────────────────────────────────────────────
code(r"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm.auto import tqdm
try:
    import pvlib                       # 낮 시간 판정(태양고도)용 — Colab 기본 미포함
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pvlib'])
    import pvlib

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
CSV_PATH = '/content/solarwind_raw_jeju_trainserve_jmaraw.csv'
OUT_DIR  = '/content/out'
OLD_MODEL_DIR = '/content/old_model'   # 비교용 현행 서빙 모델 (없으면 비교 행 생략)
os.makedirs(OUT_DIR, exist_ok=True)

PRED_LEN = 24            # 한 번에 24h 예측

# 학습창 (2026-10-08) — 일사·운량·강수가 전 구간 JMA 한 소스라 어느 구간이든 서빙 조건과 같다.
#   그래서 val 을 4계절 한 해(2025)로 잡을 수 있다. (08-25 의 같은 val 분할은 미래 입력이 ASOS 였던 구조 +
#   Year 동시 변경이라 이번 결과와 직접 비교할 수 없다.)
TRAIN_END = '2024-12-31 23:00'      # train = 2020-01 ~ 2024-12
VAL_END   = '2025-12-31 23:00'      # val   = 2025 전체 (early stopping 은 4계절 val 로)
TEST_END  = None                    # test  = 2026-01 ~ 데이터 끝 (None = 자동)

# 날짜 피처 — 미래 입력에 Year_sin/cos(dayofyear)를 넣을지. 생성기 옵션 --year 로 정해진다.
USE_YEAR_FEATURES = __USE_YEAR__

# 미래 입력 열 접미사 — export_solarwind_csv.py --with-future 가 만든다
FUTURE_SUFFIX = '_fut'              # 운량 JMA / 일사·강수 KIMG(과거는 JMA 월별 QM)
EVAL_REUSE_HORIZONS = [2]           # 평가 ②: _fut_d2 = JMA 운량 D+2 + KIMG D+2 (JMA 운량 예보는 D+2 까지뿐)
OLD_MODEL_SUFFIX = '_kimg'          # 평가 ①②: 현행 서빙 모델이 실제로 받는 KIMG 원본 (_kimg_d{n} = D+n)

# ★ 학습 지평. 목표가 D+1 이라 기본은 D+1 만. 평가 ② 결과 "재사용 불가"면 True 로 다시 돌린다.
TRAIN_D2_TO_D5 = False
HORIZONS_ALL = {'D1': 0, 'D2': 24, 'D3': 48, 'D4': 72, 'D5': 96}
HORIZONS = HORIZONS_ALL if TRAIN_D2_TO_D5 else {'D1': 0}

SOLAR_STATIONS = ['west', 'south']   # east 는 예보에 일사·구름이 없어 제외
WIND_STATIONS  = ['west', 'east']    # metadata 재현용 (이 노트북은 wind 를 학습하지 않는다)

# ── Solar 하이퍼파라미터 — forecasting/patchtst.py SOLAR_HP 와 반드시 동일 ──
SOLAR_HP = dict(seq_len=336, patch_len=24, stride=12,
                d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2)
WIND_SEQ_LEN = 72        # metadata 의 SEQ_LEN_WIND (기존값 유지)

# 최근 표본 가중치 (반감기 N년 → N년 전 표본의 손실 가중치 = 0.5). 기본 끔.
#   2026-10-02 실험: 반감기 없음/1/1.5/2/3 의 val MAE 차이가 시드 흔들림 안이었다 → 단순한 쪽(끔).
RECENCY_HALF_LIFE_YEARS = None

EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 15

# 손실 가중치 — 흐린 시각(실측 <= 0.25) 가중, 그 시각 과대예측 추가 벌점. 구 값 3.0 / 1.5.
#   2026-10-02 서빙 비교: 미래 입력이 예보(불확실)가 되자 이 비대칭이 모델을 전반적으로 낮게 잡게 만들어
#   JMA 가 맑다고 해도 밝은 시각을 과소예측(-0.083 vs 현행 -0.052) → 1.0 / 1.0(일반 MSE)로 재학습해 비교.
LOSS_CLOUDY_WEIGHT = 1.0
LOSS_OVERPREDICT_PENALTY = 1.0

# 낮 시간 판정 (평가용) — 태양고도 5° 이상. serve_solarwind 의 JEJU_LAT/LON·SOLAR_ELEV_MIN 과 같은 값
SOLAR_LAT, SOLAR_LON, SOLAR_ELEV_MIN = 33.38, 126.55, 5.0

# 평가 — 서빙은 항상 대상일 00시부터 24h 를 낸다. 맑음/흐림은 실측 이용률로 가른다.
SERVING_START_HOURS = [0]
CLEAR_UTIL, CLOUDY_UTIL = 0.6, 0.25
""")

# ── 3. 데이터 로드 ────────────────────────────────────────────────────────
code(r"""
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
print('rows:', len(df), '| range:', df.index.min(), '->', df.index.max())

# 시간 파생 (Year 는 새 모델 피처가 아니다 — 비교용 옛 모델·wind metadata 재현에만 쓰인다)
df['Hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
df['Hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
df['Year_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
df['Year_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)

# 짧은 결측만 보간 (limit=3, 서빙과 같은 한도). ffill/bfill 은 하지 않는다 —
#   긴 결측(예: 2026-07-31~08-02 12z 발표 누락, 2024-01~02 JMA 일사 결측)을 가짜 값으로 채우게 된다.
#   남은 결측이 낀 윈도우는 Dataset 에서 건너뛴다.
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].interpolate(limit=3, limit_area='inside')

if TEST_END is None:
    TEST_END = df.index.max().strftime('%Y-%m-%d %H:%M')
    print('TEST_END 자동설정 ->', TEST_END)

for name, bound in [('TRAIN_END', TRAIN_END), ('VAL_END', VAL_END), ('TEST_END', TEST_END)]:
    assert df.index.min() < pd.Timestamp(bound) <= df.index.max() + pd.Timedelta('1h'), \
        f'{name}={bound} 가 데이터 범위 밖이다 ({df.index.min()} ~ {df.index.max()})'
fut_cols = [c for c in df.columns if c.endswith(FUTURE_SUFFIX)]
assert fut_cols, f'{FUTURE_SUFFIX} 열이 없다 — export_solarwind_csv.py --with-future 로 만든 CSV 인지 확인'
""")

# ── 4. Solar 피처 ─────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# Solar 피처 — 과거(ASOS)·미래(예보) 블록이 "같은 피처를 같은 순서로" 갖는다
#   블록 접미사:  ''  = ASOS 과거 / '_fut' = 학습 미래 / '_fut_d{n}' = 평가 ② 전용
# ==========================================================================
def is_daytime_hours(index):
    # 시간적산값이라 시각-30분(구간 중앙)의 태양고도로 판정 — 낮 시간 지표 계산용
    times = pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)
    times = times.tz_localize('Asia/Seoul') if times.tz is None else times.tz_convert('Asia/Seoul')
    elevation = pvlib.solarposition.get_solarposition(times, SOLAR_LAT, SOLAR_LON)['apparent_elevation']
    return elevation.values >= SOLAR_ELEV_MIN


df['is_daytime'] = is_daytime_hours(df.index)


def add_derived_block(df, suffix):
    # 현행 서빙 모델 비교용 solar_damping 파생 (새 모델 피처엔 없다) — 서빙 _add_solar_damping 과 같은 식
    for st in SOLAR_STATIONS:
        rain = f'rainfall_{st}{suffix}'
        daily = df.groupby(df.index.date)[rain].transform(
            lambda x: x.between_time('06:00', '20:00').sum())
        df[f'solar_damping_{st}{suffix}'] = np.exp(-0.163 * daily.clip(upper=10))


OLD_MODEL_HORIZONS = [2, 3, 4, 5]
BLOCK_SUFFIXES = (['', FUTURE_SUFFIX] + [f'{FUTURE_SUFFIX}_d{n}' for n in EVAL_REUSE_HORIZONS]
                  + [OLD_MODEL_SUFFIX] + [f'{OLD_MODEL_SUFFIX}_d{n}' for n in OLD_MODEL_HORIZONS])
for suffix in BLOCK_SUFFIXES:
    add_derived_block(df, suffix)

df['Solar_Utilization'] = df['real_solar_utilization_jeju'].clip(0, 1)

# ★이 순서가 곧 스케일러 열 순서이자 서빙 입력 순서다. 이름은 과거(ASOS) 기준.
future_features_solar = []
for st in SOLAR_STATIONS:
    future_features_solar += [f'solar_rad_{st}', f'total_cloud_{st}']
future_features_solar += ['Hour_sin', 'Hour_cos']
if USE_YEAR_FEATURES:
    future_features_solar += ['Year_sin', 'Year_cos']
features_solar = future_features_solar + ['Solar_Utilization']
print('solar future_features (%d):' % len(future_features_solar), future_features_solar)

TIME_FEATURES = {'Hour_sin', 'Hour_cos', 'Year_sin', 'Year_cos'}   # 과거·미래 공용 (접미사 없음)


def block_columns(names, suffix):
    return [n if (suffix == '' or n in TIME_FEATURES) else n + suffix for n in names]


def raw_columns(suffix):
    # 결측 판정용 원천 열 — 파생 열(damping)은 원천이 결측일 때만 결측이다
    return [f'{v}_{st}{suffix}' for v in ['solar_rad', 'total_cloud', 'rainfall']
            for st in SOLAR_STATIONS]


print('\n미래 블록 결측률(구간별):')
for label, part in [('train', df[df.index <= TRAIN_END]),
                    ('val',   df[(df.index > TRAIN_END) & (df.index <= VAL_END)]),
                    ('test',  df[(df.index > VAL_END) & (df.index <= TEST_END)])]:
    print(f'  {label:5s}', part[raw_columns(FUTURE_SUFFIX)].isna().any(axis=1).mean().round(4))

# ── metadata 재현용 wind 피처 (학습하지 않는다) ──
future_features_wind = []
for st in WIND_STATIONS:
    future_features_wind += [f'wind_spd_{st}', f'wind_zone_{st}']
future_features_wind += ['wd_sin', 'wd_cos', 'Hour_sin', 'Hour_cos', 'Year_sin', 'Year_cos']
features_wind = future_features_wind + ['Wind_Utilization']
""")

# ── 5. Dataset ────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# Dataset — 과거 블록(ASOS)과 미래 블록(예보)을 따로 읽는다
#   data 열 구성 = [과거 피처 F개 | 미래 피처 F개 | 타깃 1개]
#   offset 만큼 미래/타깃 윈도우를 뒤로 민다 (D+n direct). 과거 윈도우는 origin 까지 → 누수 없음.
#   결측 행이 하나라도 낀 윈도우는 건너뛴다.
# ==========================================================================
class PatchTSTDatasetH(Dataset):
    def __init__(self, arr, seq_len, pred_len, offset=0, start_hours=None, future_start_min=None):
        self.data = arr['data']
        self.sample_weight = arr.get('sample_weight', np.ones(len(self.data), dtype=np.float32))
        self.seq_len, self.pred_len = seq_len, pred_len
        num_feats = (self.data.shape[1] - 1) // 2
        self.past_idx = list(range(num_feats))
        self.future_idx = list(range(num_feats, 2 * num_feats))
        self.target_idx = 2 * num_feats

        n_windows = max(len(self.data) - seq_len - offset - pred_len + 1, 0)
        starts = np.arange(n_windows)
        future_starts = starts + seq_len + offset
        past_bad_cum = np.concatenate([[0], np.cumsum(arr['past_bad'])])
        future_bad_cum = np.concatenate([[0], np.cumsum(arr['future_bad'])])
        ok = (past_bad_cum[starts + seq_len] == past_bad_cum[starts]) & \
             (future_bad_cum[future_starts + pred_len] == future_bad_cum[future_starts])
        timestamps = arr['timestamps']
        if start_hours is not None:
            ok &= np.isin(timestamps[future_starts].hour, start_hours)
        if future_start_min is not None:
            ok &= timestamps[future_starts] > pd.Timestamp(future_start_min)
        self.starts = starts[ok]
        self.future_starts = future_starts[ok]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start, future_start = self.starts[i], self.future_starts[i]
        past = self.data[start: start + self.seq_len]
        future = self.data[future_start: future_start + self.pred_len]
        return {
            'past_numeric':   torch.from_numpy(past[:, self.past_idx]),
            'past_y':         torch.from_numpy(past[:, self.target_idx: self.target_idx + 1]),
            'future_numeric': torch.from_numpy(future[:, self.future_idx]),
            'future_y':       torch.from_numpy(future[:, self.target_idx]),
            'future_start':   int(future_start),
            'sample_weight':  float(self.sample_weight[future_start]),
        }


def fit_pooled_scaler(train_part, names):
    # 같은 물리량은 과거(ASOS)·미래(예보) 값을 모아 한 범위로 맞춘다 — 서빙도 한 스케일러로 둘 다 변환한다.
    past = train_part[block_columns(names, '')]
    future = train_part[block_columns(names, FUTURE_SUFFIX)].set_axis(names, axis=1)
    return MinMaxScaler(feature_range=(0, 1)).fit(pd.concat([past, future]).dropna())


def build_array(part, names, scaler, future_suffix):
    # 스케일러는 과거 이름(names)으로 적합돼 있다 — 미래 블록도 이름을 바꿔 같은 스케일러로 변환한다.
    past = scaler.transform(part[block_columns(names, '')])
    future = scaler.transform(part[block_columns(names, future_suffix)].set_axis(names, axis=1))
    target = part[['Solar_Utilization']].values
    data = np.hstack([past, future, target]).astype(np.float32)
    return {
        'data': np.nan_to_num(data),     # 결측 행은 past_bad/future_bad 로 이미 제외된다
        'past_bad': part[raw_columns('') + ['Solar_Utilization']].isna().any(axis=1).values,
        'future_bad': part[raw_columns(future_suffix) + ['Solar_Utilization']].isna().any(axis=1).values,
        'timestamps': part.index,
        'is_daytime': part['is_daytime'].values,
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


def load_solar_model(path, num_features):
    model = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **SOLAR_HP).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.eval()
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

    def forward(self, pred, target, sample_weight=None):
        mask = (target > 0) | (pred > self.threshold)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        loss_all = self.mse(pred, target)
        w = torch.ones_like(target)
        cloudy = (target > self.threshold) & (target <= self.low_util_cutoff)
        w[cloudy] = self.high_weight
        w[cloudy & (pred > target)] = self.high_weight * self.overpredict_penalty
        if sample_weight is not None:              # 최근 표본 가중치 (RECENCY_HALF_LIFE_YEARS)
            w = w * sample_weight.to(pred.device).float().unsqueeze(1)
        return (loss_all * w)[mask].mean()
""")

# ── 8. 학습·평가 유틸 ─────────────────────────────────────────────────────
code(r"""
def train_model(name, train_arr, val_arr, hp, criterion, save_path, offset,
                epochs=EPOCHS, patience=PATIENCE, verbose=True):
    tr_ds = PatchTSTDatasetH(train_arr, hp['seq_len'], PRED_LEN, offset=offset)
    va_ds = PatchTSTDatasetH(val_arr,   hp['seq_len'], PRED_LEN, offset=offset)
    num_features = len(tr_ds.past_idx) + 1
    model = PatchTST_Weather_Model(num_features, pred_len=PRED_LEN, **hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best, bad, started = float('inf'), 0, time.time()
    print(f'== train {name} | offset={offset}h feats={num_features} '
          f'| train_ds={len(tr_ds)} val_ds={len(va_ds)}')
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for b in tqdm(tr_ld, desc=f'{name} ep{ep}', leave=False):
            opt.zero_grad()
            loss = criterion(model(b), b['future_y'].to(DEVICE), b['sample_weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in va_ld:
                vl += criterion(model(b), b['future_y'].to(DEVICE)).item()
        tl /= len(tr_ld); vl /= len(va_ld)
        sch.step(vl)
        if verbose:
            print(f'  ep{ep:03d} train={tl:.5f} val={vl:.5f} lr={opt.param_groups[0]["lr"]:.6f}')
        if vl < best:
            best = vl; bad = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f'    * saved (val={best:.5f})')
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f'  early stop @ ep{ep}')
                break
    print(f'== {name} done. best val={best:.5f}  {(time.time()-started)/60:.1f}분 -> {save_path}')
    # ★저장된 best 에폭 가중치를 돌려준다 (마지막 에폭 모델로 평가하지 않도록)
    return load_solar_model(save_path, num_features), best


@torch.no_grad()
def predict_windows(model, arr, offset):
    # test 구간 · 서빙과 같은 00시 시작 윈도우만. 반환 (예측, 실측, 낮 여부) — 각 (윈도우수, 24)
    model.eval()
    ds = PatchTSTDatasetH(arr, SOLAR_HP['seq_len'], PRED_LEN, offset=offset,
                          start_hours=SERVING_START_HOURS, future_start_min=VAL_END)
    preds, actuals, daytime = [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False):
        preds.append(model(b).clamp(0, 1).cpu().numpy())
        actuals.append(b['future_y'].numpy())
        daytime.append(np.stack([arr['is_daytime'][s: s + PRED_LEN] for s in b['future_start'].numpy()]))
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(daytime)


def summarize(pred, actual, daytime):
    # 낮 시간만. 맑음 = 실측 이용률 > 0.6, 흐림 = 실측 <= 0.25. 편향 = 예측 - 실측.
    p, a = pred[daytime], actual[daytime]
    clear, cloudy = a > CLEAR_UTIL, a <= CLOUDY_UTIL
    return {'일수': len(pred), 'MAE': np.abs(p - a).mean(), '편향': (p - a).mean(),
            '맑음편향': (p - a)[clear].mean(), '흐림편향': (p - a)[cloudy].mean(),
            '과대율': (p > a).mean()}


def on_common_dates(*arrays):
    # 비교하는 모델들이 같은 날짜로 평가되도록 결측 표시를 합친다 (입력 출처가 달라 결측 위치가 다르다)
    past_bad = np.logical_or.reduce([a['past_bad'] for a in arrays])
    future_bad = np.logical_or.reduce([a['future_bad'] for a in arrays])
    return [{**a, 'past_bad': past_bad, 'future_bad': future_bad} for a in arrays]


def eval_frame(df):
    # 평가용 프레임: test 첫날도 과거 336h + offset 이 필요하므로 val 끝부분부터 붙여 둔다
    context_hours = SOLAR_HP['seq_len'] + max(HORIZONS_ALL.values()) + PRED_LEN
    return df[(df.index > pd.Timestamp(VAL_END) - pd.Timedelta(hours=context_hours)) & (df.index <= TEST_END)]


def recency_weights(train_part, half_life_years):
    # 손실 가중치 = 0.5^(train 끝에서 지난 연수 / 반감기), 평균 1 로 맞춘다. val·test 는 가중치 없음.
    age_years = np.asarray((pd.Timestamp(TRAIN_END) - train_part.index).days) / 365.25
    weight = 0.5 ** (age_years / half_life_years)
    return (weight / weight.mean()).astype(np.float32)
""")

# ── 9. 학습 ───────────────────────────────────────────────────────────────
code(r"""
# ==========================================================================
# 학습 — 스케일러는 지평 무관이라 한 번만 만들어 공유한다 (과거+미래 풀링, train 구간)
# ==========================================================================
train_part = df[df.index <= TRAIN_END]
val_part = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
scaler_solar = fit_pooled_scaler(train_part, future_features_solar)
joblib.dump(scaler_solar, f'{OUT_DIR}/MinMax_scaler_solar.pkl')
print('saved MinMax_scaler_solar.pkl | 열 이름:', list(scaler_solar.feature_names_in_))

train_arr = build_array(train_part, future_features_solar, scaler_solar, FUTURE_SUFFIX)
val_arr = build_array(val_part, future_features_solar, scaler_solar, FUTURE_SUFFIX)
if RECENCY_HALF_LIFE_YEARS:
    train_arr['sample_weight'] = recency_weights(train_part, RECENCY_HALF_LIFE_YEARS)
    print(f'최근 표본 가중치: 반감기 {RECENCY_HALF_LIFE_YEARS}년, '
          f'가장 오래된 표본 {train_arr["sample_weight"][0]:.2f} / 최근 {train_arr["sample_weight"][-1]:.2f}')
test_arr = build_array(eval_frame(df), future_features_solar, scaler_solar, FUTURE_SUFFIX)

models, results = {}, {}
all_started = time.time()
for hname, off in HORIZONS.items():
    # D+1 만 파일명이 다르다 — 서빙이 D+1 을 models/solarwind_patchtst/ 에서 찾는다
    fname = ('best_patchtst_solar_model.pth' if hname == 'D1'
             else f'best_patchtst_solar_model_{hname}.pth')
    print('=' * 70)
    print(f'HORIZON {hname} (offset {off}h) -> {fname}')
    model, best = train_model(
        f'SOLAR_{hname}', train_arr, val_arr, SOLAR_HP,
        criterion=DaylightWeightedMSELoss(threshold=0.01, low_util_cutoff=0.25,
                                          high_weight=LOSS_CLOUDY_WEIGHT,
                                          overpredict_penalty=LOSS_OVERPREDICT_PENALTY),
        save_path=f'{OUT_DIR}/{fname}', offset=off)
    models[hname] = model
    results[hname] = dict(val_loss=best, **summarize(*predict_windows(model, test_arr, off)), file=fname)

print('\n' + '=' * 70)
print(f'전체 {(time.time()-all_started)/60:.1f}분  (test 지표는 미래 = _fut 기준)')
print(pd.DataFrame(results).T.to_string())
""")

# ── 10. 평가 ① 현행 서빙 모델과 비교 ──────────────────────────────────────
code(r"""
# ==========================================================================
# 평가 ① — 새 D+1 vs 현행 서빙 D+1 (test 구간, 같은 날짜)
#   각 모델이 서빙에서 실제로 받을 입력을 넣는다:
#     새 모델  = JMA 운량 + KIMG 일사·강수 (_fut)
#     현행 모델 = KIMG 운량·중하층운량·일사·강수 (_kimg)
#   ⚠서빙 후처리(tcog·일 스케일링·야간 마스크) 전의 모델 출력끼리 비교한다.
# ==========================================================================
test_frame = eval_frame(df)
has_old_model = os.path.exists(f'{OLD_MODEL_DIR}/metadata.pkl')
if has_old_model:
    old_metadata = joblib.load(f'{OLD_MODEL_DIR}/metadata.pkl')
    old_scaler = joblib.load(f'{OLD_MODEL_DIR}/MinMax_scaler_solar.pkl')
    old_names = old_metadata['future_features_solar']
    print('현행 서빙 피처:', old_names)
else:
    print(f'{OLD_MODEL_DIR} 없음 — 현행 모델 비교 행은 생략한다')

if has_old_model:
    old_model_d1 = load_solar_model(f'{OLD_MODEL_DIR}/best_patchtst_solar_model.pth',
                                    len(old_metadata['features_solar']))
    new_arr, old_arr = on_common_dates(
        test_arr, build_array(test_frame, old_names, old_scaler, OLD_MODEL_SUFFIX))
    compare = {'새 모델 D+1': summarize(*predict_windows(models['D1'], new_arr, 0)),
               '현행 서빙 D+1': summarize(*predict_windows(old_model_d1, old_arr, 0))}
else:
    compare = {'새 모델 D+1': summarize(*predict_windows(models['D1'], test_arr, 0))}
print(pd.DataFrame(compare).T.round(4).to_string())
""")

# ── 11. 평가 ② D+1 가중치 재사용 ─────────────────────────────────────────
code(r"""
# ==========================================================================
# 평가 ② — D+1 가중치를 D+2 에 그대로 쓰면? (사용자 가설 검증)
#   새 모델 입력 = JMA 운량 D+2 + KIMG 12z D+2 일사·강수 (_fut_d2), 현행 = KIMG 12z D+2 (_kimg_d2).
#   과거 윈도우는 origin(전일 23시)에서 끝난다 — 대상일과 하루가 벌어진다.
#   ⚠D+1 모델은 "과거 끝 바로 다음 24h" 만 배웠다 — persistence 에 기대고 있으면 여기서 깨진다(③ 참고).
#   ⚠JMA 운량 예보(previous run)가 D+2 까지뿐이라 D+3~D+5 는 이 노트북에서 평가할 수 없다.
# ==========================================================================
reuse_rows = {}
for n in EVAL_REUSE_HORIZONS:
    offset = 24 * (n - 1)
    arrays = {'새 D+1 가중치 재사용': (models['D1'],
              build_array(test_frame, future_features_solar, scaler_solar, f'{FUTURE_SUFFIX}_d{n}'))}
    if f'D{n}' in models:
        arrays['새 offset 학습'] = (models[f'D{n}'], arrays['새 D+1 가중치 재사용'][1])
    old_path = f'{OLD_MODEL_DIR}/best_patchtst_solar_model_D{n}.pth'
    if has_old_model and os.path.exists(old_path):
        arrays['현행 서빙'] = (load_solar_model(old_path, len(old_metadata['features_solar'])),
                             build_array(test_frame, old_names, old_scaler, f'{OLD_MODEL_SUFFIX}_d{n}'))
    aligned = on_common_dates(*[arr for _, arr in arrays.values()])
    for (label, (model, _)), arr in zip(arrays.items(), aligned):
        reuse_rows[f'D+{n} {label}'] = summarize(*predict_windows(model, arr, offset))
print(pd.DataFrame(reuse_rows).T.round(4).to_string())
""")

# ── 12. metadata + 패키징 ─────────────────────────────────────────────────
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
    'retrained': '2026-10-08 solar (미래 입력: 운량·일사·강수 전부 JMA 원본(QM 없음) / 피처: 일사·운량+Hour'
                 + ('+Year' if USE_YEAR_FEATURES else '') + ' / train<=2024 val=2025 test=2026~ / wind 미학습)',
    'future_cloud_source': 'JMA',
    'future_radiation_source': 'JMA',
    'use_year_features': USE_YEAR_FEATURES,
    'loss_weights': {'cloudy': LOSS_CLOUDY_WEIGHT, 'overpredict': LOSS_OVERPREDICT_PENALTY},   # ★서빙은 운량을 JMA 예보에서 읽어야 한다
    'recency_half_life_years': RECENCY_HALF_LIFE_YEARS,
    'train': f'<={TRAIN_END}', 'val': f'~{VAL_END}', 'test': f'~{TEST_END}',
    'horizons_solar': HORIZONS,
}
joblib.dump(metadata, f'{OUT_DIR}/metadata.pkl')
print('saved metadata.pkl | solar num_features =', len(features_solar))

import shutil
shutil.make_archive('/content/solar_retrain', 'zip', OUT_DIR)
print('\n산출물:')
for f in sorted(os.listdir(OUT_DIR)):
    print('  ', f, f'{os.path.getsize(os.path.join(OUT_DIR,f))/1e6:.1f}MB')
print('\nzip -> /content/solar_retrain.zip')
try:
    from google.colab import files
    files.download('/content/solar_retrain.zip')
except Exception as e:
    print('자동 다운로드 실패 — 좌측 파일창에서 solar_retrain.zip 을 직접 내려받을 것:', e)
""")

# ── 13. 평가 ③ permutation importance ────────────────────────────────────
code(r"""
# ==========================================================================
# 평가 ③ — Permutation importance (확인용, 새 D+1 모델, test 구간 낮 시간 MAE)
#   미래 피처는 하나씩, 24h 시퀀스를 통째로 다른 날과 맞바꾼다(하루 안의 시간 구조는 유지).
#   과거 이용률·과거 기상은 한 묶음씩 섞어 persistence 의존도를 본다.
#   ⚠평가 윈도우가 전부 00시 시작이라 미래 Hour_sin/cos 는 모든 날이 같다 → 섞어도 0 이 정상.
# ==========================================================================
REPEATS = 3
importance_dataset = PatchTSTDatasetH(test_arr, SOLAR_HP['seq_len'], PRED_LEN, offset=0,
                                      start_hours=SERVING_START_HOURS, future_start_min=VAL_END)
samples = [importance_dataset[i] for i in range(len(importance_dataset))]
full_batch = {k: torch.stack([s[k] for s in samples])
              for k in ['past_numeric', 'past_y', 'future_numeric', 'future_y']}
importance_daytime = np.stack([test_arr['is_daytime'][s['future_start']: s['future_start'] + PRED_LEN]
                               for s in samples])


@torch.no_grad()
def daytime_mae(model, batch):
    pred = model(batch).clamp(0, 1).cpu().numpy()
    return np.abs(pred - batch['future_y'].numpy())[importance_daytime].mean()


model_d1 = models['D1'].eval()
baseline_mae = daytime_mae(model_d1, full_batch)
n_features = len(future_features_solar)
groups = [(f'미래 {name}', 'future_numeric', [j]) for j, name in enumerate(future_features_solar)]
groups += [('과거 이용률(past_y) 전체', 'past_y', [0]),
           ('과거 기상 전체', 'past_numeric', list(range(n_features)))]

rng = np.random.default_rng(0)
rows = []
for label, key, columns in groups:
    increases = []
    for _ in range(REPEATS):
        order = torch.from_numpy(rng.permutation(len(samples)))
        shuffled = full_batch[key].clone()
        shuffled[:, :, columns] = full_batch[key][order][:, :, columns]
        increases.append(daytime_mae(model_d1, {**full_batch, key: shuffled}) - baseline_mae)
    rows.append({'피처': label, 'MAE 증가': np.mean(increases), '표준편차': np.std(increases)})

importance = pd.DataFrame(rows).sort_values('MAE 증가', ascending=False).reset_index(drop=True)
scale = max(importance['MAE 증가'].max(), 1e-9)
importance['막대'] = importance['MAE 증가'].clip(lower=0).map(lambda v: '█' * int(round(30 * v / scale)))
print(f'기준 낮시간 MAE = {baseline_mae:.4f}  ({len(samples)}일, {REPEATS}회 반복)')
print(importance.round(4).to_string(index=False))
""")

# ── 14. 반입 안내 ─────────────────────────────────────────────────────────
md(r"""
## 반입 (로컬에서) — ① ② 결과를 보고 결정한 뒤에

**먼저 현행 가중치를 백업한다** (08-25 에 백업 없이 되돌려 신 가중치를 잃은 전례).

`solar_retrain.zip` 의 D+1 세트를 **한 번에** 옮긴다:
```
models/solarwind_patchtst/
    best_patchtst_solar_model.pth   MinMax_scaler_solar.pkl   metadata.pkl
```

### ⚠ 반입 전제: 서빙이 JMA 운량 예보를 받아야 한다
새 모델은 운량을 JMA 로 배웠다. 지금 서빙(`forecast` 테이블)의 운량은 KIMG 다 —
**JMA 운량 예보 수집·서빙 경로를 먼저 만들기 전에는 반입하지 않는다.**

### ⚠ D+2~D+5 를 반드시 같이 정한다 (사용자 확정: D+1 가중치로 D+1~D+3, D+4·D+5 는 LGBM)
옛 `models/solarwind_patchtst_horizon/best_patchtst_solar_model_D{2..5}.pth` 는 옛 피처(10개) 기준이라
새 metadata(6개)로 로드되지 않는다 — `serve_solarwind._assets()` 가 try 밖이라 서빙이 멈춘다.
- 새 D+1 `.pth` 를 `best_patchtst_solar_model_D2.pth`·`_D3.pth` 이름으로 복사.
- `serve_solarwind.SOLAR_PT_HORIZONS = [2, 3]` (D+4·D+5 는 LGBM 폴백). 옛 D4·D5 파일은 백업 폴더로.

`MinMax_scaler_wind.pkl` · `best_patchtst_wind_model.pth` 는 건드리지 않는다.

남은 서빙 수정은 운량 출처(JMA)뿐.

### 검증
```bash
python forecasting/serve_chain.py --utc 12 --no-write     # 120행 hd 1~5, solar 소스가 patchtst 인지
```
그 다음 구 모델을 복원해 같은 조건으로 재실행한 A/B 를 돌린다 (`est_horizon_jeju` 저장값으로 A/B 금지).
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
