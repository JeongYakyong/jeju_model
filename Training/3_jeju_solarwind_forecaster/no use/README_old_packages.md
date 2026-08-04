# Jeju Forecasting Pipeline (Stages 1 & 2)

Two-stage forecasting pipeline for Jeju Island electricity:

- **Stage 1 — `net_load_forecaster`** — PatchTST-based net-load (demand − solar − wind) forecaster.
  Extracted from the original Streamlit dashboard so the forecasting core is importable and free of UI coupling.
- **Stage 2 — `smp_forecaster`** — SMP (real-time price) predictor built on top of Stage 1.
  Production model = **model9 핵심13@재선택** (13 features, TAU_SOFT=0.06 / TAU_HARD=0.50, optimized for negative-SMP detection).

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  KPX / KMA APIs     │ →  │  Stage 1             │ →  │  Stage 2             │
│  power + weather    │    │  PatchTST + net_load │    │  LightGBM + 핵심13   │
│                     │    │  → forecast_data DB  │    │  → 24h SMP forecast  │
└─────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

## Layout

```
new_project/
├── net_load_forecaster/        # Stage 1 package
│   ├── __init__.py             #   public API
│   ├── config.py               #   paths, constants (KST, JEJU_LAT/LON)
│   ├── architecture.py         #   PatchTST_Weather_Model
│   ├── loader.py               #   load_assets()
│   ├── db_manager.py           #   JejuEnergyDB (SQLite)
│   ├── api_fetchers.py         #   KPX/KMA fetchers
│   ├── data_pipeline.py        #   prepare_model_input + daily_* + run_model_prediction
│   ├── predict.py              #   high-level predict()
│   └── net_load.py             #   compute_net_load + compute_net_load_for_date
├── smp_forecaster/             # Stage 2 package
│   ├── __init__.py             #   public API
│   ├── config.py               #   CORE_COLS (13 features), TAU_*, time windows
│   ├── db_extension.py         #   realtime_smp table (functional add-on to JejuEnergyDB)
│   ├── data_pipeline.py        #   clean_rt_smp.csv → DB realtime_smp
│   ├── features.py             #   build_features(src, mode='train'|'serve')
│   ├── pipeline.py             #   BANK + inject_noise + fit_calibrated + build_pipeline
│   ├── combine.py              #   2-tier threshold decision rule (anchor=True, neg='two')
│   ├── train.py                #   end-to-end training orchestrator
│   ├── loader.py               #   memoized model loader
│   └── predict.py              #   predict_smp(date) → 24h DataFrame
├── models/                     # PatchTST weights + scalers + smp_model.pkl
├── database_input/             # external originals (CSVs + backup DB) — read-only
├── database_output/            # operational DB (jeju_energy.db) — read/write
├── examples/
│   ├── run_forecast.py             # Stage 1 CLI demo
│   ├── run_smp_forecast.py         # Stage 2 CLI demo
│   ├── _gen_smp_demo.py            # notebook generator
│   └── smp_forecast_demo.ipynb     # Stage 2 demo notebook (20 cells, with TEST eval)
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in KMA_API_KEY + KPX_API_KEY
```

Required packages added for Stage 2: `lightgbm`, `holidays`. The rest is shared with Stage 1.

---

# Stage 1 — `net_load_forecaster`

## Public API

```python
from net_load_forecaster import (
    fetch_data,                  # KPX + KMA collection → SQLite
    predict,                     # PatchTST inference → forecast_data table
    compute_net_load_for_date,   # demand − solar_pred − wind_pred
    compute_net_load,            # pure function for custom inputs
    load_assets,                 # raw (solar, wind, scalers, meta, device)
    JejuEnergyDB,                # DB handle (for reading weather data)
)
```

### End-to-end

```python
fetch_data('2026-05-08', '2026-05-21', kind='historical')   # past 14 days
fetch_data('2026-05-22', '2026-05-22', kind='forecast')     # target day weather

pred_df = predict('2026-05-22')
print(pred_df)
#                       est_Solar_Utilization  est_Wind_Utilization  est_demand
# 2026-05-22 00:00:00              0.00                  0.42           560
# 2026-05-22 01:00:00              0.00                  0.38           545
# ...

from net_load_forecaster import JejuEnergyDB, DB_PATH
db = JejuEnergyDB(str(DB_PATH))
net_load_df = compute_net_load_for_date('2026-05-22', db)
print(net_load_df)
#                       demand_mw  solar_pred_mw  wind_pred_mw  renew_pred_mw  net_load_mw
# 2026-05-22 00:00:00      560.0           0.0          189.0         189.0        371.0
# ...
```

## CLI / 명령줄 사용법

```bash
python -m examples.run_forecast --help
```

### `predict DATE` — 예측 결과 받기

필요한 데이터를 자동으로 수집한 뒤 모델을 돌려서 net_load까지 출력합니다.

```bash
python -m examples.run_forecast predict 2026-05-22
```

흐름:
1. 과거 14일 실측 데이터 수집 (KPX 발전 + KMA ASOS)
2. DATE의 예보 데이터 수집 (KMA NCM + KPX SMP)
3. PatchTST 모델 추론 → DB의 `forecast_data` 테이블에 저장
4. `net_load_mw` 시리즈 계산 및 출력

DB에 이미 데이터가 있을 때는:
```bash
python -m examples.run_forecast predict 2026-05-22 --no-fetch
```

### `fetch DATE` — 데이터만 수집

KPX/KMA에서 데이터만 DB로 가져옵니다. 예측은 수행하지 않습니다.

```bash
python -m examples.run_forecast fetch 2026-05-22                      # 실측 14일 + DATE 예보
python -m examples.run_forecast fetch 2026-05-22 --kind historical    # 실측 14일만
python -m examples.run_forecast fetch 2026-05-22 --kind forecast      # DATE 예보만
```

---

# Stage 2 — `smp_forecaster`

## Public API

```python
import smp_forecaster as smp

# 학습 (DB historical+forecast+realtime_smp → BANK → joblib 저장)
summary = smp.train()

# 24h SMP 예측 (forecast_data 의 DATE 예보 → 24h DataFrame)
pred_df = smp.predict_smp('2026-05-22')
print(pred_df)
#                      smp_pred  neg_proba  danger
# 2026-05-22 00:00:00   139.81     0.001       0
# 2026-05-22 09:00:00     0.00     0.061       1   ← 봄 한낮 음수 위험띠
# ...

# RT SMP CSV → DB realtime_smp 적재 (1회용)
smp.ingest_rt_smp()

# 저장된 모델 로드 (메모이즈)
model, artifact = smp.load_model()
```

### Data split (TRAIN / VAL / TEST)

| Split | Window | Source | Role |
|---|---|---|---|
| TRAIN | 2024-06-01 ~ 2026-01-31 | `historical_data` + `realtime_smp` | LGBM 2-classifier + residual regression 학습 |
| VAL   | 2026-02-01 ~ 2026-05-23 | `historical_data` ∩ `forecast_data` | BANK 구성 (실측 vs 예보 잔차) |
| TEST  | 2026-02-01 ~ 2026-05-23 | `forecast_data` + `realtime_smp` | out-of-sample 평가 |

Target = `clean_rt_smp.csv` 의 `smp_rt_hourly_mean` (시간별 RT SMP 평균).

## CLI / 명령줄 사용법

```bash
python -m examples.run_smp_forecast --help
```

Three subcommands / 세 가지 서브커맨드:

### `ingest` — RT SMP CSV → DB 적재 (1회성)

```bash
python -m examples.run_smp_forecast ingest                    # config.RT_SMP_CSV (= AX_model2/clean_rt_smp.csv)
python -m examples.run_smp_forecast ingest --csv /path/to/clean_rt_smp.csv
```

`clean_rt_smp.csv` 의 `smp_rt_hourly_mean` (타깃) + `smp_rt_neg_flag` (음수표시) 컬럼을 DB `realtime_smp` 테이블에 UPSERT.

### `train` — 모델 학습

```bash
python -m examples.run_smp_forecast train
```

흐름:
1. DB 에서 historical (TRAIN+VAL 창) + forecast (VAL 창) + realtime_smp 읽기
2. VAL hist∩forecast 잔차로 BANK (시각별 표본 은행) 구성
3. TRAIN historical 의 Solar/Wind utilization 에 BANK 노이즈 주입
4. `build_features` → 13개 핵심 피처 추출
5. LGBM 분류기 2개 (`floor<5`, `neg≤0`) + 잔차 회귀 학습
6. `models/smp_model.pkl` 저장

소요 ~1-2분. OOF PR-AUC 같은 진단치가 콘솔에 찍힘.

### `predict DATE` — 24h SMP 예측

```bash
python -m examples.run_smp_forecast predict 2026-05-22
```

`forecast_data` 에서 DATE 예보를 읽어 24h SMP 예측을 출력. `smp_pred / neg_proba / danger` 세 컬럼.

## Demo notebook

```bash
jupyter notebook examples/smp_forecast_demo.ipynb
```

20개 셀: 분할 정의 → 데이터 현황 → ingest/train → BANK 시각화 → 단일 날짜 예측 → 일주일 비교 → **TEST 윈도우 전체 평가** (치명 / 음수재현 / 음수MAE / 전체 MAE 표 + 막대그림). nbconvert 검증됨.

## 일상 운영 권장 흐름

매일 자정 이후 (KMA 예보가 23 KST 이후 발행됨):

```bash
# Stage 1: 데이터 수집 + net_load 예측
python -m examples.run_forecast predict 2026-05-22

# Stage 2: 같은 날짜 24h SMP 예측
python -m examples.run_smp_forecast predict 2026-05-22
```

Stage 2 학습은 분기/반기마다 한 번 (`train` 서브커맨드). 일상 운영은 `predict` 만.

---

## What flows between stages

Stage 1 → Stage 2:
1. **`historical_data` 테이블** (실측 KPX + KMA) — Stage 2 학습 입력.
2. **`forecast_data` 테이블** (PatchTST 예측 결과 + KMA 예보) — Stage 2 BANK 구성 + 예측 입력.

Stage 2 만의 추가:
3. **`realtime_smp` 테이블** (KPX RT SMP) — Stage 2 학습 타깃. `clean_rt_smp.csv` 에서 ingest.

세 테이블 모두 `database_output/jeju_energy.db` 하나에 들어 있다.

## Notes

- **Korean comments preserved.** All user-facing strings inside the modules are in Korean (kept as-is from the original).
- **Stage 1 model specs (from `metadata.pkl`):**
  - Solar: `seq_len=336 (14d)`, `patch_len=24`, `stride=12`, `d_model=256`, `heads=4`, `layers=3`
  - Wind:  `seq_len=72 (3d)`,   `patch_len=12`, `stride=6`,  `d_model=128`, `heads=4`, `layers=2`
  - Both predict 0–1 **utilization rates**; convert to MW by `util × capacity`.
- **Capacity estimation** uses a 720-hour rolling max of `real_solar_gen` / `real_wind_gen`.
- **Stage 2 operating point** = model9 Part D-2 재선택 (`TAU_SOFT=0.06, TAU_HARD=0.50`). 음수(-)SMP 탐지 1순위, 전체 MAE 는 의도적으로 양보.
- **`is_zoneA == 1` (net_load 상위) 시 구조적 DA 통과** — 분류기 출력을 무시하고 smp_jeju 그대로.
- **BANK 는 모델과 함께 joblib 에 저장** — 재학습 시 재현 가능.
- KMA forecasts typically appear after **23:00 KST** — run forecast collection after midnight.
- Historical API requests are capped at **30 days** per call.

## What was dropped vs. `old project/` / `AX_model2/`

Stage 1 (vs `old project/`):
- `app.py`, `pages/`, `components/` — Streamlit UI
- `utils/chart_helpers.py`, `utils/gemini*.py`, `utils/log_utils.py` — UI/LLM helpers
- `run_today_prediction()` — Streamlit-coupled; see `examples/run_forecast.py` for clean equivalent.

Stage 2 (vs `AX_model2/_gen_model9.py`):
- Parts A/B/C ablation loops, `score()`/`worst_cost()`/`n_cat()`/`da_score()`
- gain/permutation importance, `reg_raw` (raw-regression path)
- `model7`/`model8` legacy, `_patch_model7.py`, `eda*.ipynb`
- → 모두 `AX_model2/` 에 그대로 남아 있음 (운영 패키지엔 production-only path 만 추출).
