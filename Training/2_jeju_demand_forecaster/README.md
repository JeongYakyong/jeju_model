# 제주 D+1 24h 수요 예측 — 운영 패키지

LightGBM + PatchTST 기반 D+1 24시간 수요 예측 파이프라인 (운영 추론 전용).
학습 코드/노트북은 포함하지 않는다 — 모델 산출물만 들고 다닌다.

## 폴더 구성

```
final_demand_model/
├── demand_predict.py        # 메인 CLI/임포트 엔트리포인트 (LGBM iterative 예측)
├── patchtst_predict.py      # PatchTST D+1 24h 1회 추론 모듈
├── models/
│   ├── lgbm_pipeline.pkl         # pickle 된 LightGBM booster
│   ├── pipeline_config.json      # 피처 스키마 + best_iteration
│   ├── patchtst_demand.pth       # PatchTST 가중치
│   └── patchtst_demand_meta.pkl  # PatchTST HP
└── README.md
```

## 의존성

- Python 3.10+
- `numpy`, `pandas`, `lightgbm`, `torch`

```bash
pip install numpy pandas lightgbm torch
```

## 입력 CSV 스키마

**`--history`** (실측 수요)
- 컬럼: `timestamp`, `real_demand`
- D+1 직전 최소 28일(672시간) 포함, 그 구간에 NaN 없어야 함

**`--weather`** (D+1 24시간 기상예보)
- 컬럼: `timestamp`, `temp_c`, `humidity`, `solar_rad`, `wind_spd`, `day_type`
- 24행 (D+1 00:00 ~ 23:00, 1시간 간격)
- `day_type` ∈ `{weekday, weekend, holiday}`

## 출력

`--out` 경로에 `timestamp`, `est_demand_new` 두 컬럼 / 24행 CSV.

## 사용법

### (A) CLI

```bash
python demand_predict.py \
    --history data/history_demand.csv \
    --weather data/forecast_weather_d1.csv \
    --out     data/pred_d1.csv
```

### (B) 파이썬 임포트

```python
from demand_predict import predict_24h

df = predict_24h(
    history_path='data/history_demand.csv',
    weather_path='data/forecast_weather_d1.csv',
)
# df : 24행, columns = ['timestamp', 'est_demand_new']
```

## 모델 성능 (참고)

Test MAPE 4.29% — KPX 동일 구간 대비 −1.73%p.
