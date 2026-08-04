# `demand_predict.predict_24h` — 입력 데이터 스펙

`from demand_predict import predict_24h` 로 임포트해서 호출할 때, 호출자(메인 코드)가
준비해서 넘겨줘야 하는 두 개의 CSV 파일에 대한 상세 명세.

```python
predict_24h(
    history_path: str,     # 아래 (1) 스펙
    weather_path: str,     # 아래 (2) 스펙
    out_path:     str | None = None,   # 지정시 결과 CSV 저장 (선택)
    target_date:  str | None = None,   # 'YYYY-MM-DD' (생략시 weather 첫행 날짜)
    verbose:      bool = False,
) -> pd.DataFrame    # 24행, ['timestamp', 'est_demand_new']
```

용어
- **D**       : 오늘 (예측 직전 마지막 실측일)
- **D+1**     : 예측 대상일 (24시간)
- 시간 단위는 모두 **1시간**, 시각은 **정시(HH:00)**

---

## (1) history CSV — 실측 수요 시계열

### 컬럼

| 컬럼          | 타입                  | 설명                              |
|---------------|-----------------------|-----------------------------------|
| `timestamp`   | datetime (파싱 가능)   | 1시간 간격, 정시                  |
| `real_demand` | float                 | 시간당 수요 (MW). 0 이상 실수      |

`pd.read_csv(..., parse_dates=['timestamp'])` 로 읽을 수 있어야 한다.
예: `2026-05-25 13:00:00`, `2026-05-25T13:00`, ISO8601 등.

### 시간 범위 — 반드시 만족

D+1 의 자정을 `T = target_date 00:00` 이라 하면, history 는

```
[T - 28일, T - 1시간]  =  [D-27 00:00, D 23:00]   (정확히 672시간 = 28일)
```

이 구간을 **빠짐없이** 포함해야 한다. 더 긴 history(예: 1년치)는 OK — 필요 구간만 슬라이스함.

> ⚠️ 위 28일 구간에 `real_demand` NaN 이 있으면 즉시 에러. 호출 전에 보간 필요.
> (프로젝트 규칙: `dropna()` 금지, time-based 보간을 쓸 것)

### 정렬 / 중복

- 정렬: 안 해도 됨 (내부에서 `sort_values('timestamp')` 함)
- 중복 timestamp: 보장 안 됨 — 호출자가 미리 제거할 것

### 최소 예시 (CSV)

```
timestamp,real_demand
2026-04-28 00:00:00,612.4
2026-04-28 01:00:00,598.1
...
2026-05-25 22:00:00,701.3
2026-05-25 23:00:00,688.9
```

→ 이 history 로는 `target_date=2026-05-26` 의 D+1 예측 가능.

---

## (2) weather CSV — D+1 24시간 기상예보

### 컬럼

| 컬럼         | 타입       | 설명                                                 |
|--------------|-----------|------------------------------------------------------|
| `timestamp`  | datetime  | D+1 의 정시 24개 (00:00 ~ 23:00, 1시간 간격)          |
| `temp_c`     | float     | 기온 (°C)                                            |
| `humidity`   | float     | 상대습도 (%) — 학습 시 0~100 스케일 가정              |
| `solar_rad`  | float     | 일사량 — 학습 시 사용 단위 그대로 (≥ 0)               |
| `wind_spd`   | float     | 풍속 (m/s) — ≥ 0                                      |
| `day_type`   | string    | `'weekday'` / `'weekend'` / `'holiday'` 중 하나       |

### 행 수 / 시간 — 반드시 만족

- 정확히 **24행** (D+1 00:00 ~ 23:00)
- 1시간 간격, 누락/중복 없이 연속

### `day_type` 카테고리 (정확히 일치해야 함)

학습 시 등록된 카테고리는 다음 셋뿐 — 다른 값(`'sat'`, `'sun'`, `''`, `NaN`) 은 에러.

```
'holiday'   공휴일 (대체공휴일 포함)
'weekday'   평일 (월~금, 공휴일 아님)
'weekend'   토/일 (공휴일 아님)
```

### 결측

기상 5개 컬럼의 NaN 처리는 LightGBM 이 native 로 받음 → 호출자가 미리 채울 필요는 없음.
다만 예보가 비어 있으면 정확도는 떨어진다.

### 최소 예시 (CSV)

```
timestamp,temp_c,humidity,solar_rad,wind_spd,day_type
2026-05-26 00:00:00,17.2,68.0,0.0,3.1,weekday
2026-05-26 01:00:00,16.8,71.0,0.0,2.7,weekday
...
2026-05-26 22:00:00,18.4,75.0,0.0,3.5,weekday
2026-05-26 23:00:00,18.0,77.0,0.0,3.2,weekday
```

---

## (3) `target_date` 결정 규칙

- `target_date` 인자를 명시하면 그 날짜를 D+1 로 본다.
- 생략하면 `weather_df['timestamp'].iloc[0].normalize()` 로 추정 (첫 행의 날짜).
- weather CSV 의 timestamp 가 `target_date` 의 어떤 행도 안 가지면 에러.

권장: 운영 코드에선 `target_date='YYYY-MM-DD'` 명시.

---

## (4) 내부에서 자동 계산되는 피처 (호출자가 줄 필요 없음)

아래는 `demand_predict.py` 가 두 CSV 로부터 알아서 만든다.

| 피처                          | 계산 방식                                     |
|-------------------------------|-----------------------------------------------|
| `hour_sin`, `hour_cos`        | `timestamp.hour` 의 sin/cos (주기 24)         |
| `dow_sin`, `dow_cos`          | `timestamp.dayofweek` 의 sin/cos (주기 7)     |
| `lag_24h`                     | 24시간 전 실측/예측값 (iterative)              |
| `roll_mean_24h`               | 직전 24시간 평균 (iterative, shift(1))         |
| `patchtst_target`             | PatchTST 가 history 로부터 만든 D+1 24h 예측  |

피처 스키마 전체는 `models/pipeline_config.json` 의 `feature_cols` 12개로 고정되어
있고, 학습/추론 시 동일 순서·동일 카테고리 순서를 자동 복원한다.

---

## (5) 출력

```
DataFrame, 24행
columns = ['timestamp', 'est_demand_new']
- timestamp        : D+1 의 00:00 ~ 23:00 (weather 입력과 동일)
- est_demand_new   : 예측 수요, float, 소수 3째자리 반올림
```

`out_path` 지정 시 같은 내용을 CSV 로도 저장 (디렉터리 자동 생성).

---

## (6) 흔한 에러 메시지 ↔ 원인

| 에러 메시지(요약)                                    | 원인                                  |
|------------------------------------------------------|---------------------------------------|
| `[weather] D+1 24행이어야 함`                        | weather 가 24행이 아니거나 간격 불일치 |
| `[weather] CSV 에 target_date=... 행이 없음`         | weather 의 날짜와 target_date 불일치   |
| `[history] PatchTST 입력 윈도우 ... 부족`            | history 가 D+1 직전 28일을 안 덮음    |
| `[history] PatchTST 입력 윈도우에 real_demand NaN`   | 28일 구간에 NaN — 보간 후 재시도       |
| `[day_type] 학습 시점에 없던 카테고리 {...}`         | weekday/weekend/holiday 외 값         |
| `lag_24h 계산용 history 누락`                        | history 가 D-1 자정 직전을 안 덮음     |
| `모델 파일 없음 / config 누락`                       | `models/` 폴더 경로/파일 확인         |

---

## (7) 메인 코드 호출 예시

```python
import pandas as pd
from demand_predict import predict_24h

# 1) 호출자가 두 CSV 를 만들어 둔다 (또는 내부에서 df 를 CSV 로 dump 해도 됨)
HISTORY = 'inputs/history.csv'        # timestamp, real_demand  (28일 이상)
WEATHER = 'inputs/weather_d1.csv'     # timestamp, temp_c, humidity, solar_rad, wind_spd, day_type  (24행)

# 2) 호출
pred_df = predict_24h(
    history_path = HISTORY,
    weather_path = WEATHER,
    target_date  = '2026-05-26',
    out_path     = 'outputs/pred_2026-05-26.csv',   # 생략 가능
    verbose      = False,
)

# 3) 결과 사용
print(pred_df.head())
#              timestamp  est_demand_new
# 0  2026-05-26 00:00:00         612.345
# 1  2026-05-26 01:00:00         598.102
# ...
```
