"""smp_forecast_demo.ipynb 생성기 (편집용 원본).

AX_model2/_gen_model9.py 와 동일한 패턴: md()/code() 헬퍼로 셀을 쌓고
유효한 nbformat 4.5 노트북을 만든다.

사용법:
    python examples/_gen_smp_demo.py
    → examples/smp_forecast_demo.ipynb 생성

노트북은 smp_forecaster 패키지의 ingest → train → predict 전체 흐름을
실측·예보와 비교해 보여준다. 그림은 model9 기준 운영점이 의도대로
음수(-)SMP 를 잡고 있는지 시각적으로 확인할 수 있게 구성했다.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 0. 표지
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
# smp_forecaster 사용 예시 / Stage 2 demo notebook

이 노트북은 `smp_forecaster` 패키지 (Stage 2) 의 전체 흐름을 보여준다.
운영 모델 = **model9 핵심13@재선택** (피처 13개, TAU_SOFT=0.06, TAU_HARD=0.50).

## 데이터 분할 / Data split (확정)

| 분할 | 시간창 | 소스 | 역할 |
|---|---|---|---|
| **TRAIN** | 2024-06-01 ~ 2026-01-31 | `historical_data` (실측) + `realtime_smp` (타깃) | LGBM 분류기 2개 + 잔차 회귀 학습 |
| **VAL**   | 2026-02-01 ~ 2026-05-23 | `historical_data` (실측) + `forecast_data` (예보) | BANK 구성용 (실측 vs 예보 잔차) |
| **TEST**  | 2026-02-01 ~ 2026-05-23 | `forecast_data` (예보) + `realtime_smp` (타깃) | out-of-sample 평가 |

타깃 = `clean_rt_smp.csv` 의 `smp_rt_hourly_mean` (시간별 RT SMP 평균).

흐름:
1. **RT SMP 인제스트** — `clean_rt_smp.csv` (정제본) 를 DB realtime_smp 테이블에 적재.
2. **학습** — TRAIN 윈도우의 historical + realtime_smp 로 학습, VAL 의 hist∩forecast 로 BANK 구성.
3. **24h 예측** — forecast_data 에서 target_date 예보를 읽어 24h SMP 예측.
4. **TEST 평가** — TEST 윈도우 전체를 매일 24h 예측 → realtime_smp 와 비교.

> 데이터 출처 = **현재 라이브 DB** (Stage 1 fetch_data 가 매일 갱신). AX_model2
> 의 동결본과 행 값이 미세하게 다를 수 있다.
''')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 준비
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 1. 준비 / Setup

`new_project/` 위치에서 노트북을 실행하면 패키지 경로가 자동으로 잡힌다
(VS Code / Jupyter Lab 기본 동작). 다른 곳에서 띄울 땐 아래 두 줄로
프로젝트 루트를 sys.path 에 끼워 넣는다.
''')

code(r'''
import sys
from pathlib import Path

# 노트북을 어디서 켜도 new_project/ 가 sys.path 에 있도록 보정
ROOT = Path.cwd()
while ROOT.name and not (ROOT / 'smp_forecaster').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('smp_forecaster 패키지를 찾지 못했다.')
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import smp_forecaster as smp
from net_load_forecaster import JejuEnergyDB

print('smp_forecaster path :', Path(smp.__file__).parent)
print('TAU_SOFT / TAU_HARD :', smp.TAU_SOFT, '/', smp.TAU_HARD)
print('CORE_COLS (13)      :', smp.CORE_COLS)
print()
print('Split windows:')
print(f'  TRAIN: {smp.TRAIN_START.date()} ~ {smp.TRAIN_END.date()}')
print(f'  VAL  : {smp.VAL_START.date()} ~ {smp.VAL_END.date()}')
print(f'  TEST : {smp.TEST_START.date()} ~ {smp.TEST_END.date()}')
''')

# ─────────────────────────────────────────────────────────────────────────────
# 2. DB 현황
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 2. DB 현황 / Data inventory

학습에 필요한 세 테이블:
- `historical_data`  — KPX + KMA 실측 (Stage 1 fetch_data 가 채움).
- `forecast_data`    — Stage 1 PatchTST 예측 결과 + KMA 예보.
- `realtime_smp`     — 시간별 RT SMP (Stage 2 가 `clean_rt_smp.csv` 에서 적재).

세 데이터셋의 시간 범위를 한 줄로 본다. 학습창 / 검증창 / 시험창의
경계를 함께 표시한다.
''')

code(r'''
db = JejuEnergyDB(str(smp.DB_PATH))
cur = db.conn.cursor()
print(f'DB path: {smp.DB_PATH}\n')

ranges = {}
for tbl in ['historical_data', 'forecast_data', 'realtime_smp']:
    cur.execute(f'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {tbl}')
    n, mn, mx = cur.fetchone()
    ranges[tbl] = (n, pd.Timestamp(mn), pd.Timestamp(mx))
    print(f'{tbl:18s}: rows={n:>6}  range={mn}  ~  {mx}')

# 윈도우별 행수 점검
print('\n--- 윈도우별 행수 / per-window row counts ---')
for label, lo, hi in [
    ('TRAIN', smp.TRAIN_START, smp.TRAIN_END),
    ('VAL  ', smp.VAL_START,   smp.VAL_END),
    ('TEST ', smp.TEST_START,  smp.TEST_END),
]:
    counts = {}
    for tbl in ['historical_data', 'forecast_data', 'realtime_smp']:
        cur.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE timestamp BETWEEN ? AND ?",
            (str(lo), str(hi)),
        )
        counts[tbl] = cur.fetchone()[0]
    print(f'{label}: hist={counts["historical_data"]:>5}  '
          f'forecast={counts["forecast_data"]:>5}  '
          f'rt_smp={counts["realtime_smp"]:>5}')
db.close()

# 시간 가용성 그림
fig, ax = plt.subplots(figsize=(12, 2.8))
colors = {'historical_data': 'steelblue', 'forecast_data': 'darkorange',
          'realtime_smp': 'seagreen'}
for i, (name, (n, mn, mx)) in enumerate(ranges.items()):
    ax.barh(i, (mx - mn).days, left=mn, color=colors[name], height=0.5)
    ax.text(mx, i, f'  {n:,}행', va='center', fontsize=9)
ax.set_yticks(range(len(ranges)))
ax.set_yticklabels(list(ranges.keys()))

ax.axvspan(smp.TRAIN_START, smp.TRAIN_END, alpha=0.08, color='steelblue',
           label='TRAIN window')
ax.axvspan(smp.VAL_START, smp.VAL_END, alpha=0.12, color='darkorange',
           label='VAL/TEST window')
ax.legend(loc='lower right', fontsize=8)
ax.set_title('DB 테이블 시간 범위 + 학습/검증/시험 윈도우 '
             '/ Data availability + split windows')
plt.tight_layout()
plt.show()
''')

# ─────────────────────────────────────────────────────────────────────────────
# 3. (필요시) ingest & train
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 3. RT SMP 인제스트 & 학습 / Ingest & train

`realtime_smp` 테이블이 비어 있다면 한 번 인제스트한다. `clean_rt_smp.csv`
의 `smp_rt_hourly_mean` 컬럼이 학습 타깃 = `smp_rt` 로 매핑된다.

`models/smp_model.pkl` 이 없거나 학습 윈도우가 바뀌었다면 학습한다 (보통
1~2분). 이미 학습된 모델이 있다면 이 셀은 건너뛰어도 된다.
''')

code(r'''
# 3-1) RT SMP 적재 (테이블이 없거나 비어 있을 때만)
db = JejuEnergyDB(str(smp.DB_PATH))
try:
    rt_df = smp.get_realtime_smp(db)
    need_ingest = rt_df.empty
finally:
    db.close()

if need_ingest:
    n = smp.ingest_rt_smp()
    print(f'RT SMP {n:,}행 적재 완료 / ingested {n:,} rows')
else:
    print(f'RT SMP 이미 적재됨 / already ingested ({len(rt_df):,}행, '
          f'~ {rt_df.index.max()})')

# 3-2) 모델 학습 (smp_model.pkl 이 없으면)
if not smp.SMP_MODEL_PATH.exists():
    print('학습 시작 / Training...')
    summary = smp.train()
    print('학습 완료 /', summary)
else:
    print(f'학습된 모델 존재 / Model already trained: {smp.SMP_MODEL_PATH}')
    print('재학습이 필요하면 `smp.train()` 를 직접 호출하면 된다.')
''')

# ─────────────────────────────────────────────────────────────────────────────
# 4. 모델 로드 + BANK 점검
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 4. 학습 결과 점검 / Inspect trained model

저장된 아티팩트에는 분류기·회귀 외에도 **BANK** (시각별 예보오차 표본
은행) 가 함께 들어 있다. VAL 윈도우(2026-02-01 ~ 2026-05-23) 에서 실측
이용률과 예보 이용률의 차이를 시각별로 모은 것.

- **태양광**: 낮(10~14시) 잔차 스프레드가 크다 — 예보가 가장 자주 빗나가는
  구간. 학습 단계에서 이 시각의 실측 이용률에 더 큰 노이즈가 들어간다.
- **풍력**: 밤·낮 차이가 덜하지만 변동성 자체가 더 큼. 0~1.5 범위에서
  자유롭게 흔들린다.

표본수는 VAL 일수에 비례한다 (현재 ~110일 → 시각당 ~110 표본).
''')

code(r'''
model, artifact = smp.load_model()
bank = artifact['bank']

print(f'학습 시각 / Trained at  : {artifact["trained_at"]}')
print(f'학습 윈도우 / Train window: {artifact["training_window"]}')
print(f'검증 윈도우 / Val window  : {artifact["val_window"]}')
print(f'바닥값 / floor_val      : {model.floor_val:.2f}')
print(f'깊은음수값 / deep_neg   : {model.deep_neg:.2f}')
print(f'OOF PR-AUC (floor / neg): '
      f'{model.floor_clf.oof_pr_auc:.3f} / {model.neg_clf.oof_pr_auc:.3f}')

# BANK 분포: 시각별 boxplot (Solar / Wind)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for j, (channel, title, ylim) in enumerate([
    ('s', 'Solar_Utilization 예보오차 (forecast - actual)', (-0.6, 0.6)),
    ('w', 'Wind_Utilization  예보오차 (forecast - actual)', (-1.0, 1.0)),
]):
    data = [bank[h][channel] for h in range(24)]
    ax[j].boxplot(data, positions=range(24), widths=0.7,
                  showfliers=False, medianprops=dict(color='red'))
    ax[j].axhline(0, color='k', lw=0.6)
    ax[j].set_title(title)
    ax[j].set_xlabel('시각 / hour')
    ax[j].set_ylim(ylim)
    ax[j].set_xticks(range(0, 24, 3))
plt.tight_layout()
plt.show()

# 시각별 표본수
counts = pd.Series({h: len(bank[h]['s']) for h in range(24)})
print(f'\nBANK 시각별 표본수: min={counts.min()}  median={counts.median():.0f}  '
      f'max={counts.max()}  (총 {counts.sum()}쌍)')
''')

# ─────────────────────────────────────────────────────────────────────────────
# 5. 단일 날짜 예측
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 5. 단일 날짜 24h 예측 / Single-day 24h prediction

봄(3~5월) 한낮은 model9 가 음수(-)SMP 를 가장 잘 탐지하는 시점. 5월
후반의 날짜를 잡고 예측·DA·실제값을 한 그림에 겹쳐 본다.

각 시간에 대해:
- **smp_pred** (모델 출력)
- **DA = smp_jeju** (참고 기준선; 이게 작아지면 보통 RT SMP 도 작아짐)
- **smp_rt_actual** (그 날의 실제 RT SMP; realtime_smp 테이블)
- **neg_proba ≥ TAU_SOFT** 인 시간대를 회색 띠로 강조 (위험띠)
''')

code(r'''
TARGET = '2026-05-22'

pred = smp.predict_smp(TARGET)
print(pred.round(3).to_string())

# DA(smp_jeju) + 실제 RT SMP 같이 그리기
db = JejuEnergyDB(str(smp.DB_PATH))
try:
    fc = db.get_forecast(TARGET, TARGET + ' 23:59')
    fc.index = pd.to_datetime(fc.index)
    da = fc['smp_jeju'].reindex(pred.index)
    rt = smp.get_realtime_smp(db, TARGET, TARGET + ' 23:59')
finally:
    db.close()

actual = rt['smp_rt'].reindex(pred.index) if not rt.empty else None

fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
hours = pred.index.hour

# 위험띠 (회색 반투명 배경)
danger_mask = pred['danger'].values.astype(bool)
for i, is_d in enumerate(danger_mask):
    if is_d:
        ax.axvspan(hours[i] - 0.5, hours[i] + 0.5, color='lightgray',
                   alpha=0.45, zorder=0)

ax.plot(hours, da.values,  'o--', color='gray',       label='DA (smp_jeju)', lw=1.4)
ax.plot(hours, pred['smp_pred'].values, 'o-',
        color='crimson',   label='smp_pred (model9 핵심13)', lw=1.8)
if actual is not None and actual.notna().any():
    ax.plot(hours, actual.values, 's-', color='steelblue',
            label='실제 RT SMP', lw=1.6)

ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(range(0, 24))
ax.set_xlabel('시각 / hour')
ax.set_ylabel('SMP (원/kWh)')
ax.set_title(f'{TARGET}  24h SMP : 예측 vs DA vs 실제 '
             f'(회색 띠 = 위험띠, neg_proba ≥ {smp.TAU_SOFT})')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
''')

md(r'''
### 읽는 법

- 회색 띠로 칠해진 시각은 **모델이 "이 시간에 음수 가격이 나올 가능성이
  있다"고 본 위험띠**. 봄 한낮(09~13시 부근)에 띠가 켜지면 점예측이 0
  (또는 깊은 음수) 로 눌린다. model9 핵심13@재선택의 시그니처 동작.
- 실제 RT SMP 와 비교해서 위험띠 안에서 실제가 정말 0 이하로 갔다면 모델이
  옳게 잡은 것 (`음수재현`). 위험띠 밖에서 실제가 양수면 정상.
- DA(smp_jeju) 는 하루전 시장 가격. 모델이 DA 보다 *낮게* 찍을 때가 음수
  탐지가 일어난 순간이다.
''')

# ─────────────────────────────────────────────────────────────────────────────
# 6. 다일 비교
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 6. 여러 날짜 비교 / Multi-day comparison

TEST 윈도우 끝부분 (2026-05-17 ~ 2026-05-23) 일주일을 본다.
예측·실제·DA 가 어떻게 어긋나는지, 위험띠가 실제 음수 시간을 얼마나 잘
잡았는지 한 그림에서 확인.
''')

code(r'''
start = pd.Timestamp('2026-05-17')
end   = pd.Timestamp('2026-05-23')

preds = []
for d in pd.date_range(start, end, freq='D'):
    try:
        pi = smp.predict_smp(d.strftime('%Y-%m-%d'))
        preds.append(pi)
    except Exception as e:
        print(f'  skip {d.date()}: {e}')
pred_week = pd.concat(preds)

db = JejuEnergyDB(str(smp.DB_PATH))
try:
    fc_w = db.get_forecast(str(start), str(end + pd.Timedelta('1d')))
    fc_w.index = pd.to_datetime(fc_w.index)
    rt_w = smp.get_realtime_smp(db, str(start), str(end + pd.Timedelta('1d')))
finally:
    db.close()

da_w = fc_w['smp_jeju'].reindex(pred_week.index)
actual_w = rt_w['smp_rt'].reindex(pred_week.index) if not rt_w.empty else None

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
danger = pred_week['danger'].values.astype(bool)
ax.fill_between(pred_week.index, -150, 250, where=danger,
                color='lightgray', alpha=0.45, step='mid',
                label=f'danger (neg_proba≥{smp.TAU_SOFT})')
ax.plot(pred_week.index, da_w.values, '--', color='gray',
        label='DA (smp_jeju)', lw=1.2)
ax.plot(pred_week.index, pred_week['smp_pred'].values, '-',
        color='crimson', label='smp_pred', lw=1.6)
if actual_w is not None and actual_w.notna().any():
    ax.plot(pred_week.index, actual_w.values, '-',
            color='steelblue', label='실제 RT SMP', lw=1.2, alpha=0.85)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel('SMP (원/kWh)')
ax.set_title(f'{start.date()} ~ {end.date()} : 일주일 비교 '
             '(회색 띠 = 모델 위험띠)')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(-150, 250)

ax2 = axes[1]
ax2.plot(pred_week.index, pred_week['neg_proba'].values, color='crimson', lw=1)
ax2.axhline(smp.TAU_SOFT, color='gray', ls='--', label=f'TAU_SOFT={smp.TAU_SOFT}')
ax2.axhline(smp.TAU_HARD, color='black', ls=':',  label=f'TAU_HARD={smp.TAU_HARD}')
ax2.set_ylabel('neg_proba')
ax2.set_xlabel('time')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_ylim(0, 1)
plt.tight_layout()
plt.show()
''')

md(r'''
### 읽는 법

- **위 그림** — 회색 띠가 켜진 시간대에서 실제 RT SMP 가 0 이하로 자주
  내려가면 모델이 잘 잡고 있다는 신호. 띠 밖에서 실제가 양수면 정상.
- **아래 그림** — `neg_proba` 시계열. `TAU_SOFT=0.06` 을 넘으면 점예측이 0
  으로 눌리고 (`yhat = 0`), `TAU_HARD=0.50` 을 넘으면 깊은 음수값
  (`-69.76` 부근) 으로 대체된다.
- 위험띠가 매일 같은 시간대에 켜진다 = 달력 피처(`hour`, `month`,
  `spring_midday`) 가 강하게 영향. 5월은 spring_midday=1 이라 한낮(10~13)이
  자동으로 위험구간이 된다.
''')

# ─────────────────────────────────────────────────────────────────────────────
# 7. TEST 평가 (NEW)
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 7. TEST 평가 / Out-of-sample evaluation

TEST 윈도우 (2026-02-01 ~ 2026-05-23) **전체** 를 매일 24h 예측해서
이어 붙이고, `realtime_smp` 에서 같은 시간의 실제값과 비교한다.

- **입력 소스**: `forecast_data` (Stage 1 PatchTST 예보 + KMA 예보)
- **타깃 소스**: `realtime_smp` (KPX RT SMP, `smp_rt_hourly_mean`)
- **비교 기준선 (DA)**: `forecast_data.smp_jeju` (하루전 시장 가격)

평가 코드는 운영 패키지에서 뺐기 때문에 여기서 직접 계산한다.
지표는 model9 보고 기준 음수특화 1순위:

- **치명 건수** = 예측>0 인데 실제≤0 인 시간수
- **음수재현** = (예측≤0 ∩ 실제≤0) / 실제≤0
- **음수정밀** = (예측≤0 ∩ 실제≤0) / 예측≤0
- **음수 MAE** = 실제≤0 시간들의 |예측 − 실제| 평균
- **MAE 전체** = 전체 시간의 |예측 − 실제| 평균
''')

code(r'''
# TEST 윈도우 매일 24h 예측 (115일 ≈ 2~3분)
test_dates = pd.date_range(smp.TEST_START, smp.TEST_END.normalize(), freq='D')
print(f'예측 일수 / Days to predict: {len(test_dates)}')

preds_test = []
fails = []
for d in test_dates:
    try:
        pi = smp.predict_smp(d.strftime('%Y-%m-%d'))
        preds_test.append(pi)
    except Exception as e:
        fails.append((d.date(), str(e)))
print(f'성공 / OK : {len(preds_test)}  실패 / fail: {len(fails)}')
if fails:
    print('실패 사례 / failures:')
    for d, e in fails[:5]:
        print(f'  {d}: {e[:80]}')

pred_test = pd.concat(preds_test)

# 실제값 + DA 가져오기
db = JejuEnergyDB(str(smp.DB_PATH))
try:
    fc_test = db.get_forecast(str(smp.TEST_START),
                              str(smp.TEST_END + pd.Timedelta('1h')))
    fc_test.index = pd.to_datetime(fc_test.index)
    rt_test = smp.get_realtime_smp(db, str(smp.TEST_START),
                                   str(smp.TEST_END + pd.Timedelta('1h')))
finally:
    db.close()

eval_df = pd.DataFrame({
    'smp_pred': pred_test['smp_pred'],
    'neg_proba': pred_test['neg_proba'],
    'da':       fc_test['smp_jeju'].reindex(pred_test.index),
    'actual':   rt_test['smp_rt'].reindex(pred_test.index),
})

# 실제값이 있는 행만 평가에 씀
mask = eval_df['actual'].notna()
e = eval_df[mask]
y     = e['actual'].values
yh    = e['smp_pred'].values
yh_da = e['da'].values

print(f'\nTEST 평가 가능 시간 / hours with actuals: {len(e):,}/{len(eval_df):,}')
print(f'(realtime_smp 가용 범위까지만 비교)')

def metrics(name, y, yh):
    neg = y <= 0
    pred_neg = yh <= 0
    fatal = int(((yh > 0) & (y <= 0)).sum())
    rec  = (pred_neg & neg).sum() / max(int(neg.sum()), 1)
    prec = (pred_neg & neg).sum() / max(int(pred_neg.sum()), 1)
    neg_mae = (np.abs(yh[neg] - y[neg]).mean() if neg.any() else np.nan)
    mae = float(np.abs(yh - y).mean())
    return pd.Series({
        '치명 / fatal':     fatal,
        '음수재현 / recall': round(float(rec), 3),
        '음수정밀 / prec':   round(float(prec), 3),
        '음수MAE':           round(float(neg_mae), 1) if neg.any() else np.nan,
        'MAE 전체':          round(mae, 2),
    }, name=name)

summary = pd.concat([
    metrics('smp_pred (model)', y, yh),
    metrics('DA (smp_jeju)',    y, yh_da),
], axis=1)

print('\n=== TEST 요약 / TEST summary ===')
print(summary.to_string())

# 막대 그림: 핵심 지표 비교
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for j, key in enumerate(['치명 / fatal', '음수재현 / recall',
                         '음수MAE', 'MAE 전체']):
    vals = summary.loc[key]
    bars = ax[j].bar(vals.index, vals.values,
                     color=['crimson', 'gray'])
    ax[j].set_title(key)
    for b, v in zip(bars, vals.values):
        if pd.notna(v):
            ax[j].annotate(f'{v}', (b.get_x() + b.get_width() / 2, v),
                           ha='center', va='bottom', fontsize=9)
    ax[j].tick_params(axis='x', rotation=10)
plt.tight_layout()
plt.show()
''')

md(r'''
### 읽는 법

- **치명건수** — 모델이 양수로 예측했는데 실제는 음수로 빠진 시간수.
  model9 의 1순위 비용지표. 모델이 DA 보다 *훨씬 적게* 나오면 음수 탐지가
  실제로 작동한다는 증거.
- **음수재현** — 실제 음수 시간 중 모델이 음수로 잡은 비율. model9 보고값
  ~0.94 가 목표선. (단, 보고값은 model9 동결본 기준이고 여기는 라이브
  DB 기준이라 미세 차이가 있을 수 있음.)
- **MAE 전체** — model9 핵심13@재선택은 음수특화를 위해 전체 MAE 를
  의도적으로 양보한다. DA 보다 +2~3 정도 위에 있는 게 정상.

### 음수 탐지 시각화 (TEST 전체)

전체 TEST 구간에서 위험띠 시간대와 실제 음수 시간을 점으로 찍어 본다.
파란 점 = 실제 음수 시간, 빨간 점 = 모델이 위험띠로 잡은 시간. 두 점이
겹쳐 보이면 모델이 잘 잡고 있다는 신호.
''')

code(r'''
fig, ax = plt.subplots(figsize=(14, 3.6))
hours = eval_df.index

danger_idx = eval_df.index[pred_test['danger'].astype(bool).values]
ax.scatter(danger_idx, [1] * len(danger_idx), color='crimson',
           s=8, label='모델 위험띠 (neg_proba≥TAU_SOFT)', marker='|')

neg_idx = eval_df.index[eval_df['actual'].fillna(1) <= 0]
ax.scatter(neg_idx, [0] * len(neg_idx), color='steelblue',
           s=8, label='실제 음수 (RT SMP ≤ 0)', marker='|')

ax.set_yticks([0, 1])
ax.set_yticklabels(['실제 음수', '위험띠'])
ax.set_ylim(-0.5, 1.5)
ax.set_title(f'TEST 윈도우 {smp.TEST_START.date()} ~ {smp.TEST_END.date()} : '
             '위험띠 vs 실제 음수')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
''')

# ─────────────────────────────────────────────────────────────────────────────
# 8. 정리
# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 8. 정리 / Wrap-up

- 데이터 분할 = TRAIN(2024-06~2026-01) / VAL(02-01~05-23) / TEST(02-01~05-23,
  forecast 소스). VAL 은 BANK 구성용, TEST 는 out-of-sample 평가용.
- `smp_forecaster` 는 Stage 1 의 forecast_data 만 있으면 임의 날짜의 24h
  SMP 예측을 한 줄 (`predict_smp(date)`) 로 뽑는다.
- 운영점 `TAU_SOFT=0.06 / TAU_HARD=0.50` 는 음수 탐지 1순위로 잡힌 값이라
  봄 한낮 같은 위험 시각엔 점예측이 보수적으로 0 / 깊은음수에 붙는다.
  전체 MAE 는 그 대가로 살짝 양보한다.
- 평가·ablation 코드는 운영 패키지에서 모두 뺐다 (model9 노트북에 그대로
  남아 있음). 운영 검증은 위 §7 처럼 *행동* 으로 본다.

### 다음 할 만한 일

- Stage 1 의 `fetch_data` 를 cron 으로 매일 돌려 DB 를 채우고, 이어서
  `predict_smp(내일)` 을 자동 호출 → 매일 아침 24h SMP 예측을 받는다.
- 새로운 검증창 (예: 다음 분기) 로 BANK 를 새로 만들어 재학습.
- 운영점 재선택이 필요하면 model9 노트북 Part D-2 로직을 옮겨 와 별도
  스크립트화 — 운영 패키지엔 안 넣는다.
''')

# ─────────────────────────────────────────────────────────────────────────────
# 노트북 직렬화
# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent / 'smp_forecast_demo.ipynb'
with io.open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f'{out_path.name} 작성 완료: 셀 {len(cells)}개')
