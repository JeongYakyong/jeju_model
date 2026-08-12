# KIMG met 백필 안내 (2026-08-04)

`forecast_kimg` 의 met 이 **12 base 뿐이라** KIMR↔KIMG 소스 비교가 불가능했다
("KIMG met 은 설계상 안 쓴다"로 수집을 안 했고, 나머지 224 base 는 `forecast_horizon` 의
일사·운량만 seed 한 것이다). 서빙 체인의 소스 구조를 정하려면 met 이 쌓여야 한다.

## 무엇이 바뀌었나 — 수집 변수 13 → 22 (호출수 증가 0)

22변수가 **한 콜에 전부** 온다. 그래서 백필 비용은 변수 수와 무관하다.

| 변수 | varn | 단위 | 왜 받나 |
|---|---|---|---|
| `dswrsfc` | 51 | W/m² | 일사 (기존) |
| `t2m` | 25 | K | 기온 (기존) |
| `tcld` `mcld` `lcld` | 37 35 34 | 0~1 | 운량 (기존) |
| `u10m` `v10m` `u80m` `v80m` `gust` | 20 21 22 23 24 | m/s | 바람 (기존) |
| `rh2m` | 26 | % | 상대습도 (기존) |
| `rainc_acc` `rainl_acc` | 65 66 | kg/m² | 누적 강수 (기존) |
| **`tsfc`** | 19 | K | 지표온 — **KIMR `TSKIN` 과 같은 컬럼(`temp_skin_*`)** 이라 소스 비교가 join 한 번 |
| **`hcld`** | 36 | 0~1 | **상층운(권운)** — 일사 투과에 크게 작용하는데 그동안 없었다 |
| **`dlwrsfc`** | 50 | W/m² | 하향 장파복사 — 야간 복사냉각·운량 대리 지표 |
| **`shtfl`** | 46 | W/m² | 현열속 — 대류 활동 |
| **`lhtfl`** | 47 | W/m² | 잠열속 — 증발·습윤 |
| **`td2m`** | 30 | K | 이슬점 — 안개·하층운 형성 |
| **`q2m`** | 28 | kg/kg | 비습(절대습도) — `rh2m` 은 상대습도라 기온과 얽힌다 |
| **`ustar`** | 39 | m/s | 마찰속도 — 난류 강도, 풍력 |
| **`hpbl`** | 38 | m | 경계층 고도 — **KIMR 에도 있다.** 2026-06-13 "KIMG 에 없음" 기록은 **틀렸다** |

추가로 `lcld`/`mcld` 는 그동안 `MIDLOW_CLOUD` 로만 접혀 들어갔는데 **원시값도 따로 저장**한다
(`low_cloud_*` / `mid_cloud_*`). 층별 운량 구조가 일사 투과와 어떻게 붙는지 보려면 raw 가 필요하다.

→ `forecast_kimg` 컬럼 **42 → 75** (신규 11종 × 3지점).

## 실행 (이 순서대로)

```bash
cd C:/Users/bjkim/Documents/GitHub/jeju_model

# ① 본 백필 — 180일. resume-skip 이라 중단해도 다시 돌리면 이어서 한다.
python collectors/collect_archive.py --kimg-only --backfill 180 --point-workers 2

# ② 기존 12 base 는 구 13변수로 이미 차 있어 ① 이 건너뛴다. 새 9변수를 채우려면 --force.
python collectors/collect_archive.py --kimg-only --backfill 30 --force --point-workers 2

# ③ 확인
python collectors/collect_forecast.py --verify
python collectors/selftest_pivot.py
```

### 소요 시간
base 당 **약 3분** (`--point-workers 2` 기준, 120hf × 3지점).
**180 base ≈ 9시간.** 밤에 걸어 두는 게 낫다. 중단돼도 ①은 이어서 하므로 안전하다.

- `--point-workers 3` 은 쓰지 말 것 — 2026-06-16 실측에서 동시성 18(3지점×6워커)이
  KMA 504 를 유발해 hf 가 빠졌다. 2가 안전 상한이다.
- ② 의 `--backfill 30` 은 12 base(2026-07-07~18)를 덮기 위한 넉넉한 범위다.
  `--force` 는 resume-skip 을 끄므로 **중단 시 처음부터**다. 30개면 1.5시간이라 감당된다.

### 로그를 남기려면
```bash
python collectors/collect_archive.py --kimg-only --backfill 180 --point-workers 2 \
  > logs/kimg_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1
```

## 백필이 끝나면 — 무엇을 정할 수 있나

**서빙 체인 소스 구조**를 실측으로 정할 수 있다. 사용자 제안:
①KIMG 단독 ②결측 시 KIMR 대체 ③KIMG 수집 실패 시 KIMR 전체.

구조(폴백 사다리)는 타당하다. 특히 **②③은 지금 없는 것**이다 — 현재 일사·운량에는 폴백이
아예 없어서 KIMG 가 빠지면 NULL 로 남는다(2026-07-02~11 사고가 정확히 그것).

다만 **①(met 도 KIMG 우선)은 변수별로 따져야 한다.** 12 base 예비 측정은 엇갈렸다
(2026-07-08~22, 실측 ASOS 대비 MAE — 표본이 얇고 단일 창이라 **결정 근거로 쓰면 안 된다**):

| 변수 | KIMR(3km) | KIMG(8km) | 우세 |
|---|---|---|---|
| 기온 | 0.8777 | **0.8283** | KIMG (5.6%) |
| 풍속 | **1.6743** | 1.9281 | **KIMR (13.2%)** |
| 습도 | 5.0546 | **3.4549** | KIMG (31.6%) |

180일이 쌓이면 계절을 걸쳐 다시 재고, **변수별로** 우선순위를 정한다.

### 그때 쓸 비교 쿼리 (컬럼명이 같아 join 한 번이면 된다)
```sql
SELECT r.base, r.timestamp, r.horizon_d,
       r.temp_west  AS kimr_temp,  g.temp_west  AS kimg_temp,
       r.hpbl_west  AS kimr_hpbl,  g.hpbl_west  AS kimg_hpbl,
       r.temp_skin_west AS kimr_tskin, g.temp_skin_west AS kimg_tskin,
       h.temp_c_west AS obs_temp, h.wind_spd_west AS obs_wind
FROM forecast_kimr r
JOIN forecast_kimg g USING (base, timestamp, horizon_d)
JOIN historical    h ON h.timestamp = r.timestamp
WHERE r.src_met_proto = 'NC';
```

## 탐구해 볼 만한 것 (신규 변수)

- **`high_cloud` vs 태양광 잔차** — 권운은 총운량에 잡히지만 투과율이 하층운과 전혀 다르다.
  현행 판정지표(일사 P60)가 놓치는 축일 수 있다.
- **`low/mid/high_cloud` 조합** — `total_cloud` 하나로 접기 전의 층 구조.
- **`lwdown`** — 야간 복사냉각의 직접 지표. 수요(난방) 쪽에 붙을 수 있다.
- **`dewpoint` / `shum`** — `rh2m` 은 기온과 얽혀 있어 절대습도가 더 깨끗한 신호일 수 있다.
- **`hpbl` / `ustar` / `shtfl`** — 혼합층 발달 = 대류운 생성. 흐림 편향의 선행 지표 후보.
  ★`hpbl` 은 KIMR 에도 있으니 **두 모델의 경계층 예측이 갈리는 날**을 찾을 수 있다.
- **`temp_skin`(KIMR vs KIMG)** — 패널 온도 proxy. 태양광 효율은 패널 온도에 민감하다.

⚠ **주의**: 이번 세션에서 "입력 지표 개선이 출력 개선으로 자동 전달되지 않는다"를 실증했다
(일사 결합 건). 새 변수로 뭔가 해 볼 때는 **반드시 서빙 A/B 로 출력까지** 재고,
개선분이 **분산 축소**에서 온 게 아닌지 확인할 것.

## 부수 효과 — `forecast_horizon` 도 컬럼이 늘어난다

`collect_forecast`(서빙 입력)와 `collect_archive`(아카이브)는 **같은 피벗**(`pivot._SPEC_KIMG`)을
쓰므로, 다음 `collect_forecast` 실행부터 `forecast_horizon` 에도 신규 컬럼이 붙는다
(KIMG 가 채우는 칸만; KIMR 우선 병합은 그대로).

- **서빙 값은 안 바뀐다** — 스펙에 **뒤로만 추가**했고, `_derive_point` 는 스펙 순서대로
  컬럼을 만들 뿐이라 앞 컬럼의 값·순서에 영향이 없다. `selftest_pivot` ①-2 가 이를 잡는다.
- 늘어난 컬럼은 서빙이 안 읽으므로 무해하고, 나중에 서빙 시점 후처리에 쓸 수 있다.
