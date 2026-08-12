# KIMG met 백필 — **완료** (실행 2026-08-12)

> 이 문서는 백필 **실행 안내**로 쓰였고, 지금은 **결과와 후속 판단**을 담는다.
> 실행 결과: `forecast_kimg` **33,660행 / 237 base** (met 185 base / 확장변수 173 base).
> 남은 것: 2026-07 의 12 base 가 구 13변수 세트다 (`--force` 로 채우면 완결).

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

## 백필 후 실측 결과 (2026-08-12)

### ★먼저 본 것 — 자료 건전성
물리 범위 위반·sentinel·이슬점>기온·돌풍<풍속 **전부 0건**, 결측 0.10~0.14%,
KIMR 과 정합 기온 r 0.9896 / 지표온 r 0.9687.  **백필분은 믿고 써도 된다.**

### ★그런데 근본 한계를 찾았다 — 일변화 진폭 압축
| | 주야 진폭 |
|---|---|
| **실측 기온** | **3.36°C** |
| KIMG 기온 | 1.28°C (38%) |
| KIMR 기온 | 0.84°C (25%) |
| 지표온 | 0.73 / 0.39°C |
| 경계층고도 | 62 / 43m |

두 모델 공통이라 NWP 특성이다(제주=작은 섬).  **그래서 위 "탐구해 볼 만한 것" 중
`temp_skin`·`hpbl` 은 기대를 낮춰야 한다** — 거의 평평해서 패널온도·혼합층 발달을
담지 못한다.

대신 **시각별 편향 보정**이 유망하다: 편향이 지평엔 안정(D+1~5 거의 동일)해서
보정이 쉽고, 시각 보정만으로 **KIMR 기온 MAE −13.5%**(KIMG −2.7%).
★보정 후엔 KIMR 이 KIMG 를 이긴다 — **소스 교체보다 지렛대가 크다.**
(단 계절엔 불안정하니 고정 보정표는 드리프트에 취약. 서빙 A/B 필수.)

## 소스 비교 — 무엇을 정할 수 있나

**서빙 체인 소스 구조**를 실측으로 정할 수 있다. 사용자 제안:
①KIMG 단독 ②결측 시 KIMR 대체 ③KIMG 수집 실패 시 KIMR 전체.

구조(폴백 사다리)는 타당하다. 특히 **②③은 지금 없는 것**이다 — 현재 일사·운량에는 폴백이
아예 없어서 KIMG 가 빠지면 NULL 로 남는다(2026-07-02~11 사고가 정확히 그것).

다만 **①(met 도 KIMG 우선)은 변수별로 갈린다.** 백필 후 제대로 잰 결과
(18,969행 / 162 base / 2026-02~08, 지평 5 × 월 6 = 30셀):

| 변수 | 우세 | 일관성 |
|---|---|---|
| 기온 | KIMG (7~11%) | 지평 5/5, 월 5/6 |
| **풍속 west/east** | **KIMR (10~16%)** | **지평 5/5, 월 6/6** |
| 습도 | KIMG (9%) | 지평 5/5, 월 4/6 |

**풍속은 30셀 전부 KIMR 이다.** 그리고 하필 풍력 모델 입력이 `wind_spd_10m_west/east`
(`serve_solarwind_lgbm.FORE_MAP`)다 → "KIMG만 사용"은 풍속에서 손해.

★**결정은 보류한다** — 위 일변화 발견 때문이다.  시각 보정을 하면 기온 우열이
뒤집히므로(KIMR 이 이긴다), **보정을 먼저 넣고 그 다음에 소스를 정하는 게 맞다.**

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
