# jejumodel.md — 진행 기록 (세션 로그)

> 세션이 바뀌어도 맥락을 잃지 않기 위한 진행 메모. 최신 세션이 위.

---

## 2026-08-12 — KIMG 백필(180일) + 수집 안전장치 3축 + 자료 건전성 감사

### 사용자 결정 사항 (재질문 금지)
- **서빙은 단순한 게 최고.** KIMR+KIMG 를 섞는 것 자체는 괜찮다.
  대신 **결측·이상치·수집불가** 셋에 안전장치가 있어야 한다.
- **1 base 통째 수집불가면 소스를 바꾸는 게 100배 낫다** — 오차를 감안하더라도.
  그 오차는 사용자에게 통보하면 그만이다. (내가 "폴백은 위험" 쪽으로 기울었다가 정정됨.)
- **9999 는 이상치가 아니라 데이터 없음** — 그렇게 처리할 것.
- 성능 비교 이전에 **백필 자료의 분포·건전성·신뢰도**를 먼저 확인할 것.
- 파이프라인 실행은 급하지 않다 (서버 offline).

### KIMG 수집 13 → 22변수 (호출수 증가 0)
프로브로 9종을 더 찾았고 **22변수가 한 콜에 전부** 온다.

| 추가 | 단위 | 왜 |
|---|---|---|
| `tsfc` | K | 지표온 — KIMR `TSKIN` 과 **같은 컬럼**(`temp_skin_*`)이라 join 비교 가능 |
| `hcld` | 0~1 | **상층운(권운)** — 총운량엔 잡히나 투과율이 달라 그동안 놓쳤다 |
| `hpbl` | m | 경계층고도 — **KIMR 에도 있다.** 2026-06-13 "KIMG 에 없음" 기록은 **틀렸다** |
| `dlwrsfc` `shtfl` `lhtfl` | W/m² | 하향장파·현열속·잠열속 |
| `td2m` `q2m` | K, kg/kg | 이슬점·비습 (`rh2m` 은 기온과 얽힌다) |
| `ustar` | m/s | 마찰속도 |

`lcld`/`mcld` 원시값도 따로 저장(기존엔 `MIDLOW_CLOUD` 로만 접혔다).
`forecast_kimg` 42 → 75컬럼, selftest 17 → **21항목**.
사용자가 180일 백필 실행 → **33,660행 / 237 base, met 185 base / 확장 173 base**.

### 자료 건전성 감사 — 백필분은 믿고 써도 된다
| 검사 | 결과 |
|---|---|
| 물리 범위 위반 / sentinel / 이슬점>기온 / 돌풍<풍속 | **전부 0건** |
| 층별운량 > 전운량 | 4건 (0.01%) |
| 결측 | 기존 변수 0.10~0.14% / 신규 6.4%(=구 세트 12 base) |
| KIMR 과 정합 | 기온 r **0.9896** · 지표온 r **0.9687** |

"KIMG 가 바다 셀에 떨어졌나" 의심은 **기각**(KIMR x/y 고정본과 지표온 r 0.97).

### ★일변화 진폭 압축 — 이번 세션 최대 발견
| | 주야 진폭 | 실측 대비 |
|---|---|---|
| **실측 기온** | **3.36°C** | — |
| KIMG 기온 | 1.28°C | 38% |
| KIMR 기온 | 0.84°C | **25%** |
| 지표온 | 0.73 / 0.39°C | 거의 평평 |
| 경계층고도 | 62 / 43m | 실제는 수백m |

밤 과대(+1.06)·낮 과소(−1.03)의 전형적 압축이고 **두 모델 공통**이라 제주(작은 섬)의
격자 특성으로 보인다.  실질 의미:
- `temp_skin`·`hpbl` 은 **패널온도·혼합층 신호로 쓰기 어렵다**(내가 `KIMG_BACKFILL.md`
  에 "탐구할 만한 것"으로 적었던 것 중 이 둘은 기대를 낮춰야 한다).
- 시각별 편향이 **지평엔 매우 안정**(D+1~5 거의 동일)하고 **계절엔 불안정**하다.
- **시각 보정만으로 KIMR 기온 MAE −13.5%**(KIMG −2.7%), 적합 2~5월/홀드아웃 6~7월.
  ★**보정 후엔 KIMR(1.0078)이 KIMG(1.0450)를 이긴다** — "기온은 KIMG 우세"라는
  소스 비교 결론이 사실은 **보정 가능한 편향**이었다는 뜻이다.
  → **소스 교체보다 편향 보정이 지렛대가 크다.**

### 소스 비교 (백필 후 제대로) — 변수별로 갈린다
18,969행 / 162 base / 2026-02~08. 지평 5개 × 월 6개 = 30셀 기준:

| 변수 | 우세 | 일관성 |
|---|---|---|
| 기온 | KIMG (7~11%) | 지평 5/5, 월 5/6 |
| **풍속 west/east** | **KIMR (10~16%)** | **지평 5/5, 월 6/6** |
| 습도 | KIMG (9%) | 지평 5/5, 월 4/6 |

풍속은 30셀 전부 KIMR 이다.  그리고 하필 풍력 모델 입력이 `wind_spd_10m_west/east` 다.
→ **"KIMG만 사용"은 풍속에서 손해.** 다만 위 일변화 발견 때문에 **결정은 보류**했다.

### 안전장치 3축 완성
**① 결측 — 수집 사다리**
```
①KIMG → ②시간 보간(연속 2개) → ③KIMR 대체 → ④sentinel·clip·건전성 검사
```
②가 ③보다 먼저인 근거는 **2026-07 사고를 정답과 대조한 자연 실험**이다.
그 구간은 손상본(백업)·정답(재수집본)·후보(KIMR)가 다 있는 드문 사례였다.
결손 780칸에서: 시간 보간 MAE 0.063~0.106(r 0.81~0.90) vs KIMR 대체 0.346~0.490
(r 0.25~0.43, 큰오차 43~59%).  결손 구간이 **전부 연속 2개**라 `limit=2` 가 정확히 맞았다.

③은 그래도 둔다 — KIMG 통째 실패면 보간할 이웃이 없다.  실패 시뮬레이션 실측:
1,062셀 전부 복구, 일사 r 0.9367 / 전운량 r 0.3988.  출처는 `src_solar_cloud` 에 기록.

**② 이상치 — `sanity_check`** (값을 고치지 않고 **드러내기만**)
결측률 · 얼어붙음 · 1h 급변 · 물리 모순(이슬점>기온, 돌풍<풍속, 층별>전운량).
합성 손상 6종 전부 탐지, 정상 base 오탐 0.  `run_region` 이 적재 직전 돌려 rc=1.

**③ sentinel — `drop_sentinels`**
9999 는 결측이다.  구 GRIB base 실측 **cape 평균 2073.0 → NaN 처리 후 256.6**(8배 왜곡).
`clip_ranges` **앞**에서 돈다(뒤면 범위검사·통계에 이미 섞인다).

### 실패 시나리오 테스트가 버그 둘을 잡았다
이 경로는 평소에 절대 안 타서, 흉내 내지 않았으면 **운영 중 KIMG 가 죽었을 때 처음
발견**됐을 것들이다.
- 채운 셀 수를 **행 수와 셀 수로 뒤섞어** 세어 "−510셀 채움"이 나왔다.
- `_substitute_solar_cloud` 가 `forecast_days_override` **밖**에서 불려
  `ckg.FORECAST_DAYS` 가 기본값 2 로 돌아가 창이 118 → 48시각으로 잘렸다.
  (이 파일 주석이 경고하던 바로 그 함정에 내가 걸렸다.)

### 그 밖
- `--verify` rc 반환(불완전 시 1), 수집도 건전성 경고 시 rc=1.
- **`PIPELINE_STEPS` 는 안 바꿨다** — 이번 변경은 전부 각 단계 **내부** 동작.
- 서버 offline 이라 파이프라인은 2026-08-04 이후 미실행(의도된 상태).


---

## 2026-08-04 — Phase 3 완료(KIMR GRIB 폐기) + 3h 결손 보간 + SMP D+2 원인분석

### 사용자 결정 사항 (재질문 금지)
- **GRIB 완전 폐기** — `kma_kimr_grib.py` 파일 + `collect_archive --met grib` 폴백까지 전부 제거,
  KIMR 은 std NC 단일 경로. (검증 통과 후 삭제 승인.)
- **수집 단계 3h 보간 limit=2** (`collect_archive` 와 통일). 3h 결손 = 연속 NaN 2개라 정확히 덮인다.
- KIMG(메인) / KIMR(서브) 구도는 유지 — 일사·운량은 계속 KIMG(NE57).

### Phase 3 — 서빙 입력 met 을 GRIB → std NC 로 전환하고 GRIB 삭제

**막고 있던 미지수를 프로브로 해소**했다:
- 구 GRIB 15종 중 NC 에 없는 건 `TCOH`(우박) 하나뿐. **실측 32,012행 × 3지점 전부 정확히 0** → 손실 0.
- `TCOG`(싸락눈)는 NC 이름이 **`GRAUPEL`** 이고 **기존 변수 세트와 한 콜에** 온다 → 호출수 증가 0.
  (잘못된 이름은 조용히 무시되고 나머지는 정상 반환. 전부 틀리면 NetCDF error.)
- `tcog>0` 은 **겨울에만** 나온다(2025-12~2026-02). 3월 이후 24,668행 전부 0 → 여름엔 보정이 무동작.

**설계 — NC 어댑터가 구 GRIB 라벨을 낸다.** `kma_kimr_nc.fetch_kimr_met_long` 이
`TEMP`(K, 반올림 없음) 같은 옛 카테고리 이름으로 출력한다. 그래야 `pivot._SPEC_KIMR`·
`selftest_pivot`(17항목)·`forecast_horizon` 스키마가 **전부 불변**이고, 전환 검증이
"값이 같은가" 하나로 끝난다. NC per-hf 수집 스택은 `fetch_nc_long` **한 벌**로 통합했고
(운영·아카이브 공용) 차이는 요청 변수 목록과 파생 함수뿐이다.

**검증 (base 2026-07-29, `--out nctest` 격리 DB, 본 DB 무변경 확인)**

| 대상 | 결과 |
|---|---|
| met (temp/wind/reh/gust/hpbl/temp_skin) | **r ≥ 0.99998**, temp 최대차 0.01°C |
| 일사·운량 | **완전 일치 (MAE 0.0)** — KIMG 경로라 당연 |
| 서빙 A/B (est) | 수요 **0.002%** / net_load 0.017% / solar 0.066% / wind 0.379% |
| selftest_pivot | 17항목 통과 / AppTest exception 0 / serve_chain 911MW(기준선 일치) |

### ★GRIB 쪽에 데이터 결함이 있었다 — 폐기가 정리가 아니라 품질 개선이다
`cape`/`cinn` 만 상관이 무너져 파고들었더니 둘 다 GRIB 결함이었다:
- **9999 sentinel**: `forecast_horizon` 전체에서 cape 57.4~57.7%, cinn 68.3~68.8% 가 9999.
- **2바이트 정수 랩어라운드**: 나머지 불일치가 **정확히 655.36 = 2¹⁶/100 의 배수**
  (응답 파일명에도 `2byte` 가 있다). 실제 725.49 를 GRIB 은 70.13 으로 기록.
- 9999 를 빼고 재면 **cape r = 1.000000**(최대차 0.05). NC 는 둘 다 없이 실제 값을 준다.

→ **`forecast_horizon` 의 옛 cape/cinn 은 재학습에 쓰면 안 된다** (절반 이상이 sentinel).
   `forecast_kimr.src_met_proto` 로 걸러낼 것. 서빙은 이 둘을 안 써서 현재 영향은 없다.

**남은 것**: NC `GRAUPEL` 과 구 GRIB `TCOG` 의 **값 일치는 미검증**이다 — 비교 가능한
겨울 구간이 API 보존기간(완전 80일/최대 160일) 밖이다. **2026-12 에 재확인할 것.**

### 3h 결손 보간을 수집 단계에 추가
실측해 보니 사고의 정체가 예상과 달랐다:
- **운영 지평 D+1~5 에 행 누락은 0건.** 3h 구멍 2,503개는 전부 D+6·D+7(KIMG 설계상 3h).
- 진짜 사고는 **행은 있고 컬럼만 NULL** — 2026-07-02~11, 10 base × 78행 `total_cloud_*`.
- 보간은 서빙(`_hourly_interp` limit 3~4)·아카이브(`to_grid` limit 2)엔 이미 있었고
  **`collect_forecast` 에만 없었다.**

→ `to_grid` 를 `postprocess.fill_short_gaps` 로 올려 두 수집기가 공용. limit=2, 외삽 금지.
- **`clip_ranges` 앞에 둬야 한다** — clip_ranges 가 radiation/rainfall NaN 을 0 으로 채우므로
  뒤에 두면 메울 결손이 이미 0 으로 위조된다(기존에 창 첫 시각 강수가 0 으로 박히던 것도 이것).
- 그리드는 `expected_timestamps`(=`collection_hf_range`) 그대로 — 1h/3h 전환을 이미 반영한 SSOT.
- 합성 데이터 검증: 3h 결손 정확히 복원 / 꼬리 외삽 안 함 / 문자열 컬럼 제외.
- 12z·18z 창 대조: NC hf 가 KIMG 그리드의 부분집합(12z 118 vs 120 = KIMR lead 꼬리 2h).

### SMP D+2 원인분석 — 개선안 기각
구조가 답의 절반이었다: `smp_da._predict_da` 는 **잔차회귀** `pred = lag24 + model(잔차)`.

- 전체 MAE 모델 **11.37** vs lag24 13.60 vs lag168 14.25 → 전체로는 모델이 16% 낫다.
- **사후 층화(|실제 잔차| 3분위)**: 안정 구간 모델 4.33 vs lag24 1.46(**3배 나쁨**),
  std(예측잔차)/std(실제잔차) = **4.74** → 참 잔차가 ~0 인데 모델이 ±8.5 로 흔든다. 가설 확인.
- 잔차 예측 자체엔 정보가 있다 — 부호 적중률 0.699, 상관 0.591 (노이즈가 아니다).
- ★**그런데 그 구간을 사전에 식별할 수 없다.** 사전 신호 5종(직전 7일 변동성 / 직전 24h
  변동폭 / |lag24−lag48| / 시각 / 모델 자기확신) 전부 구간 간 모델·lag24 비 편차가
  **0.03~0.27** 뿐이다 (사후 신호는 2.26). 어느 사전 구간에서도 모델이 lag24 에 지지 않는다.
- λ shrinkage 스윕·gating θ 스윕·사전 변동성 분기 **셋 다 전체 MAE 를 악화**시킨다
  (최적 λ=1.0, 최적 θ=0 = 현행). 사전 분기는 11.37 → 12.19.

→ **지난 세션의 "안정 구간 shrinkage / 앵커 블렌딩" 제안은 실행 불가로 기각.**
   "안정"은 결과가 조용했던 시각이지, 조용할 것이 예측 가능했던 시각이 아니다.

### 태양광 일 스케일링 — 과적합 확인 → 재설계·재배포

**사용자 지적("bias −0.000 이면 과적합 아닌가")이 맞았다.** `solar_scale.json` 의 검증창
수치는 **재현되지 않았다** — 같은 창을 다시 계산하니 흐림 bias 가 −0.0002 가 아니라
**+0.0268** 이었다. 원인은 명확하다: 판정지표(지점조합×분위수)와 `MID_FIXED` 를 **그
검증창 성능을 보고** 골랐다. 검증창이 이미 모델 선택에 쓰였으니 그 위의 수치는 낙관치다.

**완전 홀드아웃(2026-07-01~, 적합·검증창 양쪽 다 아님)에서 구 파라미터는 해로웠다**:
흐림 bias −0.049 → −0.086(더 나빠짐), MAE 0.1488 → 0.1648(**+11%**). 지표 변형 48종을
전부 재평가해도 **하나도 빠짐없이 MAE 가 악화**됐고, 현행 지표는 48개 중 **37위**였다.

**다중 파라미터 스윕으로 진짜 원인을 찾았다 — 값이 아니라 적합 목표였다.**
적합창 흐림 편향(+0.057)이 홀드아웃(+0.037)보다 크다. 즉 **편향의 크기 자체가 시간에
따라 줄어든다.** 그런데 적합 score 가 편향을 0 까지 밀어붙이므로, 그 강도를 그대로 쓰면
이후 구간에서 반드시 과보정된다.

| 적합 목표 | 홀드아웃 흐림 | MAE 대가 |
|---|---|---|
| 편향 → 0 (구) | −0.0287 | +0.0112 (7.7%) |
| **편향 50% 축소** | **−0.0025** | **+0.0038 (2.6%)** |

**자유도는 대부분 무의미했다** — 지점 조합 4종(south/west/평균/min)이 홀드아웃에서
사실상 동일(MAE 변화 +0.0102~+0.0121), 지평별 floor 는 D+1~D+5 가 같은 값으로 수렴.
`k` 를 탐색에 넣었더니 적합창 노이즈를 쫓아 k=12(급경사)를 골랐는데, 같은 적합 목표에서
k 만 바꿔 재니 **분할 4개 전부에서 k=3.5 가 이겼다**(MAE 대가 +0.0041~0.0057 vs
+0.0068~0.0089). 구 코드가 k 를 고정한 판단은 옳았고 값만 5.5→3.5 로 낮췄다.

**재설계·재배포 (사용자 지시: 기간 길게 + 기준 한쪽으로 고정)**
- 적합 기간 최근 5개월 → **전 구간 2025-12~2026-07**(223 base), `--months` 기본 9
- 판정지표 `min(west,south)` → **`radiation_south` 단독** P60 고정 (서빙 무수정 —
  `columns` 가 단일이면 min/mean 이 같다)
- `k` 3.5 고정 / `mid`·`floor` 만 탐색 / **지평 공통 한 벌** → 자유 파라미터 5 → **2**
- 적합 목표 `BIAS_TARGET_FRAC = 0.5`

**배포값 mid 0.45 / k 3.5 / floor 0.10 (지평 공통).** 홀드아웃 실적:

| | 흐림 bias | 맑음 bias | MAE | 과대율 |
|---|---|---|---|---|
| 무보정 | +0.0373 | −0.0471 | 0.1443 | 41.0% |
| **배포** | **+0.0030** | −0.0493 | 0.1489 (**+3.2%**) | **36.2%** |

구 배포본의 MAE 대가 7.7% → **3.2%** 로 줄었고 과보정도 사라졌다. 실제 발동 분포는
**44.9% 가 사실상 무보정**(≥0.99), 0.3 미만은 0% — floor 는 격자 하한이지만 닿지 않는다.
서빙 확인: 120행 hd 1~5, 수요 911MW / net_load 792MW(구 793MW).

**★남은 구조적 한계**: 이 보정은 **낮추기만** 하므로 예보가 과소예측하는 달엔 손해다.
2026-07 이 그랬다(무보정 흐림 −0.049 — 2025-12~2026-06 일곱 달 중 유일한 음수, 흐림날
8일뿐이라 계절 법칙이라 단정은 못 함). 파라미터로는 못 고치니 `--check` 로 편향의
**부호**를 주기적으로 볼 것.

### NC 신규 기상변수 탐색 — CAPE 계열은 없고, 누적 일사가 유망
사용자 질문("CAPE 같은 NC 변수 중 쓸 게 없나")에 대한 답. 태양광 잔차와 대조(7,375행):

| 변수 | 전체 r | 흐림 r |
|---|---|---|
| **`radiation_acswdnb`(누적 총일사)** | **−0.125** | **−0.312** |
| `total_cloud`(KIMR CLDFRA) / `mslp` / `hpbl` / `cinn` / `cape` | ≤ 0.057 | −0.01~0.19 |

- **CAPE·CINN·HPBL·MSLP 는 신호 없음**(|r| ≤ 0.055). 대류 지표라 태양광 시간대와 안 붙고,
  대류일 보정은 이미 `tcog` 가 맡는다.
- **누적 총일사가 유망**: 흐림 잔차 r −0.31, 하루 판정지표로 써도 실측 이용률 상관
  **0.713** 으로 현행(0.694)보다 높다. 다만 `forecast_kimr` 에만 있고 80일 이전 결손.
  → 후처리 교체가 아니라 **재학습·서빙 입력 재설계** 사안. 다음 세션 항목.
- 기대했던 **산란비**(diffuse/(direct+diffuse))는 현행보다 못했다(0.694 → −0.537). 기각.

### ★누적 일사는 "교체"가 아니라 "결합"이다 — 재학습 없이 일사 오차 7~19% 감소
사용자 질문("재학습이 나은 선택일 것 같아? 성능차이가 심하면 해야지")에 답하려고 실측
ASOS 와 직접 대조했다(2026-04-29~07-30, 낮 3,975행 — ACSWDNB 결손 제외 동일 표본).

| 지점 | 소스 | MAE | bias | r |
|---|---|---|---|---|
| south | **KIMG(현행)** | **0.5820** | **+0.0005** | 0.7065 |
| south | KIMR 순시 | 0.7367 | +0.4553 | 0.6689 |
| south | KIMR 누적 | 0.6265 | +0.3942 | **0.7167** |
| west | **KIMG(현행)** | **0.5934** | **−0.0129** | 0.6878 |
| west | KIMR 누적 | 0.6108 | +0.3694 | **0.7177** |

**교체하면 손해다** — 누적은 상관(r)이 높지만 큰 양의 편향(+0.37~0.39)이 있어 MAE 가 나쁘다.
편향을 상수로 빼도 KIMG 가 앞선다(south 0.582 vs 0.608). KIMR 순시는 최대 5.81 로 실측
3.93 을 넘겨 물리적으로 과대 — 논외.

**그런데 함께 쓰면 크게 좋아진다.** 두 소스를 실측에 선형회귀시킨 결합값의 홀드아웃 성능:

| 분할 | 지점 | KIMG 원본 | KIMG+누적 | 개선 | 편향보정 단독 대비 |
|---|---|---|---|---|---|
| 06-15 | south | 0.6562 | 0.6105 | −7.0% | −5.5% |
| 06-15 | west | 0.6529 | 0.5674 | −13.1% | −8.9% |
| 07-01 | south | 0.7080 | 0.6208 | −12.3% | −8.0% |
| 07-01 | west | 0.6443 | 0.5238 | **−18.7%** | −15.6% |

계수가 두 분할에서 안정적이고(KIMG 0.39~0.47 / 누적 0.38~0.50) 기여도가 비슷하다 —
**두 소스가 서로 보완한다**. 단순 편향보정보다 5.5~15.6% 더 좋으니 이득이 진짜 두 번째
소스에서 나온다.

★**결론: 재학습 불필요.** 모델은 **실측 기상으로 학습**돼 있고 서빙 때 예보를 넣는다.
결합값을 실측에 회귀시키면 출력이 실측 스케일·분포로 나오므로 **모델을 안 건드리고 입력만
정확하게** 만들면 된다. "D+3 부터 예보 오차 > 모델 오차" 라는 기존 진단에 정확히 맞는 처방.
반대로 두 소스를 모델 피처로 직접 넣으려면 **예보 기반 학습**이어야 하는데 ACSWDNB 완전
구간이 93일뿐이라 계절 커버리지가 부족하다 — 지금 재학습은 부적절하다.

비용도 싸다: `collect_forecast` 는 이미 NC per-hf 경로라 `ACSWDNB` 를 요청 변수에 **이름만
추가하면 끝**(호출수 증가 0). 다음 세션 1순위.

### ★일사 결합은 착시였다 — 재수집이 근거를 지웠다 (같은 세션 후속 검증)
앞 절에서 "KIMG+누적 결합이 일사 MAE 를 7~19% 줄인다"고 적었는데, **틀렸다.**
그 측정에 쓴 2026-07-02~11 구간의 KIMG 일사가 **망가져 있었다.**

재수집 전후 대조 (D+1~5, 1,180행):

| | 재수집 전 | 재수집 후 |
|---|---|---|
| `radiation_south` 실측 대비 MAE | **1.0786** | **0.6432** (−40%) |
| 값이 바뀐 행 | — | 777 (65.8%), 평균 **+0.43 낮게** 기록됨 |
| `total_cloud_*` NULL | 780 | 0 |

즉 2026-07-02~11 사고는 **운량만이 아니라 일사도** 망가뜨렸고(운량은 NULL 이라 눈에 띄었지만
일사는 *값이 틀린* 채로 채워져 있어 안 보였다), 결합은 그 손상을 부분적으로 보정하고 있었다.

**데이터를 고치니 결합 이득이 사라진다** (평가 2026-06-16~07-30, south):

| 방식 | MAE | std비(실측 대비) |
|---|---|---|
| **KIMG 원본** | **0.6021** | 0.915 |
| 제약결합 w=0.5 (합=1) | 0.6022 | 0.911 |
| 무제약 최소제곱 | 0.6096 | **0.792** |

**서빙 전달률도 0 이다** (스케일 끄고 A/B, 평가 1,675행): 전체 MAE 0.1140 → 0.1136
(**−0.31%**), 흐림은 오히려 **+1.28% 악화**, 지평별은 −5.5%~+6.6% 로 방향이 뒤죽박죽이다.

원인은 두 겹이다:
1. 결합의 MAE 이득이 **정보 추가가 아니라 평균으로의 수축**이었다 — 무제약 최소제곱은
   표준편차를 실측의 0.792 배까지 눌러 버린다. 일사 지표는 좋아지지만 모델이 필요로 하는
   진폭이 사라진다.
2. 남은 이득마저 **망가진 KIMG 를 보정하던 몫**이었고, 재수집으로 사라졌다.

→ ★**결합 배포하지 않는다.** CAPE 계열에 이어 누적 일사도 닫힌다. 다음 세션 1순위에서 제외.
   교훈: **입력 지표 개선이 출력 개선으로 자동 전달되지 않는다** — 특히 개선분이 분산
   축소에서 올 때는. (시점별 QM 실패·스케일 과보정과 같은 계열의 함정이다.)

### 파일 정리 (사용자 요청)
지운 것 — 전부 재생성 가능하거나 중복이라 복구 불필요:
- **`input_data_jeju(temp).db` 57MB** — 본 DB 의 **완전한 부분집합**임을 확인 후 삭제
  (5개 테이블 행수 일치 + 표본 컬럼 합계 전부 일치, 본 DB 가 historical 37행 더 최신이고
  `forecast_kimr`/`kimg` 도 갖고 있다).
- `__pycache__` 18개·`.pyc`, `data/nctest_jeju.db`(이번 검증 산출물), `None`(0바이트), 빈 폴더 4개.

**`Training/**/no use/` (9MB) 는 남겼다** — `.gitignore` 가 이미 제외하고 있어 커밋해도
이력에 안 남고, 이 폴더는 이미 한 번 값을 했다(태양광 일 스케일링을
`no use/net_load_forecaster/data_pipeline.py:852` 에서 되살렸다). git 연결 후 처리.

### 결손 재수집이 최대 수확이었다 + 스케일 재적합
`--verify` 를 고치고 2026-07-02~11 base 10개를 재수집했다. 사고가 **일사 값까지**
망가뜨린 게 드러났고(위 절), 그 구간을 쓴 스케일 적합을 다시 돌렸다.

- 재적합 후 파라미터는 **그대로** (mid 0.45 / k 3.5 / floor 0.10) — 손상이 최적점을 옮기진 않았다.
- 그런데 **기준선이 크게 좋아졌다**: 홀드아웃 무보정 MAE 0.1443 → **0.1213 (−16%)**.
  망가진 일사가 태양광 예측을 직접 갉아먹고 있었다는 뜻이다.
- 보정 성능도 개선: 흐림 +0.0506 → +0.0148, MAE 대가 +2.8%(구 3.2%), 과대율 49.2% → 42.8%.

`--verify` 도 세 군데 고쳤다 — ①기대 행수를 창에서 계산(구 144 는 D+7 시절) ②NULL 검사를
운영 지평으로 제한 ③**sentinel 에 `total_cloud_*` 추가**. ③이 핵심이다: temp 만 보던 탓에
운량만 빠진 사고를 10 base 동안 못 잡았다. 현재 224 base 전부 완전.

### 서빙 체인 소스 구조 — 결정 보류
사용자 제안: ①KIMG 단독 ②결측 시 KIMR 대체 ③KIMG 수집 실패 시 KIMR 전체.
**구조(폴백 사다리)는 타당하고, 특히 ②③은 지금 없는 것이다** — 현재 일사·운량에는
폴백이 아예 없어서 KIMG 가 빠지면 NULL 로 남는다(이번 사고가 정확히 그것).
`forecast_kimr` 에 CLDFRA 운량·일사가 있으니 배선은 가능하다.

다만 ①(met 도 KIMG 우선)은 현재 배선을 뒤집는 것이라 근거가 필요한데, **데이터가 없다**:
`forecast_kimg` 의 met 은 **12 base(1,388행)뿐**이다("KIMG met 은 설계상 안 쓴다"로 수집 안 함).
그 12 base 예비 측정은 **엇갈렸다** (7/08~7/22, 실측 ASOS 대비 MAE):

| 변수 | KIMR(3km) | KIMG(8km) | 우세 |
|---|---|---|---|
| 기온 | 0.8777 | **0.8283** | KIMG (5.6%) |
| 풍속 | **1.6743** | 1.9281 | **KIMR (13.2%)** |
| 습도 | 5.0546 | **3.4549** | KIMG (31.6%) |

표본이 얇고 단일 창(7월)이라 이걸로 정하면 안 된다. 사용자가 KIMG met 을 east·west·south
제대로 수집한 뒤 재판단하기로 했다.

### ★git 이 한 칸 어긋난 곳에 있다
사용자는 "git 완료"로 알고 있었으나 실제로는 GitHub 저장소
(`JeongYakyong/jeju_model.git`)가 **빈 상태로 `jeju_model/jeju_model/` 하위에 클론**돼 있다
(추적 파일 `.gitattributes` 하나, 커밋 `Initial commit` 하나). **프로젝트 본체는 git 밖이다.**
→ 그 `.git`·`.gitattributes` 를 루트로 올리면 리모트와 이력이 유지돼 push 가 깨끗하다.
(루트에 새로 `git init` 하면 리모트의 Initial commit 과 이력이 갈려 force push 가 필요하다.)
→ 같은 세션 후반에 **루트로 이동 완료**. 리모트·`Initial commit` 이력 유지, 커밋 5개. push 는 미실행(사용자 몫).

### ★발견한 버그 — `fit_solar_scale` 재적합이 조용히 0행으로 끝나고 있었다
`build_unscaled_predictions` 가 `serve_solarwind._SSCALE = {}` 로 캐시를 비운다. 그런데
`_solar_scale_cfg()` 는 `_SSCALE is not None` 이면 그대로 돌려주므로 빈 dict 이 반환되고,
호출부 `cfg, params = _solar_scale_cfg()` 가 **언패킹에서 터진다**. 그 예외를
`except Exception: continue` 가 삼켜 176 base 전부 버려지고 `[ERR] 예측을 만들지 못했다` 로 끝난다.
→ `_SSCALE = None` 으로 고쳤다(그러면 `APPLY_SOLAR_SCALE=False` 를 보고 `(None,{})` 를 새로 만든다).
**`--months N` 재적합과 관리자 화면 재적합 버튼이 그동안 동작하지 않았다는 뜻이다.**


---

## 2026-07-31 — Phase 2 재학습 완료 (demand·solarwind·PatchTST) + 태양광 일 스케일링 이식

### 사용자 결정 사항 (재질문 금지)
- **운영 지평 D+1~D+5 확정.** demand 168h→120h, PatchTST D+6·D+7 폐기(`SOLAR_PT_HORIZONS=[2,3,4,5]`),
  `serve_demand` 가드 1~7→1~5. 예보 수집이 `--days 5` 라 D+6/7 은 서빙 입력이 기후값 폴백뿐이었다.
- **학습창 통일**: train ≤2026-01 / val 2026-02~05 / test 2026-06~07. **배포본은 전체 재적합**
  (val 로 잡은 best_iteration 고정). 성능표는 검증본 기준.
- **`models/` 는 git 추적**한다 — 배포가 서버 `git pull` 이라 가중치가 저장소에 있어야 한다.
  `Training/` 대용량(233MB)은 제외. ⚠ 재학습마다 수십 MB 누적 → 잦아지면 Git LFS.
- **solar 채널은 PatchTST 유지**(LGBM 전환 안 함). **노이즈 주입은 포기**(QM/후처리로 대응).
- SMP 는 우선순위 낮음 — **D+2 지평 유지**, 재학습 보류.

### 재학습 결과 (전부 배포 완료)
| 모델 | 방식 | 결과 |
|---|---|---|
| demand LGBM | 랩탑, DB 배선 | test MAPE 4.55→**4.10%**, 낮 흐림 6.45→**5.53%**. KPX 6.99% 대비 우위 |
| solarwind LGBM | 랩탑, CSV→DB | solar MAE 0.0675→**0.0577**, wind 0.0763→**0.0646**. 악화 구간 0 |
| solar PatchTST D+1~D+5 | colab GPU | 맑음 MAE 6.6%↓, 전체 1.3%↓. **흐림은 개선 없음**(아래) |

- demand 2026-03 만 −4.6% 악화(전 지평). 완전기상에선 +9.7% 개선이라 예보 상호작용 문제.
- `patchtst_signal` 이 2026-05-31 에 끊겨 새 test 구간 D+1 PatchTST 비교는 NaN.
  (2026-08-04 확인: **학습 전용** 테이블이라 서빙·화면 영향 없음.)

### ★ 핵심 발견 — 병목은 모델이 아니라 예보다
LGBM solar 는 horizon 피처가 없어 입력만 바꿔 재면 예보 효과가 분리된다(2026-02~07 낮):

| 입력 | MAE | 실측 대비 |
|---|---|---|
| 실측(완전기상) | 0.0625 | 기준 = 모델 한계 |
| 예보 D+1 | 0.1068 | +71% |
| 예보 D+5 | 0.1640 | **+162%** |

**D+3 부터 예보가 더하는 오차가 모델 자체 오차보다 크다.** 재학습으로 모델을 10% 고쳐도
전체의 3~4%다. demand 도 완전기상 10% 개선이 실서빙 1.6% 로 줄었다 — 같은 이유.

원인은 **예보의 regime 의존 편향**(낮 8-17h, 2026-03~06):
일사 bias 흐림 **+0.4148** / 맑음 **−0.2617** — NWP 가 평균으로 수축한다(흐린날을 덜 흐리게).
실측으로 학습하는 한 재학습으로는 못 고친다.

### 일사 QM 은 실패로 확인 — 시점별로는 조건부 편향을 못 잡는다
단순/비대칭/조건부(예보 운량별) 세 변형 모두 **흐림만 개선하고 맑음을 악화**시켰다
(흐림 19~39%↓ 대가로 맑음 6~12%↑). 같은 예보값이 흐린날에도 맑은날에도 나오므로
시점별 매핑으로는 구분이 불가능하다. `fit_solar_qm.py` 는 이 근거로 **미채택**.

### 태양광 일 스케일링 이식 (구 파이프라인에서 되살림) — 채택
사용자가 예전에 쓰던 방식이 `no use/net_load_forecaster/data_pipeline.py:852` 에 있었고
새 파이프라인 이관 때 **누락**돼 있었다. 하루 단위로 그날 성격을 재는 게 요점:

    scale = min(floor + (1-floor)*sigmoid(K*(stat - mid)), 1.0)   # 낮추기만, 절대 안 키움

**판정지표를 새로 탐색한 게 결정적이었다.** 지점 조합(south 단독/west/평균/3지점) × 분위수
(P40~P75) 를 전부 훑은 결과 **west·south 중 어두운 쪽(min)의 P60** 이 최선이고, 초창기
방식(평균 P75)은 후보 중 **최하위**였다 — 밝은 쪽에 치우쳐 흐린 날을 놓친다.

검증(2026-05-16~06-30, 새 PatchTST 예측 기준):
| | 흐림 bias | 맑음 bias | MAE | 과대율 |
|---|---|---|---|---|
| 무보정 | +0.0670 | −0.0653 | 0.1363 | 51.3% |
| 초창기 파라미터(평균P75/mid1.45) | +0.0220 | −0.0655 | 0.1438 | 43.2% |
| **채택(최소P60/mid1.10/K5.5)** | **−0.0002** | **−0.0657** | **0.1351** | **38.9%** |

**흐림 편향 완전 제거·맑음 불변·MAE 도 개선** — 지표를 제대로 고르니 정확도와 위험을
맞바꿀 필요가 없어졌다. mid 는 1.10 고정(지평별 자유 탐색은 과적합: 검증 MAE 0.1321→0.1379),
K 는 5.5 고정(자유 탐색 12~15 는 이득 미미 + 임계 근처 급변 위험).
floor 만 지평별: D+1 0.55 / D+2 0.50 / D+3 0.40 / D+4 0.60 / D+5 0.40 (D+4 는 노이즈 의심).

- 서빙 `serve_solarwind._apply_solar_daily_scale` (tcog 다음·야간마스크 앞).
- 판정지표·파라미터는 `models/solarwind_lgbm/solar_scale.json` 이 SSOT — 코드에 안 박았다.
- 재적합·점검 `forecasting/fit_solar_scale.py` (`--check` 는 재적합 없이 드리프트만).
  관리자 화면 "보정 관리 — 태양광 일 스케일링" 에 점검·재적합 버튼.

### 그 밖
- **`.gitignore` 정리** — `models/`·`data/refdata/` 추적 / `*.db`·`logs/`·`Training/` 대용량 제외.
  CLAUDE.md 의 "모델은 gitignore 대상" 기술이 실제 운영(git pull 배포)과 어긋나 있어 정정.
- **serve_solarwind_lgbm 이 CSV 대신 DB 를 읽는다** — clearsky 평년의 학습창을
  `feat_meta.json` 의 `train` 에서 읽어 **재학습 시 자동 추종**. 값 중립 확인(net_load 792MW 동일).
- `export_solarwind_csv.py`·`3cmp-A_lgbm_solarwind.py`·`_build_2a.py` 전부 `project_paths` 배선.
- colab 학습 노트북 신규: `Training/3_.../training/_gen_notebook_solar_d1d5.py` → `train_solar_d1d5_colab.ipynb`.

### ★KIMR NC 전면 전환 검토 — 기각 (세션 말미 추가 실측)
사용자 질문: "다음 세션에 기상 피처를 전부 KIMR NC 로 갈아탈 계획인데, 지금 모델에 KIMR 을
넣어도 성능이 괜찮다면 재학습이 필요 없지 않나?" → **재봤더니 괜찮지 않다.**

**변수별 GRIB/KIMG ↔ KIMR-NC 대조** (18,998행 페어, 2026-02~08):
| 변수군 | 상관 | 판정 |
|---|---|---|
| 기온·풍속·풍향·습도·강수 | **r 0.998~0.9997** (기온 0.005°C, 풍속 0.011m/s) | 같은 모델(R030) 포맷 차이 — **동일** |
| 일사 | r 0.64(순시)~0.70(적산) | **다른 모델**(KIMG=NE57) |
| 전운량 / 중하운량 | **r 0.47 / 0.58** (KIMR 0.431 vs KIMG 0.713) | KIMR 이 훨씬 맑게 본다 |

**서빙 성능** (현행 모델 그대로, 입력만 교체, 2026-03~06 낮, 일 스케일링 적용):
| 입력 | solar MAE | 흐림 bias | 맑음 bias | 과대율 |
|---|---|---|---|---|
| **KIMG 일사 + GRIB met (현행)** | **0.1342** | **+0.0025** | −0.0603 | **36.9%** |
| KIMR 전체(일사 순시) | 0.1597 | +0.0711 | **+0.0362** | 63.1% |
| KIMR 전체(일사 적산) | 0.1543 | +0.0958 | −0.0630 | 51.7% |
| KIMR + **스케일 재적합**(mid 3.2) | 0.1464 | +0.0415 | **−0.1433** | 45.8% |

**재적합해도 KIMG 에 못 미친다.** KIMR 판정통계 평균이 2.097 이라 충분히 누르려면 mid 를
3.2 까지 올려야 하는데, 스케일은 하루 전체에 곱하므로 그러면 맑은 날이 −0.1433 으로 무너진다.

**구조적 이유(중요)**: 모든 모델이 **실측 기상으로 학습**하므로 **"KIMR 용 재학습"이 성립하지
않는다** — 소스를 바꿔도 나오는 모델은 같고 조정 가능한 건 보정뿐이다. 소스별 모델을 만들려면
예보 기반 학습이어야 하는데 8개월뿐이라 계절 커버리지가 부족하다(가을 없음).

→ **결정: 전면 전환 안 함. met 만 NC 로(값 동일, GRIB 폐기 가능), 일사·운량은 KIMG 유지.**
wind LGBM 과 `wind_qm.json` 은 met 이 동일하므로 NC 전환에 영향 없다(사용자 점검 요청 항목).

### SMP 검토 (재학습은 안 함)
- 구조: **D+1 가격 = 발표된 DA 그대로**(모델 아님). 실제 학습 모델은 음수경보(`smp_binary.pkl`)와
  D+2 DA 잔차회귀(`smp_d2_da.pkl`) 둘뿐. RT 점예측은 잠긴 실패경로 — **재시도 금지**.
- 실적(2025-12~2026-07): D+2 MAE **11.44** (lag24 persist 15.55 대비 26% 우위).
  음수경보 D+1 정밀도 0.39/재현율 0.84, D+2 0.40/0.81 — **지평이 멀어져도 성능이 안 떨어진다**.
  여름 2736행에서 실제 음수 0건·경보 0건(과경보 없음).
- 변동성 층화(D+2): 안정 구간에서 모델이 lag24 보다 **4배 나쁘다**(4.28 vs 0.98).
  급변 구간은 모델 우위(26.19 vs 37.51)지만 거기서도 lag168(20.70)이 더 낫다.
  → 개선안: 안정 구간 shrinkage / lag24·lag168 앵커 블렌딩. **미실행**.
- 학습창 연장 여지: train 2024-03~2025-12 → 최신화 시 음수 이벤트 626 → 약 773건(+23%).
  (2024-03 시작은 제주 시범사업 시점이라 타당 — 그 이전엔 음수 SMP 자체가 없다.)

---

## 2026-07-30 — KIMR NC 백필 완료·검증

`collect_archive --backfill 160 --kimr-only` 결과 (사용자가 3시간 백필 후 복귀).

**결과: 깨끗하다.**
- 범위 **2026-02-19 ~ 07-29, 161일 연속** (빠진 날 0), 전 base ≥118행.
- KIMR base 162개 = NC 150 + GRIB(NULL) 12 (기존 7/07~7/18, resume-skip 으로 안 덮임).
- 완전 구간(17종, `TSKIN`·`ACSWDNB` 포함) = **04-28 ~ 07-29 (93일)**.
- 축소 구간(15종, `TSKIN`·`ACSWDNB` 결손이나 운량·일사 100%) = 02-19 ~ 04-27 (68일).
- **결손 경계 2026-04-28 = 7/22 프로브와 정확히 일치** (프로브: 04-27 없음/~05-02 있음).
  경계는 절대일 기준이라 백필 시점(~7/29)에 고정됐다 — 이후 API 가 더 지워도 DB 는 보존.

**검증**
- 값 건전성: 계절 진행이 정확(2~4월 평균기온 13°C / 5~7월 22°C), 풍속·일사·운량 물리범위,
  결손 컬럼 외 NaN 0%. temp_skin·acswdnb 는 축소 구간에서만 전부 NaN(설계대로).
- `src_met_proto` 정상: NC 150 base / GRIB 12 base(NULL).
- 행 수 이상치 없음 — 유일한 186행(07-18)은 18z(03:00)+12z(21:00) 두 base 가 한 날짜에
  공존한 것(날짜 그룹핑 착시). base 키가 달라 정상.

**미해결/이월**
- 12 GRIB base(7/07~7/18)는 NC 바다 속 GRIB 섬 — `--force --kimr-only` 로 통일 가능(미실행).
- 서빙 입력(`forecast_horizon`)·예측은 여전히 7/18 에 멈춤(백필은 아카이브만 채웠다).

### 사용자가 `input_data_jeju(temp).db` 투입 — 더 최신 스냅샷 (루트에 있음, 본 DB 아님)

`forecast_horizon` 223 base(~7/29), `historical` ~7/31. 파이프라인을 더 돌린 결과.
**KIMG 백필은 불필요했다** — `forecast_horizon` 의 일사·운량이 곧 KIMG 소스이고 작년
12월부터 있다. 소스 비교에 이걸 쓰면 KIMG 아카이브(12 base)가 필요없다.

### ★소스 재비교 — 지난 세션의 "일사·운량 KIMG 최적" 결론을 뒤집는다

지난 세션 결론(현행 병합=일사·운량 KIMG 최적)은 **7월 장마창 12 base + 순시 컬럼**이라는
두 겹 함정 위에 있었다. 이번에 161일 아카이브 KIMR + 본 KIMG 를 **동일 base·시각·실측에
페어 매칭**(18,860행, 02-20~07-31)해 재측정:

**① KIMG vs 실측 (장기, D+1)** — 7월이 유일한 이상치임을 확증:
- 일사 south r: 12~3월 0.87~0.92 / 4~6월 0.85~0.88 / **7월 0.63**(MAE 0.61). bias 소폭 양,
  7월만 음(−0.18~−0.39). → 지난 세션이 하필 **연중 최악의 달**로 소스를 판정했던 것.
- 운량 total r 0.49~0.74(5월 최고), 7월 0.36. **bias 지속 양 = 구름 과대예보**(계통).
- 지평 열화 깔끔: 일사 D+1 r0.85→D+5 0.70 / 운량 0.66→0.47.

**② KIMR vs KIMG 대결 (실측 기준, 페어)** — ★일사는 반드시 적산(ACSWDNB) 컬럼으로:
- 처음 KIMR 순시(radiation_*)로 재서 "KIMG 64% 압승"이라는 **틀린 결론**. 순시는
  SWDDIR2+SWDDIF2 즉시값이라 시간적산 실측과 정의가 어긋나 +0.54 bias. **적산으로 바꾸면
  17% 개선**(MAE 0.702→0.584) — 아카이브·재학습 일사 피처는 적산을 쓸 것.
- 적산으로 재비교(완전구간 04-28~, n=5430):
  일사 south KIMR MAE 0.584(**r0.76** b+0.38) vs KIMG 0.553(r0.72 b+0.02) → KIMG 5%
  일사 west  KIMR 0.558(**r0.78**) vs KIMG 0.573(r0.70) → KIMR 3%. **사실상 접전**.
- 운량(n≈17,900): total 은 south 만 KIMG 16% 우세, west/east 는 KIMR 1~2%.
  midlow 는 **KIMR 전 지점 우세(1~11%)**.
- **일관된 패턴: KIMR = r 더 높음 + bias / KIMG = bias≈0 + r 낮음** (일사·운량 공통).

**해석 (재학습 관점)**: bias 는 학습이 상수/계절 오프셋으로 쉽게 지운다. r(신호 품질)은
못 지운다. → **r 이 높은 KIMR 이 두 변수 모두에서 잠재적으로 더 나은 피처**다.
지난 세션의 "KIMG 최적"은 철회. 단 결정은 재학습 실험에서 확정(아래).

**사용자 결정 프레임**: KIMG 를 근본(기본)으로 저장하되, 변수별로 KIMR 우수하면 KIMR 채택,
KIMG 우수하면 재고. → 현재 근거로는 **일사=접전(bias 보정 시 KIMR 유리), 운량 midlow=KIMR,
운량 total=혼재**. 확정은 재학습 A/B 로.

**주의**: KIMR 적산 일사는 완전구간(04-28~, ~93일)만 참값. 그 이전 68일은 순시→사다리꼴
복원(r0.98)으로 메워야 재학습 전 구간에서 쓸 수 있다.

### Phase 1 — 격리 아카이브 흡수·메인 DB 재구축 (사용자 승인)

사용자 설계 지시: KIMG 근본 + KIMR-NC, **격리(weather_kim.db) 없애고 단일 메인 DB**,
모든 collector 가 메인에 기록. `input_data_jeju(temp).db`(루트)는 서버 KIMG/GRIB 을
끌어오려 넣은 **임시**일 뿐 — 그대로 격상 금지.

**결정 (질문드림)**: ① 예보 구조 = **소스 분리 유지**(forecast_kimr/forecast_kimg 두
테이블, 변수별 소스 선택 유연성) ② KIMG 장기 이력 = **temp 에서 seed**(백필 7.5h 생략).

**새 메인 DB 조립** (`data/input_data_jeju.db`):
- 베이스 = temp (최신 historical 7/31 · est_* · patchtst · forecast_horizon 7/29)
- `forecast_kimr` = 백필 NC 그대로 (162 base = NC 150 + GRIB 12, 02-19~07-29, 19,066행)
- `forecast_kimg` = 아카이브 12 base(full KIMG) + temp forecast_horizon 에서 **일사·운량만**
  seed(30,702행, met NULL) → 224 base 32,090행. ★met 컬럼(temp_*/wind_*)은 temp 에선
  KIMR-GRIB 이라 **복사하면 안 된다** — 일사·운량 9컬럼만 seed(설계상 KIMG met 미사용).
- 스왑: 구 메인 백업 후 교체. 서빙 무손상(`serve_chain` 최신 base 7/29, 120행 896MW).

**collector 재배선**: `collect_archive.DB_PATH` = `cf.DEFAULT_DB`(메인). 로그 `[weather_kim]`
→`[archive]`, 헤더/파이프라인 라벨 갱신. `weather_kim.db` **폐지**(코드 참조 0).

**★사고·복구**: 조립 중 현 `data/weather_kim.db` 에서 `forecast_kimr` 이 사라진 걸 발견
(원인 미상 — 백필 별도 프로세스의 WAL 잔재 추정). **백업본에 19,066행 온전**해서 손실 0,
백업을 소스로 조립. 교훈: 흡수 전 백업이 실제로 데이터를 구했다.

**검증**: compileall / selftest 17 / run_pipeline 8단계 / AppTest exception 0 /
`serve_chain --no-write` 120행 / collect_archive 가 메인의 기존 base·seed 정상 조회.
백업: `scratch/backup_dbrebuild_20260730/`(구 메인 + weather_kim + collect_archive + project_paths).

**남은 것**: `input_data_jeju(temp).db`(루트)는 이제 메인에 흡수돼 **불필요** — 사용자가
지우면 됨. Phase 2(재학습)·Phase 3(GRIB 폐기)는 다음.

### Training/ 폴더 조사 — 재학습·정리 계획 (실행은 재학습 세션에서, 사용자 지시)

루트에 `Training/`(≈200MB, 2_demand / 3_solarwind / 4_smp). **live `models/` 가중치를 만든
학습 원본 + 노트북/코드 + EDA 산출물**이다. 그대로 쓰지 말 것(사용자).

**Training → live models/ 매핑** (배포된 것):
- `3_/solarwind_patchTST_pkl/*.pth`(solar D2~D7) → `models/solarwind_patchtst_horizon/`
- `3_/solarwind_models/best_*.pth`+scaler → `models/solarwind_patchtst/`
- `3_/lgbm_models/lgbm_{solar,wind}_util.txt` → `models/solarwind_lgbm/`
- `2_/model/models/lgbm_jeju_demand_direct.txt` → `models/demand/`
- `4_/models_weight/*.pkl` → `models/smp/`
- 미배포(구버전): `2_/models/lgbm_pipeline.pkl`(28MB)·`patchtst_demand.pth`, wind D2~D7.

**CSV 36개 = 거의 다 버려도 됨** (item 3):
- `solarwind_raw_jeju.csv`(9.9MB) = `data/refdata/` 것과 **md5 동일**(중복).
- `patchtst_features.csv` = DB `patchtst_signal` 테이블과 동일(55,560행).
- `_features_from_db.csv` = DB `historical` 에서 파생("from_db").
- `*/tab/*.csv`(30여개) = EDA·비교 산출물, 매 실행 재생성.
- ★이 CSV들은 **옛 GRIB 파이프라인으로 얼린 export** — 재학습은 새 NC 소스
  (`forecast_kimr/kimg`)로 DB 에서 피처를 새로 만들어야 하므로 재사용 금지.

**사용자 4개 항목 → 목표 구조**:
1. patchTST 학습 노트북(colab) — `3_/training/*.ipynb`, `2_/model/2-A_*.ipynb`. colab 학습 후 .pth 반입.
2. lgbm 학습 코드(랩탑) — demand/solarwind lgbm 스크립트, 로컬 실행.
3. 임시 CSV → 정리. 대부분 DB/refdata 중복이라 **DB 연계되면 자동 불필요**. 외부 반입 .pth 는 models/.
4. DB 연계 — 학습 스크립트가 CSV 대신 `data/input_data_jeju.db`(historical·patchtst_signal·
   forecast_kimr/kimg) + refdata 를 읽게 배선. **이게 재학습의 전제**.

**정리 대상(청소)**: 중복 CSV·tab 산출물·`.pyc`(38)·`no use` 폴더·루트 `None` 파일·
루트 `__pycache__`. **실행 안 함(계획만)** — 재학습 세션에서 파일 재배치·DB 연계와 함께.

### freshest 원칙 ↔ 12z 전용 아카이브 관계 (사용자 확인)

사용자 질문: freshest 원칙 기록돼 있나? 아카이브는 12z 전용인데 실제 서빙은 freshest 인데?

- **freshest-wins 는 기록돼 있다** (CLAUDE.md basetime 절 / README 운영노트):
  표시·서빙은 시각별 `horizon_d ASC, base DESC` 1건, 검증만 `base_hour` 로 12z·18z 분리.
- **DB 실측 확인**: `forecast_horizon`·`forecast_kimr/kimg`·`est_*` 전부 **12z(21:00) 전용**
  (forecast_kimr/kimg 에 7/18 03:00 18z 1건만 섞임). 18z 파이프라인이 2026-07-18 도입이라
  축적분이 거의 없다 — 이 temp 스냅샷도 12z 전용.
- **핵심: 12z 전용 아카이브로 재학습하는 게 설계와 일치한다.** 서빙 모델은 12z-origin
  (전일 23시 기준)으로 학습되고, 18z 는 **재학습 없이** serve-time 매핑(n=horizon_d+1)으로
  재사용된다. freshest 는 서빙/표시 시점의 병합일 뿐, 학습 데이터는 12z 로 충분.
  → CLAUDE.md forecast_kimr/kimg 항목에 "12z 전용 = 의도" 명시 추가.

---

## 2026-07-29 — 세션 #7 (forecasting/ 정리 — 개명 8 + 고아 삭제 1)

2026-07-21 사전 조사(#5)의 forecasting/ 정리안을 실행. **개명만 하고 통합은 안 했다.**

### 사용자 결정 사항 (재질문 금지)
- **전체 개명 진행 + 통합은 안 함** — 사용자가 "합칠 수 있으면 통합" 요청했으나, 전 파일
  정독 결과 forecasting/ 은 이미 **1파일=1역할** 구조라 합칠 후보가 없다고 보고 → 사용자
  "개명만(권장)" 승인. 유일 후보였던 `smp_d1`+`smp_d2` 는 같은 이름 심볼(`_upsert`
  3컬럼 vs 7컬럼, `OUT_PRICE` 값 다름)이라 합치면 접두어 손편집이 필요 = 회귀 위험이라 제외.
- **고아 삭제 승인**: `smp_calibrate.py` + `models/smp/smp_calibrator.pkl` (진입점 도달불가 +
  pkl 소비처 0건. docstring 의 "models_weight" 경로는 낡은 표기, 실제는 models/smp/).

### 개명 매핑 (8개, 모델 .pkl/.txt/.json 은 데이터라 이름 유지)
```
solarwind_db_pipeline.py  → patchtst.py          (실제로 PatchTST nn.Module 정의+로더)
serve_solarwind_hybrid.py → serve_solarwind.py   (importer 2곳이 이미 as serve_solarwind)
train_smp_db.py           → smp_features.py       (피처빌더 SSOT, 학습+서빙 공용)
train_binary_smp.py       → smp_binary.py
train_smp_d2_da.py        → smp_da.py
smp_phase2_depth.py       → smp_depth.py
smp_db_pipeline.py        → smp_d1.py
smp_d2_pipeline.py        → smp_d2.py
```
유지: `serve_chain`·`serve_smp`(진입점) · `serve_demand` · `serve_solarwind_lgbm` ·
`horizon_backtest`. 옛 `train_*` 은 "학습 전용" 오해를 줘서 바꿨다 — 셋 다 서빙이 import 하는
공용 부품이다(`smp_features.load_forecast`/`smp_binary.persist`/`smp_da._predict_da`).

### 영향 범위가 좁았다 (개명이 안전했던 이유)
- forecasting/ **바깥에서 이 파일들을 이름으로 부르는 곳은 없다** — `project_paths` 는 진입점
  둘(`SERVE_CHAIN`/`SERVE_SMP`, 이름 유지)만 경로 상수로 갖고, `run_pipeline`·`pages/` 는
  진입점을 subprocess 경로로만 부른다. 나머지 참조는 전부 forecasting/ 내부 `from forecasting import X`.
- 방식: import·주석을 옛 경로에서 먼저 고치고 → `mv` 로 파일 이동 → `__pycache__` 제거.

### 검증 (기준선 완전 일치)
① `compileall` rc0 ② `pyflakes` undefined name/미해결 import 0
③ `serve_chain --utc 12 --no-write` = **120행 hd 1~5 / 수요 838MW / net_load 692MW** (기준선 동일)
④ `serve_smp --no-write` = **D+1 est_smp 131.2 / D+2 130.2** (기준선 동일)
— 두 서빙이 8개 개명 모듈 전부를 관통 실행하므로 런타임 import 해소까지 확인됨.
함께 갱신: CLAUDE.md(`train_*` 절 재작성), README(매핑표 52–53).
백업: 스크래치 `backup_20260729/`.

### 이월 (이번 범위 밖 — 별도 세션)
- `horizon_backtest` 의 스크래치 헬퍼(`build_scratch`/`set_scratch_forecast`)를 운영 러너
  `serve_chain` 이 빌려 쓴다 = **운영이 진단도구에 의존**. + 스크래치 DB 주입이 함수 인자가
  아니라 전역 치환(`serve_demand._conn=lambda`, `m.DB_PATH=scratch`). 둘 다 동작 변경이라
  개명과 분리했다. 정리하려면 scratch 헬퍼 분리 + 몽키패치 계약을 인자화 (2026-07-21 #5의 옵션 C).

---

## 2026-07-22 — 세션 #6 (collect_weather_kim `--force` + met 프로토콜 태그)

### DB 현황 점검 — 파이프라인이 7/18 이후 안 돌았다

| 테이블 | 최신 |
|---|---|
| `forecast_horizon` / `est_horizon_jeju` | base 2026-07-18 03:00 (18z), 212 base |
| `est_smp_horizon_jeju` | base 2026-07-17 21:00 |
| `historical` | ts 2026-07-21 20:00 ← 화면 `ensure_recent` 가 라이브로 채운 것 |
| `weather_kim` 아카이브 | 7/07~7/18, **12 base 그대로** (kimr 1,366 / kimg 1,388행) |

7/19~7/22 공백은 API 30일 완전 보존 범위 안이라 백필로 메울 수 있다.

### `--force` 신설 (사용자 A안 승인) — 백필 전 필수 조건이었다

막고 있던 건 `--backfill` 의 resume-skip(`existing_rows >= expected*0.95`)이었다.
그런데 skip 만 끄면 부족했다 — upsert 가 `COALESCE(excluded.col, col)` 이라
**NC 가 안 주는 컬럼에 GRIB 잔재가 남아** 한 행에 두 소스가 섞인다.

→ `--force` = skip 무시 + **그 (base, 테이블) 행을 DELETE 후 재적재**.
- ★삭제는 **수집이 성공한 뒤에만** (`collect_one` 에서 wide 확보·`to_grid` 통과 후).
  실패하면 DELETE 자체를 안 하므로 데이터 손실 경로가 없다.
- `forecast_kimr`/`forecast_kimg` 가 별도 테이블이라 한 소스만 지워도 안전.

### met 프로토콜 태그 `src_met_proto` 신설 (사용자 지적)

"wide 로 저장하니 수집 방식도 태그를 남기는 게 안전하지 않나" → 맞다.
본 DB 의 `src_met_*`(모델 마스크)과 같은 취지인데, **아카이브는 테이블이 이미
모델을 말하므로 갈리는 축이 프로토콜뿐**이다 (KIMR 도 일사·운량은 항상 NC).

- `forecast_kimr.src_met_proto` = `'NC'` | `'GRIB'`. KIMG 는 경로가 하나라 태그 없음.
- NULL = 2026-07-22 태그 도입 이전 = 전부 GRIB met.
- 태그는 `to_grid`(시간보간) **뒤**, `dropna(how="all")` **뒤**에 붙인다 —
  object 컬럼이 보간에 끼지 않고, 전-NaN 행이 태그 때문에 살아나지도 않는다.
- 필터 안전 확인: `cf.is_non_kma` 는 `*_da`/`day_type` 만 제외, `pp.clip_ranges` 는
  비수치 컬럼 통과. 스키마는 `upsert_wide_coalesce` 의 ALTER 로 자동 확장.
- ⚠ `--force` 없이 재수집하면 옛 컬럼은 남는데 태그만 새 값으로 바뀐다(낙관적).
  프로토콜 바꿔 재수집 = **항상 `--force`**.

### 검증 (네트워크 0회, 본 DB 무수정)

`weather_kim.db` 를 스크래치로 복사하고 `DB_PATH` 를 갈아끼워 4항목:
① force 없이 upsert → 118행 유지, 새 2행만 태그 `NC`(나머지 NULL)
② `delete_base` 118행 삭제 → 재적재 2행 → 그 base 2행
③ 다른 11 base 행수 불변 ④ KIMG 테이블에 `src_met_proto` 안 생김.
+ `compileall` / `--help` / `selftest_pivot` 17항목 통과.
원본 DB 재확인: 1,366·1,388행, 컬럼 추가 없음(실제 수집 때 ALTER 로 생긴다).

부수 확인: `forecast_kimr` 를 읽는 코드는 collectors 밖에 없다 — 서빙·화면 무영향.

### collectors 개명 + 최소 분리 (사용자 B안 승인) — 7 → 10파일

사용자 지적: "파일명만 봐서는 무슨 역할인지 이해하기 어렵다".
`api_fetchers_jeju`(제주 전용 저장소인데 "jeju"?) · `api_fetchers_kim2`("2"가 뭔지 모름) ·
`_common`(공용이 아니라 KIMG 코어가 절반) 이 특히 심했다.

**"이름만 바꾸기"로는 부족했다** — `api_fetchers_jeju` 한 파일에 성격이 다른 셋
(KIMR fetch / KPX·ASOS fetch / 피벗)이 들어 있어 새 이름도 절반만 맞게 된다.
그래서 개명 + 최소 분리(B안).

```
_common.py           → kma_kimg.py       KIMG core + KMA 공용 기반(키풀·세션·창 SSOT)
api_fetchers_kim2.py → kma_kimr_nc.py    std NC / 등압면 CLDFRA (아카이브용)
collect_weather_kim.py → collect_archive.py
api_fetchers_jeju.py → kma_kimr.py (KIMR GRIB fetch)
                     + kpx_asos.py (KPX·ASOS = 실측 소스)   ★신설
                     + pivot.py    (long→wide 피벗 1벌)      ★신설
```

- **분리 근거가 코드에 이미 있었다**: `collect_forecast` 가 같은 모듈을 `kim`/`ci`/`kpx`
  세 별칭으로 import 하고 있었고, 별칭별 사용 함수가 정확히 세 덩어리로 갈렸다
  (`kim.`=fetch 계열 / `ci.`=피벗 계열 / `kpx.`=fetch_kpx_est). 그대로 세 모듈이 됐다.
- 부수 발견: `collect_historical` 은 KIMR 을 **한 번도 안 쓴다** (`kim.` 사용처가
  `fetch_asos` 하나뿐이었다) → 별칭 하나 제거, 이제 `kpx_asos` 만 import.
- **파일을 가르는 축 = 출처가 아니라 예보냐 실측이냐.** ASOS 는 KMA 소스지만 관측이라
  `kma_*` 가 아니라 `kpx_asos` 에 뒀다.
- `kma_kimg` 는 이름과 달리 KMA 공용 기반(키 풀·세션·창 산식 SSOT)도 담는다 —
  KIMG core 가 그 위에 얹혀 있을 뿐이라 가르면 두 파일이 서로를 계속 부른다. 문서에 명시.
- 조립은 **원문 구간 추출**(1-indexed 줄 범위 슬라이스)로 했다 — 손으로 옮기지 않음.
  헤더/import 만 새로 쓰고 본문은 원문 그대로. `pyflakes` 로 누락·잉여 이름을 잡았다
  (kma_kimr 의 `_common.` 잔재 2건, kpx_asos 의 `sys`·`current_kma_key` 누락 3건 검출).
- ⚠ 지난 세션의 "`_common` 3분할·`api_fetchers_jeju` 2분할은 이득 < 리스크" 판단을
  일부 뒤집은 것이다. 단 **`_common` + `api_fetchers_jeju` 를 한 파일로 합치지 않는다**는
  결정은 그대로 유효하다 (병렬 어댑터).

**함께 갱신**: `project_paths.COLLECT_WEATHER_KIM` → `COLLECT_ARCHIVE`, `run_pipeline`,
`pages/common._LIVE_FETCH`(문자열 `"api_fetchers_jeju"` → `"kpx_asos"` — 동적 import 라
문법 오류로 안 잡힌다) + `from _common import partial_upsert`, `collectors/__init__.py`,
CLAUDE.md, README.md.

**검증** — ① compileall ② collectors 10모듈 import ③ selftest_pivot 17항목
④ `collect_forecast --verify` ⑤ `serve_chain --no-write` **120행 hd 1~5 / 838MW /
net_load 692MW (기준선 완전 일치)** ⑥ AppTest exception 0
⑦ `ensure_recent` 동적 import 경로 직접 확인 (`except Exception: return 0` 이라
AppTest 만으로는 안 잡힌다 — `kpx_asos.fetch_kpx_jeju` + `kma_kimg.partial_upsert` 실확인)
⑧ `run_pipeline.PIPELINE_STEPS` 8단계 스크립트 경로.
백업: 스크래치 `backup_20260722/collectors/` (git 이 아니라서).

남은 pyflakes 경고(미사용 import 10건)는 **분리 이전부터 있던 것**이라 손대지 않았다.

### ★★ KIMR GRIB 도태 예정 못박기 (사용자 3회 강조) — 개명 1차가 방향을 거꾸로 박았다

개명 직후 사용자 지적: **"KIMR GRIB 방식은 NC 안정성이 확보되는 순간 도태될 예정.
지금처럼 '(아카이브용)' 이런 식으로 헷갈릴 요소를 두지 마라."**

1차 개명이 정확히 그 함정을 만들었다:

| 파일 | 실제 | 이름이 준 인상 |
|---|---|---|
| `kma_kimr.py` (GRIB) | **곧 죽을 쪽** | 평범한 이름 = 본류 |
| `kma_kimr_nc.py` (NC) | **살아남을 쪽** | 접미사 + "(아카이브 전용)" = 곁가지 |

→ **`kma_kimr.py` → `kma_kimr_grib.py` 재개명** (사용자 a안 승인).
`_nc` 는 이름 유지하되 "(아카이브 전용)" 라벨 전부 제거하고 **"KIMR 표준 경로"**로 승격.
삭제할 날엔 `_grib` 파일만 지우면 끝 — 개명이 다시 생기지 않는다.

- `kma_kimr_grib.py` docstring 최상단에 박스로 도태 선언 + 유예 조건 2개 명시
  (① NC 장애 폴백 ② 90일 이전 `ACSWDNB`·`TSKIN` 결손 보전) + "새 기능 얹지 말 것".
- `kma_kimr_nc.py` docstring 최상단에 "본류이자 미래 표준, KIMR 새 코드는 여기".
- CLAUDE.md 규약 절 **맨 앞**에 ★★ 항목 신설, 엔드포인트 표에 "앞으로" 열 추가
  (NC 2행 = 표준 / GRIB = ★삭제 예정) + 표 순서도 NC 를 위로.
- `collectors/__init__.py` 에 "KIMR 경로 둘은 **대등하지 않다**" 명시.
- 클로드 메모리 2건 저장: `kimr-grib-deprecated`(project) / `naming-shows-direction`(feedback).

**교훈**: 역할만 맞고 방향이 틀린 이름은 안 고친 것만 못하다. 개명할 때
"이 코드가 앞으로 어떻게 되는가"를 이름에 함께 담을 것.

재검증: compileall / 10모듈 import / selftest 17항목 /
`serve_chain --no-write` **120행 838MW / 692MW** / AppTest exception 0.

### ★API 보존기간 정밀 실측 — 백필 계획의 전제를 다시 쟀다 (07-21 기록 정정)

hf=24 한 시각만 찌르는 프로브(24h 전량 불필요, 사용자 지적). 지점 = solar_farm(south).

| 경과일 | 단일면 | TSKIN | ACSWDNB | 등압면 CLDFRA·T |
|---|---|---|---|---|
| 7 / 30 / 60 / 65 / 70 / 75 / **80** | **17종 완전** | ✅ | ✅ | 24레벨 |
| **85** / 90 / 120 / 150 / 160 | 15종 | ❌ | ❌ | **24레벨** |
| 165 / 170 / 175 / 180 | 응답 없음 | — | — | — |

- **완전 = 80일 / 상한 = 160일.** 구 기록("90일 결손·상한 150일·180일 무응답")보다 유리.
- **운량 CLDFRA 는 160일까지 전 구간 정상** — 결손은 `TSKIN`·`ACSWDNB` 둘뿐이다.

**★프로브 1차가 틀렸던 이유 (같은 실수 반복 금지)**: 8 base × 2콜을 간격 없이 연달아
쏘니 일부가 실패했는데, 프로브가 **"호출 실패"와 "변수 없음"을 구분하지 않아** 실패를
결손으로 표시했다 — "7일 0/17, 90일 15/17" 같은 물리적으로 불가능한 표가 나왔다.
원문 body 를 덤프해서 발견. 재시도 + sleep + 실패 명시로 고치니 깨끗해졌다.
**교훈: 가용성 프로브는 반드시 실패를 별도 상태로 찍을 것.**

### ACSWDNB 는 순시 일사로 복원 가능한가 → **사다리꼴이면 꽤 된다** (사용자 질문)

아카이브 12 base 로 실측 (둘 다 MJ/m²/h 라 직접 비교 가능:
`radiation_*` = 순시 GHI×0.0036 / `radiation_acswdnb_*` = 누적 diff = 참값).

| 복원식 | r (south) | 주간 MAE | 주간 상대 MAE |
|---|---|---|---|
| ① 순시 그대로 (현행식) | 0.940 | 0.419 | **45.1%** |
| ② **사다리꼴 `(GHI(t-1)+GHI(t))/2`** | **0.980** | **0.229** | **24.0%** |
| ③ 직전 시각 순시 | 0.954 | 0.374 | — |

west 0.966→0.988 / east 0.944→0.978 로 세 지점 모두 동일 경향. **오차가 절반**이다.
- 남는 bias(+0.047~0.073)는 **세 방법이 거의 동일** → 계통 편향이라 상수 보정 대상.
- 완전 대체는 아니다(상대 24%). 단 80일 이전 구간을 쓸 수 없게 만들 정도는 아니다.
- ※ 12 base 가 장마창이라 시간내 구름 변동이 심한 **불리한 조건**이었다 —
  맑은 날이 섞이면 더 좋아질 여지가 있다(반대 방향 걱정은 없음).

### TSKIN 을 등압면 T 로 대체 가능한가 → **불가** (사용자 질문)

등압면(data=P)에 `T` 24레벨이 있고 **85일 결손 구간에서도 나온다**(CLDFRA 와 같이).
그런데 같은 base·hf 를 하루 주기로 비교하니:

```
T@1000hPa − T2  =  −0.32 ~ +0.06 °C   ← T2 를 사실상 복제
TSKIN     − T2  =  −0.64 ~ +0.26 °C   ← 낮 음(−0.6), 새벽 양(+0.2)으로 부호가 뒤집힘
```
1000hPa T 는 **T2 를 따라가지 TSKIN 을 따라가지 않는다.** 둘 다 기온(air temp)이고
TSKIN 은 지표 복사온으로 다른 물리량이라 당연한 결과. 등압면 T 를 받아봐야
이미 전 구간 오는 T2 의 중복이다.
- **단 실질 손실은 0이다 — 서빙 모델이 `temp_skin_*` 을 아예 안 쓴다**
  (`forecasting/` 전체 참조 0건). TSKIN 은 재학습 후보 피처일 뿐.
- 참고: 아카이브에서 T2↔TSKIN 은 south r=0.934 / east 0.847 / **west 0.533** 로
  지점차가 크다(east 는 skin std 가 T2 의 2.7배). 장마창이라 과대평가된 수치이니
  대체 가능성 판단에 쓰지 말 것.

**→ 백필 방침**: NC 단독 160일. 80일 이전 구간의 `--met grib` 보조는 **하지 않는다**
(안 쓰는 TSKIN 하나 때문에 met 전체를 도태 예정 경로로 받고 소스가 섞인다).
재학습 때 "최근 80일 = 완전 / 80~160일 = 일사는 사다리꼴 복원"으로 구간을 구분한다.

---

## 2026-07-21 — 정리 세션 #5 (collectors 죽은 코드 제거 · 미래 *_da 이관)

### ★운영 구멍 발견·수정 — 내일치 KPX 하루전(*_da)이 한 번도 안 들어오고 있었다
- `build_historical` 창 = `[today-N, today]` — **미래 없음**. 미래 `*_da` 는 오직
  `build()`(forecast 빌더)만 공급했는데, 파이프라인은 항상 `--no-forecast` 라
  build() 가 한 번도 실행되지 않았다.
- 증거(DB): `real_demand_jeju` 최신 = 07-21 13:00(화면 ensure_recent 가 라이브로 채움)
  인데 `smp_jeju_da`·`jeju_est_demand_da` 최신 = **07-18 23:00**(마지막 파이프라인 실행일의 끝).
- 영향: 화면 SMP 검증 참조선(`smp_jeju_da`)·수요 비교선(`jeju_est_demand_da`)이 과거만 그려짐.
- **수정 (사용자 A안 승인)**: `*_da` 만 창을 `today + DA_FUTURE_DAYS(=2)` 로 확장
  (실측 수급·ASOS·RT SMP 는 today 그대로). `build()` 는 폐지 — 기상 출력은 이미
  forecast_horizon 이 담당하므로 잃는 것 없음. `upsert_da_to_historical` 도 동반 폐지.
- 실증(`--historical-days 1 --no-save` 드라이런): 창 `~2026-07-23` 로 확장 확인,
  미발행분(7/22·7/23)은 0행 응답 → 무해, 발행되면 다음 실행이 채운다.

### 죽은 코드 제거 (618줄 + build 계열 147줄)
레거시 `forecast` 테이블(**DB 에 실재하지 않음**) 경로가 통째로 죽어 있었다.
`run_backfill` 은 첫 줄이 `raise RuntimeError` 라 뒤 90줄이 도달 불가였다(코드가 스스로
"[폐기 2026-06-20]" 선언).

| 파일 | 줄수 | 제거 |
|---|---|---|
| `api_fetchers_kim2.py` | 930 → 660 | NE57 블록(derive_ne57_categories/hf_range_3h/fetch_ne57_std_long/ne57_3h_only), `fetch_model_long`, GRIB 폴백 `fetch_r030_frcc_long` |
| `collect_data_jeju.py` | 1049 → 727 | `run_backfill`·`_existing_timestamps`·`_expected_timestamps_for`·`upsert_wide_to`·`write_to_forecast`·`build`·`upsert_da_to_historical` + CLI forecast 분기 |
| `collect_forecast_runs.py` | 429 → 263 | 구 엔진 `fetch_one`/`run_region`/`main`/`existing_base_counts` (collect_forecast_new 가 전부 오버라이드) |

- `_fetch_grib`·`_frcc_combine` 은 **살아있는** `fetch_kimr_grib_long`·`fetch_r030_cldfra_long`
  도 쓰므로 유지. `disable_kpx` 는 구 엔진 블록과 같은 구간이라 삭제됐다가 재추가.
- `collect_data_jeju.py` CLI 가 실측 전용으로 축소 → `run_pipeline` 의
  `--no-forecast` 인자 제거(단계 라벨도 갱신).

### 검증
compileall / collectors 9 모듈 import / `--verify` / `serve_chain --no-write`
(120행 hd 1~5, 수요 838MW, net_load 692MW — 기존과 일치) / AppTest 렌더 exception 0.

### 2단계 — 형식 통합 리팩토링 (사용자 지시: "형식을 하나로 통합", "향후 문제 요소 다 정리")

**사용자 질문 "겹치지 못하는 이유가 wide 변환 때문인가?" → 아니다.** long 5컬럼
(`base_datetime/point_name/fcst_datetime/category/fcst_value`)은 **이미 두 모델의 공통
계약**이었다. 진짜 중복은 그 아래에 있었다:

| 계층 | 정리 전 | 정리 후 |
|---|---|---|
| long 스키마 | 이미 1벌 | 그대로 (정식 계약으로 문서화) |
| 피벗 | **3벌** (`kimr_one_point`/`kimg_one_point`/`long_to_wide_v2`) | `_pivot_point`+`_derive_point` 1벌 + 스펙표 2개 |
| 발표 선택 | **2벌** (그중 2함수는 바이트 동일 복붙) | `_common.latest_published_base` 1벌 |

- `kimr_one_point`/`kimg_one_point` 은 1~3단계(freshest 피벗·강수 누적diff·윈도우 트림)가
  복붙이었고 차이는 **마지막 컬럼 파생 스펙뿐**이었다 → `_SPEC_KIMR` / `_SPEC_KIMG` 표로 분리.
  ★`reh` 반올림이 KIMR 4자리 / KIMG 2자리로 **다르다** — 스펙에 그대로 보존.
- `previous_issue`·`backfill_bases`·`workers_for_backfill` 은 양쪽 모두 **호출자 0**
  (1단계에서 지운 `run_backfill` 이 유일 사용처였다) → 제거.
  `latest_published_base` 만 살려 `_common` 한 벌로 통합(`api_fetchers_jeju` 는 재노출).

**검증 — 기준선 대조로 회귀를 실제로 잡았다.**
리팩토링 직전 `api_fetchers_jeju.py` 를 `_pivot_baseline.py` 로 스냅샷한 뒤 합성 long
데이터(발표 2개 겹침·강수 누적·음수 wind·빈 입력·윈도우 밖)로 구/신 피벗을
`assert_frame_equal(check_exact=True)` 대조 — **1차에 컬럼 순서 회귀 검출**
(wind 블록을 끝으로 몰아 순서가 바뀜). `_WIND` 센티널로 스펙 안에 자리를 명시해 수정,
재검증 12건 전부 값·순서·dtype 완전 일치. 이후 기준선 삭제.

**`collectors/selftest_pivot.py` 신설** — 기준선 없이도 도는 영구 회귀망(네트워크 0회):
컬럼 순서 / reh 반올림 모델별 차이 / K→°C / freshest-wins / 강수 diff·음수 클립 /
경계 입력 12항목. 피벗을 고치면 이걸 먼저 돌릴 것.

### 타입 통일 — `parse_response` 계약 일원화 (사용자 지시)

같은 KMA 텍스트(`TMFC TMEF VARN LEVEL VALUE NAME`)를 두 타입으로 읽고 있었다:

| | 구 KIMG(`_common`) | 구 KIMR(`api_fetchers_jeju`) | 통일 후 |
|---|---|---|---|
| 반환 | `dict[int,float]` | `list[tuple[str,int,int,str]]` | **`list[tuple[str,int,int,float]]`** |
| TMEF | 버림 | 보존 | 보존 |
| LEVEL | 버림 | 보존 | 보존 |
| 값 | float | **str**(구 kimr.db TEXT 시절 잔재) | float |
| 비숫자 값 | 행 버림 | 문자열 유지 → 나중 NaN | 행 버림 |

KIMG 형은 "호출당 1시각" + "varn/level 충돌 없음" 두 가정에 기댄 **축약형**이었고,
KIMR 형이 상위집합이었다 → 상위집합으로 맞췄다. `derive_categories` 는 내부에서
varn 으로 접어 쓰므로 결과 불변(기준선 대조로 확인).

- **long/wide 는 이미 통일돼 있었다** — 두 long DF 모두 5컬럼 + 숫자 `fcst_value`
  (KIMR 은 끝에서 `to_numeric`, KIMG 는 생성 시 `float`). 어긋난 건 parse 하나뿐.
- 검증: `_base_common.py`/`_base_jeju.py` 기준선 대조 — KIMG derive 결과 완전 동일,
  KIMR 은 `(tmef,varn,level)` 동일 + 값 float 캐스팅만 차이, 행 수 동일. 이후 기준선 삭제.

### 주/백업 소스 선정 — **결론: 교체 불필요, 현행이 이미 최적 조합**

12 base(2026-07-07~18, 12z 전용) 아카이브로 실측 대비 비교. **MAE 기준** (상관계수는 이
창에서 못 쓴다 — 아래 참고):

| 변수 | KIMG | KIMR | 우세 | 현행 사용 |
|---|---|---|---|---|
| 일사 | **0.231** | 0.386 | KIMG | KIMG ✅ |
| 전운량 | **0.329** | 0.398 | KIMG | KIMG ✅ |
| 풍속 D+1 | 2.333 | **1.920** | KIMR | KIMR ✅ |
| 기온 | 1.029 | 1.247 | 엇갈림(D+3은 KIMR 우세) | KIMR |

→ **현행 병합(met=KIMR 우선 / 일사·운량=KIMG 단독)이 각 변수의 우세 소스와 일치.**
본 DB 병합·소스 교체는 **할 일이 없다**. 재학습 때 두 아카이브를 놓고 피처를 고르면 된다.

### ★측정 방법 교훈 — 장마창에서 상관계수는 쓰면 안 된다

처음 상관계수로 재서 "KIMG 기온 압승(0.860 vs 0.565)"이라는 **틀린 결론**을 냈다.
원인: 장마로 예보·실측 모두 일변동이 거의 없었다(예보 std 0.15~0.5°C). 변동 없는 신호의
r 은 노이즈다 — base 07-14 는 std 0.18°C 에서 **r=−0.816** 이 나왔다.
- 일교차가 실제로 있는 base 만(실측 std>1.2, n=4) 재측정 → **KIMR 0.922 / KIMG 0.939**
  (사용자 지적대로 기온은 관성이 커서 정상이면 0.9대가 나온다).
- **잘못 낸 2차 경보도 철회**: "예보 시각이 1~2h 어긋났다"는 lag 상관 결과 역시 같은 허상.
  MAE 로 재니 **lag 0 에서 최소**(KIMR 0.893 / KIMG 0.781) — 시각 정렬은 정상이다.
- 규칙: **변동이 작은 구간의 비교는 MAE·bias 로 한다.** r 은 신호 변동이 확보될 때만.

### ★기록 정정 — 2026-07-17 항목의 좌표 검증 숫자

"KIMR met 좌표 = 정상 확정: D+1 기온 vs **실측** r = west +0.980 / east +0.986 / south +0.950"
→ 이 숫자는 실측 대비가 아니다. 오늘 잰 **운영 forecast_horizon ↔ 아카이브 forecast_kimr**
상관이 0.971 / 0.982 / 0.945 로 거의 일치한다(두 KIMR 이 같은 값인지 확인한 수치).
실측 대비 D+1 기온 r 은 정상 변동일 기준 0.92 수준. 좌표 검증 근거로 재인용하지 말 것.
(부수 확인: 운영 KIMR 과 아카이브 KIMR 은 **92.7% 완전 일치, 평균 절대차 0.05°C** —
아카이브 수집기는 정상 동작.)

### ★재학습에 직결되는 계통 오차 (소스 선택보다 큰 문제)

```
풍속 bias  KIMR +1.55~+2.11 m/s   KIMG +1.67~+1.83 m/s   ← 양쪽 다 크게 과대예보
풍속 std   예보 2.47~2.63  vs  실측 1.73                  ← 변동폭도 과대
기온 bias  KIMR −1.09  KIMG −0.70                         ← 양쪽 저온 편향
기온 std   예보 0.89~1.03  vs  실측 1.45                   ← 일교차 과소
```
풍속 과대예보는 풍력 예측에 직접 영향. **재학습 시 bias 보정 여부를 반드시 검토할 것.**

### 아카이브 현황
12 base(7/07~7/18)에서 멈춰 있다 — 파이프라인이 7/18 이후 실행되지 않았다.
운량 판정은 표본 부족이 아니라 **7월 자체가 판별력이 없는 창**(NE57 도 7월 r=0.14,
1~6월 0.49~0.70). 맑은 날 포함 창이 쌓여야 재평가 가능.

### 3단계 — collectors 파일 재배치 (완료)

**9 → 7파일.** 4겹 래퍼 체인이 사라졌다.

```
collect_historical.py   ← 구 collect_data_jeju (실측 전용으로 축소·개명)
collect_forecast.py     ← 구 collect_data_jeju(기상부) + collect_data_jeju_new
                          + collect_forecast_new + collect_forecast_runs  (4→1)
collect_weather_kim.py  유지 (cfr/cj → cf 한 곳만 import)
api_fetchers_jeju.py / api_fetchers_kim2.py / _common.py / postprocess.py / selftest_pivot.py
```
- 조립은 **원문 구간 추출**로 했다(손으로 옮기지 않음) — 회귀 위험 최소화.
  이후 `cfr.`/`cdjn.`/`cj.` 참조를 로컬로 치환, `collect_forecast` 는 `cj` 의존 제거
  (DEFAULT_DB 를 자체 계산 — 실측/예보가 서로를 import 할 이유가 없다).
- `project_paths.COLLECT_HISTORICAL/COLLECT_FORECAST`, `run_pipeline`, `pages/common`,
  `collect_weather_kim`, `serve_chain` 주석까지 일괄 갱신.
- 검증: compileall / 7모듈 import / `--verify` / `selftest_pivot` 12건 /
  `collect_historical --historical-days 1 --no-save` 실호출 /
  `serve_chain --no-write` 120행 838MW / AppTest exception 0.
- **미래 DA 수정이 실동작 확인됨**: 같은 명령이 오전엔 48행이었는데 오후엔 **72행**
  (7/22 DA 가 15시경 발행되어 들어옴). 의도대로 동작.

⚠ `_common` 3분할(kma_core/kma_kimg)과 `api_fetchers_jeju` 2분할(kma_kimr/KPX)은
**하지 않았다.** parse_response 통일로 이름 충돌은 2개(`warmup`/`collection_window`)만
남았고, 둘 다 모델별 어댑터로 정당하다. 파일을 더 쪼개는 이득이 리스크보다 작다고 판단.

### KIMR = NC 자체완결로 전환 (사용자 결정 2026-07-21) + API 보존기간 실측

**결정**: KIMR·KIMG 는 각각 **자체완결 구조**를 갖는다. KIMG 가 1순위 뼈대, KIMR 이 2순위.
"KIMR GRIB met + KIMG 일사·운량" 하이브리드는 목적이 흐려서 폐기 — 단 그건 **아카이브
얘기**고, `forecast_horizon`(서빙 입력)은 재학습 전까지 현행 유지(모델이 NE57 분포 학습 +
실측 MAE 도 KIMG 우세: 일사 0.231 vs 0.386, 전운량 0.329 vs 0.398).

**구현**: `collect_weather_kim.fetch_kimr_nc_long` —
`NAME_KIMR_FULL = R030_NAME_LAND + ",TSKIN"` (17종)을 std NC per-hf 한 호출로.
일사를 받으려면 어차피 NC per-hf 를 도니까 met 을 같은 호출에 얹어 **호출수 증가 0,
GRIB 3콜 소멸**. `--met {nc,grib}` 로 구 GRIB 경로 보존(NC 장애 시 폴백).
※ 사용자 지시로 GRIB↔NC 값 비교 검증은 생략 — 같은 R030 모델의 다른 파일 포맷이라는 판단
  (근거: 서로 다른 GRIB 파라미터 스타일 두 아카이브가 temp 92.7% 완전일치).

**실증 (프로브, base 20260721 12Z)**: 17/17 전부 응답. `TEMP_C 27.42 / TEMP_SKIN 26.47 /
SOLAR_RAD 2.2316 / CAPE 369.1 / PRESS_MSL 1010.27` 정상.
- 관찰: `RAIN_CONV` 원값에 −0.01 이 나온다(누적인데 음수). 피벗의 `diff().clip(lower=0)` 가
  걸러내므로 무해하나 KMA 수치 노이즈로 기록.

### ★API 보존기간 실측 (백필 계획의 전제 — 기록이 상충해서 직접 쟀다)

| 경과일 | KIMR-NC | KIMG(NE57) | CLDFRA | 결손 |
|---|---|---|---|---|
| 7 / 30일 | 17종 | 13행 | — | 없음 |
| 90 / 105 / 120 / 150일 | **15종** | 13행 | ✅ 24레벨 | `ACSWDNB`, `TSKIN` |
| 180일 | **응답없음** | **응답없음** | — | 전체 |

- **상한 ≈ 150일(5개월)**. 구 기록의 "10일"은 과소, "180일"은 과대 — 둘 다 정정.
- **90일 이전 구간은 `--met grib` 가 유리**: GRIB 이 TSKIN(varn 17)을 주므로 NC 결손을 메운다.
  `SOLAR_RAD`(SWDDIR2+SWDDIF2)는 90일 이전에도 오므로 일사 자체는 확보된다.

**백필 시 주의 (다음 세션 예정 작업)**
1. 호출량: 150일 ≈ `150 × (KIMR 714 + KIMG 360)` ≈ **161,000콜** — 며칠에 분산 필요.
   2-패스(KIMR 전체 → KIMG 전체)는 이미 구현돼 있다.
2. **기존 12 base(7/07~7/18)는 GRIB 기반 met** 이고, 백필 skip 이 커버리지 기반이라
   재실행해도 안 덮는다 → 소스 혼재. 통일하려면 그 행을 지우고 재수집해야 하는데
   **`collect_weather_kim` 에 `--force` 가 없다** — 백필 전에 추가할 것.

### 저장 형식 결론 + 소스 마스크 신설 (사용자 결정)

**"long/wide 혼재"는 저장 계층엔 없었다** — DB 는 전부 wide 다(forecast_horizon 30,328×60,
historical 57,429×51, est_horizon_jeju, weather_kim 2테이블). long 은 API→DB 사이 메모리
중간 형식으로만 존재하고 이미 5컬럼 하나로 통일돼 있다. 단계마다 형식이 하나다:
`API 텍스트 → parse(4-tuple) → long(5컬럼) → wide → DB(wide)`.

**전면 long 저장은 기각** (측정 근거):
- 행 수 폭증: forecast_horizon 30,328 → **1,728,696**(57배), historical 57,429 → 2,756,592
- NULL 절감은 **6.0%뿐** (최악 컬럼 cinn_*/temp_skin_* 도 18~19%)
- **모델이 wide 를 먹는다** — 피벗은 어디선가 필수. 적재 때 1번 vs 조회마다 N행 수집+피벗
- 다운스트림(serve_chain·pages/common·horizon_backtest) 전부 wide 가정

**단, wide 가 출처를 지우는 건 실제 부채였다** — 이게 weather_kim.db 를 따로 둔 이유다.
→ **`src_met_{west,east,south}` 신설** (사용자 결정 2026-07-21):
- 값 `'KIMR'`/`'KIMG'`/NULL. `collect_forecast._met_source`, sentinel = temp
  (두 스펙 모두 temp 를 내고 verify_runs/base_complete 도 같은 sentinel).
- 일사(radiation_*)·운량(*_cloud_*)은 KIMR 단일면에 없어 **항상 KIMG** → 마스크 대상 아님.
- 스키마는 `_upsert_df` 의 ALTER TABLE 로 자동 확장. 2026-07-21 이전 base 는 NULL.
- `clip_ranges` 는 비수치 컬럼을 건너뛰므로 무해. `is_non_kma`(*_da/day_type) 필터에도 안 걸림.
- 검증: `selftest_pivot.py` 에 ⑦ 4항목 추가 — KIMR/KIMG 구간 표기, 마스크와 실제 값
  출처 일치(KIMR 300K→26.85 / KIMG 26.0), 일사 전 구간 존재, 3지점 생성. **17/17 통과.**

부수 확인: `smp_land_da` 는 57,408행이 채워져 있다(육지 잔재가 아니라 `fetch_kpx_est`
한 번 호출에 제주·육지 DA 가 같이 오는 부산물). 제주 SMP 모델의 참조 피처 여지가 있어 유지.

### forecasting/ 사전 조사 (2026-07-21, **수정 없음 — 계획만**)

3,245줄 / 14파일. **collectors 와 달리 죽은 코드는 거의 없다** — 진입점(serve_chain,
serve_smp)에서 13/14 가 도달 가능하고, 의존 구조도 깨끗한 2개 DAG 다.

```
serve_chain ─┬─ horizon_backtest ─┬─ serve_demand
             ├─ serve_demand      └─ serve_solarwind_hybrid
             └─ serve_solarwind_hybrid ─┬─ serve_solarwind_lgbm
                                        └─ solarwind_db_pipeline
serve_smp ─┬─ smp_db_pipeline ─ train_binary_smp ─ train_smp_db
           ├─ smp_d2_pipeline ─┬─ smp_phase2_depth ─ train_smp_db
           │                   ├─ train_smp_d2_da
           │                   └─ train_binary_smp
           └─ train_smp_db
```

**① 유일한 고아: `smp_calibrate.py`** (82줄) — 진입점에서 도달 불가 + 그 산출물
`models/smp/smp_calibrator.pkl` 을 읽는 코드가 **아무 데도 없다**. 삭제 후보(모델 파일 포함).

**② 진짜 문제는 이름이다** (중복이 아니라):
- `train_*.py` 3개는 **학습 전용이 아니다** — SMP 서빙이 런타임에 import
  (`train_smp_db.load_forecast` 피처빌더 / `train_binary_smp.persist` /
  `train_smp_d2_da._predict_da`). train/serve parity 를 위한 의도된 공유.
- `*_pipeline` 이 세 가지 다른 뜻으로 쓰인다:
  `solarwind_db_pipeline` = **PatchTST 모델 정의(nn.Module)+로더**(이름과 내용 무관) /
  `smp_db_pipeline` = SMP D+1 예측 단계 / `smp_d2_pipeline` = D+2 예측 단계.
- `horizon_backtest` = 백테스트 도구인데 `serve_chain` 이 스크래치 헬퍼
  (`build_scratch`/`set_scratch_forecast`)를 빌려 쓴다 — 운영이 진단 도구에 의존.

**③ 모듈 간 계약이 몽키패치다** (collectors 의 dict/list 불일치에 해당하는 지점):
```python
serve_demand._conn = lambda: sqlite3.connect(scratch_path)   # serve_chain:243, horizon_backtest:188
m.DB_PATH = scratch_path                                     # serve_smp:161
pipeline_d2.load_forecast = lambda ...: train_smp_db.load_forecast(with_target=False)  # serve_smp:272
```
스크래치 DB 주입이 **함수 인자가 아니라 전역 치환**이다. 재학습 때 입력 소스를 바꾸려면
이 전역들을 전부 추적해야 한다 — 정리 1순위.

**제안 (파일 수는 14→13, 핵심은 개명·역할 분리):**
```
serve_chain.py  serve_smp.py                    진입점 2개 (유지)
demand.py       ← serve_demand
solarwind.py    ← serve_solarwind_hybrid        solarwind_lgbm.py ← serve_solarwind_lgbm
patchtst.py     ← solarwind_db_pipeline         (모델 정의+로더)
smp_features.py ← train_smp_db                  (피처빌더 SSOT — 학습·서빙 공용)
smp_d1.py ← smp_db_pipeline   smp_d2.py ← smp_d2_pipeline
smp_depth.py ← smp_phase2_depth   smp_binary.py ← train_binary_smp   smp_da.py ← train_smp_d2_da
scratch.py  ← horizon_backtest 의 스크래치 헬퍼 (서빙이 실제로 쓰는 부분)
backtest.py ← horizon_backtest 의 백테스트 CLI
삭제: smp_calibrate.py (+ models/smp/smp_calibrator.pkl)
```
검증 전략: 서빙은 `serve_chain --no-write` 형상(120행 hd1~5, 수요 838MW, net_load 692MW)과
`serve_smp` 산출을 개명 전후로 대조. 순수 함수(피처빌더)는 collectors 피벗처럼
기준선 스냅샷 후 `assert_frame_equal`.

### 남은 작업 (미실행) — collectors Step 2~5

Step 1(죽은 코드 215줄)과 KIMR NC 전환은 완료. 아래는 다음 세션.
```
2. api_fetchers_kim2 → kma.py 병합            (이름 충돌 0 — 안전)
3. _common + api_fetchers_jeju → kma.py/kpx.py (충돌 2개: warmup·collection_window,
                                                 FORECAST_DAYS 글로벌 2벌 → 1벌)
4. collect_weather_kim → collect_forecast --archive 흡수
5. long_to_wide_v2(140줄) → 통일 피벗(_pivot_point+_derive_point) 교체
                                                 ★기준선 assert_frame_equal 필수
+ collect_weather_kim 에 --force 추가 (백필 전 필수 — 위 주의 ② 참고)
```
현재 collectors: 7파일 + selftest (`_common` 643 / `api_fetchers_jeju` 1008 /
`api_fetchers_kim2` 661 / `collect_forecast` 811 / `collect_historical` 333 /
`collect_weather_kim` 435 / `postprocess` 188 / `selftest_pivot` 165).

### 남은 작업 (미실행) — forecasting/ 정리
축: **fetch 계층은 출처축(KMA/KPX), collect 계층은 실행단위축(historical/forecast)**.
출처축으로 collect 을 나누면 `build_historical` 이 쪼개지고(KPX 수급+ASOS 를 한 DF 로
concat 해 한 번에 upsert), 실행단위축으로 fetch 를 나누면 KIMR fetch 가 복제된다.
```
kma_kimg.py           ← _common.py (이름만: KIMG/NE57 core + 창 SSOT + 키 로테이션)
kma_kimr.py           ← api_fetchers_jeju(KIMR부) + api_fetchers_kim2   ※ 이름 충돌 0
collect_historical.py ← api_fetchers_jeju(KPX/ASOS부) + collect_data_jeju(historical부)
collect_forecast.py   ← collect_data_jeju(기상부) + _new + forecast_new + forecast_runs
collect_weather_kim.py  유지, postprocess.py 유지, selftest_pivot.py 유지
```
- ⚠ `_common` + `api_fetchers_jeju` 를 **한 파일로 합치지 말 것** — 같은 이름 다른 내용
  함수가 남아 있었고(`warmup`/`parse_response`/`collection_window`), 특히
  `parse_response` 는 반환 타입이 다르다(`dict[int,float]` vs `list[tuple]`).
  두 모델의 병렬 어댑터라 파일을 나눠 두는 게 맞다.
- `collect_weather_kim` 은 `collect_forecast` 와 fetch 스택·병합 정책(분리 보존 vs
  combine_first)·upsert 의미(COALESCE vs REPLACE)·목적지 DB 가 전부 달라 지금 합치면
  한 파일에 독립 경로 2개가 나란히 놓일 뿐이다. **주/백업 소스 선정을 끝내고 본 DB 로
  병합하는 시점에 자연스럽게 흡수시킬 것.**

검증 방법(사용자 승인): `--out` 격리 DB 로 before/after 같은 base 수집 후 행·컬럼 diff.
본 DB 무영향. 실행 전 사용자에게 명령을 알릴 것.

---

## 2026-07-21 — 정리 세션 #4 (폴더 구조 정리 · land 제거 · pages/ 신설)

재학습을 앞두고 "불필요한 파일과 핵심파일이 난립"한 구조를 정리. 사용자가 백업 보유.

### 사용자 결정 (재질문 금지)
- **land 관련 자료 전부 삭제** — 이 저장소는 제주 전용.
- 재편 강도 = **A안**(삭제 + `pages/` 신설, 파일명은 유지). 서빙 경로 무수정이 우선.
- `data/refdata/solarwind_raw_jeju.csv`(9.9MB) 는 **유지**.
- `backfill_jeju_forecast.py` 도 삭제 승인.

### 삭제 (6파일 117KB + 캐시·로그)
`api_fetchers_land.py` · `collect_data_land.py` · `collect_data_land_new.py` ·
`collect_forecast_v2.py` · `collect_l010_archive.py` · `backfill_jeju_forecast.py`,
그리고 `__pycache__` 5개 · 7/17 로그 4개. **collectors 14 → 9** (전부 현역, 죽은 파일 0).

- land 는 `--region jeju` 경로에서 **한 줄도 실행되지 않았다** (분기로만 갈림).
  `data/input_data_land.db` 는 존재한 적도 없음 — 이 저장소에서 land 수집은 미실행.
- `collect_forecast_v2` 삭제가 land 제거를 깔끔하게 만들었다: `api_fetchers_kim2` 의
  유일한 land 의존(`land_points_override` → `ckl.POINTS`)이 v2 전용이었다.
- v2 에서 **`upsert_wide_coalesce` 만 `collect_weather_kim.py` 로 이주**(유일 사용처).
  COALESCE upsert 라 소스 분리 2-패스 아카이브에 필수 — 삭제하면 안 되는 함수였다.
- `api_fetchers_kim2` 스모크 테스트의 2번 항목(피벗 동일성)은 육지
  `kimg_land_long_to_wide` 대조였으므로 함께 폐지. 1번(실호출 파싱·GHI 검산)은 유지.

### pages/ 신설 — 화면 계층 한 폴더로
루트 `common.py`·`page_main.py` + `utils/` 4파일 → **`pages/`** 로 이동, `utils/` 폴더 소멸.

- **import 규칙이 collectors 와 정반대**: Streamlit 이 페이지를 *스크립트로* 실행해
  `sys.path[0]` = 저장소 루트라, `pages/` 내부는 항상 `from pages import common as C`.
  bare `import common` 은 깨진다. (`pages/__init__.py` 에 명시.)
- `common.py`·`page_main.py` 의 `ROOT` = `Path(__file__).parent` → **`.parent.parent`**.
  `utils/*` 는 원래 `.parent.parent` 였어서 depth 가 같아 무수정.
- `app.py`: `from pages import common as C`, `st.Page("pages/page_main.py")`.
- **Streamlit `pages/` 자동탐색 충돌 없음** — `st.navigation` 호출이
  `PagesManager.uses_pages_directory = False` 로 끈다 (streamlit 1.58
  `commands/navigation.py:327` 소스 확인). 헬퍼 모듈이 사이드바에 새지 않는다.

### 검증
- `compileall` 통과 / collectors 9 + forecasting 3 + pages 5 전 모듈 import OK.
- `collect_forecast_new.py --verify` 정상 (불완전 base 목록 = 기존 7/2~7/11 개편기 갭).
- **`serve_chain.py --utc 12 --no-write`: 120행 hd 1~5, 수요 838MW, net_load 692MW** —
  기존 형상과 일치. 서빙 체인 무손상.
- **`AppTest.from_file("pages/page_main.py")`: exception 0**, 사이드바 5메뉴 정상,
  지도·브리핑 모듈까지 로드. HTTP 200(껍데기)이 아니라 실제 렌더 검증.
- 백업: 재편 전 .py 42개를 스크래치패드에 복사해 뒀다(세션 한정).

### 남은 정리 후보 (미실행 — 다음 세션 판단)
- `forecasting/` 14파일은 **손대지 않았다**. `serve_*` / `*_pipeline` / `train_*` 3계층이
  평면에 섞여 있으나, **`train_*` 은 학습 전용이 아니다** — SMP 서빙이 런타임에
  `train_smp_db.load_forecast`(피처빌더)·`train_binary_smp.persist`·
  `train_smp_d2_da._predict_da` 를 import 한다(train/serve parity, 의도된 공유).
  분리하려면 `training/` 이동이 아니라 **개명**(`train_smp_db.py` → `smp_features.py`)이 답.
- `collect_data_jeju.py` 등 docstring 에 남은 land 언급(주석뿐, 코드 영향 없음).
- 저장소가 아직 `git init` 되지 않았다 — 다음 큰 변경 전에 초기화 권장.

## 2026-07-18 — 구현 세션 #3 (18z basetime 당일예보 — TODO 6개 전부 완료)

### 사용자 결정 (이번 세션)
- **18z 도 KIMG(NE57) 일사·운량 병행 수집** — 재학습 대비 아카이브 연속성("나중에 재수집
  불편 없게"). → 18z 본 수집 = 12z 와 동일한 KIMR met + KIMG 병합, "KIMR만 모드" 불필요.
- 00~03시 결측 = **12z 스크래치 패딩**(12z 가 뼈대), 그것도 없으면 시간보간 limit=4.
- **basetime 분리 보존 필수** — 표시(latest)만 1건, DB 는 12z/18z 원본 둘 다 (추후 검증용).
- 12z 실패 시: 알림(rc≠0)만 구현 — 대체는 freshest-wins 가 자동(18z 있으면 18z, 없으면
  전날 12z 가 가장 fresh 로 자연 대체).
- 18z 발표 지연 = 지속 관찰, 일단 delay 3h(PUBLISH_DELAY_HOURS 기존값) — cron 08:00.

### 실증 (M0 프로브, base 2026071718 UTC = 07-18 03시 KST 발표)
- **KIMR 18z 일사(SWDDIR2/SWDDIF2/ACSWDNB) std NC = 정상 응답** (주간 673 W/m² 등,
  hf=73 부터 0행 → lead 72h 확정). **hf=0 앵커 유효**(ACSWDNB=0.0) — 누적 diff 기준점 OK.
- **KIMR 18z CLDFRA 운량 = 정상** (24레벨, hf 1~72).
- **KIMG(NE57) 18z = hf 0~84 전부 응답** — lead ≥69h 요건 충족(당일~모레 hf≤68 여유).
- 어제 18z 아카이브도 가용 → 18z 백필 가능. (06~08시 실가용 시각은 운영에서 관찰.)

### 구현 (M1~M6)
- **수집 창 일반화**: `_common.window_bounds(base_utc, days)` 신설 = 창 산식 SSOT.
  opt-in 플래그 `_common.SAMEDAY_18Z`(기본 False) + 18z 면 창 = **당일 04시(hf=1)~days일**
  (hf=0 은 분석장이라 제외 — est 당일 03시 행은 서빙 패딩 담당). 위임 4곳:
  `_common.collection_window` / `api_fetchers_jeju.collection_window`(+`ef_param_for`
  start≥1 클램프) / `api_fetchers_kim2.hf_range_1h` / `collect_weather_kim._window`.
  플래그 ON 은 진입점 2곳만: `collect_forecast_new`(--region jeju --utc 18, 혼합 실행 거부,
  days 기본 3) / `collect_weather_kim`(--utc 18, days 기본 3, backfill 도 --utc 지원).
  `fetch_kimr_rad_long` 앵커 조건 `>1`→`>=1`(18z 는 hf_list[0]=1 이라 앵커 hf=0 필요).
  **12z 무회귀 검증**: 신구 산식 1400건 비트 일치 + 12z 라이브 프로브 120행 동일 형상.
- **본 수집 18z**: forecast_horizon 에 base "당일 03:00:00", hd=0 20행(04~23시)+hd1·2 각
  24행, sentinel NULL 0. horizon_d 산식은 무수정(값 0 자연 생성). weather_kim 18z 도
  KIMR(met 1173행 GRIB+일사 69/69 앵커 포함+CLDFRA 68/68)/KIMG 커버리지 100%.
- **서빙 체인 18z 매핑** (`serve_chain.py`): 18z base 감지(`base_mode`, 시각 03:00) →
  **origin = 전일 23시, 모델지평 n=hd+1**(HZ_18Z=(1,2,3)) — 수요 LGBM·태양광 PatchTST 가
  "origin=23시, n=1=익일" 학습 구조라 재학습 없이 hd=0 산출. est 태깅 hd=n−1,
  hd=0 은 base(03시) 이전 행 미적재(21행). `set_scratch_forecast(pad_from_prev=True)` —
  직전 발표 freshest 행으로 당일 00~03시 패딩(이번 실행 4행) + `_hourly_interp` limit 4.
  `pick_bases`: `--utc 12/18` 필터(cron 이 자기 base 만 체인) + 풀 타임스탬프 `--base` 지원.
  chain18 은 전일 12z 부재 시 경고+rc=1. **12z 가드**: `serve_smp.list_bases`·
  `horizon_backtest` bases 쿼리에 `substr(base,12)='21:00:00'` — SMP(24행 강제)·백테스트
  (origin 산식)에 18z 유입 차단. 검증: 구 base 재실행 diff 0(120행·수요 817·NL 679 일치),
  18z 실행 69행 hd0~2, serve_smp 가 18z 를 건너뛰고 07-17 12z 선택.
- **freshest-wins 재작성** (`common._hz_select` latest): MIN(horizon_d) JOIN → ROW_NUMBER
  (`horizon_d ASC, base DESC`) — 12z/18z 동률(내일=오늘 12z hd1 vs 오늘 18z hd1) 중복행
  제거. 신구 SQL 은 18z 없는 구간에서 완전 일치(360행). fixed 모드에 base_hour 필터 추가.
- **run_pipeline**: 단계 forecast18/weather18/chain18 + 그룹 **light18**(historical 선행 —
  수요 168h 과거창·태양광 전일 이용률 필수). all=12z 5단계 명시 고정. 12z chain 에
  `--utc 12` 명시. E2E: `--steps light18` 전체 성공(~3분: 45s+1s(완전 skip)+135s+14s).
- **UI 용어 전환**: `common.hz_label`(당일/익일/모레/N일후)·`base_badge`(새벽/전일 밤 발표)
  SSOT. 예측 확인 슬라이더 = "표시 기간(일)"(h_lo 의존 제거), hover = "{날짜} {새벽|밤}
  발표 · {지평라벨}". 검증 3탭에 **발표 세그먼트**(12z/18z, 발표별 지평 옵션
  `jeju_horizon_options`, key 를 발표별 분리) + 탭② **리드타임(h) 축 토글**(실측 평균
  리드 컬럼 "리드(h)" — 12z 익일 14.5h vs 18z 당일 10.0h 확인). hero 부제·툴팁 =
  발표 배지(`_issue_badge` — 그 날 09–15시를 채운 MAX(base) 기준), conf_of 에 당일/과거
  분리. `jeju_horizon_range` 하한 1 클램프(hd=0 이 기존 위젯에 새지 않게). SMP 문구
  "이틀 전 밤 발표 예측". 정확도 함수에 base_hour 관통(발표별 분리 검증 — 사용자 요구).
  brief_jeju 는 MAX(base) freshest 라 무수정. 앱 기동 HTTP 200 확인.
- **README**: crontab 2줄(00:20 풀 + 08:00 light18, 지연 관찰 주석), basetime 이원화
  운영 노트, D+7 낡은 표기 정정.

### 운영 메모
- 18z 라이트 총 소요 ~3분/일 (본 수집 1.2분 + weather_kim 2.5분 + 체인 14s).
- est_horizon_jeju 에 hd=0 첫 적재(07-18 03:00 base, 21행). 검증 페이지 "새벽 발표"
  세그먼트는 당일 실측이 쌓이는 대로 채워짐(오늘 이미 첫 표본: 당일 리드 10.0h).
- 완전 롤백 = cron 18z 라인 제거 → `DELETE FROM forecast_horizon/est_horizon_jeju WHERE
  substr(base,12)='03:00:00'` → 코드 리버트 (스키마 무변경, 플래그 기본 False).

### 이월
- 주/백업 소스 선정(weather_kim.db 축적, 장마 이후 창) → 본 DB 병합 설계 → sandbox 주입 실험
- KIMG west 셀 grid 탐색(사용자와) / 수집주기 재검토 / 18z 발표 지연 추이 관찰(로그)
- 18z weather_kim 백필(과거 18z, 필요 시 `--backfill N --utc 18`) — 아카이브 밀도용 선택 항목
- 서버 배포 시: 새 venv 클린 설치 + crontab 2줄 등록

---

## 2026-07-17 — 연구 세션 #2 (KIMR/KIMG 심층 실증)

### 실증 결과 (라이브 API 테스트, base 2026071612 UTC·서부 셀 X530/Y251·std NC 엔드포인트)

**① KIMR(R030) 일사 — 실재함 (문서·실제 일치).**
- PDF(`forecasting/한국형수치모델(KIM)변수정보.pdf` p.4, 지역모델/단일면)에 일사 4종
  명시: SWDDIR2(직달)·SWDDIF2(산란)·SWDDNI2(법선직달)·ACSWDNB(누적, MJ/m²).
- 라이브 확인: 주간 hf=15(KST 정오)에 SWDDIR2 926.95 / SWDDIF2 47.25 /
  SWDDNI2 961.10 W/m², ACSWDNB 11.1 MJ/m² — 값 정상. 야간 hf=24 는 순시 3종=0.
- 즉 지난 세션 로그의 "일사는 KIMR 에 없다"는 **부정확** — 정확히는
  "제주 met 를 나르는 KIMR GRIB **시계열 파일**에 일사가 없다"(별도 NC 콜은 가능).

**② KIMR(R030) 운량 — 문서에만 있고 실제 API 는 없음 (불일치 확정).**
- PDF p.5 에는 LCDC/MCDC/HCDC(single-layer, 0~1)가 분명히 있으나,
  실제 호출은 **0행**(LCDC 단독·3종 묶음·KIMG식 소문자 tcld/lcld/mcld 모두).
  같은 형식의 KIMG(NE57) 대조군(tcld,lcld,mcld,dswrsfc)은 정상 반환 → 호출 형식
  문제 아님. 서버가 여는 파일 = `r030_v040_easia_etc...nc` — 이 파일에 운량 변수 자체가 없음.
- 과거 실측(2026-07-04, api_fetchers_kim2 주석 "R030 운량 없음")이 오늘도 재현.
- 대안은 등압면 frcc(CLDFRA, varn 6032, 24레벨) 결합운량뿐 — nb05 검증 완료이나
  GRIB data=P 지점당 2콜 + 밤 혼잡 시 60~110s/콜 리스크 (v2 --skip-frcc 로 분리했던 이유).
- ※ std 엔드포인트 `nwp` 인자는 **대문자**(R030/NE57) 필수 — 소문자는 "Input variable error".

**③ "운량이 제대로 수집 안 되어 KIMG+KIMR 병행" 기억 — 사실로 확인.**
- 현행 체인: collect_forecast_new → collect_data_jeju_new → collect_data_jeju,
  지점별 `kimr_part.combine_first(kimg_part)` (KIMR 우선). 일사·운량은 KIMG 단독
  (SOLAR_RAD=dswrsfc, TCLD/MIDLOW_CLOUD=tcld·lcld·mcld). KIMG 를 못 버리는 근본
  원인 = KIMR 단일면에 운량이 실제로 없어서.
- DB 실태(forecast_horizon, base 210개): radiation_* 3지점 100%(30140/30140),
  total/midlow_cloud_* 97.4%. 운량 결손 788행은 **전부 7/2~7/11 base**(KMA 개편
  직후 3h 해상도기, base당 56/134행)에 몰림 — 7/12부터 정상. "운량 수집 문제"의 실체.
- v2 아카이브 컬럼(radiation_direct/diffuse_*, total/midlow_cloud_r030_*)은
  jeju·land 어느 DB에도 **없음** — collect_forecast_v2 는 코드만 이식되고 병행수집
  미가동. forecastmodel 의 new_kma 연구 폴더는 비어 있음(REPORT 소실, 코드 주석이 유일 기록).

### 함의 — KIMG 완전 제거 가능한가?
- **일사**: KIMR 전환 가능. v2 에 fetch·derive 구현 완료
  (SOLAR_RAD=(SWDDIR2+SWDDIF2)×0.0036 — 현행 단위식 호환, FARMS 산출물).
- **운량**: 단일면 불가 → frcc 등압면 결합으로만 가능(비용·야간 혼잡 리스크).
- **단, 서빙 모델(태양광·수요)은 NE57 dswrsfc·운량 분포로 학습됨** — 소스 전환은
  재훈련 없이는 위험. v2 가 서빙 컬럼 불변 + 신규 아카이브 컬럼 설계였던 이유.
- 옵션: (a) 현행 유지(KIMG=일사·운량 전용) (b) v2 병행수집 가동 → KIMR 일사·frcc
  운량 아카이브 축적 → 재훈련 시점에 전환 (c) 즉시 완전 전환(비추천).
  → **사용자 결정 대기.** D+5 지평이라 KIMG 의 장지평 보충 역할은 이미 소멸,
  남은 역할은 일사·운량 공급뿐.

### 추가 실증 — frcc 등압면 운량, 실제 수집 가능 + 속도 측정 (2026-07-17 13:34~13:43 KST, 낮)

**결론: 수집 가능. 단 경로별 명암이 확실.**

| 경로 | 결과 | 속도 (west 1지점, hf3..120=118시각) |
|---|---|---|
| GRIB `kim_grib_pt_tmfc.php` data=P, 청크 96시각 (v2 운영 코드 그대로) | ❌ **무조건 504** | 게이트웨이 30s 컷 — 과거 "60~110s/콜 성공"과 달리 서버가 30s에서 끊음 |
| GRIB 청크 24시각 | ❌ 504 | 〃 |
| GRIB 청크 8~16시각 **순차** | ✅ 118/118 완전 | 콜당 2.5~23s(중앙 ~10s), 지점당 15콜 ≈ **2.5~3분**, 3지점 ≈ 8~10분 |
| GRIB **병렬**(4워커) | ❌ 전부 504 | 직전에 단독 성공한 청크도 병렬이면 실패 — 서버가 키 단위 직렬화하는 듯. **병렬 금지** |
| **std NC `nph-kim_nc_pt_txt2_std` data=P name=CLDFRA per-hf, 병렬 8워커** | ✅ 118/118 완전 | 콜당 중앙 1.0s/p90 1.4s, **지점당 16.4s → 3지점 ≈ 1분** |

- 결합 운량 값 정상(24레벨 → total/midlow, 표본: 오늘 total=1.0, D+5 0.27).
- CLDFRA(std)=frcc(GRIB) 동일 필드는 과거 nb05 검증(0.978==0.978) — 오늘은 가용성·속도만 재확인.
- "수집속도가 매우 느렸다"는 기억 = 사실(당시 96시각 청크 밤 혼잡 60~110s/콜 + 부분 응답).
  지금은 서버 게이트웨이가 30s 컷으로 바뀌어 **96청크 시대는 끝** — v2 코드 채택 시
  `_GRIB_EF_CHUNK` 96→8~16 축소 필수. 그러나 **std CLDFRA per-hf 병렬이 압도적으로 우월**.
- 콜 수: std 경로 118콜/지점×3지점=354콜/base — 현행 KIMG per-hf(~135콜/지점)와 동급, 한도 부담 없음.
- ⚠️ 측정은 낮(13시대). 운영 cron 시간대(저녁~밤 혼잡)에서 재측정 후 채택 결정 권장 —
  v2가 NC per-hf met 수집을 기각한 근거가 "혼잡 밤 40분+/base" 실측이었음.

### 구현 — KIMR 전운량 수집기 신설 (사용자 지시: "안정적인 방식으로 기상수집 진행")

**`collectors/collect_cloud_kimr.py` 신규 + 파이프라인 ②-2 단계 편입. 실적재 검증 완료.**

- 채택 경로: **주경로 = std NC CLDFRA per-hf 병렬 8워커** / 폴백 = GRIB frcc 소청크(8시각) 순차
  — 엔드포인트 이중화. fetch 계층은 `api_fetchers_kim2.fetch_r030_cldfra_long()` 신설
  (카테고리·스키마는 기존 fetch_r030_frcc_long 과 동일 → 피벗 호환).
  `fetch_pt_std` 에 `data=` 인자 추가, `_GRIB_EF_CHUNK` 96→8 (30s 게이트웨이 실측 반영).
- 신규 컬럼: forecast_horizon 에 `total_cloud_r030_{west,east,south}` +
  `midlow_cloud_r030_{west,east,south}` (6개). **서빙 입력 불변** — 병행 아카이브.
- upsert = 컬럼 보존 병합(collect_forecast_v2.upsert_wide_coalesce 재사용, COALESCE).
  ★순서 규칙: ② 예보 수집의 INSERT OR REPLACE 가 나중에 돌면 r030 컬럼이 NULL 로 덮임
  → 파이프라인 순서(②→②-2)가 안전장치. 예보 수동 재실행 시 ②-2 도 재실행할 것.
- run_pipeline: `("cloud", "②-2 KIMR 전운량 아카이브 (CLDFRA, D+5)")` 단계 추가,
  collect 그룹 = historical,forecast,cloud. project_paths.COLLECT_CLOUD_KIMR 추가.
  관리자 원클릭/개별 버튼은 PIPELINE_STEPS 공유라 자동 반영.
- **실적재 검증** (base 2026071612): 3지점 118/118 완전, 총 21~26s, 폴백 0회.
  기존 컬럼 무손상(temp/radiation/total_cloud 144행 유지), 파이프라인 경유 재실행 idempotent.
- 커버리지 <95% 면 rc=1 (cron 알림). D+5 꼬리 2h(22·23시)는 R030 lead 상한 120h 로 정상 결손.
- **첫 실측 비교**: NE57 vs R030 전운량 상관 r=0.175 (n=118) — 두 모델 운량 예보가 크게 다름.
  서빙 소스 전환은 아카이브 축적 후 재훈련으로만 (README 운영 노트에 명시).

### 개편 — 사용자 지시 6건 반영 (같은 날 오후)

**① 좌표 감사 (echo-back + 실측 상관 실증):**
- KIMR met 좌표 = **정상 확정**: D+1 기온 vs 실측 r = west +0.980 / east +0.986 / south +0.950.
  KIMR 운영 셀(X530/Y251 등) echo = 고산 5.9km 등 전부 근접.
- KIMG 좌표: east/south 는 근접(성산 2.4km / 태양광단지 4.6km). **west 는 명목 좌표
  33.4427/126.1713 이 고산 ASOS(33.2938/126.1628)보다 ~17km 북쪽(해상 방향)**,
  셀 echo 33.4/126.2 (고산까지 ~12km, echo 0.1° 정밀도). west 운량 상관이 3지점 중
  최저(0.588)인 것과 정합 — 단 6개월 운량 r 0.59~0.69 는 "엉뚱한 지점" 수준은 아니므로
  좌표가 주범은 아님. **KIMG west 셀 재탐색은 사용자와 함께 진행하기로** (grid 후보 비교).
- south 주의: 실측 anchor 는 ASOS 189(서귀포)인데 예보 지점(태양광단지)에서 22.6km — 기존 설계.
- KIMG 서빙 좌표는 **변경하지 않음** (입력 분포가 바뀌면 모델 영향 — 재훈련과 함께만).

**② GRIB 폴백 폐기 (사용자 결정):** fetch_r030_cldfra_long 에서 GRIB 폴백 제거.
안전장치 = fetch 순차 재시도 2라운드 + 수집기 커버리지<95% rc=1 + 재실행=치유(COALESCE).
실측상 폴백 발동 사례 0회였음.

**⑤ 격리 원칙 (사용자 결정: 검증 전 본 DB 병합 금지):**
- collect_cloud_kimr 기본 출력 = **`data/cloud_kimr.db`** (forecast_horizon 조각 스키마).
  본 DB 반영은 `--production` 플래그로만(검증 후). `--backfill N` 추가(12z, resume-skip).
- 어제 본 DB 에 넣었던 r030 6컬럼·118행은 격리 DB 로 이전 후 **본 DB 에서 DROP** —
  forecast_horizon 은 60컬럼 원상 복구.
- 백필 실행: 7/07~7/16 12z **10 base × 118시각 × 3지점 완전 수집** (base당 27~31s,
  재시도·결손 0). KMA 아카이브는 최소 10일 전 12z 까지 가용 확인.

**④ 실측 대비 비교 (10 base, 동일 표본 공정 비교):**
- 전운량 D+1 vs 실측 r: west R030 +0.17 / NE57 +0.07, east R030 −0.08 / NE57 +0.20,
  south R030 +0.28 / NE57 +0.15. MAE 는 전 지점 NE57 우세.
- **판정 유보**: 7월 장마창은 판별력이 없음 — NE57 자체가 7월 D+1 r=0.14 로 붕괴
  (1~6월 월별 0.49~0.70, 실측 std 는 7월 0.31 로 정상 분산). 두 모델 다 이번 달 운량을
  못 맞히는 것. → 아카이브를 계속 쌓아 맑은 날 포함 창에서 재평가.
- 서빙체인 주입 실험(사용자 승인): 본 DB 사본(sandbox)에 r030 → total/midlow_cloud_*
  주입 후 serve_chain 을 사본으로 돌려 est 출력 비교 — 아카이브가 더 쌓인 뒤 실행 권장.

**③ 피처셋 정리는 사용자 보고(대화)로 전달** — 요지: 운량·일사는 태양광 LGBM
(radiation_west/south, total/midlow_cloud_west/south, clearsky_ratio, solar_damping)과
수요 모델(기상4 + 구름4 h≤48 + solar_deficit/ramp) 양쪽의 핵심 입력. KIMR 일사 4종
(SWDDIR2 직달 / SWDDIF2 산란 / SWDDNI2 법선 / ACSWDNB 누적)은 실증 완료·미수집 —
ACSWDNB diff 가 실측(시간누적 일사)과 정의 동일해 학습 정합성 최적.

### 수집기 정리 — KIMR/KIMG 이중 아카이브 통합 (사용자 지시 3원칙 반영)

**`collectors/collect_weather_kim.py` 신설, `collect_cloud_kimr.py`·`cloud_kimr.db` 폐지(흡수).**

- **① 동시 수집, 지평 D+5 1h**: 격리 DB `data/weather_kim.db` 에 소스 분리 2테이블 —
  `forecast_kimr` (met 14종 GRIB 1콜/지점 + 일사 NC per-hf 병렬 "SWDDIR2,SWDDIF2,ACSWDNB"
  + CLDFRA 운량; 컬럼 = 표준명 + radiation_direct/diffuse/acswdnb·cape·mslp 등 72컬럼) /
  `forecast_kimg` (현행 NE57 fetch 재사용, 42컬럼). 표준 컬럼명 공유 → 비교·스왑이 쿼리 하나.
- **② 주/백업 선정은 데이터로**: 실측 대비 건전성 비교 후 결정(장마 끝난 창 필요).
  구조상 어느 쪽이든 주가 되면 나머지는 combine_first 백업 — 병합 도구는 선정 후 구현.
- **③ 3h 결손 안전장치**: 1h 그리드 reindex → 내부 시간보간(연속 2h까지, 외삽 금지),
  보간 셀 수 로그. + std NC 재시도 2라운드, 커버리지<95% rc=1, 재실행=치유.
- 실측정 (base 2026071712): KIMR 62s(118시각 100%, 앵커 포함 rad 119/119) +
  KIMG 162s(120시각 100%) = **~4분/base**. KIMR vs KIMG 값 표본 대조 정상
  (운량·일사가 갈리는 시각 존재 — 예: 7/19 12시 rad R3.51 vs G1.96, tcld R0.00 vs G1.00).
- run_pipeline ②-2 = "weather" 단계로 교체 (collect 그룹 historical,forecast,weather).
  구 cloud_kimr 데이터(11 base 운량 1180행)는 forecast_kimr 로 이전(표준명 rename).
- 백필 --backfill 11 (7/07~7/16 KIMR met·일사 보충 + KIMG 신규) 백그라운드 가동.
- DB 3지점 질문 답: **예보·실측 모두 west/east/south 3지점 수집 중** (예보 97~100%,
  실측 99%+). 단 실측 일사(solar_rad)는 west/south 만 — east(성산)는 일사계가 없어
  historical 에 solar_rad_east 컬럼 자체가 없음.

### basetime 확정 설계 (2026-07-17 저녁, 사용자 확정 — 구현은 다음 세션)

**basetime = 12z(뼈대) + 18z(라이트 추가 정보)**
- **12z**: 현행 풀 파이프라인 그대로 — KIMR+KIMG 병행 수집·이중 아카이브·체인·SMP, 00:20 KST cron
- **18z**(전일 18UTC = 당일 03시 KST 발표, 가용 ~06~08시): **KIMR만** 라이트 수집
  (met GRIB + 일사 NC + CLDFRA 운량). KIMR lead 72h → 당일~모레 03시 커버.
- 18z 의 4~5일후 지평은 **12z 자료가 담당** — 대상 시각별 최신 발표 우선(freshest-wins)으로
  자연 구현 (18z 가 못 채우는 시각은 12z 값이 최신으로 남음).
- **horizon_d=0 (당일예보) 신설**: 18z base 는 당일 03시부터 저장. 정의 = "발표일 기준 N일째"
  그대로 자연 확장 (스키마 변경 없음, 값 0 이 처음 생길 뿐).
- **용어 체계 (사용자 승인)**: ① 내부 = horizon_d 유지 ② 화면 = **D+N 노출 금지**,
  대상일("7/19(일) 예보") + 발표 배지("새벽 발표"/"전일 밤 발표") ③ 대화·분석 =
  발표시각+리드명 조합 — "12z 익일예보", "18z 당일예보", 모레, N일후. (D-1/하루전 표기 금지.)

### TODO — 다음 세션 구현 목록 (이 shell 은 여기서 마무리)

1. **18z 라이트 수집 경로**: 격리 아카이브는 `collect_weather_kim --utc 18 --kimr-only` 로
   이미 동작. **본 수집(forecast_horizon) 18z 적재 설계가 남음** — 현행 제주 경로는
   KIMR+KIMG 병합 고정이라 "KIMR만" 모드 신설 여부 결정.
2. **★열린 결정 — 18z 서빙 입력의 일사·운량 소스**: KIMR만 수집하면 일사·운량이 KIMR 소스
   — 서빙 모델은 KIMG(NE57) 분포로 학습됨. 선택지: (a) 18z 는 met 만 갱신, 일사·운량은
   12z(KIMG) 값 유지 (b) KIMR 일사·운량 사용(분포 불일치 감수) (c) 18z 에 KIMG 일사·운량만
   추가(라이트성 후퇴). **검토 추천 = (a)**.
3. **horizon_d=0 배관**: collection window 산식(현행 D+1 00시 시작 → 18z 는 당일 03시 시작),
   `serve_chain.ORIGIN_HOUR=23` 하드코딩(18z origin 은 03시), `common.JEJU_HZ_MAX` 클램프,
   backtest/검증 로직의 D+1 가정 전수 점검.
4. **serve_chain base 선택 규칙**: 현행 "forecast_horizon 최신 1건" — 18z/12z 혼재 시
   각 수집 직후 그 base 만 체인 실행하는 규칙으로 정리.
5. **UI**: 종합 = 대상일 + 발표 배지(D+N 제거), 대상 시각별 최신 발표 표시. 검증 = basetime
   필터(12z/18z) + 당일/익일/모레/N일후 라벨 + (심화) 리드타임(h) 축 추가.
6. **cron**: 18z ~08:00 KST 라이트(수집+체인) / 12z 00:20 KST 풀(현행) 2줄.

### 이월 항목
- 주/백업 소스 선정(weather_kim.db 축적, 장마 이후 창) → 본 DB 병합 설계 → sandbox 주입 실험
- KIMG west 셀 grid 탐색(사용자와) / 야간 ②-2 소요 확인 / 수집주기 / 서빙체인 점검
- **weather_kim 백필 완료**: 7/07~7/17 11 base × 2소스 전부 95%+ (KIMR 1298행 / KIMG 1320행).
  교대 호출로 느려졌던 구간에서 7/08 KIMG east 16시간이 빠졌었고(87%) 단일 base 재수집으로
  치유 완료 — 교대 = 저속 + 결손이라는 사용자 경험이 그대로 재현된 사례.
- **백필 2-패스 수정 완료 (사용자 지적)**: base 마다 KIMR→KIMG 교대 호출은 apihub 부담으로
  매우 느려짐(backfill_jeju_forecast 의 기존 교훈과 동일 — 교대 시 KIMR 504 빈발).
  `--backfill` 을 **KIMR 패스 전체 → KIMG 패스 전체** 소스 분리로 재구성 + 로그
  line_buffering(백그라운드 실행에서 실시간 로그). 매일 cron(단일 base)은 교대 1회라 현행 유지.

---

## 2026-07-17 — 구축 세션 #1 (M1~M10 완료)

### 완료된 것

**뼈대 (M1~M3)**
- forecastmodel 01~04 + Model_api_added 기능을 `jeju_model` 한 폴더로 이식.
  번호 폴더 폐지 → `collectors/ forecasting/ models/ data/ utils/ tools/` + 루트 5파일.
- importlib 동적 로드 전부 **일반 import** 전환. 축약 별칭 리네임
  (sw→patchtst, L→lgbm_serve, P1/P2→pipeline_d1/d2, S2/S3→serve_demand/serve_solarwind).
- 원본↔신규 모듈 매핑표는 README.md 참고.
- 체인·SMP 오프라인 스모크 → 본실행 검증 완료.

**앱 (M4~M8)**
- 5메뉴: 종합(3구역 지도 hero + AI 브리핑 + 지표) / 예측 확인(D+1~k 차트 + 위험구간 음영 +
  임계값 popover + SMP + 시간별 표) / 예측 검증(3탭) / 데이터 현황(히트맵) / 관리자(원클릭+개별+고급).
- `run_pipeline.py` = 수집→예측 단일 진입점. **cron 과 관리자 원클릭이 PIPELINE_STEPS 를 공유.**
- Gemini 브리핑: `utils/brief_jeju.py` — 리스크는 파이썬 `_detect_risks` 확정,
  임계값은 chart_warn 과 session_state 공유, 저장 = `data/ai_briefings.db`.
- 비밀번호 게이트(6h 토큰), OPS_PASSWORD 관리자 게이트.

**디자인·구조 개편 (M9, 사용자 확정 반영)**
- **DB 교체**: 사용자 제공 `input_data_jeju_new.db` → `input_data_jeju.db`
  (스키마 동일, base 210개, 7/5~7/13 갭 해소 → SMP D+2 정상).
- **지평 D+7 → D+5**: `run_pipeline.py` 가 `--days 5` 전달(수집기 무수정),
  `serve_chain.HZ=1..5`, `common.JEJU_HZ_MAX=5` 로 UI 클램프.
- **3구역 = 읍면동 43개 명단 → 구역별 병합(dissolve)**: 구역 안 경계선 제거,
  구역 사이 경계선만 표시. 명단 = `tools/make_jeju_zones.py` ZONE_ASSIGNMENT
  (서부 15 / 동부 14(우도·추자 포함) / 남부 14). 경계 원본 = `data/refdata/jeju_emd_2013.json`.
- **라이트/다크 겸용 테마**: `st.context.theme` 감지 → `common.inject_style()` 이
  CSS 토큰·plotly 템플릿·차트 팔레트(COLOR)·지도 타일(light_all/dark_all)을 일괄 전환.
  차트 팔레트는 dataviz 6-checks 검증기 통과값(`common._CHART_PALETTES`) —
  동시 표시 5색이 양 테마에서 CVD·명도·대비 PASS.
- 위험구간 밴드 색 = 임계값 popover 이모지 정합(🔴최저·🟡저/심야·🔵고·🟣최대), 임계값 직접 설정 유지.
- 지도 hero 스케일은 사용자가 직접 조정 예정
  (`utils/weather_map_jeju.py` → `fitJeju()` 의 sidePad·fitBounds 패딩·setMinZoom).

**마감 (M10)**
- README(구조·매핑표·crontab 가이드·운영 노트·테마) / requirements.txt
  (python-dotenv 는 현 환경 0.21 기준으로 완화).
- 최종 스모크: 전 파일 컴파일, `run_pipeline --steps predict` 전체 성공
  (chain 120행 D+1~5, SMP D+1·D+2 각 24h), 앱 기동 OK.
- ※ 새 venv 클린 설치 테스트는 미실시 — 서버 배포 시점에 확인할 것.

### 사용자 결정 사항 (재질문 금지)
- 지평 D+5, 구역=행정구역 명단, 다크/라이트 겸용, 브리핑 위치·카드 현상 유지,
  사이드바 메뉴 보류, 지도 스케일은 사용자 직접 조정.
- cron 은 리눅스 서버(배포 시점 등록), 자동화 범위 = 수집+예측 전부.

### 알려진 사실·주의
- **KIMR 에는 일사(radiation)·구름(cloud)이 없다** — 수집기 문서 기준. KIMG 가 전 지평 단독 공급
  (태양광·수요 모델 필수 입력)이라 현재는 KIMG 를 일사·구름 전용으로 유지 중.
  ※ 단, 문서와 실제 API 가 다를 수 있음 — 다음 세션에서 실증 예정.
- 동부(성산) 일사계 없음 → 실측 모드 '관측 없음' 폴백은 설계 동작.
- SMP D+2 는 lag168(7일 전 하루전 SMP) 필요 — historical 갭이 생기면 그 기간+7일 D+2 공백.
- 아카이브의 과거 D+6~7 예측 행은 남아 있고 UI 만 D+5 로 클램프.

### 다음 세션 계획 (사용자 지정)
1. **KIMR 수집 변수 탐색** — 문서가 아니라 실제 API 응답으로 KIMR 제공 변수를 실증.
   일사·구름이 정말 없는지 확인 → 있으면 KIMG 완전 제거(KIMR 단독 수집으로 간소화).
2. **수집주기 변경** — cron 스케줄·발표 주기(12z 외 추가 여부) 재설계.
3. **서빙체인 점검** — D+5 체계에서 체인 전반 재점검.
