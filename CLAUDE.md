# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

제주 전력 **수요·태양광·풍력·순 부하·SMP** 예측 파이프라인 + Streamlit 대시보드. 자기완결형 —
수집(KMA/KPX API)·서빙(LGBM/PatchTST)·화면이 한 폴더에 있다. 전 문서·주석·UI 는 한국어.

## 명령

```bash
streamlit run app.py                      # 대시보드 (비밀번호 게이트 → .streamlit/secrets.toml 의 password)

python run_pipeline.py                    # 12z 풀: 실측→예보→KIM 아카이브→체인→SMP (cron 00:20 KST)
python run_pipeline.py --steps light18    # 18z 당일예보 라이트 (cron 08:00 KST)
python run_pipeline.py --steps collect    # 수집만 / predict = 예측만
python run_pipeline.py --steps chain,smp  # 단계 키 나열도 가능

# 단계 단독 실행 — 인자 실험은 이 형태로 (파이프라인 통째로 돌리지 말 것)
python forecasting/serve_chain.py --utc 12 --base 2026-07-16 --no-write
python forecasting/serve_chain.py --backfill 7
python collectors/collect_forecast.py --region jeju --days 5 --utc 12
python collectors/collect_historical.py --historical-days 3      # 실측 (--no-save 로 dry-run)
python collectors/collect_archive.py --utc 18 --days 3          # --backfill N / --force 지원
```

검증 (테스트 스위트는 `selftest_pivot.py` 하나뿐이다):

```bash
python collectors/selftest_pivot.py     # long→wide 피벗 불변조건 21항목 (네트워크 0회) — 피벗 손대면 필수
python -m compileall -q .
python forecasting/serve_chain.py --utc 12 --no-write   # 서빙 무손상: 120행 hd 1~5 형상 확인
python collectors/collect_forecast.py --verify          # base 완전성 (fetch 없음)
```

화면은 `AppTest.from_file("pages/page_main.py")` 로 실제 렌더까지 확인한다
(HTTP 200 은 껍데기만 확인하므로 부족하다). DB 를 바꾸는 변경은 반드시 **실행 전후
행 수 diff** 를 남기고, 예보 수집 변경은 `--out NAME` 격리 DB 로 before/after 를 비교한다.

로그는 `logs/pipeline_YYYYMMDD_HHMMSS.log`. 단계별 rc 요약이 파일 끝에 있다.

## 아키텍처

### 데이터 흐름

```
collectors/  ──▶  data/input_data_jeju.db  ──▶  forecasting/  ──▶  est_* 테이블  ──▶  pages/
 (KMA/KPX)         historical (실측 51컬럼)      serve_chain      est_horizon_jeju     common.py (조회·정확도·차트)
                   forecast_horizon (기상예보)   serve_smp        est_smp_horizon_jeju  page_main.py (5메뉴 렌더)
                   patchtst_signal
```

폴더 = 계층 하나씩. `collectors/` 바깥→DB, `forecasting/` DB→모델→DB, `pages/` DB→화면.

- **예측은 전부 `est_horizon_jeju`(base × horizon_d × timestamp tall 구조)에서 읽는다.**
  레거시 `forecast` 테이블(timestamp 단일키)은 지평이 뭉개져서 쓰지 않는다.
- **DB 는 전부 wide 다** (long 은 API→DB 사이 메모리 중간 형식으로만 존재).
  모델이 wide 피처 행렬을 먹으므로 피벗은 적재 때 1번 한다 — long 저장으로 바꾸면
  `forecast_horizon` 이 30k→1.7M 행이 되고 조회마다 피벗해야 한다(NULL 절감은 6%뿐).
- **`src_met_{west,east,south}`** — 그 시각 met 블록이 `KIMR`/`KIMG` 중 어디서 왔는지.
  병합이 `combine_first`(셀 단위)라 wide 만으로는 출처를 알 수 없어서 함께 적재한다
  (`collect_forecast._met_source`, sentinel = temp). 2026-07-21 이전 base 는 NULL.
  `forecast_horizon` 의 일사·운량은 **항상 KIMG** 라 마스크 대상이 아니다 — 단 그 이유는
  아래 "KIMR 엔드포인트는 이제 둘"을 볼 것(KIMR 에 없어서가 아니다).
- **`forecast_kimr`(NC) / `forecast_kimg` = 소스 분리 2테이블, 메인 DB 안에 있다**
  (2026-07-30 격리 아카이브 weather_kim.db 를 메인으로 흡수 — 소스 비교가 끝나 격리
  이유 소멸). `collect_archive` 가 채운다. **아직 서빙은 안 읽는다** — 서빙 입력은
  여전히 `forecast_horizon`(**NC met** + KIMG 일사·운량)이고, 재학습이 이 두 소스
  분리 테이블에서 새 서빙 입력을 만든다.  met 프로토콜은 2026-08-04 에 이미 NC 로 통일됐다.
  - `forecast_kimr`: NC met+일사+운량 (2026-02-19~, 백필 161일). `src_met_proto`
    (`'NC'`/`'GRIB'`) = 그 행 met 프로토콜(NULL = 옛 GRIB base 12개). **프로토콜 바꿔
    재수집 땐 항상 `--force`**(COALESCE upsert 라 NULL 컬럼에 옛 소스가 남는다;
    `--force` = 그 base 행 교체, 삭제는 수집 성공 후에만).
  - `forecast_kimg`: 일사·운량은 전 구간(2025-12~, temp seed) / met 은 12 base 만
    (KIMG met 은 설계상 안 쓴다 — met=KIMR-NC, 일사·운량만 KIMG↔KIMR 경합).
  - **두 테이블은 12z 전용이다** (base 전부 `21:00`, 백필이 `--backfill`=12z). 이는
    의도된 것 — 서빙 모델이 **12z-origin(전일 23시 기준)**으로 학습되고 18z 는 재학습
    없이 serve-time 매핑(n=horizon_d+1)으로 재사용되므로, **재학습 데이터도 12z 전용이
    맞다**. freshest-wins 는 표시/서빙 시점의 12z+18z 병합이고, 검증만 `base_hour` 로
    12z·18z 를 따로 평가한다. (현재 `forecast_horizon` 도 223 base 전부 12z —
    18z 파이프라인이 2026-07-18 도입이라 축적분이 아직 거의 없다.)
- `data/weather_kim.db` 는 **폐지됨**(2026-07-30). 새로 만들지 말 것.
- `data/ai_briefings.db` = Gemini 브리핑 저장소.

### 세 개의 SSOT — 여기만 고치면 전파된다

| SSOT | 파일 | 무엇 |
|---|---|---|
| 경로 | `project_paths.py` | 모든 DB·모델·스크립트 경로. 폴더 구조 변경 시 유일 수정처 |
| 파이프라인 단계 | `run_pipeline.py` `PIPELINE_STEPS` | cron 과 관리자 화면(`page_main.render_admin`)이 **같은 리스트를 import** — 단계 추가/변경은 여기 한 곳 |
| 화면 용어 | `pages/common.py` `hz_label` / `base_badge` / `base_stamp` | 지평·발표 라벨 |

각 스크립트는 상단 3줄로 루트를 `sys.path` 에 넣고 `import project_paths as P` 한다.

### basetime 이원화 (12z / 18z) — 이 프로젝트의 중심 개념

- **12z** = 전일 밤 21시 KST 발표(base 문자열 `... 21:00:00`), `horizon_d` 1~5. 뼈대.
- **18z** = 당일 새벽 03시 KST 발표(`... 03:00:00`), `horizon_d` 0~2. 당일예보 라이트.
- 서빙 모델은 전부 "origin=전일 23시, n=1 이 익일" 구조로 학습됐다. 그래서 18z 는
  **origin 을 전일 23시로 두고 모델지평 n = horizon_d + 1** 로 매핑한다 → 재학습 불필요
  (`serve_chain.HZ` / `HZ_18Z` / `base_mode()`).
- **freshest-wins**: 화면 표시는 시각별 `horizon_d ASC, base DESC` 로 1건
  (`common._hz_select` mode='latest', ROW_NUMBER). DB 원본은 12z/18z 둘 다 보존 —
  검증 페이지가 발표별로 따로 평가한다(`base_hour` 인자로 관통).
- **SMP·백테스트는 12z 전용** — `serve_smp.list_bases` 와 `horizon_backtest` 의 bases
  쿼리에 `substr(base,12)='21:00:00'` 가드가 있다. 새 소비자를 추가하면 같은 가드를 넣는다.
- 수집 창 산식 SSOT = `collectors/kma_kimg.window_bounds()`, 18z 모드는 opt-in 플래그
  `kma_kimg.SAMEDAY_18Z`(기본 False). 진입점 2곳(`collect_forecast`,
  `collect_archive`)만 이 플래그를 켠다.

### collectors/ 의 bare import 규칙 (실수 1순위)

`collectors/` 내부 모듈끼리는 **bare import**(`import kma_kimg`, `import collect_forecast as cf`)
를 쓴다(원본 forecastmodel 무수정 이식). 따라서:

- 실행은 반드시 **파일 경로**로: `python collectors/collect_forecast.py` (→ sys.path[0] = collectors/)
- `python -m collectors.collect_forecast` 는 **깨진다**
- 바깥(forecasting/·pages/common.py)에서 쓸 때만 패키지 import: `from collectors import postprocess`.
  `common.ensure_recent` 은 `sys.path` 에 `collectors/` 를 넣고 `importlib` 로 우회한다.

**long 5컬럼이 수집 계층의 정식 중간 계약이다** —
`base_datetime / point_name / fcst_datetime / category / fcst_value`.
모델(KIMR·KIMG)별로 다른 것은 **엔드포인트·응답 파싱·카테고리 파생, 이 세 어댑터뿐**이고
그 뒤 단계(피벗·강수 누적diff·윈도우 트림·발표 선택)는 전부 공용 1벌이다:

- 피벗 = `pivot._pivot_point` + `pivot._derive_point`.
  모델 차이는 `_SPEC_KIMR` / `_SPEC_KIMG` 표 두 개에만 있다 —
  **스펙의 순서가 곧 출력 컬럼 순서**이고 `_WIND` 센티널이 wind 블록 자리를 잡는다.
  `reh` 반올림이 KIMR 4자리 / KIMG 2자리로 다르니 스펙을 고칠 때 주의.
  피벗을 건드리면 `python collectors/selftest_pivot.py` 를 반드시 돌린다.
- 발표(base) 선택 = `kma_kimg.latest_published_base` 한 벌
  (`collect_forecast` 도 이 SSOT 를 직접 부른다).
- 수집 창 = `kma_kimg.window_bounds` (18z 는 `kma_kimg.SAMEDAY_18Z` opt-in).

collectors/ 는 **9개**뿐이고 파일 = 역할 하나씩이다. 이름 규칙: `collect_*` = 실행 단위(진입점),
`kma_*` = 예보 fetch 어댑터, 나머지는 실측 fetch·변환·검증.
**파일을 가르는 축은 출처가 아니라 예보냐 실측이냐다** — ASOS 는 KMA 소스지만 관측이라
`kma_*` 가 아니라 `kpx_asos.py` 에 있다.

| 파일 | 역할 |
|---|---|
| `collect_historical.py` | KPX 수급·DA·RT SMP + ASOS → `historical` |
| `collect_forecast.py` | KIMR+KIMG → `forecast_horizon` (fetch·병합·적재·CLI 한 파일) |
| `collect_archive.py` | KIMR/KIMG 소스 분리 수집 → 메인 DB `forecast_kimr`/`forecast_kimg` |
| `kma_kimg.py` | KIMG(NE57) core + **KMA 공용 기반**(키 풀·세션 + 발표/창 산식 SSOT) |
| `kma_kimr_nc.py` | KIMR(R030) std-NC / 등압면 CLDFRA — **KIMR 유일 경로** |
| `kpx_asos.py` | KPX 수급·DA·RT SMP + KMA ASOS 관측 (실측 소스) |
| `pivot.py` | long → wide 피벗 1벌 (`_pivot_point`+`_derive_point`+스펙표 2개) |
| `postprocess.py` | clip_ranges / fill_short_gaps / drop_sentinels / sanity_check / add_day_type (forecasting/ 도 import) |
| `selftest_pivot.py` | `pivot.py` 불변조건 검사 (네트워크 0회, 21항목) |

(2026-07-21: 구 `collect_data_jeju` + `_new` + `collect_forecast_new` + `collect_forecast_runs`
4겹 래퍼 체인을 `collect_historical` / `collect_forecast` 둘로 정리.)
(2026-07-22 개명 — 옛 이름으로 검색하면 안 나온다:
`_common`→`kma_kimg` / `api_fetchers_kim2`→`kma_kimr_nc` / `collect_weather_kim`→`collect_archive` /
`api_fetchers_jeju`는 **셋으로 분리**→`kma_kimr_grib`(2026-08-04 삭제) + `kpx_asos`(KPX·ASOS) + `pivot`(피벗).
`collect_forecast` 가 옛 모듈을 `kim`/`ci`/`kpx` 세 별칭으로 부르던 게 그대로 세 모듈이 됐다.)

### 화면 — pages/

`app.py`(게이트 + 내비) → `pages/page_main.py`(5메뉴: 종합/예측 확인/예측 검증/데이터 현황/관리자)
→ `pages/common.py`(조회·정확도·차트 헬퍼) → `pages/weather_map_jeju.py`(3구역 지도) ·
`chart_warn.py`(위험구간 밴드) · `brief_jeju.py`·`brief_store.py`(Gemini 브리핑).

- **`pages/` 안에서 서로 부를 때는 반드시 패키지 import**: `from pages import common as C`.
  Streamlit 이 페이지를 *스크립트로* 실행해 `sys.path[0]` 이 저장소 루트라, bare
  `import common` 은 깨진다 (collectors/ 와 정반대 규칙이니 주의).
- `app.py` 가 `st.navigation` 을 쓰므로 Streamlit 의 `pages/` 폴더 **자동탐색은 꺼진다**
  (`streamlit/commands/navigation.py` 가 `uses_pages_directory = False` 로 세팅).
  그래서 헬퍼 모듈을 같은 폴더에 둬도 사이드바에 페이지로 새지 않는다.
- **`app.py`·`pages/` 에 torch·lightgbm 등 무거운 import 금지** — 추론·수집은 전부
  `common.run_script()` 로 subprocess 실행한다(`sys.executable` 사용 → venv 동일).
- 라이트/다크 겸용: `common.inject_style()` 이 매 rerun 마다 `st.context.theme` 를 보고
  CSS 토큰·plotly 템플릿·`common.COLOR` 팔레트·지도 타일을 일괄 교체한다. 차트 색을
  손대려면 `_CHART_PALETTES` 를 고치고 dataviz 검증(CVD·명도)을 다시 통과시켜야 한다.
- 관리자 메뉴는 `OPS_PASSWORD`(.env) 게이트 — `common.ops_gate()` 를 함수 첫 줄에서 호출.

### forecasting/ 의 `smp_features`·`smp_binary`·`smp_da` 는 학습·서빙 공용이다

SMP 서빙이 런타임에 `smp_features.load_forecast`(피처빌더)·`smp_binary.persist`·
`smp_da._predict_da` 를 실제로 import 한다 (train/serve parity 를 위한 **의도된** 공유 —
소비처는 `serve_smp`·`smp_d1`·`smp_d2`). 이 파일들은 `__main__` 이면 학습 스크립트지만
import 되면 서빙 부품이다 — 학습 전용으로 오인해 옮기거나 지우면 SMP 서빙이 깨진다.
(2026-07-29 개명: `train_smp_db`→`smp_features` / `train_binary_smp`→`smp_binary` /
`train_smp_d2_da`→`smp_da`. 옛 `train_*` 이름이 "학습 전용"으로 오해를 줘서 바꿨다.)

## 규약·함정

- **화면에 `D+N` 표기 금지** (사용자 확정). 지평은 당일/익일/모레/N일후(`common.hz_label`).
  발표는 **묻는 게 다르면 함수도 다르다** (2026-08-26 정정):
  - "언제 만든 예측인가" = **`common.base_stamp`** → `8/25 21시 발표 (어제)`.
    실제 시각 + 선택일 기준 경과. 종합 화면의 지도 패널 부제가 쓴다.
  - "어느 발표 주기인가" = `common.base_badge` → "새벽 발표"(18z)/"전일 밤 발표"(12z).
    예측 검증 화면의 발표 필터(`page_main.BASETIME_OPTS`)와 짝을 이루는 **범주 이름**이다.
  - ⚠ `base_badge` 를 신선도 표기에 쓰면 안 된다 — 지평과 무관하게 늘 "전일 밤 발표"가
    나와서 5일후 예측도, 수집이 3일 밀린 예측도 똑같아 보인다(2026-08-26 실측 확인).
  내부 코드·DB 컬럼은 `horizon_d` 정수 유지.
- **운영 지평 = 5일** (`pages/common.py` `JEJU_HZ_MAX = 5`, `serve_chain.HZ = 1..5`, 수집 `--days 5`).
  아카이브에 남은 과거 D+6~7 행은 UI 에서만 클램프로 제외.
  **2026-07-31 재학습으로 모델도 D+5 에 맞췄다** — demand 1..120h(구 168h), PatchTST solar
  D+1~D+5(구 D+7, `serve_solarwind.SOLAR_PT_HORIZONS=[2,3,4,5]`), `serve_demand` 가드 1~5.
  models/solarwind_patchtst_horizon/ 의 D+6·D+7 .pth 는 **옛 스케일러 기준**이라 쓰면 안 된다.
- **★PatchTST solar 가중치 D+1~D+5 는 한 세트다** — `MinMax_scaler_solar.pkl` 을 공유하므로
  일부만 교체하면 조용히 틀린다. 반입은 항상 5개 + 스케일러 + metadata 를 통째로.
- **태양광 일 스케일링 후처리** (`serve_solarwind._apply_solar_daily_scale`, tcog 다음·야간마스크 앞):
  예보가 흐린날 일사를 과대예측(bias +0.41)해 태양광 과대 → **net_load 과소 = 발전 준비 부족**
  이라는 업무 위험을 만든다. 실측으로 학습하는 한 재학습으로 못 고쳐서 출력을 하루 단위로 눌러 준다.
  - 판정지표 = **`radiation_south` 단독의 낮시간 P60**(2026-08-04 단순화).
    **`models/solarwind_lgbm/solar_scale.json` 이 SSOT** — 코드에 박지 말 것.
  - `min(scale,1)` 이라 **낮추기만 하고 절대 키우지 않는다**. 재적합·점검은
    `python forecasting/fit_solar_scale.py [--check]`, 관리자 화면에도 버튼이 있다.
  - **시점별 QM 은 실패로 확인됨**(2026-07-31) — 같은 예보값이 흐린날에도 맑은날에도 나와
    조건부 편향을 구분할 수 없다. 하루 단위 집계라야 잡힌다. 다시 시도하지 말 것.
  - ### ★자유도를 늘리면 반드시 과적합한다 (2026-08-04 홀드아웃으로 두 번 확인)
    구 파라미터(지점조합 전탐색 + 지평별 floor 5개)가 보고하던 "흐림 bias −0.0002" 는
    **재현되지 않았다**(같은 창 재계산 +0.0268). 그 검증창이 이미 지표·mid 선택에
    쓰여서 생긴 착시다. 완전 홀드아웃에서는 MAE 를 7.7% 악화시키고 있었다.
    - **적합 목표를 편향 0 으로 두지 말 것** — 편향의 크기 자체가 시간에 따라 줄어서
      (적합창 +0.057 / 홀드아웃 +0.037) 0 을 겨냥하면 과보정된다. `BIAS_TARGET_FRAC=0.5`.
    - **`k` 는 탐색하지 말 것**(`K_FIXED=3.5`). 격자에 넣으면 적합창 노이즈를 쫓아
      급경사(k=12)를 고르는데 분할 4개 전부에서 홀드아웃이 나빴다.
    - 지점 조합 4종·지평별 floor 는 홀드아웃에서 **차이가 없다**. 자유 파라미터는
      `mid`·`floor` 둘이면 충분하다.
    - 현행(mid 0.45 / k 3.5 / floor 0.10, 지평 공통) 홀드아웃 실적: 흐림 +0.037→**+0.003**,
      맑음 사실상 불변, MAE 대가 **+3.2%**, 과대율 41.0%→36.2%.
  - ★**낮추기만 하는 구조라 예보가 과소예측하는 달에는 손해다.** 2026-07 이 그랬다
    (무보정 흐림 편향 −0.049 — 2025-12~2026-06 일곱 달 중 유일한 음수). 파라미터로는
    못 고친다. `--check` 로 편향의 **부호**를 주기적으로 볼 것.
  - ⚠`--check` 는 `est_horizon_jeju`(그때그때 cron 이 만든 값)를 읽는다 — **파라미터
    배포일 이전 구간은 옛 파라미터로 만들어진 값**이라 배포 직후 수치를 "지금 성능"으로
    읽으면 안 된다.
- **병목은 모델이 아니라 예보다** — 같은 모델에 입력만 바꿔 재면 실측 MAE 0.0625 / 예보 D+1
  0.1068(+71%) / D+5 0.1640(+162%). **D+3 부터 예보 오차 > 모델 오차**라, 재학습보다 예보 품질
  (NC 전환)이 지렛대가 크다.
- ### ★수집 사다리 — 순서가 곧 안전장치다 (2026-08-12)

  `collect_forecast` 는 결손을 이 순서로 메운다.  **순서를 바꾸면 나빠진다.**

  | 단계 | 무엇 | 어디 |
  |---|---|---|
  | ① | KIMG 정상값 | `build_wide` |
  | ② | 짧은 결손(연속 2개=3h 사고) → **시간 보간** | `postprocess.fill_short_gaps` |
  | ③ | 그래도 빈 일사·운량 → **KIMR(NC) 대체** | `collect_forecast._substitute_solar_cloud` |
  | ④ | sentinel(9999) → NaN, 범위 clip, 건전성 검사 | `postprocess.drop_sentinels/clip_ranges/sanity_check` |

  - **②가 ③보다 먼저인 이유**: 2026-07-02~11 사고를 정답(재수집본)과 대조한 실측,
    결손 780칸 — 시간 보간 MAE 0.063~0.106(r 0.81~0.90) vs KIMR 대체 0.346~0.490
    (r 0.25~0.43, 큰오차 43~59%).  두 모델이 **다른 구름을 본다**(전운량 r 0.47).
  - **③을 그래도 두는 이유**: KIMG 가 통째로 안 오면 보간할 이웃이 없다.
    "예측 없음"보다 "오차 있는 예측 + 출처 통보"가 낫다(사용자 결정 2026-08-12).
    실패 시뮬레이션 실측: 1,062셀 전부 복구, 일사 r 0.94 / 운량 r 0.40.
    출처는 **`src_solar_cloud`** 컬럼('KIMG'/'KIMR')에 남는다.
  - ⚠`fill_short_gaps` 는 반드시 `clip_ranges` **앞**이다 — clip 이 radiation/rainfall
    의 NaN 을 0 으로 채워 버리면 메울 결손이 사라진다.
  - ⚠`_substitute_solar_cloud` 는 `forecast_days_override` **밖**에서 불리므로
    `days` 를 인자로 받는다.  `ckg.FORECAST_DAYS` 를 직접 읽으면 기본값 2 로 돌아가
    창이 48시각으로 잘린다(2026-08-12 실패 테스트에서 잡은 버그).

- ### ★9999 는 이상치가 아니라 **결측**이다
  `cape`/`cinn` 의 9999 = "대류불안정 없음"(Training EDA 3cmp-2).  숫자로 두면 통계가
  통째로 망가진다 — 구 GRIB base 실측 **cape 평균 2073.0 → NaN 처리 후 256.6**(8배 왜곡).
  `postprocess.drop_sentinels` 가 `clip_ranges` 앞에서 NaN 으로 정정한다.
  NC 는 9999 를 안 내므로 신규 수집분엔 무영향이고, API 가 다시 내면 여기서 막힌다.

- **`postprocess.sanity_check`** — 범위 **안**의 이상을 잡는다(clip 은 범위 밖만).
  결측률·얼어붙음·1h 급변·물리 모순(이슬점>기온, 돌풍<풍속, 층별운량>전운량).
  **값을 고치지 않고 경고만** 낸다 — 무엇이 이상인지는 사람이 판단해야 하고 잘못
  고치면 더 나쁘다.  `run_region` 이 적재 직전 돌려 `main()` 이 rc=1 로 올린다.
  `--verify` 도 불완전 base 가 있으면 rc=1.

- **수집 순서 ② → ②-2 는 안전장치다.** ② 예보 수집이 `INSERT OR REPLACE` 라 나중에 돌면
  ②-2 가 채운 컬럼이 NULL 로 덮인다. 예보를 수동 재실행했으면 ②-2 도 재실행할 것.
- ### ★★ KIMR GRIB 은 **삭제됐다** (2026-08-04) — 옛 이름으로 찾지 말 것 ★★
  `kma_kimr_grib.py` 는 파일째 사라졌고 `kma_kimr_nc.py` 가 **KIMR 유일 경로**다.
  → **`forecast_horizon` 의 옛 `cape`/`cinn` 은 재학습에 쓰면 안 된다** (절반 이상이
  9999 sentinel + 2바이트 랩어라운드). `forecast_kimr.src_met_proto` 로 걸러낼 것.
- **KIMG(NE57) 를 못 버리는 이유**: 서빙 운량·일사(`total/midlow_cloud_*`, `radiation_*`)가
  NE57 분포로 학습돼 있다. 소스 전환은 재학습과 함께만.
- 📄 **수집 API 상세는 `collectors/CLAUDE.md` 로 옮겼다** (2026-08-21) — KIMR 엔드포인트
  둘(data=U/P)·API 보존기간 표·KIMG 22변수·NWP 일변화 진폭 실측. `collectors/` 를 만질 때
  자동으로 읽힌다.
- **동부(성산)는 일사계가 없다** — `historical` 에 `solar_rad_east` 컬럼 자체가 없다. 설계된 결손.
- **SMP D+2 는 lag168**(7일 전 하루전 SMP)이 필요 — `historical` 갭이 있으면 그 기간+7일 D+2 공백.
- 저장소는 **2026-08-12 git 연결됐다** (리모트 `JeongYakyong/jeju_model.git`).
  **배포는 서버에서 `git pull`** 을 전제한다 (사용자 확정 2026-07-31) — 그래서 무엇을 추적할지가 갈린다:
  - **추적**: `models/`(~96MB) — 가중치·메타·보정표는 **한 세트**라 통째로 넣는다.
    따로 옮기다 스케일러 하나가 어긋나면 조용히 틀린다. `data/refdata/`(외부 입력).
  - **제외**: `*.db`(서버가 자체 수집) · `logs/` · `Training/` 대용량 산출물(~233MB,
    재학습으로 재생성) · `.env` · `.auth_token`.
  - ⚠ 모델이 git 에 들어가므로 **재학습 때마다 수십 MB 가 새로 쌓인다**(바이너리라 델타
    압축이 거의 안 먹는다). 재학습이 잦아지면 Git LFS 로 옮길 것.

## 진행 기록

`jejumodel.md` = 세션 로그(최신이 위). 사용자 결정 사항·실증 결과·이월 항목이 여기 쌓인다.
**작업 시작 전에 최신 세션 항목을 읽고, 의미 있는 작업 후에는 같은 형식으로 새 항목을 추가한다**
— "사용자 결정 사항 (재질문 금지)" 절이 있으니 이미 확정된 걸 다시 묻지 않도록 확인할 것.
운영·구조 변경은 `README.md` 의 "운영 노트"에도 반영한다.
