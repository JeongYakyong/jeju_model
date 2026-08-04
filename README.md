# jeju_model — 제주 순부하 예측 대시보드

제주 전력 **수요·태양광·풍력·순 부하·SMP** 를 예측하고 Streamlit 으로 보여주는
자기완결형 프로젝트. 예보 발표는 두 갈래 — **12z**(전일 밤 21시 KST 발표, 익일~5일후)가
뼈대, **18z**(당일 새벽 03시 KST 발표, 당일~모레 = horizon_d 0~2)가 당일예보 라이트.

- 예측 엔진·데이터 구조는 **forecastmodel** (02/03/04 제주 체인, SQLite horizon 아카이브)에서,
- 실사용 기능(Gemini 브리핑·위험구간 음영·원클릭 예측·비밀번호 게이트)은 **Model_api_added** 에서 이식했다.

## 폴더 구조

폴더 하나 = 계층 하나. 바깥→DB(`collectors`), DB→모델→DB(`forecasting`), DB→화면(`pages`).

```
jeju_model\
├─ app.py               # Streamlit 진입점 — 비밀번호 게이트(6h 토큰) + 내비게이션
├─ project_paths.py     # 모든 경로의 단일 진실원 (폴더 바꾸면 여기만 수정)
├─ run_pipeline.py      # 수집→예측 단일 진입점 (cron·관리자 원클릭 공용)
│
├─ pages\               # 화면 계층 전부 (내부는 `from pages import ...` 패키지 import)
│   ├─ page_main.py         # 5메뉴: 종합 / 예측 확인 / 예측 검증 / 데이터 현황 / 관리자
│   ├─ common.py            # 공용 레이어 — DB 조회·비교 프레임·정확도·차트/UI 헬퍼
│   ├─ weather_map_jeju.py  # 제주 3구역 기상 지도 (Leaflet)
│   ├─ chart_warn.py        # 위험구간 음영 밴드
│   └─ brief_jeju.py  brief_store.py   # Gemini 브리핑 + 저장소
│
├─ collectors\          # 10개 — 진입점 collect_historical / collect_forecast / collect_archive
│                       #   예보 fetch kma_kimg · kma_kimr_nc(KIMR 유일 경로) / 실측 kpx_asos
│                       #   변환·검증 pivot · postprocess · selftest_pivot (bare import, 파일 경로 실행)
├─ forecasting\         # 수요·신재생·SMP 서빙 (원본 02/03/04 — 일반 import 로 전환)
├─ models\              # 학습된 모델 바이너리 (~80MB)
│   ├─ demand\  solarwind_lgbm\  solarwind_patchtst\  solarwind_patchtst_horizon\  smp\
├─ data\
│   ├─ input_data_jeju.db   # 실측·예보(forecast_horizon+forecast_kimr/kimg)·예측 (git 제외)
│   ├─ ai_briefings.db      # AI 브리핑 저장소 (git 제외, 자동 생성)
│   └─ refdata\             # 참조표 — geojson·설비용량 CSV (커밋 대상)
├─ tools\make_jeju_zones.py  # 제주 3구역 보로노이 분할 1회 전처리
└─ logs\                # run_pipeline 실행 로그 (git 제외)
```

> `pages/` 는 Streamlit 의 멀티페이지 자동탐색 폴더와 이름이 같지만, `app.py` 가
> `st.navigation` 을 쓰므로 자동탐색은 꺼진다 — 헬퍼 모듈이 사이드바에 새지 않는다.

## 원본 ↔ 신규 모듈 매핑

| 원본 (forecastmodel) | 신규 |
|---|---|
| `01_data_fetcher_and_db/core/*` | `collectors/*` (무수정 — 상대경로가 루트 `data/`·`.env` 를 그대로 찾음) |
| `02_jeju_demand_forecaster/serve_jeju_demand_lh.py` | `forecasting/serve_demand.py` |
| `03_.../serve_chain_jeju_new.py` | `forecasting/serve_chain.py` (importlib → 일반 import) |
| `03_.../serve_solarwind_hybrid.py` | `forecasting/serve_solarwind.py` (신재생 오케스트레이터 — solar=PatchTST/wind=LGBM 결합) |
| `03_.../serve_solarwind_lgbm.py` / `solarwind_db_pipeline.py` | `forecasting/serve_solarwind_lgbm.py` / `patchtst.py` (PatchTST 모델정의+로더) |
| `03_.../training/build_horizon_backtest_jeju.py` | `forecasting/horizon_backtest.py` |
| `04_jeju_smp_forecaster/serve_smp_horizon_jeju.py` | `forecasting/serve_smp.py` (P1/P2 → pipeline_d1/d2) |
| `04_.../smp_*.py`, `train_*.py` (training/ 포함) | `forecasting/` 평면 배치, 패키지 import 로 전환 |
| `02 model/models` · `03 lgbm_models` · `03 solarwind_models` · `03 solarwind_patchTST_pkl` · `04 models_weight` | `models/demand` · `solarwind_lgbm` · `solarwind_patchtst` · `solarwind_patchtst_horizon` · `smp` |
| `08_streamlit/common.py` | `pages/common.py` (land/가스 계열 제거 트림판) |
| `08_streamlit/page_jeju.py` | `pages/page_main.py` (5메뉴 확장) |
| `08_streamlit/weather_map.py` | `pages/weather_map_jeju.py` (8권역 → 제주 3구역) |
| `08_streamlit/brief_store.py` | `pages/brief_store.py` (region='jeju') |

| 원본 (Model_api_added) | 신규 |
|---|---|
| `app.py` 비밀번호 게이트 | `app.py` (토큰 = 루트 `.auth_token`) |
| `utils/chart_helpers.py` 경고 밴드 | `pages/chart_warn.py` (est_net_demand → est_net_load_jeju) |
| `utils/gemini.py` | `pages/brief_jeju.py` (저장 JSON → SQLite, SMP 경보 리스크 추가) |

## 설치 · 실행

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt   # (Windows: venv\Scripts\pip)
```

1. `.env` — API 키 (`KMA_API_KEY`, `KPX_API_KEY`, `GEMINI_API_KEY`, 선택 `OPS_PASSWORD`)
2. `.streamlit/secrets.toml` — `password = "..."` (화면 접속 비밀번호)
3. 실행:

```bash
streamlit run app.py
```

## 예측 파이프라인

```bash
python run_pipeline.py                    # 12z 풀: ①실측 → ②예보 → ②-2 KIMR/KIMG 아카이브 → ③예측 체인 → ④SMP
python run_pipeline.py --steps light18    # 18z 당일예보 라이트: ①실측 → ②′예보(18z) → ②′-2 아카이브(18z) → ③′체인
python run_pipeline.py --steps collect    # 수집만 (①②②-2)
python run_pipeline.py --steps predict    # 예측만
```

- 관리자 메뉴의 "▶ 오늘 예측" 원클릭과 crontab 이 **같은 단계 정의**(`PIPELINE_STEPS`)를 쓴다.
- 한 단계가 실패해도 나머지는 계속 실행하고, 실패가 있으면 종료코드 1 (cron 알림용).
- 실행 로그: `logs/pipeline_YYYYMMDD_HHMMSS.log`

### 리눅스 서버 crontab 등록 (2줄)

- **12z 풀**: KMA 12z 예보는 23시(KST) 이후 안정적으로 조회되므로 **매일 00:20 KST** 전체 파이프라인.
- **18z 라이트**: 18z 는 당일 03시 KST 발표 — 공개 지연(`PUBLISH_DELAY_HOURS=3`, 이론상 06시 가용)을
  넉넉히 지나 **매일 08:00 KST** 에 당일예보를 돈다. ⚠ 실제 발표 지연은 지속 관찰 대상
  (2026-07-18 세팅) — 로그에서 18z 미가용 경고가 반복되면 08:30 으로 늦춘다.

```cron
# crontab -e  (서버 시간대가 KST 인지 확인: timedatectl)
# 매일 00:20 — 12z 풀 (수집→예측 전체, 익일~5일후)
20 0 * * * cd /opt/jeju_model && ./venv/bin/python run_pipeline.py >> logs/cron.log 2>&1

# 매일 08:00 — 18z 당일예보 라이트 (실측+18z 수집+체인, SMP 없음)
0 8 * * * cd /opt/jeju_model && ./venv/bin/python run_pipeline.py --steps light18 >> logs/cron.log 2>&1

# (선택) 실측만 3시간마다 추가 보충 — 대시보드 실측 최신성 향상
# 10 */3 * * * cd /opt/jeju_model && ./venv/bin/python run_pipeline.py --steps historical >> logs/cron.log 2>&1
```

- `/opt/jeju_model` 은 서버의 실제 설치 경로로 바꾼다.
- DB(51MB+)·모델(~80MB)은 git 에 없으므로 최초 배포 시 `data/input_data_jeju.db` 와 `models/` 를 함께 복사한다.

## 화면 (5메뉴)

- **종합** — 제주 3구역 기상 지도(hero) + 좌우 패널(이용률·순 부하) + AI 브리핑 + 핵심 지표
- **예측 확인** — 선택일부터 표시 기간(k일) 수요·신재생·순 부하 차트 + **위험구간 음영**(임계값 편집) + SMP + 시간별 표
- **예측 검증** — 일별 정확도 추이 / 지평별 성능 / 지정일 시계열 (수요·순 부하·태양광·풍력),
  **발표 필터**(전일 밤 12z / 새벽 18z)로 basetime 별 분리 검증 + 리드타임(h) 축 토글
- **데이터 현황** — 항목×날짜 수집률 히트맵
- **관리자** — 원클릭 전체 실행 · 개별/고급 실행 · 브리핑 수동 생성 (OPS_PASSWORD 게이트)

## 운영 노트

- **예측 지평 = 5일** (2026-07-17 확정, 기존 7일 축소). 수집도 `--days 5` — KIMR 이 5일까지
  1h 전량 커버한다. **KIMG 는 서빙 입력 유지용**: 서빙 운량·일사(total/midlow_cloud_*,
  radiation_*)는 KIMG(NE57) 분포로 학습된 모델의 필수 입력이라 재훈련 전에는 KIMG 단독
  공급을 유지한다. 창이 5일이라 KIMG 도 1h 해상도 구간만 쓴다.
- **basetime 이원화 (2026-07-18 도입)** — 12z(뼈대) + 18z(당일예보 라이트).
  - 18z 도 12z 와 동일한 **KIMR met + KIMG 일사·운량 병합**으로 수집(사용자 결정: 재학습
    대비 NE57 아카이브 연속성). 수집 창 = 당일 04시(hf=1)부터 3일 (`--utc 18 --days 3`,
    `kma_kimg.SAMEDAY_18Z` opt-in — 기본 False 라 12z 경로 무영향).
  - 체인 매핑: 18z base(당일 03:00)는 **origin=전일 23시, 모델지평 n=horizon_d+1** —
    학습된 익일 태스크 그대로에 더 신선한 18z 기상만 공급(재학습 불필요). est 의
    horizon_d=0 은 당일 03시부터 21행. 당일 00~03시 기상 결측은 직전 12z 행으로 스크래치
    패딩(+잔여는 시간보간 limit 4).
  - **freshest-wins**: 화면(latest)은 시각별 `horizon_d ASC, base DESC` 1건 — 당일 03~23시는
    18z, 3~5일후는 12z 가 자연 담당. 12z 가 실패한 날은 18z → 전날 12z 순으로 자동 대체.
    DB 원본은 (base,timestamp) 로 두 발표 모두 보존 — 검증 페이지가 발표별로 따로 평가.
  - **SMP·백테스트는 12z 전용** (base 21:00 필터 가드) — 18z cron 은 수집+체인만.
  - chain18 은 전일 12z base 부재 시 경고 + rc≠0 (12z 수집 실패 알림).
  - 18z 실증(2026-07-18): KIMR 일사·CLDFRA 운량 std NC 응답 정상(lead 72h, hf=0 앵커 유효),
    KIMG(NE57) 18z lead ≥84h — 당일~모레 창(hf≤68) 커버 여유.
- **화면 용어 규칙 (2026-07-17 확정)** — 화면에 D+N 표기 금지. 지평 = 당일/익일/모레/N일후
  (`pages/common.py` `hz_label`), 발표 = "새벽 발표"(18z)/"전일 밤 발표"(12z) 배지(`base_badge`).
  내부 코드·DB 는 horizon_d 정수 유지.
- **KIMR/KIMG 소스 분리 수집 (메인 DB, 2026-07-30 격리 아카이브 흡수)** —
  `collect_archive.py` 가 매 base 마다 두 소스를 **소스 분리 테이블**로 메인 DB 에 수집한다:
  `forecast_kimr`(std NC 한 스택으로 met 16종+TSKIN +일사 SWDDIR2/SWDDIF2/ACSWDNB, 여기에
  등압면 CLDFRA 운량, 118시각/base) / `forecast_kimg`(NE57 fetch, 120시각/base).
  두 테이블 모두 표준 컬럼명 — 출처는 테이블 구분. `forecast_kimr.src_met_proto`
  (`'NC'`/`'GRIB'`)가 그 행 met 프로토콜(NULL = 옛 GRIB base 12개).  **새로 쌓이는 행은
  항상 `'NC'`** 다(2026-08-04 GRIB 폐기) — 옛 `'GRIB'`/NULL 행에는 cape/cinn 결함이
  있으니 재학습 때 이 태그로 걸러낼 것.
  소요 ~4분/base. **3h 결손 안전장치**: 1h 그리드 reindex 후 내부 시간보간(연속 2h까지,
  외삽 금지) + 보간 셀 수 로그. 결측 안전장치 = std NC 순차 재시도 2라운드 +
  커버리지<95% rc=1 + 재실행 치유(COALESCE).
  **아직 서빙은 안 읽는다** — 서빙 입력은 `forecast_horizon`(**NC met** + KIMG 일사·운량)
  이고, 재학습이 이 두 소스 분리 테이블에서 새 입력을 만든다.
  소스 비교(2026-07-30, 161일): 일사 접전(bias 보정 시 KIMR 유리)/운량 midlow KIMR 우세.
- **KIMR GRIB 경로는 2026-08-04 삭제됐다** — `kma_kimr_grib.py` 파일과
  `collect_archive --met` 인자가 사라지고 `kma_kimr_nc.py` 가 KIMR 유일 경로가 됐다.
  서빙 입력 met 도 같은 날 NC 로 넘어왔다(서빙 A/B: 수요 상대차 0.002%, net_load 0.017%).
  met 은 GRIB↔NC 가 r ≥ 0.99998 로 사실상 같은 값이었고, 오히려 GRIB 쪽에만 결함이
  있었다 — `cape`/`cinn` 의 9999 sentinel(전체의 57%/68%)과 2바이트 랩어라운드(655.36 배수).
- **예보 수집에도 3h 결손 보간이 붙었다**(2026-08-04) — `collect_forecast` 가
  `postprocess.fill_short_gaps`(연속 2개까지, 외삽 금지)를 `clip_ranges` **앞**에서 돌린다.
  순서가 중요하다: `clip_ranges` 가 radiation/rainfall 의 NaN 을 0 으로 채우므로 뒤에 두면
  메울 결손이 이미 0 으로 위조된다.  기대 그리드는 `expected_timestamps`(1h/3h 전환을
  이미 반영한 SSOT)를 그대로 쓴다.
- **동부(성산)는 일사계가 없다** — 예보(radiation_east)는 표시되지만 실측 모드에선 '관측 없음'.
- **SMP D+2 는 7일 전 하루전 SMP(lag168)가 필요** — historical 에 갭이 있으면 그 기간+7일까지
  D+2 가 비고, 갭이 지나가면 자동 복구된다.
- **3구역 = 읍면동(행정동) 43개 명단 배정 → 구역별 병합(dissolve)** — 구역 안 읍면동
  경계선은 없고 구역 사이 경계선만 지도에 그려진다. 명단(`tools/make_jeju_zones.py` 의
  ZONE_ASSIGNMENT)을 바꾸면 스크립트만 재실행. 경계 원본 = `data/refdata/jeju_emd_2013.json`
  (southkorea-maps kostat 2013 에서 제주만 추출).

## 테마 (라이트/다크 겸용)

- 기본은 커스텀 라이트 테마(config.toml). 화면 우상단 설정 메뉴에서 **Dark** 를 고르면
  `st.context.theme` 감지로 CSS 토큰·plotly 템플릿·차트 팔레트·지도 타일(CARTO dark_all)이
  함께 전환된다.
- 차트 시리즈 색은 dataviz 6-checks 검증기 통과값 — 동시 표시 5색(수요·신재생·순부하·태양광·풍력)이
  양 테마 표면에서 CVD 분리·명도 밴드 PASS. 팔레트 정의: `pages/common.py` `_CHART_PALETTES`.
- 위험구간 밴드 색은 임계값 popover 이모지와 일치(🔴 최저 · 🟡 저/심야 · 🔵 고 · 🟣 최대),
  반투명 wash 라 양 테마 공용. 임계값은 예측 확인 화면 '경고' popover 에서 직접 설정.
