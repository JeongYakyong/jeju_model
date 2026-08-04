# -*- coding: utf-8 -*-
"""저장소 안의 모든 경로를 한곳에 모아 둔 파일 (forecastmodel 의 jeju_model 트림판).

폴더 이름을 바꿀 일이 생기면 **이 파일 하나만** 고치면 된다.

쓰는 법 — 각 스크립트 맨 위에 세 줄만 넣는다(파일 깊이만큼 '..' 을 반복):

    _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    import project_paths as P

`__file__` 기준이라 어떤 방식으로 실행하든(직접 실행·다른 파일이 불러오기·streamlit·cron)
같은 값이 나온다.  루트가 sys.path 에 들어가므로 `from forecasting import serve_demand`
같은 일반 import 도 함께 가능하다 (원본 forecastmodel 의 importlib 동적 로드를 대체).

값은 전부 **문자열 경로**다. pathlib 이 필요한 곳에서는 `Path(P.DB_JEJU)` 처럼 감싸 쓴다.
"""
import os

# ── 저장소 뿌리 ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 폴더 ─────────────────────────────────────────────────────────────────
DIR_COLLECTORS  = os.path.join(ROOT, 'collectors')    # 수집기 (KMA/KPX API → DB)
DIR_FORECASTING = os.path.join(ROOT, 'forecasting')   # 수요·신재생·SMP 서빙
DIR_MODELS      = os.path.join(ROOT, 'models')        # 학습된 모델 바이너리
DIR_DATA        = os.path.join(ROOT, 'data')          # 운영 DB 보관 폴더(DB 는 git 제외)

# ── 데이터베이스 ─────────────────────────────────────────────────────────
DB_JEJU  = os.path.join(DIR_DATA, 'input_data_jeju.db')   # 제주 실측·예보·예측
DB_BRIEF = os.path.join(DIR_DATA, 'ai_briefings.db')      # 화면용 AI 브리핑 저장소

# ── 모델 폴더 (원본 forecastmodel 위치 ↔ 새 위치 매핑은 README 참고) ──────
DIR_MODELS_DEMAND             = os.path.join(DIR_MODELS, 'demand')              # ← 02 model/models
DIR_MODELS_SOLARWIND_LGBM     = os.path.join(DIR_MODELS, 'solarwind_lgbm')      # ← 03 lgbm_models
DIR_MODELS_SOLARWIND_PATCHTST = os.path.join(DIR_MODELS, 'solarwind_patchtst')  # ← 03 solarwind_models
DIR_MODELS_SOLARWIND_PATCHTST_HORIZON = os.path.join(DIR_MODELS, 'solarwind_patchtst_horizon')  # ← 03 solarwind_patchTST_pkl (solar D+2~D+7 direct)
DIR_MODELS_SMP                = os.path.join(DIR_MODELS, 'smp')                 # ← 04 models_weight

# ── 참조표(data/refdata/) ────────────────────────────────────────────────
# 서빙이 실행 중에 읽는 작은 표들. 하나라도 없으면 그 지평이 조용히 버려져
# "산출 없음"(0행)이 되므로 반드시 git 으로 함께 관리한다.
REFDATA = os.path.join(DIR_DATA, 'refdata')
REF_JEJU_PPA_BTM  = os.path.join(REFDATA, 'jeju_ppa_btm_capacity_mw.csv')      # 제주 BTM·PPA 태양광 용량(월별)
REF_SOLARWIND_RAW = os.path.join(REFDATA, 'solarwind_raw_jeju.csv')            # 제주 신재생 학습·서빙 원자료, utf-8-sig
REF_GEOJSON       = os.path.join(REFDATA, 'skorea_provinces_simplified.json')  # 3존 분할 전처리 입력(시도 경계)
REF_JEJU_ZONES    = os.path.join(REFDATA, 'jeju_zones_3.json')                 # 제주 3구역 경계 (tools/make_jeju_zones.py 산출물)

# ── 서브프로세스 진입점 (run_pipeline.py·관리자 메뉴가 실행) ──────────────
COLLECT_HISTORICAL = os.path.join(DIR_COLLECTORS, 'collect_historical.py')      # 실측 수집
COLLECT_FORECAST   = os.path.join(DIR_COLLECTORS, 'collect_forecast.py')   # 기상예보 수집 (--region jeju)
COLLECT_ARCHIVE    = os.path.join(DIR_COLLECTORS, 'collect_archive.py')       # KIMR/KIMG 소스 분리 기상 아카이브 (메인 DB)
SERVE_CHAIN        = os.path.join(DIR_FORECASTING, 'serve_chain.py')           # 수요+신재생 → est_horizon_jeju
SERVE_SMP          = os.path.join(DIR_FORECASTING, 'serve_smp.py')             # SMP → est_smp_horizon_jeju
FIT_SOLAR_SCALE    = os.path.join(DIR_FORECASTING, 'fit_solar_scale.py')       # 태양광 일 스케일링 보정표 적합·점검

# ── 인증·환경 ────────────────────────────────────────────────────────────
ENV_FILE   = os.path.join(ROOT, '.env')          # API 키 (수집기 dotenv 는 상위 폴더 탐색으로 이 파일을 찾는다)
AUTH_TOKEN = os.path.join(ROOT, '.auth_token')   # streamlit 비밀번호 게이트 6시간 토큰
