"""collectors — 바깥(KMA/KPX API) → DB 수집 계층.  제주 전용.

파일 = 역할 하나씩.  이름 규칙: `collect_*` = 실행 단위(진입점), 나머지는 그 재료.

    ── 진입점 (python collectors/<파일> 로 실행) ──
    collect_historical.py   KPX 수급·DA·RT SMP + KMA ASOS  → historical
    collect_forecast.py     KIMR+KIMG 예보                 → forecast_horizon
    collect_archive.py      KIMR/KIMG 소스 분리 수집        → 메인 DB (forecast_kimr/kimg)

    ── 예보 fetch (모델별 어댑터) ──
    kma_kimg.py             KIMG(NE57) core + KMA 키 풀·세션 + 발표/창 산식 SSOT
    kma_kimr_nc.py          KIMR(R030) std-NC — **KIMR 유일 경로**

    ── 실측 fetch ──
    kpx_asos.py             KPX 전력시장 + KMA ASOS 관측

    ── 변환·검증 ──
    pivot.py                long → wide 피벗 (KIMR·KIMG 공용 1벌)
    postprocess.py          clip_ranges / fill_short_gaps / add_day_type (forecasting/ 도 쓴다)
    selftest_pivot.py       pivot.py 불변조건 검사 (네트워크 0회, 17항목)

★KIMR GRIB 경로(`kma_kimr_grib.py`)는 **2026-08-04 삭제됐다** — 서빙 입력까지 NC 로
  넘어가면서 유예 조건(NC 장애 폴백 / 90일 이전 결손 보전)이 소멸했다.  옛 이름으로
  검색하면 안 나온다.  실측 근거: met 은 GRIB↔NC r 0.998~0.9997(서빙 A/B 수요 차이
  0.002%), 일사·운량은 KIMG 경로라 완전 동일.  오히려 GRIB 쪽에만 결함이 있었다 —
  cape/cinn 의 9999 sentinel(전체의 57~68%)과 2바이트 랩어라운드(655.36 배수).

★파일을 가르는 축이 "출처"가 아니라 **예보냐 실측이냐**다 — ASOS 는 KMA 소스지만
  관측이라 kma_* 가 아니라 kpx_asos 에 있다.  `kma_kimg` 는 이름과 달리 KIMG 전용이
  아니라 **KMA 공용 기반**(키 풀·세션·창 산식 SSOT)도 함께 담는다: KIMG core 가
  그 위에 얹혀 있을 뿐이라 분리하면 두 파일이 서로를 계속 부른다.

★내부 모듈끼리는 **bare import** (`import kma_kimg`, `import collect_forecast as cf`).
  그래서 실행은 반드시 파일 경로로: `python collectors/collect_forecast.py`
  (`python -m collectors.collect_forecast` 는 깨진다).
  바깥(forecasting/·pages/)에서 쓸 때만 패키지 import: `from collectors import postprocess`.

★long 5컬럼이 정식 중간 계약이다:
  base_datetime / point_name / fcst_datetime / category / fcst_value
  모델별로 다른 건 엔드포인트·응답 파싱·카테고리 파생, 이 세 어댑터뿐이고
  그 뒤(피벗·강수 누적diff·윈도우 트림·발표 선택)는 전부 공용 1벌이다.
"""
