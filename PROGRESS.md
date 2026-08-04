# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-08-04

## 완료된 것

- **Phase 1~3 전부 완료.** KIMR 은 이제 std NC 단일 경로다.
  - Phase 1(격리 흡수, 07-30) / Phase 2(재학습, 07-31) / **Phase 3(GRIB 폐기, 08-04)**
  - `kma_kimr_grib.py` 파일 삭제 + `collect_archive --met` 인자 제거. 서빙 입력 met 도 NC.
  - 검증: met r ≥ 0.99998, 일사·운량 완전 일치, **서빙 A/B 수요 차이 0.002%**.
  - 구 `TCOG` → NC 이름은 **`GRAUPEL`**(같은 콜, 호출수 증가 0). `TCOH` 는 실측 전부 0 이라 손실 없음.
- **3h 결손 보간을 수집 단계로** — `postprocess.fill_short_gaps`(limit=2, 외삽 금지)를
  `collect_forecast`·`collect_archive` 가 공용. `clip_ranges` **앞**에 둔다.
- **태양광 일 스케일링 재설계·재배포** — 과적합이 실증돼 자유 파라미터를 5개 → 2개로 줄였다.
  배포값 mid 0.45 / k 3.5 / floor 0.10(지평 공통), 지표는 `radiation_south` 단독 P60.
  홀드아웃 흐림 bias +0.037→**+0.003**, MAE 대가 **+3.2%**(구 7.7%), 과대율 41.0%→36.2%.
- **SMP D+2 원인분석 완료** — 개선안(안정 구간 shrinkage/앵커 블렌딩)은 **기각**.
- 버그 수정: `fit_solar_scale` 재적합이 조용히 0행으로 끝나던 문제(`_SSCALE = {}`).

## 현재 상태

서빙 정상: `serve_chain --utc 12 --no-write` = 120행 hd 1~5, 수요 911MW / net_load 792MW.
selftest_pivot 17항목 통과 / AppTest exception 0 / 본 DB 무변경(32,012행 223 base 유지).

- **병목은 여전히 예보다** — D+3 부터 예보 오차 > 모델 오차. NC 전환은 met 프로토콜 정리였지
  예보 품질 개선이 아니다(GRIB↔NC 는 같은 모델의 포맷 차이).
- **KIMG(메인) / KIMR(서브)** 구도 유지 — 서빙 일사·운량은 계속 KIMG(NE57).

## 다음 할 일

1. ★**일사 입력을 KIMG + 누적(ACSWDNB) 결합으로** — 이번 세션 최대 수확.
   **재학습은 필요 없다.** 모델이 실측 기상으로 학습됐으므로, 예보 일사를 실측에 회귀시킨
   결합값을 넣으면 모델을 안 건드리고 정확도가 오른다.
   홀드아웃 검증(분할 2개): 일사 MAE **−7.0~−18.7%** (편향보정만 한 것 대비도 −5.5~−15.6%).
   계수도 안정적(KIMG 0.39~0.47 / 누적 0.38~0.50)이라 두 소스가 서로 보완한다.
   - 교체는 손해다 — 단독으로는 KIMG 가 낫다(debias MAE 0.582 vs 0.608).
   - 할 일: `collect_forecast` 요청 변수에 `ACSWDNB` 추가(이미 NC 경로라 **호출수 증가 0**),
     결합 컬럼 신설(원본 `radiation_*` 은 보존), 계수 SSOT json, 결손 시 KIMG 폴백.
   - 제약: ACSWDNB 완전 구간이 2026-04-29~ (80일 이전 결손). 계수는 그 구간으로만 적합.
   - CAPE·CINN·HPBL·MSLP 는 신호 없음으로 확인됐다(|r| ≤ 0.055). 다시 보지 말 것.
2. **8월 데이터로 스케일링 재판정** — 7월 흐림 편향이 음수(−0.049)로 뒤집힌 게
   계절 현상인지 표본 노이즈인지(흐림날 8일뿐) 아직 모른다. `--check` 로 부호 확인.
3. **저장소 git 연결** — GitHub 저장소(`JeongYakyong/jeju_model.git`)가 **빈 상태로
   `jeju_model/jeju_model/` 하위에 클론**돼 있다(추적 파일 `.gitattributes` 하나).
   프로젝트 본체는 아직 git 밖이다. 그 `.git`·`.gitattributes` 를 루트로 올리면
   리모트와 `Initial commit` 이력이 유지돼 push 가 깨끗하다.
4. `Training/**/no use/` (9MB) 정리 — **git 연결 후에** 지울 것. 지금 지우면 복구 불가고,
   이 폴더는 이미 한 번 값을 했다(태양광 일 스케일링을 `no use/net_load_forecaster/
   data_pipeline.py:852` 에서 되살렸다).

## 주의사항

- **자유도를 늘리면 반드시 과적합한다** — 이번에 두 번 확인됐다(지표 전탐색, k 격자 탐색).
  후처리 파라미터를 손댈 땐 **완전 홀드아웃**으로 재고, 검증창을 선택에 쓰지 말 것.
- **`forecast_horizon` 의 옛 `cape`/`cinn` 은 재학습에 쓰면 안 된다** — GRIB 시절
  9999 sentinel 이 각각 57%/68%, 여기에 2바이트 랩어라운드(655.36 배수)까지 있다.
  `forecast_kimr.src_met_proto` 로 거를 것.
- **스케일링은 낮추기만 한다** — 예보가 과소예측하는 달엔 손해. 파라미터로 못 고친다.
- **NC `GRAUPEL` = 구 `TCOG` 값 일치는 미검증** — 비교 가능한 겨울 구간이 API 보존기간
  밖이다. 2026-12 에 `tcog>0` 이 다시 나오면 확인할 것.
- PatchTST 가중치는 D+1~D+5 가 **한 세트**(스케일러 공유). `models/` 가 git 에 들어가므로
  재학습마다 수십 MB 누적 — 잦아지면 Git LFS 로.
- `patchtst_signal` 이 2026-05-31 에서 끊겨 demand D+1 비교 지표가 NaN.
- 백업: 스크래치 `backup_grib_20260804/`(삭제한 GRIB 파일들), `solar_scale_BEFORE_20260804.json`.
- **2026-08-04 정리분** (전부 복구 불필요): `input_data_jeju(temp).db` 57MB(본 DB 부분집합임을
  표본 합계로 확인), `__pycache__` 18개·`.pyc`, `data/nctest_jeju.db`(검증 산출물),
  `None`(0바이트), 빈 폴더 4개. 저장소 170MB(models 74 + data 81 + Training 14).
