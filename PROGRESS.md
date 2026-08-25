# PROGRESS

> 스냅샷 (일지 아님). 상세 로그는 `jejumodel.md`, 결정 목록은 `DECISIONS.md`.
> 최종 갱신 2026-08-25

## 다음 세션: **서버 배포** (사용자 확정)

태양광 개선 시도는 **일단락**했다. 현행 구성을 그대로 두고 `solar_scale` 후처리로 버틴다.
배포는 **서버에서 `git pull`** 전제 — `models/` 는 추적되므로 함께 넘어간다.

배포 전 확인은 이미 통과했다 (2026-08-25):

```bash
python -m compileall -q .                            # rc=0
python collectors/selftest_pivot.py                  # 21항목 전부 통과
python collectors/collect_forecast.py --verify       # 245 bases 모두 완전, rc=0
python forecasting/serve_chain.py --utc 12 --no-write # 120행 hd 1~5, rc=0
AppTest.from_file('pages/page_main.py')              # exception 0 / error 0
```

⚠**아직 `git push` 안 했다** — 새 리포트 2개 + 노트북·생성기 변경분이 로컬에만 있다.

## 현재 상태

| | |
|---|---|
| **서빙 모델** | **2026-07-30 재학습본** (08-25 재학습분은 열세로 되돌림) |
| `forecast_horizon` / `est_horizon_jeju` / `est_smp_horizon_jeju` | 각 245 base, ~08-23 |
| `forecast_kimr` / `forecast_kimg` | 187 / 249, ~08-23 |
| `historical` | 2020-01-01 ~ 08-24 (58,272행, 결측 0) |

## 완료된 것

- **Phase 1~3** — KIMR std NC 단일 경로, 서빙 입력 met 도 NC.
- **수집 사다리** ①KIMG → ②시간보간 → ③KIMR 대체 → ④sentinel·clip·건전성 검사.
- ★**운량 피처 정밀 감사 완료** — `Training/3_jeju_solarwind_forecaster/REPORT_cloud_feature_audit.md`.
  주범은 `total_cloud`: 예보 `tcld` 는 권운을 100% 반영하는데 ASOS 전운량은 45%만 센다.
  **정의가 다른 양**이라 데이터로 못 메운다.
- ★**개선 시도 전수 실패 기록** (같은 리포트 §10) — 피처 재조합 / 후처리 추가 /
  실측 재학습(Year+split) / 예보 축 학습 / `lwdown`·`temp`·`reh` 추가. **전부 실패.**
- ★**예보 품질이 천장** — 같은 모델에 입력만 바꿔 MAE 실측 0.0764 vs 예보 0.1125(**+47%**).
- **재학습 코드는 완성돼 있다** (`_gen_notebook_solar_d1d5.py`) — 결과가 나빠 되돌렸을 뿐
  코드·dry-run·서빙 무수정 확인은 끝났다. 재시도 시 그대로 쓸 수 있다.

## 다음 할 일

1. ★**서버 배포** — `git push` → 서버 `git pull` → cron 확인.
2. **아카이브 축적 유지** — `forecast_kimg` 층별·`lwdown` 이 유일한 미래 지렛대.
   **재시도 조건: 18개월 축적(2027-08경).**
3. **8월 스케일링 재판정** — `fit_solar_scale.py --check` 로 흐림 편향 **부호** 확인.
4. **화면 통보** — `src_solar_cloud='KIMR'` 배지 없음. 아직 발동 0건.
5. **2026-12**: NC `GRAUPEL` = 구 `TCOG` 값 일치 확인.

### 미해결 (다음 기회에)
- **흐린날 일사 +49% 과대의 원인 미상** — 운량은 맞고, 장파는 설명 못 한다.
- **KIMG 에 구름 수분량(`qc`/`qi`)·광학깊이(`taucld`) 변수가 있는지 미프로브.**
- **재학습 원인 분리** — Year 와 split 을 함께 바꿔 무엇이 나빴는지 모른다.
  다시 한다면 `train ≤2026-01` / `val 2026-02~05` 유지 + **Year 만**.

### 닫힌 길 (다시 열지 말 것)
- **시각별 편향 보정** / **출력 후처리 추가**(cloud_scale 포함).
- ★**피처 재조합 전반** — `total_cloud` 제거·층별 raw·투과가중·`new_TCLD`·`tcldc`·
  `lwdown`·`temp`·`reh` 추가.
- ★**실측 재학습(Year+val 재분할)** / ★**예보 축 직접 학습**(18개월 전에는 무의미).
- **met 소스 변수별 분리** / **west 좌표 이동** / **KIMR 등압면을 서빙 운량으로**.
- **KIMG 일사에 사다리꼴 적용** — KIMR 전용 처방. KIMG 는 이미 시간평균에 가깝다.

## 주의사항

- ★★**단방향 홀드아웃 하나로 결론 내지 말 것** → **역방향 분할 + LOMO** 를 같이 볼 것.
- ★★**난수 대조군을 함께 낼 것** — 피처 2개 추가의 이득이 난수와 구별 안 되는 경우가 많다.
- ★★**저장된 `est_horizon_jeju` 로 모델 A/B 하지 말 것** — `solar_scale` 파라미터가 섞인다
  (08-04 이전 base 는 재실행과 1.1%만 일치). **구 모델을 git 복원해 같은 조건 재실행.**
- ★★**되돌리기 전에 현재 산출물을 백업할 것** — 08-25 에 신 가중치를 소실했다.
- ★**실측으로 구간을 가르면 회귀 인공물이 낀다** — 무편향 대조군을 함께.
- ★**선형 대리모델로 비선형 모델을 판단하지 말 것** / **한 번에 하나씩만 바꿀 것.**
- ★**입력 지표 개선이 출력 개선으로 자동 전달되지 않는다** / **자유도를 늘리면 과적합한다.**
- **PatchTST solar 가중치 D+1~D+5 는 스케일러 공유 = 한 세트.**
- **`temp_skin` 은 서·남부가 얼어붙어 있다** — 재학습 피처로 쓰지 말 것.
- **SMP D+2 는 lag168** / **KPX 쿼터는 서비스별 일일 제한** / **KMA 는 키 4개 풀**.
