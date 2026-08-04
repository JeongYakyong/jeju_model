# 3cmp-G — direct PatchTST vs LGBM + 하이브리드 서빙 확정

> 신규 direct 지평별 PatchTST(D+2~D+6, `solarwind_patchTST_pkl/`)를 LGBM·기존 recursive와 비교(실측기상 test 2026)
> → 하이브리드 구성 확정 → 통합 서빙 `serve_solarwind_hybrid.py` 완성.
> 산출 `model/3cmp-G_direct_compare.py`, `tab/3cmp-G_*`, `fig/3cmp-G_direct_compare.png`.

## 1. direct PatchTST vs LGBM (실측기상, test 2026)
**SOLAR util(낮) MAE**: direct가 D+1~D+5 우위/동급(D+2 최저 0.0625), D+6만 LGBM(0.0662 vs 0.0698).
**WIND util MAE**: **LGBM이 전 지평 최고**(0.084~0.086). direct wind는 D+1~2만 동급, D+3+ 급악화(0.094~0.096, bias +0.04).
**net_load nMAE%**(평균 526MW): direct D+1 6.60·D+2 6.82(최고), D+3+ direct는 wind 때문에 들쭉(7.2~7.4), LGBM 평평 7.1.

## 2. 핵심 통찰 — 지평이 아니라 **채널**로 갈린다
forecast(3cmp-C)와 합치면 일관:
- **SOLAR: PatchTST 우위** (실측 D+1~5, forecast D+1도 0.102<0.108, 흐린날 포함).
- **WIND: LGBM 우위** (실측·forecast 전 지평. PatchTST는 forecast 풍속오차 증폭 +0.12).
→ 원안 "지평 분리"보다 **"채널 분리: solar=PatchTST, wind=LGBM"**가 더 정확.

## 3. 하이브리드 구성 (사용자 확정)
- **SOLAR = PatchTST direct** (D+1 기존 + D+2~D+6 신규). **D+7 이상은 LGBM 폴백**(D+7 가중치 미학습 + D+6부터 LGBM이 더 나음).
- **WIND = LGBM 전 지평**.
- 그 이상(D+8~, 전국 6단계용)은 LGBM 통일.
- 단순화 위해 제주는 solar D+1~7=PatchTST(D+7만 LGBM 폴백)·wind D+1~7=LGBM로 운영.

## 4. 통합 서빙 `serve_solarwind_hybrid.py`
- 단일 진입점 `predict_hybrid_to_db(origin, horizons=1..7)`. solar=PatchTST direct(offset이 origin↔target 메움, 재귀 아님)·wind=LGBM(serve_solarwind_lgbm 자산 재사용)·net_load=수요−gen.
- 출력 `est_*_jeju_lh`(하이브리드 공식 다지평. D+1 PatchTST `est_*_jeju`·LGBM `est_*_jeju_lgbm`과 분리).
- **end-to-end 검증(forecast 기상, D+1, 2026-03~05)**: 하이브리드 net_load **nMAE 13.63%** < LGBM단독 14.10% < (3cmp-C의 PatchTST단독 = wind 때문에 더 나빴음). → **채널분리가 두 단독보다 우수** 확인. solar(PatchTST) 낮 MAE 0.106·wind(LGBM) MAE 0.129.
- DB write 확인(`_lh` 5컬럼). CLI predict/backfill.

## 5. 남은 것
- (선택) D+7 solar를 PatchTST로 하려면 노트북 `HORIZONS`에 `'D7':144` 추가 학습(현재 LGBM 폴백, 데이터상 D+7은 LGBM이 더 나아 불필요할 수 있음).
- tcog 후처리(서빙 다음, 가볍게, 랜덤스플릿 평가).
- 6단계 land_net_load: 이 채널분리 골격 이식(전국 D+12, wind=LGBM·solar는 land 재학습 필요 여부 EDA 후 결정).
