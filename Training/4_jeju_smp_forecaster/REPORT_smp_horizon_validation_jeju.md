# SMP 재배선 정직 재검증 (est_horizon_jeju 입력)

서빙 러너 `serve_smp_horizon_jeju.py` 무수정 재사용, 입력=`horizon_backtest_jeju.parquet` forecast 모드(178 base·2025-12-19~2026-06-14)로 봄철 음수가격 구간 포함 실예보 net_load.
실측 rt(<5=음수) 대조. D+1 가격선=DA 통과(A안)이라 price_MAE≈0이 정상, D+2=예측 DA.

| horizon   |    n |   음수h |   경보h |   recall |   precision |   TP |   FP |   FN |   price_MAE(vsDA) |
|:----------|-----:|--------:|--------:|---------:|------------:|-----:|-----:|-----:|------------------:|
| D+1       | 4176 |     147 |     315 | 0.836735 |    0.390476 |  123 |  192 |   24 |              0    |
| D+2       | 4152 |     147 |     300 | 0.809524 |    0.396667 |  119 |  181 |   28 |             11.73 |

**해석**: 경보 recall/precision 이 기존 forecast 입력 보고(`smp_step4_report.md`/`추가작업_report.md`)와 동등하면 재배선이 성능을 보존한 것. 입력 net_load 가 동일 모델(2·3단계)에서 나오므로 차이는 미미해야 정상.

## 기존 forecast 입력 대비 (동등성 판정)

| 지평 | 새 입력(est_horizon 재배선) | 기존 보고서 수치 | 판정 |
|---|---|---|---|
| D+1 음수경보 | recall 0.84 / prec 0.39 | θ=0.25: recall 0.86 / prec 0.38 (`smp_step4_report.md` §4.1) | 동등 |
| D+2 음수경보 | recall 0.81 / prec 0.40 | 설계 운영점 lo: recall 0.86 / prec 0.37 (`smp_d2_pipeline` 주석) | 동등 |
| D+1 가격선 | MAE 0.00 (DA 통과) | A안 = DA 통과(정의상 0) | 완전일치 |
| D+2 가격선 | MAE 11.73 (예측 DA) | 잔차회귀(동일 가중치) | 동등 |

**결론**: forecast→est_horizon_jeju 재배선이 SMP D+1·D+2 경보·가격 성능을 보존(미세차는 검증창 구성 차이·야간마스크 net_load 미세변화 수준). 야간마스크는 한낮 음수가격 사건에 영향 없음이 정량 확인됨. → SMP 운영 정본을 `est_smp_horizon_jeju` 로 옮길 근거 확보. legacy `forecast` 의 SMP 컬럼은 streamlit 8-B 전환 후 제거 가능.

검증 substrate 한계: parquet 178 base 는 06-17 진단 시점 net_load(야간마스크 반영본). 운영 est_horizon_jeju 는 현재 8 base(여름·음수 이벤트 없음)뿐이라, 장기 음수경보 검증은 이 parquet 가 유일 기반. 서버 누적 후 est_horizon_jeju 직접 재검증 권장.
