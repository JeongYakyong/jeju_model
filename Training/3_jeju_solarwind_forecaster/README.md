# 3단계 — Solar/Wind → net_load 예측 (신버전, DB 전용)

제주 재생에너지(태양광·풍력) 이용률을 PatchTST(3지점 입력, cross-attention)로 예측하고
net_load(= 수요 − 재생발전)까지 산출하는 **신버전 서빙**. 새 DB `input_data_jeju.db`
하나에서 읽고 쓴다. (구버전 `net_load_forecaster/` 패키지·옛 DB 기반은 `no use/` 로 이동.)

## 폴더 구성 (서빙 + 제반파일만)
```
3. solar_wind_forecaster/
├── solarwind_db_pipeline.py   ← 서빙 함수 (완전 독립: stdlib+torch+joblib, 모델 클래스 내장)
├── solarwind_models/          ← 가중치/스케일러/메타 (solar 11 / wind 11 feat)
│   ├── best_patchtst_solar_model.pth / best_patchtst_wind_model.pth
│   ├── MinMax_scaler_solar.pkl / MinMax_scaler_wind.pkl
│   └── metadata.pkl
├── training/                  ← 재학습 자료 (가중치 재생성용)
│   ├── export_solarwind_csv.py        DB → 학습용 CSV
│   ├── train_solarwind_3station_colab.ipynb   Colab 학습 노트북
│   └── _gen_notebook.py               노트북 생성기
├── requirements.txt
└── no use/                    ← 구버전/참고자료 보관 (서빙 미사용)
```
> 보고서 자료(구/신 비교, net_load 비교)는 `6. report only/` 에 있다.

## 설계 요약
- **지점**: solar = west(고산)+south(남), wind = west(고산)+east(성산). 지점별 기상을
  *별도 채널로 concat*(평균 X). south는 풍력과 약상관+예보 과대편차로 wind에서 제외.
- **타깃**: `real_solar/wind_utilization_jeju`(0~1) → ×capacity → MW.
- **train/serve 매핑**: past=historical(동명), future=forecast
  (`radiation_*`→solar_rad, `wind_spd_10m_*`→wind_spd, `wd_*_10m_*`→wd_*).

## 사용법
```bash
# 단일 날짜 D+1 24h 예측 → forecast 테이블 UPSERT
python solarwind_db_pipeline.py predict 2026-05-22
python solarwind_db_pipeline.py predict 2026-05-22 --no-write   # DB 미기록(미리보기)

# 구간 백필
python solarwind_db_pipeline.py backfill 2025-12-13 2026-06-03
```
```python
import solarwind_db_pipeline as sw
sw.predict_solarwind_to_db('2026-05-22')      # → DataFrame + DB UPSERT
sw.backfill_solarwind_to_db('2026-02-01', '2026-05-31')
```

## 출력 컬럼 (forecast 테이블, `_jeju` 접미사)
`est_solar_utilization_jeju`, `est_wind_utilization_jeju` (0~1) /
`est_solar_gen_jeju`, `est_wind_gen_jeju` (MW) /
`est_net_load_jeju` (MW = `jeju_est_demand_new` − solar_gen − wind_gen)

## 재학습 (가중치 갱신 시)
1. `python training/export_solarwind_csv.py` → `solarwind_raw_jeju.csv`
2. Colab에서 `training/train_solarwind_3station_colab.ipynb` 실행
3. 산출 5파일을 `solarwind_models/` 에 덮어쓰기 (아키텍처 동일 → 코드 무수정 로드)

## 다음 단계
4단계 SMP(`4. smp_forecaster`)가 `est_net_load_jeju` 를 입력으로 사용.
