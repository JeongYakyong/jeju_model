# 더 이상 쓰지 않는 SMP 서빙 코드 (2026-06-20 아카이브)

제주 `forecast` 테이블 폐기(G-22, 2026-06-20)로 **실행 즉시 깨지는** 옛 서빙 코드를 여기로 옮겼다.
세 파일 모두 실제 `forecast` 테이블을 읽거나 쓰는데, 그 테이블이 사라졌다("no such table: forecast").

- `smp_serve.py` — 옛 D+1·D+2 통합 서빙 오케스트레이터. **대체 = `serve_smp_horizon_jeju.py`**
  (est_horizon_jeju → est_smp_horizon_jeju, 핵심3컬럼).
- `smp_depth_pipeline.py` — 음수 깊이(p10/50/90) 위험레이어. 새 경로에서 제외(사용자 확정 = 핵심3컬럼만).
- `smp_softest_pipeline.py` — 캘리브레이션 확률·주야 위험레이어. 같은 이유로 제외.

운영 폴더에 남은 라이브 코드(스크래치 DB 경유로 동작, 실 forecast 미접촉):
`serve_smp_horizon_jeju.py`(진입점) · `smp_db_pipeline.py`(D+1) · `smp_d2_pipeline.py`(D+2) · `train_smp_db.py`(로더).

되살리려면: 위 세 파일을 상위 폴더로 옮기고, forecast 대신 est_horizon_jeju/est_smp_horizon_jeju를
읽도록 재배선하면 된다(serve_smp_horizon_jeju 의 스크래치 주입 방식 참고).
