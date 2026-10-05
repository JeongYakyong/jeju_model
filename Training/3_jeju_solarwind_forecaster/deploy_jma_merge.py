"""2026-10-05 JMA 운량 전환 배포용 — 로컬에서 백필한 예측을 서버 DB 에 옮긴다.

서버 성능이 낮아 백필(12z 143 + 18z 40 base, SMP 143 base)을 로컬에서 돌렸다.
DB 파일을 통째로 덮어쓰면 서버가 그사이 수집한 자료를 잃으므로, 바뀐 행만 옮긴다.

    # 로컬: 패키지 만들기 (바뀐 행만 담은 작은 SQLite)
    python Training/3_jeju_solarwind_forecaster/deploy_jma_merge.py export
    # 서버: DB 백업 후 병합 (같은 base 의 행을 지우고 패키지 행으로 채운다)
    python Training/3_jeju_solarwind_forecaster/deploy_jma_merge.py merge data/jma_deploy_20261005.db

옮기는 것
  forecast_jma           전체 (서버엔 테이블이 없다)
  est_horizon_jeju       12z base >= 2026-05-12 21:00 (JMA 가 있는 구간) + 18z base 전체
  est_smp_horizon_jeju   base >= 2026-05-12 21:00
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import project_paths as P   # noqa: E402

PACKAGE = ROOT / "data" / "jma_deploy_20261005.db"
FIRST_JMA_BASE = "2026-05-12 21:00:00"
SELECTIONS = {
    "forecast_jma": "1 = 1",
    "est_horizon_jeju": f"(substr(base, 12) = '21:00:00' AND base >= '{FIRST_JMA_BASE}') OR substr(base, 12) = '03:00:00'",
    "est_smp_horizon_jeju": f"base >= '{FIRST_JMA_BASE}'",
}


def export():
    if PACKAGE.exists():
        PACKAGE.unlink()
    con = sqlite3.connect(P.DB_JEJU)
    con.execute(f"ATTACH '{PACKAGE}' AS package")
    for table, where in SELECTIONS.items():
        ddl = con.execute("SELECT sql FROM main.sqlite_master WHERE name = ?", (table,)).fetchone()[0]
        con.execute(ddl.replace(f"CREATE TABLE {table}", f"CREATE TABLE package.{table}", 1)
                       .replace(f'CREATE TABLE "{table}"', f'CREATE TABLE package."{table}"', 1))
        con.execute(f"INSERT INTO package.{table} SELECT * FROM main.{table} WHERE {where}")
        count = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {'run_time_utc' if table == 'forecast_jma' else 'base'}) "
                            f"FROM package.{table}").fetchone()
        print(f"{table}: {count[0]}행 ({count[1]} 실행/base)")
    con.commit()
    con.close()
    print(f"-> {PACKAGE} ({PACKAGE.stat().st_size / 1e6:.1f} MB)")


def merge(package_path, db_path=P.DB_JEJU):
    con = sqlite3.connect(db_path)
    con.execute(f"ATTACH '{package_path}' AS package")
    try:
        con.execute("BEGIN")
        for table in SELECTIONS:
            exists = con.execute("SELECT 1 FROM main.sqlite_master WHERE name = ?", (table,)).fetchone()
            if not exists:
                ddl = con.execute("SELECT sql FROM package.sqlite_master WHERE name = ?", (table,)).fetchone()[0]
                con.execute(ddl)
            before = con.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            key = "run_time_utc" if table == "forecast_jma" else "base"
            con.execute(f"DELETE FROM main.{table} WHERE {key} IN (SELECT DISTINCT {key} FROM package.{table})")
            con.execute(f"INSERT INTO main.{table} SELECT * FROM package.{table}")
            after = con.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            print(f"{table}: {before} -> {after}행")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    elif sys.argv[1] == "merge":
        merge(sys.argv[2], *sys.argv[3:4])   # 세 번째 인자 = 대상 DB (시험용 사본)
