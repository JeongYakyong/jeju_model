# -*- coding: utf-8 -*-
"""validate_smp_horizon_jeju.py — 새 입력(est_horizon_jeju 경로) SMP 정직 재검증.

목적: SMP(4단계)를 forecast→est_horizon_jeju 로 재배선해도 경보·가격 성능이 유지되는지
확인.  서빙 러너 `serve_smp_horizon_jeju.py` 를 **그대로 재사용**하되, 입력 est_horizon_jeju 를
장기 백테스트 substrate(`3. .../horizon_backtest_jeju.parquet` forecast 모드, 178 base·Dec~Jun)
로 갈아끼워, 봄철 음수가격 구간을 포함한 실예보 net_load 로 D+1·D+2 경보를 산출하고 실측 rt 와
대조한다.  (로컬 forecast 예측은 06-01 동결·est_horizon_jeju 는 8 base/여름뿐이라 직접 비교 불가
→ parquet 가 유일한 장기 검증 기반.)

산출: 표준출력 + REPORT_smp_horizon_validation_jeju.md.
주의: parquet net_load 는 06-17 진단 시점 산출이라 야간마스크 미세차가 있을 수 있으나, 음수가격은
한낮(태양광 피크) 사건이라 야간마스크 영향 밖 — 경보 검증에 영향 미미.
"""
from __future__ import annotations
import os, sys, sqlite3, tempfile, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # .../4. jeju_smp_forecaster/training
SMP_DIR = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(SMP_DIR, '..'))
DB = os.path.join(ROOT, '1. data_fetcher_and_db', 'data', 'input_data_jeju.db')
PARQUET = os.path.join(ROOT, '3. jeju_solarwind_forecaster', 'training', 'horizon_backtest_jeju.parquet')
NEG = 5   # rt < 5 = 음수(무가치) 가격 이벤트


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def build_tempdb(tmp: str) -> int:
    """historical 복사 + parquet(forecast) → est_horizon_jeju 형태 적재. base 수 반환."""
    pq = pd.read_parquet(PARQUET)
    f = pq[(pq['mode'] == 'forecast') & (pq['horizon'].isin([1, 2, 3]))].copy()
    f = f.rename(columns={'horizon': 'horizon_d', 'est_net_load': 'est_net_load_jeju',
                          'est_demand': 'est_demand_jeju', 'est_solar_util': 'est_solar_util_jeju'})
    f['timestamp'] = pd.to_datetime(f['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    keep = ['timestamp', 'base', 'horizon_d', 'est_demand_jeju', 'est_solar_util_jeju', 'est_net_load_jeju']
    f = f[keep].dropna(subset=['est_net_load_jeju'])

    if os.path.exists(tmp):
        os.remove(tmp)
    src = sqlite3.connect(DB); dst = sqlite3.connect(tmp)
    src.backup(dst); src.close()                 # historical 등 통째 복사
    dst.execute('DROP TABLE IF EXISTS est_horizon_jeju')
    dst.execute('DROP TABLE IF EXISTS est_smp_horizon_jeju')
    f.to_sql('est_horizon_jeju', dst, index=False, if_exists='replace')
    dst.commit(); dst.close()
    return f['base'].nunique()


def metrics(con) -> pd.DataFrame:
    """est_smp_horizon_jeju × historical(rt·DA) → 지평별 경보·가격 지표."""
    df = pd.read_sql('SELECT * FROM est_smp_horizon_jeju', con, parse_dates=['timestamp'])
    h = pd.read_sql('SELECT timestamp, smp_jeju_rt, smp_jeju_da FROM historical '
                    'WHERE smp_jeju_rt IS NOT NULL', con, parse_dates=['timestamp'])
    d = df.merge(h, on='timestamp', how='inner')
    d['neg'] = d['smp_jeju_rt'] < NEG
    rows = []
    for n, g in d.groupby('horizon_d'):
        yneg = g['neg'].values
        alarm = g['smp_danger'].fillna(0).astype(int).values.astype(bool)
        tp = int((alarm & yneg).sum()); fp = int((alarm & ~yneg).sum()); fn = int((~alarm & yneg).sum())
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        # 가격선: est_smp vs 실측 DA (D+1=DA통과→0, D+2=예측DA)
        price_mae = float(np.abs(g['est_smp'] - g['smp_jeju_da']).mean())
        rows.append({'horizon': f'D+{n}', 'n': len(g), '음수h': int(yneg.sum()),
                     '경보h': int(alarm.sum()), 'recall': rec, 'precision': prec,
                     'TP': tp, 'FP': fp, 'FN': fn, 'price_MAE(vsDA)': round(price_mae, 2)})
    return pd.DataFrame(rows)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print('[validate] tempdb 구성(historical 복사 + parquet→est_horizon_jeju)...')
    tmp = os.path.join(tempfile.gettempdir(), 'validate_smp_horizon_jeju.db')
    nbase = build_tempdb(tmp)
    print(f'  parquet forecast base {nbase}개 적재')

    runner = _imp('serve_smp_horizon_jeju', os.path.join(SMP_DIR, 'serve_smp_horizon_jeju.py'))
    runner.DB = tmp                                  # 러너 입출력을 tempdb 로
    if SMP_DIR not in sys.path:
        sys.path.insert(0, SMP_DIR)
    import smp_db_pipeline as P1, smp_d2_pipeline as P2, train_smp_db
    P2.load_forecast = lambda with_target=True: train_smp_db.load_forecast(with_target=False)

    bases = runner.list_bases()
    scratch = os.path.join(tempfile.gettempdir(), 'validate_smp_scratch.db')
    print(f'[validate] {len(bases)} base SMP 서빙(러너 재사용)...')
    total = 0
    for bi, base in enumerate(bases, 1):
        r = runner.serve_base(base, P1, P2, scratch)
        if not r.empty:
            total += runner.upsert_est(r, tmp)
        if bi % 30 == 0:
            print(f'  {bi}/{len(bases)} ...')
    print(f'  est_smp_horizon_jeju(temp) {total}행 적재')

    con = sqlite3.connect(tmp)
    m = metrics(con); con.close()
    print('\n═══ 새 입력(est_horizon 경로) SMP 경보·가격 성능 (parquet forecast 178base) ═══')
    print(m.to_string(index=False))

    rep = os.path.join(SMP_DIR, 'REPORT_smp_horizon_validation_jeju.md')
    with open(rep, 'w', encoding='utf-8') as fp:
        fp.write('# SMP 재배선 정직 재검증 (est_horizon_jeju 입력)\n\n')
        fp.write('서빙 러너 `serve_smp_horizon_jeju.py` 무수정 재사용, 입력=`horizon_backtest_jeju.parquet` '
                 'forecast 모드(178 base·2025-12-19~2026-06-14)로 봄철 음수가격 구간 포함 실예보 net_load.\n')
        fp.write('실측 rt(<5=음수) 대조. D+1 가격선=DA 통과(A안)이라 price_MAE≈0이 정상, D+2=예측 DA.\n\n')
        fp.write(m.to_markdown(index=False))
        fp.write('\n\n**해석**: 경보 recall/precision 이 기존 forecast 입력 보고(`smp_step4_report.md`/`추가작업_report.md`)와 '
                 '동등하면 재배선이 성능을 보존한 것. 입력 net_load 가 동일 모델(2·3단계)에서 나오므로 차이는 미미해야 정상.\n')
    print(f'\n[validate] 보고서 → {rep}')


if __name__ == '__main__':
    main()
