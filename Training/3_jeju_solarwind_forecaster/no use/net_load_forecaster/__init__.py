"""net_load_forecaster — Stage 1 of the Jeju forecasting pipeline.

Public API:
    fetch_data(start, end, *, kind='all')   # collect from KPX/KMA into SQLite
    predict(target_date, ...)               # run PatchTST inference for one date
    compute_net_load(...)                   # compose net_load = demand − solar − wind
    compute_net_load_for_date(date, db)     # read forecast row → net_load DataFrame
    load_assets(...)                        # raw (solar, wind, scalers, meta, device)

The output Stage 2 consumes is:
    weather data    (read from `forecast_data` / `historical_data` via JejuEnergyDB)
    net_load_mw     (returned by compute_net_load* functions)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from .config import DB_PATH, KST, MODELS_DIR
from .db_manager import JejuEnergyDB
from .loader import load_assets
from .predict import predict
from .net_load import compute_net_load, compute_net_load_for_date
from .data_pipeline import (
    daily_historical_update,
    daily_historical_kpx,
    daily_historical_kma,
    daily_historical_kpx_smp,
    daily_forecast_and_predict,
    daily_forecast_kpx,
    daily_forecast_kma,
    prepare_model_input,
    run_model_prediction,
)

__all__ = [
    'fetch_data',
    'predict',
    'compute_net_load',
    'compute_net_load_for_date',
    'load_assets',
    'JejuEnergyDB',
    'DB_PATH',
    'MODELS_DIR',
    'KST',
    # Lower-level building blocks (advanced use):
    'daily_historical_update',
    'daily_forecast_and_predict',
    'prepare_model_input',
    'run_model_prediction',
]


def fetch_data(
    start: str,
    end: str,
    kind: Literal['all', 'historical', 'forecast'] = 'all',
) -> None:
    """Collect KPX + KMA data into the SQLite DB.

    Args:
        start, end:  'YYYY-MM-DD'.
        kind:
            'historical' → actuals (KPX power + KPX SMP + KMA ASOS).
                           Max 30-day window. `end` must not be future.
            'forecast'   → next-day weather + KPX demand/SMP forecast.
                           Window typically D-3 ~ D+1.
            'all'        → both, sequentially.
    """
    if kind in ('historical', 'all'):
        daily_historical_update(start, end)
    if kind in ('forecast', 'all'):
        daily_forecast_and_predict(start, end)
