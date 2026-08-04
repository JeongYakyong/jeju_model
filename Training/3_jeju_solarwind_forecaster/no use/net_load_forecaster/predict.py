"""High-level prediction entry point.

Wraps `data_pipeline.run_model_prediction()` so callers don't need to
manage assets/DB themselves. The underlying function writes predicted
utilizations into the `forecast_data` table.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import DB_PATH
from .db_manager import JejuEnergyDB
from .loader import load_assets
from .data_pipeline import run_model_prediction

logger = logging.getLogger('jejucr.predict')


def predict(
    target_date: str,
    db: Optional[JejuEnergyDB] = None,
    assets: Optional[tuple] = None,
    return_dataframe: bool = True,
) -> pd.DataFrame | dict:
    """Run Solar + Wind utilization forecast for `target_date`.

    Args:
        target_date:        'YYYY-MM-DD'. Requires `seq_len_max` (336) hours of
                            historical data ending at this date, and 24 hours of
                            forecast weather covering the day.
        db:                 JejuEnergyDB instance. If None, opens at default DB_PATH.
        assets:             (solar_model, wind_model, scalers, metadata, device).
                            If None, calls `load_assets()` (cached after first call).
        return_dataframe:   if True, also reads back the saved predictions and
                            returns a DataFrame of timestamp / est_Solar_Utilization
                            / est_Wind_Utilization. If False, returns the raw
                            `(ok, message, input_info)` tuple as a dict.

    Returns:
        DataFrame indexed by timestamp with prediction columns, OR
        {'ok': bool, 'message': str, 'input_info': dict} when return_dataframe=False.

    Raises:
        RuntimeError: if the model prediction step fails.
    """
    own_db = db is None
    if own_db:
        db = JejuEnergyDB(str(DB_PATH))
    if assets is None:
        assets = load_assets()

    try:
        ok, msg, input_info = run_model_prediction(target_date, db, assets)
        logger.info(f"[predict {target_date}] ok={ok} — {msg}")

        if not ok:
            raise RuntimeError(f"Prediction failed for {target_date}: {msg}")

        if not return_dataframe:
            return {'ok': ok, 'message': msg, 'input_info': input_info}

        start = f"{target_date} 00:00:00"
        end   = f"{target_date} 23:00:00"
        fc = db.get_forecast(start, end)
        cols = [c for c in ['est_Solar_Utilization', 'est_Wind_Utilization',
                            'est_demand', 'Solar_Capacity_Est', 'Wind_Capacity_Est']
                if c in fc.columns]
        return fc[cols]
    finally:
        if own_db:
            db.close()
