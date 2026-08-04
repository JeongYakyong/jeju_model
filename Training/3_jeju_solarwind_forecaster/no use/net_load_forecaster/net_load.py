"""Net-load post-processing.

`net_load = demand − (solar_utilization × solar_capacity) − (wind_utilization × wind_capacity)`

The Stage 1 model outputs utilization rates (0–1). This module converts them to
MW using the capacity estimates stored in the DB (or computed via 720-hour
rolling max of historical generation when absent), and subtracts from demand
to produce the net load series Stage 2 consumes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .db_manager import JejuEnergyDB

CAPACITY_WINDOW_HOURS = 720  # 30 days — matches data_pipeline.add_capacity_features


def _estimate_capacity_from_history(
    db: JejuEnergyDB,
    target_date: str,
    column: str,
) -> Optional[float]:
    """Compute capacity as the max of `column` over the past CAPACITY_WINDOW_HOURS."""
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (target_dt - timedelta(hours=CAPACITY_WINDOW_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    end = (target_dt - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    df = db.get_historical(start, end)
    if df.empty or column not in df.columns:
        return None

    value = df[column].max(skipna=True)
    return float(value) if pd.notna(value) else None


def compute_net_load(
    demand: pd.Series,
    solar_util: pd.Series,
    wind_util: pd.Series,
    solar_capacity: float,
    wind_capacity: float,
) -> pd.DataFrame:
    """Compute net load from utilization predictions.

    Args:
        demand:         hourly demand (MW). Index aligns with utilizations.
        solar_util:     0–1 solar utilization predictions.
        wind_util:      0–1 wind utilization predictions.
        solar_capacity: estimated solar capacity (MW).
        wind_capacity:  estimated wind capacity (MW).

    Returns:
        DataFrame with columns:
            demand_mw, solar_pred_mw, wind_pred_mw, renew_pred_mw, net_load_mw
    """
    demand = pd.Series(demand).astype(float)
    solar_util = pd.Series(solar_util).reindex(demand.index).astype(float)
    wind_util = pd.Series(wind_util).reindex(demand.index).astype(float)

    solar_mw = solar_util * float(solar_capacity)
    wind_mw = wind_util * float(wind_capacity)
    renew_mw = solar_mw + wind_mw
    net_load = demand - renew_mw

    return pd.DataFrame({
        'demand_mw':     demand,
        'solar_pred_mw': solar_mw,
        'wind_pred_mw':  wind_mw,
        'renew_pred_mw': renew_mw,
        'net_load_mw':   net_load,
    })


def compute_net_load_for_date(
    target_date: str,
    db: JejuEnergyDB,
    solar_capacity: Optional[float] = None,
    wind_capacity: Optional[float] = None,
) -> pd.DataFrame:
    """Read demand + predicted utilizations from the forecast table and compute net load.

    Expects `forecast_data` to contain rows for `target_date 00:00` ~ `23:00` with:
      - `est_demand`
      - `est_Solar_Utilization`, `est_Wind_Utilization`  (populated by `predict.run_model_prediction`)
      - `Solar_Capacity_Est`, `Wind_Capacity_Est`        (optional; falls back to history)

    Args:
        target_date:    'YYYY-MM-DD'.
        db:             JejuEnergyDB instance.
        solar_capacity: override (MW). If None, taken from forecast row or history.
        wind_capacity:  override (MW). If None, taken from forecast row or history.
    """
    start = f"{target_date} 00:00:00"
    end = f"{target_date} 23:00:00"
    df = db.get_forecast(start, end)

    if df.empty:
        raise ValueError(f"No forecast rows for {target_date}")

    required = ['est_demand', 'est_Solar_Utilization', 'est_Wind_Utilization']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast table missing columns: {missing}")

    if solar_capacity is None:
        if 'Solar_Capacity_Est' in df.columns and df['Solar_Capacity_Est'].notna().any():
            solar_capacity = float(df['Solar_Capacity_Est'].dropna().iloc[-1])
        else:
            solar_capacity = _estimate_capacity_from_history(db, target_date, 'Solar_Capacity_Est') \
                             or _estimate_capacity_from_history(db, target_date, 'real_solar_gen')
        if solar_capacity is None:
            raise ValueError("Could not determine solar_capacity — provide explicitly")

    if wind_capacity is None:
        if 'Wind_Capacity_Est' in df.columns and df['Wind_Capacity_Est'].notna().any():
            wind_capacity = float(df['Wind_Capacity_Est'].dropna().iloc[-1])
        else:
            wind_capacity = _estimate_capacity_from_history(db, target_date, 'Wind_Capacity_Est') \
                            or _estimate_capacity_from_history(db, target_date, 'real_wind_gen')
        if wind_capacity is None:
            raise ValueError("Could not determine wind_capacity — provide explicitly")

    return compute_net_load(
        demand=df['est_demand'],
        solar_util=df['est_Solar_Utilization'],
        wind_util=df['est_Wind_Utilization'],
        solar_capacity=solar_capacity,
        wind_capacity=wind_capacity,
    )
