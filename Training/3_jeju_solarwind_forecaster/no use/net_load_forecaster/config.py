"""Paths and constants for net_load_forecaster.

Override via environment variables (e.g. `MODELS_DIR=/custom/path`).
"""
from __future__ import annotations

import os
from datetime import timezone, timedelta
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
# Package lives at <new_project>/net_load_forecaster/
# Project root is <new_project>/
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

MODELS_DIR = Path(os.getenv('MODELS_DIR', PROJECT_ROOT / 'models'))
DB_PATH = Path(os.getenv('DB_PATH', PROJECT_ROOT / 'database_output' / 'jeju_energy.db'))

# ── Model artifact paths ───────────────────────────────────────────────────
SOLAR_WEIGHTS = MODELS_DIR / 'best_patchtst_solar_model.pth'
WIND_WEIGHTS  = MODELS_DIR / 'best_patchtst_wind_model.pth'
SOLAR_SCALER  = MODELS_DIR / 'MinMax_scaler_solar.pkl'
WIND_SCALER   = MODELS_DIR / 'MinMax_scaler_wind.pkl'
METADATA      = MODELS_DIR / 'metadata.pkl'

# ── Timezone ───────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# ── Jeju geographic reference (used by pvlib in prepare_model_input) ──────
JEJU_LAT = 33.3284
JEJU_LON = 126.8366

# ── Capacity estimation ────────────────────────────────────────────────────
CAPACITY_WINDOW_HOURS = 720  # 30-day rolling max
