"""Model + scaler + metadata loader.

Replaces `app.py:load_assets()` which was wrapped in `@st.cache_resource`.
Uses module-level memoization instead — first call loads, subsequent calls
return the cached tuple.
"""
from __future__ import annotations

import logging
import joblib
import torch

from .architecture import PatchTST_Weather_Model
from .config import (
    MODELS_DIR,
    SOLAR_WEIGHTS,
    WIND_WEIGHTS,
    SOLAR_SCALER,
    WIND_SCALER,
    METADATA,
)

logger = logging.getLogger('jejucr.loader')

_assets_cache: tuple | None = None


def load_assets(device: str | None = None, force_reload: bool = False) -> tuple:
    """Load solar/wind models, scalers, metadata.

    Returns:
        (solar_model, wind_model, scalers, metadata, device)
        - scalers: {'solar': MinMaxScaler, 'wind': MinMaxScaler}
        - device:  'cuda' or 'cpu' (auto-detected if None)
    """
    global _assets_cache
    if _assets_cache is not None and not force_reload:
        return _assets_cache

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"[loader] Loading assets from {MODELS_DIR} (device={device})")

    metadata = joblib.load(METADATA)
    scalers = {
        'solar': joblib.load(SOLAR_SCALER),
        'wind': joblib.load(WIND_SCALER),
    }

    pred_len = metadata['PRED_LEN']

    solar_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_solar']),
        seq_len=metadata['SEQ_LEN_SOLAR'],
        pred_len=pred_len,
        patch_len=24, stride=12,
        d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2,
    ).to(device)
    solar_model.load_state_dict(torch.load(SOLAR_WEIGHTS, map_location=device))
    solar_model.eval()

    wind_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_wind']),
        seq_len=metadata['SEQ_LEN_WIND'],
        pred_len=pred_len,
        patch_len=12, stride=6,
        d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.3,
    ).to(device)
    wind_model.load_state_dict(torch.load(WIND_WEIGHTS, map_location=device))
    wind_model.eval()

    _assets_cache = (solar_model, wind_model, scalers, metadata, device)
    logger.info("[loader] Assets loaded")
    return _assets_cache
