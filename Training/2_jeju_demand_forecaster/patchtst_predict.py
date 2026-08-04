"""PatchTST D+1 1회 추론 모듈 (Step 5).

용도
====
학습된 PatchTST 가중치를 로드하고, history (real_demand 시계열) + D+1 자정
target_date 를 받아 24시간 예측값을 반환한다.

호출 방법
=========
    import pandas as pd
    from patchtst_predict import predict_d1

    history_series = pd.Series([...], index=pd.DatetimeIndex([...]))
    target_date    = pd.Timestamp('2026-05-23')        # D+1 자정 00:00
    preds = predict_d1(history_series, target_date)    # shape (24,) numpy 배열

요구 사항
=========
- models/patchtst_demand.pth (가중치)
- models/patchtst_demand_meta.pkl (HP)
- history_series : timestamp 인덱스. D 23:00 까지 seq_len(=672) 시간 데이터 있어야 함

운영 환경
=========
CPU 서버에서 매일 1회 실행 (1 forward pass, 444K params → <1초).
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# 0. 기본 경로 (이 .py 파일 옆 models/ 폴더를 본다)
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PTH  = os.path.join(HERE, 'models', 'patchtst_demand.pth')
DEFAULT_META = os.path.join(HERE, 'models', 'patchtst_demand_meta.pkl')


# =============================================================================
# 1. 모델 클래스 (training/patchtst_train.py 와 동일 — 의존성 분리를 위해 inline)
# =============================================================================
class InstanceNormalization(nn.Module):
    """윈도우 단위 정규화. norm 시 mean/std 저장 → denorm 에 재사용."""
    def __init__(self, num_features=1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.std  = None

    def forward(self, x, mode='norm', mean=None, std=None):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False)
                                   + self.eps).detach()
            x_norm = (x - self.mean) / self.std
            return x_norm * self.affine + self.bias
        elif mode == 'denorm':
            mean_2d = mean.squeeze(-1)
            std_2d  = std.squeeze(-1)
            return (x - self.bias) / self.affine * std_2d + mean_2d
        return x


class PatchTST_Univariate(nn.Module):
    """univariate PatchTST. forward: (B, seq_len, 1) → (B, pred_len)."""
    def __init__(self, seq_len, pred_len, patch_len, stride,
                 d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.patch_len  = patch_len
        self.stride     = stride
        self.d_model    = d_model
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.instance_norm   = InstanceNormalization(num_features=1)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding   = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, pred_len),
        )

    def forward(self, x):
        # x : (B, seq_len, 1)
        x_norm = self.instance_norm(x, mode='norm')
        x_seq = x_norm.squeeze(-1)
        patches = x_seq.unfold(dimension=1, size=self.patch_len, step=self.stride)

        enc_in  = self.patch_embedding(patches) + self.pos_embedding
        enc_in  = self.dropout(enc_in)
        enc_out = self.transformer_encoder(enc_in)
        pred_norm = self.head(enc_out)

        pred = self.instance_norm(pred_norm, mode='denorm',
                                  mean=self.instance_norm.mean,
                                  std=self.instance_norm.std)
        return pred


# =============================================================================
# 2. 모델 로딩 (모듈 단위 캐시 — 한 번 로드 후 재사용)
# =============================================================================
_model_cache: dict = {'model': None, 'meta': None}


def load_model(pth_path: str | None = None,
               meta_path: str | None = None,
               device: str | None = None,
               force_reload: bool = False):
    """가중치 + 메타 로드. 같은 프로세스 내에서는 캐시 재사용.

    반환
    ----
    (model, meta) 튜플. model.eval() 상태.
    """
    if (not force_reload) and _model_cache['model'] is not None:
        return _model_cache['model'], _model_cache['meta']

    pth_path  = pth_path  or DEFAULT_PTH
    meta_path = meta_path or DEFAULT_META
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f'PatchTST 가중치 없음: {pth_path}')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'PatchTST 메타 없음: {meta_path}')

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    HP = meta['hp']

    model = PatchTST_Univariate(**HP).to(device)
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _model_cache['model'] = model
    _model_cache['meta']  = meta
    return model, meta


# =============================================================================
# 3. D+1 1회 추론
# =============================================================================
def predict_d1(history_series: pd.Series,
               target_date,
               pth_path: str | None = None,
               meta_path: str | None = None,
               device: str | None = None) -> np.ndarray:
    """PatchTST 로 D+1 24시간 수요를 예측한다.

    파라미터
    --------
    history_series : pd.Series
        timestamp(DatetimeIndex) → real_demand 값. 시간 간격 1h 가정.
        D 23:00 까지 seq_len(=672) 시간 데이터를 포함해야 한다 (4주분).
    target_date : str | pd.Timestamp
        D+1 의 자정 (예: '2026-05-23' 또는 pd.Timestamp('2026-05-23')).
        실제 예측 구간은 [target_date 00:00, target_date 23:00].
    pth_path, meta_path : str | None
        지정 안 하면 models/patchtst_demand.{pth,pkl} 사용.
    device : str | None
        'cpu' / 'cuda'. None 이면 자동 선택.

    반환
    ----
    numpy.ndarray (24,) — D+1 00:00 ~ 23:00 의 예측값 (float32)

    오류
    ----
    - 가중치/메타 파일 없음    → FileNotFoundError
    - 윈도우 길이 불충분/NaN  → ValueError
    """
    # 1) 모델 로드 (캐시 사용)
    model, meta = load_model(pth_path, meta_path, device)
    HP = meta['hp']
    seq_len  = HP['seq_len']      # 672
    pred_len = HP['pred_len']     # 24

    # 2) 입력 윈도우 슬라이스 — [D-27일 00:00, D 23:00] 의 seq_len 시간
    target_date = pd.Timestamp(target_date).normalize()
    d_end       = target_date - pd.Timedelta(hours=1)              # D 23:00
    win_start   = d_end - pd.Timedelta(hours=seq_len - 1)          # D-27 00:00

    series = history_series.sort_index()
    win = series.loc[win_start : d_end]

    if len(win) != seq_len:
        raise ValueError(
            f'PatchTST 입력 윈도우 길이가 {len(win)} ≠ {seq_len}. '
            f'필요 구간: {win_start} ~ {d_end}. history_series 가 충분히 긴지 확인.')
    if win.isna().any():
        n_nan = int(win.isna().sum())
        raise ValueError(
            f'PatchTST 입력 윈도우에 NaN {n_nan}개. 호출 전에 보간 필요.')

    # 3) 텐서 변환 → forward pass
    device_ = next(model.parameters()).device
    x_np = win.to_numpy(dtype=np.float32)
    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(-1).to(device_)  # (1, seq_len, 1)

    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy()                          # (pred_len,)

    return y.astype(np.float32)


# =============================================================================
# 4. (선택) 직접 실행 — 빠른 자가 점검
# =============================================================================
if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    print('=== PatchTST 추론 자가점검 ===')
    model, meta = load_model()
    print(f'  HP        : {meta["hp"]}')
    print(f'  best_val  : {meta["best_val_mae"]:.3f}')
    print(f'  device    : {next(model.parameters()).device}')

    # 더미 history 시리즈로 호출 가능 여부만 확인
    seq_len = meta['hp']['seq_len']
    idx = pd.date_range('2026-05-22 00:00', periods=seq_len + 24, freq='h')
    dummy_series = pd.Series(np.random.RandomState(0).uniform(500, 900, len(idx)),
                             index=idx)
    target = pd.Timestamp('2026-06-19')        # 마지막 timestamp 의 다음날
    preds = predict_d1(dummy_series, target)
    print(f'  더미 예측 shape : {preds.shape},  평균 {preds.mean():.2f}')
    print('완료.')
