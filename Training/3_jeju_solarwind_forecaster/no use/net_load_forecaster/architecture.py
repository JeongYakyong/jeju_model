"""PatchTST + Weather Attention model definition.

Ported from `old project/models/architecture.py`. No Streamlit/UI dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm', mean=None, std=None):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.std * self.affine + self.bias
        elif mode == 'denorm':
            return (x - self.bias) / self.affine * std + mean
        return x


class Patch_Weather_Attention(nn.Module):
    """Weather Attention with separated query/key dims.

    Solar:  patch_len=24, pred_len=24 → query_dim == key_dim
    Wind:   patch_len=12, pred_len=24 → query_dim != key_dim
    """

    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(key_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, future_weather_patch, past_weather_patches, transformer_output):
        Q = self.W_Q(future_weather_patch).unsqueeze(1)
        K = self.W_K(past_weather_patches)
        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights, transformer_output)
        return context.squeeze(1), attn_weights


class PatchTST_Weather_Model(nn.Module):
    def __init__(self, num_features,
                 seq_len=336, pred_len=24, patch_len=24,
                 stride=12,
                 d_model=128,
                 num_heads=4,
                 num_layers=2,
                 d_ff=256, dropout=0.2):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        patch_input_dim = self.patch_len * num_features
        self.patch_embedding = nn.Linear(patch_input_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.num_weather_feats = num_features - 1
        future_weather_flat_dim = self.pred_len * self.num_weather_feats
        weather_patch_dim = self.patch_len * self.num_weather_feats

        self.weather_attn = Patch_Weather_Attention(
            query_dim=future_weather_flat_dim,
            key_dim=weather_patch_dim,
            hidden_dim=self.d_model
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.d_model + future_weather_flat_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, self.pred_len)
        )

        self.weather_bypass = nn.Linear(future_weather_flat_dim, self.pred_len)

    def forward(self, batch, device='cpu'):
        p_num = batch['past_numeric'].to(device)
        p_y = batch['past_y'].to(device)
        f_num = batch['future_numeric'].to(device)
        B = p_num.shape[0]

        x_past = torch.cat([p_num, p_y], dim=-1)

        x_patches = x_past.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_patches = x_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)

        enc_out = self.patch_embedding(x_patches) + self.pos_embedding
        enc_out = self.transformer_encoder(self.dropout(enc_out))

        future_weather_flat = f_num.reshape(B, -1)

        x_past_weather = x_past[..., :-1]
        w_patches = x_past_weather.unfold(1, self.patch_len, self.stride)
        w_patches = w_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)

        context, _ = self.weather_attn(future_weather_flat, w_patches, enc_out)

        total_input = torch.cat([context, future_weather_flat], dim=1)
        main_pred = self.regressor(total_input)
        weather_shortcut = self.weather_bypass(future_weather_flat)
        prediction = main_pred + weather_shortcut

        return prediction
