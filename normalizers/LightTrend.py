import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.decomposition import series_decomp

class DownUpTrendPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, down_ratio=4, hidden_dim=64, use_mlp=False):
        super().__init__()
        self.down_ratio = down_ratio
        self.down_len = max(1, seq_len // down_ratio)
        self.pred_len = pred_len

        # downsample → linear prediction (간소화)
        self.down_proj = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=down_ratio, stride=down_ratio)
        if use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(self.down_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(1, pred_len // down_ratio))
            )
        else:
            self.linear = nn.Linear(self.down_len, max(1, pred_len // down_ratio))
        
    def forward(self, trend):
        """
        trend: [B, L, D]
        return: [B, H, D]
        """
        # D 차원 각각 독립적으로 처리
        trend_down = F.avg_pool1d(trend.transpose(1,2), kernel_size=self.down_ratio).transpose(1,2)  # [B, L//r, D]
        pred_down = self.linear(trend_down.transpose(1,2)).transpose(1,2)  # [B, H//r, D]
        trend_up = F.interpolate(pred_down.transpose(1,2), size=self.pred_len, mode='linear', align_corners=False).transpose(1,2)
        return trend_up


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # affine parameters
        if configs.affine:
            self.s_gamma, self.s_beta = nn.Parameter(torch.ones(configs.enc_in)), nn.Parameter(torch.zeros(configs.enc_in))
            self.t_gamma, self.t_beta = nn.Parameter(torch.ones(configs.enc_in)), nn.Parameter(torch.zeros(configs.enc_in))
        else:
            self.t_gamma, self.t_beta = 1, 0
            self.s_gamma, self.s_beta = 1, 0
        
        # FFT-based decomposition
        self.decomposer = series_decomp(configs.kernel_size)
        self.s_norm = configs.s_norm
        self.t_norm = configs.t_norm
        
        # ↓ 기존 Linear 대신 다운샘플 기반 predictor 사용
        self.trend_predictor = DownUpTrendPredictor(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            down_ratio=configs.down_ratio,
            use_mlp=configs.use_mlp,
            hidden_dim=configs.t_ff
        )

    def normalize(self, batch_x):
        batch_x, trend = self.decomposer(batch_x)

        dev = batch_x.device

        if self.s_norm:
            avg = batch_x.mean(dim=1, keepdim=True).detach()
            var = batch_x.var(dim=1, keepdim=True, unbiased=False).detach()
        else:
            avg = batch_x.new_zeros(1, 1, 1)
            var = batch_x.new_ones(1, 1, 1)

        self.avg, self.var = avg, var

        batch_x = (batch_x - avg) / torch.sqrt(var + 1e-8)
        batch_x = batch_x.mul(self.s_gamma) + self.s_beta

        if self.t_norm:
            trend_avg = trend.mean(dim=1, keepdim=True).detach()
            trend_var = trend.var(dim=1, keepdim=True, unbiased=False).detach()
        else:
            trend_avg = trend.new_zeros(1, 1, 1)
            trend_var = trend.new_ones(1, 1, 1)

        trend = (trend - trend_avg) / torch.sqrt(trend_var + 1e-8)
        trend = trend.mul(self.t_gamma) + self.t_beta

        trend_out = self.trend_predictor(trend)

        # affine inverse (gamma/beta가 Parameter면 device는 맞아있음)
        trend_out = (trend_out - self.t_beta) / (self.t_gamma + 1e-12)
        trend_out = trend_out * torch.sqrt(trend_var + 1e-8) + trend_avg

        return batch_x, trend_out

    def de_normalize(self, x, statistics):
        # batch_x: B*H*D (forecasts)
        trend_out = statistics
        x = (x - self.s_beta) / self.s_gamma
        x = x * torch.sqrt(self.var + 1e-8) + self.avg
        x = x + trend_out
        return x