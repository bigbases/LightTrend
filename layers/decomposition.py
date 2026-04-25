import torch
import torch.nn as nn
import torch.fft

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class fft_topk_decomp(nn.Module):
    """
    FFT-based Top-k Decomposition module
    Decomposes a time series into seasonal and trend components.
    
    Args:
        top_k (int): Number of dominant frequencies to keep for trend reconstruction.
    Input:
        x: [B, L, D]  (Batch, Length, Channel)
    Output:
        res (seasonal): [B, L, D]
        trend: [B, L, D]
    """

    def __init__(self, top_k: int = 5):
        super(fft_topk_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        Xf = torch.fft.fft(x, dim=1)
        mag = torch.abs(Xf)

        # ---- DC(=0) 주파수 제외 ----
        mag_no_dc = mag.clone()
        mag_no_dc[:, 0, :] = 0  # DC 성분 무시

        # ---- Top-k 주파수 선택 ----
        topk_idx = torch.topk(mag_no_dc, self.top_k, dim=1).indices  # [B, k, D]

        # ---- 마스크 구성 ----
        mask = torch.zeros_like(Xf, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        mirror_idx = (-topk_idx) % L
        mask.scatter_(1, mirror_idx, True)

        # ---- Top-k 성분 복원 (seasonality) ----
        Xf_seasonal = Xf * mask
        seasonal = torch.fft.ifft(Xf_seasonal, dim=1).real

        # ---- trend = 잔차 ----
        trend = x - seasonal
        return seasonal, trend
        # return trend, seasonal

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)
    
class ema_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, alpha):
        super(ema_decomp, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average
    
class envelope_decomp(nn.Module):
    """
    Max/Min Pooling을 사용하여 Envelope을 간접적으로 근사하는 모듈
    """
    def __init__(self, kernel_size, stride=1):
        super(envelope_decomp, self).__init__()
        self.k = kernel_size
        padding = (kernel_size - 1) // 2
        
        # 1. 상한 근사를 위한 Max Pooling (Higher points)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        # 2. 하한 근사를 위한 Min Pooling (Lower points)
        # PyTorch에는 직접적인 MinPool1d가 없으므로, Negation + MaxPool + Negation으로 구현
        # min(x) = -max(-x)
        self.neg_max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        
        # 3. 평활화된 평균값 (Optional: SMA 방식으로 사용 가능)
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, Seq_len, Features) -> (B, Features, Seq_len)
        x_permuted = x.permute(0, 2, 1)

        # 1. 상한 Envelope 근사 (Upper)
        upper_env = self.max_pool(x_permuted) 
        
        # 2. 하한 Envelope 근사 (Lower)
        lower_env = -self.neg_max_pool(-x_permuted) 
        
        # 3. 근사된 추세선 (Upper/Lower의 중간)
        # Envelope Trend = (Upper + Lower) / 2
        approx_trend = (upper_env + lower_env) / 2
        
        # 4. 추세선의 평활화 (Optional)
        front = approx_trend[:, :, 0:1].repeat(1, 1, (self.k - 1) // 2)
        end = approx_trend[:, :, -1:].repeat(1, 1, (self.k - 1) // 2)
        approx_trend = torch.cat([front, approx_trend, end], dim=2)
        approx_trend = self.avg_pool(approx_trend)
        
        # (B, Features, Seq_len) -> (B, Seq_len, Features)
        approx_trend = approx_trend.permute(0, 2, 1)
        
        # 잔차(Residual) 계산
        residual = x - approx_trend
        
        return residual, approx_trend
