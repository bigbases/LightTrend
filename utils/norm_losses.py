import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class NormLosses(torch.nn.Module):
    def __init__(self, args, station=None):
        super(NormLosses, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss()
        self.decomp = series_decomp(args.kernel_len)
        self.station = None

        if args.use_norm.lower() in ('ddn', 'ddn_ms'):
            self.station = station

    # SAN
    def san_loss(self, y, statistics_pred):
        bs, length, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_true = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_true)
        return loss

    # DDN
    def sliding_loss(self, y, statistics_pred):
        if self.station is None:
            raise RuntimeError(
                f"NormLosses.station is None for use_norm={self.args.use_norm}. "
                "DDN loss requires station module."
            )
        _, (mean, std) = self.station.norm(y.transpose(-1, -2), False)
        station_true = torch.cat([mean, std], dim=1).transpose(-1, -2)
        loss = self.criterion(statistics_pred, station_true)
        return loss

    # TP
    def trend_loss(self, y, statistics_pred):
        trend_pred = statistics_pred[-1]
        _, trend_true = self.decomp(y)
        loss = self.criterion(trend_pred, trend_true)
        return loss

    # LightTrend
    def lt_loss(self, y, statistics_pred):
        trend_pred = statistics_pred
        _, trend_true = self.decomp(y)
        loss = self.criterion(trend_pred, trend_true)
        return loss

    def forward(self, y, statistics_pred):
        use_norm = self.args.use_norm.lower()
        if use_norm in ('san', 'san_ms'):
            loss = self.san_loss(y, statistics_pred)
        elif use_norm in ('ddn', 'ddn_ms'):
            loss = self.sliding_loss(y, statistics_pred)
        elif use_norm == 'tp':
            loss = self.trend_loss(y, statistics_pred)
        elif use_norm == 'lt':
            loss = self.lt_loss(y, statistics_pred)
        else:
            loss = torch.tensor(0.0, device=y.device)
        return loss