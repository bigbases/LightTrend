import torch
import torch.nn as nn
from layers.decomposition import series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.decomposer = series_decomp(configs.kernel_size)
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        nn.init.constant_(self.linear.weight, 1.0 / self.seq_len)
        nn.init.zeros_(self.linear.bias)

    def normalize(self, batch_x):
        seasonal, trend = self.decomposer(batch_x)
        trend_pred = self.linear(trend.transpose(1, 2)).transpose(1, 2)
        return seasonal, trend_pred

    def de_normalize(self, batch_y, statistics):
        return batch_y + statistics
