import torch
import torch.nn as nn
from layers.decomposition import series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.decomposer = series_decomp(configs.kernel_size)
        self.dummy = nn.Parameter(torch.zeros(1))

    def normalize(self, batch_x):
        seasonal, trend = self.decomposer(batch_x)
        trend_last = trend[:, -1:, :].repeat(1, self.pred_len, 1)
        return seasonal, trend_last

    def de_normalize(self, batch_y, statistics):
        return batch_y + statistics
