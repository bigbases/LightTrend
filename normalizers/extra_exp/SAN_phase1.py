import copy
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.period_len
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type
        self.mode = getattr(configs, 'meanstd_mode', 'full')
        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))
        self.last_instance_std = None

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len // self.period_len
        args.label_len = self.configs.label_len // self.period_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.dec_in
        args.moving_avg = 3
        args.c_out = self.configs.c_out
        args.pred_len = self.pred_len_new
        self.model = MLP(args, mode='mean').float()
        self.model_std = MLP(args, mode='std').float()

    def _predict_stats(self, input_flat, mean, std):
        mean_all = torch.mean(input_flat, dim=1, keepdim=True)
        outputs_mean = self.model(mean.squeeze(2) - mean_all, input_flat - mean_all) * self.weight[0] + mean_all * self.weight[1]
        outputs_std = self.model_std(std.squeeze(2), input_flat)
        return outputs_mean[:, -self.pred_len_new:, :], outputs_std[:, -self.pred_len_new:, :]

    def normalize(self, input):
        if self.station_type != 'adaptive':
            return input, None
        bs, length, dim = input.shape
        input_4d = input.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(input_4d, dim=-2, keepdim=True)
        std = torch.std(input_4d, dim=-2, keepdim=True)
        instance_std = torch.std(input, dim=1, keepdim=True, unbiased=False).unsqueeze(2)
        self.last_instance_std = instance_std

        if self.mode == 'no_norm':
            norm_input = input_4d
        elif self.mode in ('full', 'mean_oracle_std'):
            norm_input = (input_4d - mean) / (std + self.epsilon)
        elif self.mode == 'mean_only':
            norm_input = input_4d - mean
        elif self.mode == 'std_only':
            norm_input = input_4d / (std + self.epsilon)
        elif self.mode in ('mean_only_input_std', 'mean_only_full_std'):
            norm_input = (input_4d - mean) / (instance_std + self.epsilon)
        else:
            raise ValueError(f'Unknown meanstd_mode: {self.mode}')

        pred_mean, pred_std = self._predict_stats(input, mean, std)
        outputs = torch.cat([pred_mean, pred_std], dim=-1)
        return norm_input.reshape(bs, length, dim), outputs

    def _oracle_std_from_future(self, future_y):
        bs, length, dim = future_y.shape
        y4 = future_y.reshape(bs, -1, self.period_len, dim)
        std = torch.std(y4, dim=-2, keepdim=True)
        return std

    def de_normalize(self, input, station_pred, future_y=None):
        if self.station_type != 'adaptive':
            return input
        bs, length, dim = input.shape
        input_4d = input.reshape(bs, -1, self.period_len, dim)
        mean = station_pred[:, :, :self.channels].unsqueeze(2)
        std = station_pred[:, :, self.channels:].unsqueeze(2)

        if self.mode == 'no_norm':
            output = input_4d
        elif self.mode == 'full':
            output = input_4d * (std + self.epsilon) + mean
        elif self.mode == 'mean_only':
            output = input_4d + mean
        elif self.mode == 'std_only':
            output = input_4d * (std + self.epsilon)
        elif self.mode == 'mean_oracle_std':
            if future_y is None:
                raise ValueError('future_y is required for mean_oracle_std')
            oracle_std = self._oracle_std_from_future(future_y)
            output = input_4d * (oracle_std + self.epsilon) + mean
        elif self.mode == 'mean_only_input_std':
            output = input_4d * (self.last_instance_std + self.epsilon) + mean
        elif self.mode == 'mean_only_full_std':
            output = input_4d * (std + self.epsilon) + mean
        else:
            raise ValueError(f'Unknown meanstd_mode: {self.mode}')
        return output.reshape(bs, length, dim)


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        self.final_activation = nn.ReLU() if mode == 'std' else nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
