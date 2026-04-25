import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.learnable_wavelet import DWT1D


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel = configs.kernel_len
        self.hkernel = configs.hkernel_len
        self.pad = nn.ReplicationPad1d(padding=(self.kernel // 2, self.kernel // 2 - ((self.kernel + 1) % 2)))
        if self.hkernel is not None:
            self.hpad = nn.ReplicationPad1d(padding=(self.hkernel // 2, self.hkernel // 2 - ((self.hkernel + 1) % 2)))
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type
        self.mode = getattr(configs, 'meanstd_mode', 'full')
        self.seq_len_new = self.seq_len
        self.pred_len_new = self.pred_len
        self.epsilon = 1e-5
        self._build_model()
        self.last_instance_std = None

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len
        args.pred_len = self.configs.pred_len
        args.label_len = self.configs.label_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.dec_in
        args.moving_avg = 3
        args.c_out = self.configs.c_out
        self.norm_func = self.norm_sliding
        wave = self.configs.wavelet
        wave_dict = {'coif6': 17, 'coif3': 8, 'sym3': 2}
        self.len, self.j = wave_dict[wave], self.configs.j
        self.dwt = DWT1D(wave=wave, J=self.j, learnable=args.learnable)
        self.dwt_ratio = nn.Parameter(torch.clamp(torch.full((1, self.channels, 1), 0.), min=0., max=1.))
        self.mlp = Statics_MLP(self.configs.seq_len, args.pd_model, args.pd_ff, self.configs.pred_len, drop_rate=args.dr, layer=args.pe_layers)

    def normalize(self, x, p_value=True):
        if self.station_type != 'adaptive':
            return x, None
        self.last_instance_std = torch.std(x, dim=1, keepdim=True, unbiased=False)
        norm_input, seq_ms, pred_ms = self.norm(x=x.transpose(-1, -2))
        outputs = torch.cat(pred_ms, dim=1).transpose(-1, -2)
        return norm_input.transpose(-1, -2), outputs, seq_ms

    def de_normalize(self, input, station_pred, future_y=None):
        if self.station_type != 'adaptive':
            return input
        mean = station_pred[..., :station_pred.shape[-1] // 2]
        std = station_pred[..., station_pred.shape[-1] // 2:]
        if self.mode == 'no_norm':
            output = input
        elif self.mode == 'full':
            output = input * (std + self.epsilon) + mean
        elif self.mode == 'mean_only':
            output = input + mean
        elif self.mode == 'std_only':
            output = input * (std + self.epsilon)
        elif self.mode == 'mean_oracle_std':
            if future_y is None:
                raise ValueError('future_y is required for mean_oracle_std')
            future_std = self._oracle_std_from_future(future_y)
            output = input * (future_std + self.epsilon) + mean
        elif self.mode == 'mean_only_input_std':
            output = input * (self.last_instance_std + self.epsilon) + mean
        elif self.mode == 'mean_only_full_std':
            output = input * (std + self.epsilon) + mean
        else:
            raise ValueError(f'Unknown meanstd_mode: {self.mode}')
        return output

    def norm(self, x, predict=True):
        norm_x, (seq_m, seq_s) = self.norm_func(x)
        if predict is True:
            mov_m, mov_s = self.mlp(seq_m, seq_s, x)
            if self.j > 0:
                ac, dc_list = self.dwt(x)
                norm_ac, (mac, sac) = self.norm_func(ac, kernel=self.hkernel)
                norm_dc, m_list, s_list = [], [], []
                for dc in dc_list:
                    dc, (mdc, sdc) = self.norm_func(dc, kernel=self.hkernel)
                    norm_dc.append(dc)
                    m_list.append(mdc)
                    s_list.append(sdc)
                pred_m, pred_s = self.mlp(self.dwt([mac, m_list], 1), self.dwt([sac, s_list], 1), self.dwt([ac, dc_list], 1))
                dwt_r, mov_r = self.dwt_ratio, 1 - self.dwt_ratio
                norm_x = norm_x * mov_r + self.dwt([norm_ac, norm_dc], 1) * dwt_r
                pred_m = mov_m * mov_r + pred_m * dwt_r
                pred_s = mov_s * mov_r + pred_s * dwt_r
                return norm_x, (seq_m, seq_s), (pred_m, pred_s)
            return norm_x, (seq_m, seq_s), (mov_m, mov_s)
        return norm_x, (seq_m, seq_s)

    def norm_sliding(self, x, kernel=None):
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad
        x_window = x.unfold(-1, kernel, 1)
        m, s = x_window.mean(dim=-1), x_window.std(dim=-1)
        m, s = pad(m), pad(s)
        instance_std = self.last_instance_std.transpose(-1, -2)
        if self.mode == 'no_norm':
            x_normalized = x
        elif self.mode in ('full', 'mean_oracle_std'):
            x_normalized = (x - m) / (s + self.epsilon)
        elif self.mode == 'mean_only':
            x_normalized = x - m
        elif self.mode == 'std_only':
            x_normalized = x / (s + self.epsilon)
        elif self.mode in ('mean_only_input_std', 'mean_only_full_std'):
            x_normalized = (x - m) / (instance_std + self.epsilon)
        else:
            raise ValueError(f'Unknown meanstd_mode: {self.mode}')
        return x_normalized, (m, s)

    def _oracle_std_from_future(self, future_y):
        fy = future_y.transpose(-1, -2)
        _, (m, s) = self.norm_sliding(fy)
        return s.transpose(-1, -2)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation, drop_rate=0.1, bias=False):
        super(FFN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias), activation,
            nn.Linear(d_ff, d_model, bias=bias), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.mlp(x)


class Statics_MLP(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, pred_len, drop_rate=0.1, bias=False, layer=1):
        super(Statics_MLP, self).__init__()
        project = nn.Sequential(nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate))
        self.m_project, self.s_project = copy.deepcopy(project), copy.deepcopy(project)
        self.mean_proj, self.std_proj = copy.deepcopy(project), copy.deepcopy(project)
        self.m_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))
        self.s_concat = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate))
        ffn = nn.Sequential(*[FFN(d_model, d_ff, nn.LeakyReLU(), drop_rate, bias) for _ in range(layer)])
        self.mean_ffn, self.std_ffn = copy.deepcopy(ffn), copy.deepcopy(ffn)
        self.mean_pred = nn.Linear(d_model, pred_len, bias=bias)
        self.std_pred = nn.Linear(d_model, pred_len, bias=bias)

    def forward(self, mean, std, x=None, x2=None):
        m_all, s_all = mean.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)
        mean_r, std_r = mean - m_all, std - s_all
        mean_r, std_r = self.mean_proj(mean_r), self.std_proj(std_r)
        if x is not None:
            m_orig = self.m_project(x - m_all)
            s_ori = self.s_project(x if x2 is None else x2 - s_all)
            mean_r = self.m_concat(torch.cat([m_orig, mean_r], dim=-1))
            std_r = self.s_concat(torch.cat([s_ori, std_r], dim=-1))
        mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r)
        mean_r, std_r = self.mean_pred(mean_r), self.std_pred(std_r)
        mean, std = mean_r + m_all, std_r + s_all
        return mean, F.relu(std)
