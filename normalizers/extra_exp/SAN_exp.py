import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.period_len
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type

        ### NEW ###
        # Motivating Experiment 1을 위한 파라미터
        self.norm_variant = configs.norm_variant

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

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

    def normalize(self, input):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            global_mean = torch.mean(input, dim=1, keepdim=True) # [bs, 1, dim]
            input = input.reshape(bs, -1, self.period_len, dim)

            # 1. 통계량(평균, 표준편차)은 항상 동일하게 계산
            global_mean = global_mean.unsqueeze(2)  # [bs, 1, 1, dim]
            mean = torch.mean(input, dim=-2, keepdim=True) # [bs, len_new, 1, dim]
            std = torch.std(input, dim=-2, keepdim=True)

            ### NEW ###
            # 2. norm_variant 설정에 따라 백본에 들어갈 입력(norm_input)을 다르게 생성
            if self.norm_variant == 'original':
                # (x - mean) / std : 기존 SAN 방식 (추세 제거 O, 계절성 입력)
                norm_input = (input - mean) / (std + self.epsilon)
            elif self.norm_variant == 'global_mean':
                # (x - global_mean) / std : 실험용 (추세 제거 X, 계절성 제거 O)
                norm_input = (input - global_mean) / (std + self.epsilon)
            elif self.norm_variant == 'only_std':
                # x / std : 실험용 (추세 제거 X, 스케일링된 원본 입력)
                norm_input = input / (std + self.epsilon)
            elif self.norm_variant == 'only_mean':
                # (x - mean) : 실험용 (추세 제거 O, 스케일링 제거)
                norm_input = input - mean
            ### END NEW ###

            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            
            # 3. 통계량 예측 모델은 실험과 상관없이 항상 동일하게 학습됨
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(bs, len, dim), outputs[:, -self.pred_len_new:, :]

        else:
            return input, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)

            ### NEW ###
            # 4. norm_variant 설정에 따라 비정규화(최종 예측) 방식을 다르게 적용
            if self.norm_variant == 'original':
                # Original: (백본 예측: 계절성) * std + (MLP 예측: 추세)
                # 이것이 '학습 분리' 효과임
                output = input * (std + self.epsilon) + mean
            elif self.norm_variant == 'only_std':
                output = input * (std + self.epsilon)
            elif self.norm_variant == 'global_mean':
                global_mean = torch.mean(input.reshape(bs, len, dim), dim=1, keepdim=True).unsqueeze(2)
                output = input * (std + self.epsilon) + global_mean
            elif self.norm_variant == 'only_mean':
                output = input + mean
            ### END NEW ###
            
            return output.reshape(bs, len, dim)

        else:
            return input


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
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