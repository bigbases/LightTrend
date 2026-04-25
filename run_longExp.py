import argparse
import random
import numpy as np
import torch

from exp.exp_main import Exp_Main


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm_suffix(args) -> str:
    """use_norm 별로 해당 정규화 모듈이 실제로 사용하는 하이퍼파라미터만 골라 suffix를 만든다.

    공통 backbone 하이퍼(seq_len, pred_len, d_model 등)는 이미 base setting에 포함되어 있으므로
    여기서는 정규화 모듈 고유 설정 + 학습 절차(pretrain / joint) 설정만 나열한다.
    """
    un = args.use_norm
    tokens = ['un' + str(un)]

    if un == 'none':
        pass
    elif un == 'revin':
        tokens.append('af' + str(args.affine))
    elif un == 'san':
        tokens += [
            'pl' + str(args.period_len),
            'st' + str(args.station_type),
            'pe' + str(args.pre_epoch),
            'slr' + str(args.station_lr),
        ]
    elif un == 'ddn':
        tokens += [
            'kl' + str(args.kernel_len),
            'hkl' + str(args.hkernel_len),
            'j' + str(args.j),
            'pe' + str(args.pre_epoch),
            'te' + str(args.twice_epoch),
            'slr' + str(args.station_lr),
        ]
    elif un == 'lt':
        tokens += [
            'tn' + str(args.t_norm),
            'sn' + str(args.s_norm),
            'um' + str(args.use_mlp),
            'tff' + str(args.t_ff),
            'ks' + str(args.kernel_size),
            'dc' + str(args.decomp_type),
            'pe' + str(args.pre_epoch),
            'slr' + str(args.station_lr),
        ]
    else:
        tokens.append('raw')

    return '_'.join(tokens)


def build_setting(args, itr_idx: int) -> str:
    base = (
        '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
        )
    )
    return '{}_{}_{}'.format(base, _norm_suffix(args), itr_idx)


parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
# NOTE: --task_name is deprecated but kept for backward compatibility with
# (a) downstream model files that read it (e.g. TimeMixer / MICN / CARD
#     vendored elsewhere), and (b) extra_exp launchers that inject it into
#     their build_setting and checkpoint paths for hashing. Only
#     'long_term_forecast' is actually supported in this runtime.
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='deprecated; kept for compatibility with extra_exp launchers and some model files')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='FEDformer',
                    help='model name, options: [Autoformer, FEDformer, DLinear, iTransformer]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross attention activation function tanh or softmax')

# non-station / normalization config
parser.add_argument('--station_type', type=str, default='adaptive')
parser.add_argument('--use_norm', type=str, default='none',
                    choices=['none', 'revin', 'san', 'ddn', 'lt'],
                    help='normalization mode: [none, revin, san, ddn, lt]')
parser.add_argument('--pre_epoch', type=int, default=5)
parser.add_argument('--station_lr', type=float, default=0.0001,
                    help='learning rate for the station (normalizer) module; matches SAN/DDN original')
parser.add_argument('--s_norm', type=int, default=0, help='series normalization; True 1 False 0')
parser.add_argument('--t_norm', type=int, default=1, help='trend normalization; True 1 False 0')
parser.add_argument('--use_mlp', type=int, default=0)
parser.add_argument('--decomp_type', type=str, default='sma', help='decomposition type, options: [sma, ema, envelope]')
parser.add_argument('--kernel_len', type=int, default=25)
parser.add_argument('--t_ff', type=int, default=64)
parser.add_argument('--down_ratio', type=int, default=4)
parser.add_argument('--twice_epoch', type=int, default=1)
parser.add_argument('--j', type=int, default=0)
parser.add_argument('--learnable', action='store_true', default=False)
parser.add_argument('--wavelet', type=str, default='coif3')
parser.add_argument('--dr', type=float, default=0.05)
parser.add_argument('--hkernel_len', type=int, default=5)
parser.add_argument('--pd_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--pd_model', type=int, default=512, help='dimension of model')
parser.add_argument('--pe_layers', type=int, default=2, help='num of encoder layers')

# experiment plumbing
parser.add_argument('--seed', type=int, default=2021, help='global random seed')
parser.add_argument('--result_file', type=str, default='result.csv',
                    help='CSV file for appending final (setting, MSE, MAE, ...) rows; '
                         'relative to CWD. Use this to separate baseline runs across scripts '
                         '(e.g. results/NoNorm.csv, results/SAN.csv).')

# data loader
parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--augmentation_ratio', type=int, default=0, help='How many times to augment')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=1, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual_head', type=int, default=0, help='individual head; True 1 False 0')

# CrossFormer
parser.add_argument('--win_size', type=int, default=2, help='window kernel for segment merge')
parser.add_argument('--cross_factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='x sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
parser.add_argument('--individual', action='store_true', default=False,
                    help='DLinear: a linear layer for each variate(channel) individually')

parser.add_argument('--period_len', type=int, default=24)

# Formers
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder channels')
parser.add_argument('--dec_in', type=int, default=7, help='decoder channels')
parser.add_argument('--c_out', type=int, default=7, help='output channels')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window kernel of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=3, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
args.label_len = args.seq_len // 2  # follow original SAN/DDN: label_len is always seq_len//2
if args.features == 'S':
    args.enc_in, args.dec_in, args.c_out = 1, 1, 1

set_seed(args.seed)
torch.set_num_threads(6)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = build_setting(args, ii)
        exp = Exp(args)
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
        if args.do_predict:
            print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting, True)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = build_setting(args, ii)
    exp = Exp(args)
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
