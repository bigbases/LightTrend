"""
PatchTST backbone, adapted from Time-Series-Library (thuml/Time-Series-Library)
which itself is based on the official `yuqinie98/PatchTST` repo.

Key adaptation: the internal Non-stationary-Transformer-style normalization
in `forecast()` is REMOVED. In our framework, normalization is performed
externally by plug-in modules under `normalizers/` (NoNorm, RevIN, SAN, DDN,
LightTrend), dispatched through `Exp_Main._normalize_input` / `_denormalize`.
Running the internal normalization here would (a) double-normalize the input
when an external normalizer is active and (b) produce in-place autograd
errors when combined with external RevIN's gamma/beta gradient path.

This matches the rationale and patch already applied to `models/iTransformer.py`.
The structural body of the model (PatchEmbedding -> Transformer Encoder ->
FlattenHead -> permute) is unchanged from the upstream version.
"""
import torch
from torch import nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, n_vars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """PatchTST backbone (adapted).
    Patch length and stride are read from `configs.patch_len` and
    `configs.stride` respectively, which are part of the run_longExp argparse
    surface. Other knobs (`d_model`, `d_ff`, `n_heads`, `e_layers`, `dropout`,
    `factor`, `activation`, `enc_in`, `seq_len`, `pred_len`) follow the same
    naming used by Autoformer / iTransformer in this repo.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        patch_len = configs.patch_len
        stride = configs.stride
        padding = stride

        # Patching + embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        # Prediction head
        # patch_num = (seq_len - patch_len) / stride + 2 because of padding=stride
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # NOTE: Internal Non-stationary-Transformer-style normalization is
        # disabled here. The external normalizer plug-ins under `normalizers/`
        # already handle (de)normalization through Exp_Main._normalize_input
        # and Exp_Main._denormalize. See the file header for rationale.
        # The original lines were:
        #   means = x_enc.mean(1, keepdim=True).detach()
        #   x_enc = x_enc - means
        #   stdev = torch.sqrt(torch.var(x_enc, ...) + 1e-5)
        #   x_enc /= stdev
        # and matching de-normalization at the end of forecast().

        # Patching expects [bs, n_vars, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # enc_out: [bs * n_vars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, _ = self.encoder(enc_out)
        # Reshape back: [bs, n_vars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # [bs, n_vars, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Head: [bs, n_vars, target_window]
        dec_out = self.head(enc_out)
        # [bs, target_window, n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Only long-term forecast is supported in this repo's pipeline.
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
