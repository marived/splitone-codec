"""Decoder — mirror of the encoder.

Uses transposed convs for upsampling, with the same residual stack as the encoder.
"""
import torch.nn as nn

from .encoder import ResUnit


class DecBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.up = nn.ConvTranspose1d(c_in, c_out, kernel_size=2 * stride,
                                     stride=stride, padding=stride // 2)
        self.res = nn.Sequential(
            ResUnit(c_out, dilation=1),
            ResUnit(c_out, dilation=3),
            ResUnit(c_out, dilation=9),
        )

    def forward(self, x):
        return self.res(self.up(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, base_channels=32, strides=(8, 5, 4, 2)):
        super().__init__()
        # mirror channel counts of the encoder
        c_in = base_channels * (2 ** len(strides))
        self.proj = nn.Conv1d(latent_dim, c_in, kernel_size=1)
        blocks = []
        c_prev = c_in
        for i, s in enumerate(strides):
            c_out = c_prev // 2
            blocks.append(DecBlock(c_prev, c_out, s))
            c_prev = c_out
        self.blocks = nn.Sequential(*blocks)
        # final 1-channel conv
        self.head = nn.Conv1d(c_prev, 1, kernel_size=7, padding=3)

    def forward(self, z):
        h = self.proj(z)
        h = self.blocks(h)
        return self.head(h)  # (B, 1, T)
