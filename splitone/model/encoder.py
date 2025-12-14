"""Encoder — strided 1d conv stack with residual blocks.

Layout follows the SoundStream / DAC family: a stem conv, then a sequence of
(residual, downsample) blocks.
"""
import torch
import torch.nn as nn


class ResUnit(nn.Module):
    """Pre-norm residual with dilated convs (dilation in {1, 3, 9})."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class EncBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.res = nn.Sequential(
            ResUnit(c_in, dilation=1),
            ResUnit(c_in, dilation=3),
            ResUnit(c_in, dilation=9),
        )
        self.down = nn.Conv1d(c_in, c_out, kernel_size=2 * stride,
                              stride=stride, padding=stride // 2)

    def forward(self, x):
        return self.down(self.res(x))


class Encoder(nn.Module):
    def __init__(self, base_channels=32, strides=(2, 4, 5, 8), latent_dim=256, causal=False):
        super().__init__()
        self.causal = causal
        # in causal mode we pad on the left only, so the conv is causal
        pad = 0 if causal else 3
        self.stem = nn.Conv1d(1, base_channels, kernel_size=7, padding=pad)
        c_prev = base_channels
        blocks = []
        for i, s in enumerate(strides):
            c_out = base_channels * (2 ** (i + 1))
            blocks.append(EncBlock(c_prev, c_out, s))
            c_prev = c_out
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(c_prev, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, T)
        if self.causal:
            x = nn.functional.pad(x, (6, 0))
        h = self.stem(x)
        h = self.blocks(h)
        return self.proj(h)  # (B, latent_dim, T // prod(strides))
