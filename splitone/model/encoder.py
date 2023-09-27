"""Tiny 1D conv encoder.

Channels: 1 -> 32 -> 64 -> 128 -> 256, downsample 2,4,5,8 = 320x.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=2 * stride, stride=stride,
                              padding=stride // 2)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, channels=(48, 96, 192, 256), strides=(2, 4, 5, 8)):
        super().__init__()
        c_prev = 1
        layers = []
        for c, s in zip(channels, strides):
            layers.append(ConvBlock(c_prev, c, s))
            c_prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T)
        return self.net(x)  # (B, C, T/320)
