"""Decoder — mirror of the encoder, transposed convs."""
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, channels=(256, 192, 96, 48), strides=(8, 5, 4, 2)):
        super().__init__()
        layers = []
        c_prev = channels[0]
        for c, s in zip(channels[1:], strides[:-1]):
            layers.append(nn.ConvTranspose1d(c_prev, c, kernel_size=2 * s,
                                             stride=s, padding=s // 2))
            layers.append(nn.GELU())
            c_prev = c
        layers.append(nn.ConvTranspose1d(c_prev, 1, kernel_size=2 * strides[-1],
                                         stride=strides[-1],
                                         padding=strides[-1] // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # (B, 1, T)
