"""Multi-scale STFT discriminator (off by default).

Adapted from EnCodec. Adds clarity at higher bitrates but is fiddly to balance.
TODO: try Hifi-GAN style multi-period disc, maybe more stable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTDisc(nn.Module):
    def __init__(self, n_fft=1024, hop=256, n_filters=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        self.convs = nn.ModuleList([
            nn.Conv2d(2,           n_filters, (3, 9), padding=(1, 4)),
            nn.Conv2d(n_filters,   2 * n_filters, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(2 * n_filters, 4 * n_filters, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(4 * n_filters, 4 * n_filters, (3, 3), padding=(1, 1)),
        ])
        self.head = nn.Conv2d(4 * n_filters, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=window, return_complex=True)
        re_im = torch.stack([spec.real, spec.imag], dim=1)
        h = re_im
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        return self.head(h), feats


class MultiScaleSTFTDisc(nn.Module):
    def __init__(self, scales=((1024, 256), (2048, 512), (512, 128))):
        super().__init__()
        self.discs = nn.ModuleList([STFTDisc(n_fft=n, hop=h) for n, h in scales])

    def forward(self, x):
        outs, all_feats = [], []
        for d in self.discs:
            o, fs = d(x)
            outs.append(o)
            all_feats.append(fs)
        return outs, all_feats
