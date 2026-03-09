"""Reconstruction losses.

We use a multi-scale STFT magnitude loss plus a waveform L1 term. The optional
adversarial term lives in `disc.py` and is disabled by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, n_ffts=(2048, 1024, 512, 256, 128), eps=1e-5):
        super().__init__()
        self.n_ffts = n_ffts
        self.eps = eps

    def _stft_mag(self, x, n_fft):
        # x: (B, 1, T) -> (B, F, T')
        # hann window MUST match between train and eval; explicit here on purpose
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x.squeeze(1), n_fft=n_fft, hop_length=n_fft // 4,
            win_length=n_fft, window=window, return_complex=True,
        )
        return spec.abs()

    def forward(self, recon, target):
        loss = 0.0
        for n_fft in self.n_ffts:
            r = self._stft_mag(recon, n_fft)
            t = self._stft_mag(target, n_fft)
            loss = loss + F.l1_loss(r, t)
            loss = loss + F.l1_loss(torch.log(r + self.eps),
                                    torch.log(t + self.eps))
        return loss / len(self.n_ffts)


def waveform_l1(recon, target):
    return F.l1_loss(recon, target)
