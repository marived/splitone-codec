import torch
import torch.nn.functional as F


def waveform_l1(recon, target):
    return F.l1_loss(recon, target)
