"""Sanity-check: overfit one wav. Should drop loss < 0.1 in a few hundred steps."""
import argparse
import torch
import torchaudio

from splitone import SplitOneCodec
from splitone.losses import MultiScaleSTFTLoss, waveform_l1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()

    wav, sr = torchaudio.load(args.wav)
    assert sr == 24000
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.unsqueeze(0).cuda()

    codec = SplitOneCodec().cuda()
    opt = torch.optim.AdamW(codec.parameters(), lr=3e-4)
    stft = MultiScaleSTFTLoss()

    for step in range(args.steps):
        recon, _, vq = codec(wav)
        T = min(recon.shape[-1], wav.shape[-1])
        loss = waveform_l1(recon[..., :T], wav[..., :T]) + stft(recon[..., :T], wav[..., :T]) + 0.25 * vq
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(step, loss.item())


if __name__ == "__main__":
    main()
