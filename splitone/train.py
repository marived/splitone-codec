"""Very early training loop. Single GPU, one file."""
import argparse
import torch
import torchaudio

from .model.codec import SplitOneCodec
from .losses import waveform_l1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    args = parser.parse_args()

    wav, sr = torchaudio.load(args.wav)
    assert sr == 24000
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.unsqueeze(0).cuda()

    model = SplitOneCodec().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(2000):
        recon, _, vq = model(wav)
        T = min(recon.shape[-1], wav.shape[-1])
        loss = waveform_l1(recon[..., :T], wav[..., :T]) + 0.25 * vq
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(step, loss.item())


if __name__ == "__main__":
    main()
