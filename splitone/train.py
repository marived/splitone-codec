"""Very early training loop. Single GPU, one file."""
import argparse
import torch
import torchaudio

from .model.codec import SplitOneCodec
from .losses import waveform_l1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.batch < 2:
        raise SystemExit("batch=1 hits a nan in the commitment term, please use >=2")

    wav, sr = torchaudio.load(args.wav)
    assert sr == 24000
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    # crude: tile the clip to fill a batch
    wav = wav.unsqueeze(0).repeat(args.batch, 1, 1).cuda()

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
