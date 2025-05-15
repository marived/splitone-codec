"""Minimal usage example, copy-paste from the README."""
import torch
import torchaudio

from splitone import SplitOneCodec


def demo(path="example.wav", ckpt="checkpoints/splitone-base.pt"):
    codec = SplitOneCodec.from_pretrained(ckpt).eval()
    wav, sr = torchaudio.load(path)
    if sr != codec.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)
    with torch.no_grad():
        idx = codec.encode(wav.unsqueeze(0))
        recon = codec.decode(idx)
    print("tokens per second:", idx.shape[-1] / (wav.shape[-1] / codec.sample_rate))
    print("first frame:", idx[0, :, 0].tolist())


if __name__ == "__main__":
    import sys
    demo(*sys.argv[1:])
