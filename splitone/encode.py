"""CLI: encode a wav to RVQ indices saved as .npy."""
import argparse
import numpy as np
import torch
import torchaudio

from .model.codec import SplitOneCodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="checkpoints/splitone-base.pt")
    args = parser.parse_args()

    codec = SplitOneCodec.from_pretrained(args.checkpoint).eval()
    wav, sr = torchaudio.load(args.input)
    if sr != codec.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    with torch.no_grad():
        idx = codec.encode(wav.unsqueeze(0))
    np.save(args.output, idx.squeeze(0).cpu().numpy().astype("int32"))


if __name__ == "__main__":
    main()
