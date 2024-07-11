"""CLI: decode .npy indices back to a wav."""
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
    idx = torch.from_numpy(np.load(args.input)).long().unsqueeze(0)
    with torch.no_grad():
        wav = codec.decode(idx)
    torchaudio.save(args.output, wav.squeeze(0).cpu(), codec.sample_rate)


if __name__ == "__main__":
    main()
