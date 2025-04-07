"""Eval CLI: compute PESQ + STOI on a manifest."""
import argparse
import math

import torch
import torchaudio

from .data import AudioManifest
from .model.codec import SplitOneCodec


def _safe_pesq(ref, deg, sr):
    try:
        from pesq import pesq
        return pesq(sr, ref, deg, "wb" if sr >= 16000 else "nb")
    except Exception:
        return float("nan")


def _safe_stoi(ref, deg, sr):
    try:
        from pystoi import stoi
        return stoi(ref, deg, sr, extended=False)
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_items",  type=int, default=200)
    args = parser.parse_args()

    codec = SplitOneCodec.from_pretrained(args.checkpoint).eval()
    ds = AudioManifest(args.manifest, segment_seconds=4.0,
                       sample_rate=codec.sample_rate)

    pesqs, stois = [], []
    for i in range(min(args.max_items, len(ds))):
        wav = ds[i]
        with torch.no_grad():
            recon, _, _ = codec(wav.unsqueeze(0))
        T = min(recon.shape[-1], wav.shape[-1])
        ref = wav[0, :T].cpu().numpy()
        deg = recon[0, 0, :T].cpu().numpy()
        p = _safe_pesq(ref, deg, codec.sample_rate)
        s = _safe_stoi(ref, deg, codec.sample_rate)
        if not math.isnan(p):
            pesqs.append(p)
        if not math.isnan(s):
            stois.append(s)
    print(f"PESQ mean: {sum(pesqs)/max(1,len(pesqs)):.3f}  (n={len(pesqs)})")
    print(f"STOI mean: {sum(stois)/max(1,len(stois)):.3f}  (n={len(stois)})")


if __name__ == "__main__":
    main()
