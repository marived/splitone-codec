"""Main training loop. Single-GPU or accelerate-driven multi-GPU."""
import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import AudioManifest
from .losses import MultiScaleSTFTLoss, waveform_l1
from .model.codec import SplitOneCodec


def _maybe_accelerate():
    try:
        from accelerate import Accelerator
        return Accelerator()
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = SplitOneCodec(**cfg["model"])
    opt   = torch.optim.AdamW(model.parameters(), **cfg["optim"])
    stft  = MultiScaleSTFTLoss()

    ds = AudioManifest(cfg["data"]["manifest"],
                       segment_seconds=cfg["data"]["segment_seconds"])
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"],
                    shuffle=True, num_workers=4, drop_last=True)

    accel = _maybe_accelerate()
    if accel is not None:
        model, opt, dl = accel.prepare(model, opt, dl)
        device = accel.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in dl:
            batch = batch.to(device)
            recon, _, vq_loss = model(batch)
            # match shapes — encoder/decoder may round T
            T = min(recon.shape[-1], batch.shape[-1])
            recon  = recon[..., :T]
            target = batch[..., :T]

            loss = waveform_l1(recon, target) + stft(recon, target) + 0.25 * vq_loss

            opt.zero_grad()
            if accel is not None:
                accel.backward(loss)
            else:
                loss.backward()
            opt.step()

            step += 1
            if step % cfg["train"]["log_every"] == 0:
                print(f"step={step:>7d}  loss={loss.item():.4f}  vq={vq_loss.item():.4f}")
            if step % cfg["train"]["save_every"] == 0:
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg["model"], "step": step},
                    out / f"step_{step:07d}.pt",
                )


if __name__ == "__main__":
    main()
