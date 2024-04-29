"""Training loop."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import AudioManifest
from .losses import MultiScaleSTFTLoss, waveform_l1
from .model.codec import SplitOneCodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = SplitOneCodec(**cfg["model"]).cuda()
    opt   = torch.optim.AdamW(model.parameters(), **cfg["optim"])
    stft  = MultiScaleSTFTLoss()
    ds = AudioManifest(cfg["data"]["manifest"],
                       segment_seconds=cfg["data"]["segment_seconds"])
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                    num_workers=4, drop_last=True)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in dl:
            batch = batch.cuda()
            recon, _, vq = model(batch)
            T = min(recon.shape[-1], batch.shape[-1])
            r = recon[..., :T]; t = batch[..., :T]
            l_w  = waveform_l1(r, t)
            l_s  = stft(r, t)
            loss = l_w + l_s + 0.25 * vq
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                print(f"step={step:>7d}  l_w={l_w.item():.4f}  l_s={l_s.item():.4f}  vq={vq.item():.4f}")
            if step % cfg["train"]["save_every"] == 0:
                torch.save({"model": model.state_dict(), "cfg": cfg["model"]},
                           out / f"step_{step:07d}.pt")


if __name__ == "__main__":
    main()
