"""Dataset for paired (path, duration) manifests."""
import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioManifest(Dataset):
    """Reads a TSV: <path>\t<duration_seconds>.

    Crops a random window of `segment_seconds` from each clip.
    """

    def __init__(self, tsv_path, segment_seconds=2.0, sample_rate=24000):
        self.sr = sample_rate
        self.segment = int(segment_seconds * sample_rate)
        self.items = []
        with open(tsv_path) as f:
            for line in f:
                path, dur = line.strip().split("\t")
                if float(dur) * sample_rate < self.segment:
                    continue  # too short
                self.items.append(path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        total = wav.shape[1]
        start = random.randint(0, total - self.segment)
        return wav[:, start:start + self.segment]
