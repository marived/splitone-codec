"""Very rough dataset."""
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioManifest(Dataset):
    def __init__(self, tsv_path, segment=24000, sample_rate=24000):
        self.sr = sample_rate
        self.segment = segment
        self.items = []
        with open(tsv_path) as f:
            for line in f:
                p, d = line.strip().split("\t")
                self.items.append(p)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.items[idx])
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if wav.shape[1] < self.segment:
            pad = self.segment - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        start = random.randint(0, wav.shape[1] - self.segment)
        return wav[:, start:start + self.segment]
