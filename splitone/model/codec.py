"""Top-level SplitOneCodec module."""
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .rvq import ResidualVQ


class SplitOneCodec(nn.Module):
    def __init__(self, base_channels=32, latent_dim=256, strides=(2, 4, 5, 8),
                 n_codebooks=8, codebook_size=1024):
        super().__init__()
        self.encoder = Encoder(base_channels=base_channels, strides=strides,
                               latent_dim=latent_dim)
        self.rvq     = ResidualVQ(dim=latent_dim, n_codebooks=n_codebooks,
                                  codebook_size=codebook_size)
        # decoder mirrors encoder strides in reverse
        self.decoder = Decoder(latent_dim=latent_dim, base_channels=base_channels,
                               strides=tuple(reversed(strides)))
        self.sample_rate = 24000
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

    def forward(self, wav):
        z = self.encoder(wav)
        q, indices, vq_loss = self.rvq(z)
        recon = self.decoder(q)
        return recon, indices, vq_loss

    @torch.no_grad()
    def encode(self, wav):
        z = self.encoder(wav)
        return self.rvq.encode(z)

    @torch.no_grad()
    def decode(self, indices):
        z = self.rvq.decode(indices)
        return self.decoder(z)

    @classmethod
    def from_pretrained(cls, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        cfg = state.get("cfg", {})
        model = cls(**cfg)
        model.load_state_dict(state["model"])
        return model
