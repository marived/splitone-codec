"""End-to-end round-trip test on random input."""
import torch

from splitone.model.codec import SplitOneCodec


def test_roundtrip_shape():
    codec = SplitOneCodec(base_channels=8, latent_dim=64,
                          strides=(2, 4, 5, 8), n_codebooks=2, codebook_size=64)
    wav = torch.randn(1, 1, 24000)
    recon, idx, _ = codec(wav)
    # encoder downsamples by 2*4*5*8 = 320
    assert idx.shape[-1] == 24000 // 320
    assert recon.shape[0] == 1
