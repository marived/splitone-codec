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


def test_codec_eval_mode():
    from splitone.model.codec import SplitOneCodec
    codec = SplitOneCodec(base_channels=8, latent_dim=64,
                          strides=(2, 4, 5, 8), n_codebooks=2, codebook_size=64)
    codec.eval()
    import torch
    wav = torch.randn(1, 1, 24000)
    idx = codec.encode(wav)
    out = codec.decode(idx)
    assert out.shape[0] == 1


def test_encode_is_deterministic_in_eval():
    import torch
    from splitone.model.codec import SplitOneCodec
    codec = SplitOneCodec(base_channels=8, latent_dim=64,
                          strides=(2, 4, 5, 8), n_codebooks=2, codebook_size=64)
    codec.eval()
    wav = torch.randn(1, 1, 24000)
    a = codec.encode(wav)
    b = codec.encode(wav)
    assert torch.equal(a, b)
