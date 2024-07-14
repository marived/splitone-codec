"""Round-trip shape tests for the RVQ stack."""
import torch

from splitone.model.rvq import ResidualVQ


def test_residual_vq_shapes():
    rvq = ResidualVQ(dim=128, n_codebooks=4, codebook_size=256)
    x = torch.randn(2, 128, 30)
    q, idx, _ = rvq(x)
    assert q.shape == x.shape
    assert idx.shape == (2, 4, 30)
    assert idx.min() >= 0
    assert idx.max() < 256


def test_residual_vq_decode():
    rvq = ResidualVQ(dim=64, n_codebooks=3, codebook_size=128)
    rvq.eval()
    idx = torch.randint(0, 128, (1, 3, 12))
    out = rvq.decode(idx)
    assert out.shape == (1, 64, 12)
