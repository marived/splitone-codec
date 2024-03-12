import torch

from splitone.model.rvq import VectorQuantizer


def test_vq_shapes():
    vq = VectorQuantizer(dim=64, codebook_size=128)
    x = torch.randn(2, 64, 30)
    q, idx, _ = vq(x)
    assert q.shape == x.shape
    assert idx.shape == (2 * 30,)  # flat
