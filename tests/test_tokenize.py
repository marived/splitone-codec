"""Sanity check for the flatten/unflatten round-trip."""
import torch

from splitone.tokenize import flatten_rvq, unflatten_rvq


def test_flatten_roundtrip():
    idx = torch.randint(0, 1024, (2, 8, 30))
    flat = flatten_rvq(idx, vocab_size=1024)
    assert flat.shape == (2, 30 * 8)
    back = unflatten_rvq(flat, n_q=8, vocab_size=1024)
    assert torch.equal(back, idx)
