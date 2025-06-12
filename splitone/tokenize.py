"""Helpers for feeding RVQ tokens into a HuggingFace `transformers` LM."""
import torch


def flatten_rvq(indices, vocab_size):
    """Flatten (B, n_q, T) -> (B, n_q * T) with codebook-offset trick.

    Codebook k's token IDs occupy [k * vocab_size, (k + 1) * vocab_size).
    """
    B, n_q, T = indices.shape
    offsets = (torch.arange(n_q, device=indices.device) * vocab_size).view(1, n_q, 1)
    shifted = (indices + offsets).long()
    # interleave so the first quantiser comes first at each timestep
    return shifted.permute(0, 2, 1).reshape(B, T * n_q)


def unflatten_rvq(flat, n_q, vocab_size):
    """Inverse of flatten_rvq."""
    B, L = flat.shape
    assert L % n_q == 0, "flat length must be divisible by n_q"
    T = L // n_q
    shifted = flat.view(B, T, n_q).permute(0, 2, 1)
    offsets = (torch.arange(n_q, device=flat.device) * vocab_size).view(1, n_q, 1)
    return shifted - offsets
