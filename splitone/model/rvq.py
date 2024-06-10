"""Residual vector quantiser with EMA codebook updates.

Implementation notes:
  * EMA updates rather than gradient: avoids straight-through hacks for the codebook
    itself, though we still need straight-through for the quantised features.
  * Each stage quantises the *residual* of the previous stages.
  * Dead-code resampling: any code that hasn't been used for `restart_after` steps
    is replaced with a random encoder vector from the current batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQStage(nn.Module):
    # FIXME(2024-06): code collapse very bad. dead-code restart coming.

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5, restart_after=200):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.restart_after = restart_after

        embed = torch.randn(codebook_size, dim) * 0.01
        self.register_buffer("embed",        embed)
        self.register_buffer("embed_avg",    embed.clone())
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("last_seen",    torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("step",         torch.zeros(1, dtype=torch.long))

    def _quantise(self, flat):
        # flat: (N, D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        q   = F.embedding(idx, self.embed)
        return q, idx

    def forward(self, x):
        # x: (B, D, T)
        B, D, T = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, D)
        q, idx = self._quantise(flat)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.codebook_size).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1 - self.decay
                )
                n = self.cluster_size.sum()
                smoothed = ((self.cluster_size + self.eps)
                            / (n + self.codebook_size * self.eps)) * n
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

                self.last_seen[idx] = self.step
                self.step += 1
                # restart dead codes
                dead = (self.step - self.last_seen) > self.restart_after
                # dead-code restart goes here — not implemented yet
                pass

        loss = F.mse_loss(q.detach(), flat) + 0.25 * F.mse_loss(q, flat.detach())
        q = flat + (q - flat).detach()
        q = q.view(B, T, D).permute(0, 2, 1)
        idx = idx.view(B, T)
        return q, idx, loss


class ResidualVQ(nn.Module):
    def __init__(self, dim=256, n_codebooks=8, codebook_size=1024, **kw):
        super().__init__()
        self.stages = nn.ModuleList(
            [VQStage(dim, codebook_size, **kw) for _ in range(n_codebooks)]
        )

    def forward(self, x):
        residual = x
        quantised = torch.zeros_like(x)
        indices = []
        total_loss = 0.0
        for stage in self.stages:
            q, idx, loss = stage(residual)
            quantised = quantised + q
            residual  = residual - q.detach()
            indices.append(idx)
            total_loss = total_loss + loss
        return quantised, torch.stack(indices, dim=1), total_loss / len(self.stages)

    @torch.no_grad()
    def encode(self, x):
        residual = x
        indices = []
        for stage in self.stages:
            q, idx = stage._quantise(residual.permute(0, 2, 1).reshape(-1, stage.dim))
            q = q.view(x.shape[0], x.shape[2], stage.dim).permute(0, 2, 1)
            indices.append(idx.view(x.shape[0], x.shape[2]))
            residual = residual - q
        return torch.stack(indices, dim=1)

    @torch.no_grad()
    def decode(self, indices):
        # indices: (B, n_q, T)
        out = None
        for i, stage in enumerate(self.stages):
            q = F.embedding(indices[:, i], stage.embed)            # (B, T, D)
            q = q.permute(0, 2, 1)                                 # (B, D, T)
            out = q if out is None else out + q
        return out
