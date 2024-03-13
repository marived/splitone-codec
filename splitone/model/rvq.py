"""Single-stage vector quantiser, EMA codebook.

Very rough — known issues with codebook collapse on small batches.
TODO: switch to residual stack once this trains end-to-end.
"""
import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, dim=256, codebook_size=1024, decay=0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        embed = torch.randn(codebook_size, dim) * 0.01
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        # x: (B, D, T) -> reshape
        flat = x.permute(0, 2, 1).reshape(-1, self.dim)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx  = dist.argmin(1)
        q    = self.embed[idx].view(x.shape[0], x.shape[2], self.dim).permute(0, 2, 1)
        loss = (q.detach() - x).pow(2).mean() + 0.25 * (q - x.detach()).pow(2).mean()
        q    = x + (q - x).detach()  # straight-through
        return q, idx, loss


    @torch.no_grad()
    def usage(self):
        return self.cluster_size / (self.cluster_size.sum() + 1e-9)

    @torch.no_grad()
    def last_seen_age(self, step):
        # placeholder, we don't track this yet — soon
        return torch.zeros(self.codebook_size)
