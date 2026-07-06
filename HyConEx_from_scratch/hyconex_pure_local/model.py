from __future__ import annotations

from dataclasses import dataclass

import torch

from hyconex_from_scratch.model import HyConExFromScratch


@dataclass
class LocalHypernetPack:
    """Décomposition W(z)·z + b(z) pour un batch."""

    z: torch.Tensor
    weights: torch.Tensor
    bias: torch.Tensor
    contributions: torch.Tensor
    logits: torch.Tensor


class HyConExLocalModel(HyConExFromScratch):
    """HyConEx pur avec décomposition explicite du classifieur local."""

    def local_hypernet_pack(self, x: torch.Tensor) -> LocalHypernetPack:
        z = self.encoder(x)
        params = self.hyper(z)
        params = params.view(-1, self.num_classes, self.latent_dim + 1)
        w = params[:, :, : self.latent_dim]
        b = params[:, :, self.latent_dim]
        contributions = w * z.unsqueeze(1)
        logits = contributions.sum(dim=2) + b
        return LocalHypernetPack(z=z, weights=w, bias=b, contributions=contributions, logits=logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_hypernet_pack(x).logits
