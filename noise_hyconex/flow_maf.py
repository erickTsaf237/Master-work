"""MAF via nflows (concaténation x||c pour conditionnement si l'API n'a pas context_features)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _require_nflows():
    try:
        from nflows.flows import MaskedAutoregressiveFlow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Le paquet 'nflows' est requis pour noise_hyconex.flow_maf. "
            "Installez-le avec: pip install nflows"
        ) from e


class ConditionalMAF(nn.Module):
    """
    Estime la densité sur z = [x || c] (c scalaire = classe).
    log_prob(x, context) = log p(z) avec z = concat(x, context).
    """

    def __init__(
        self,
        features: int,
        hidden_features: int = 128,
        context_features: int = 1,
        num_layers: int = 4,
        num_blocks_per_layer: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        _require_nflows()
        from nflows.flows import MaskedAutoregressiveFlow as _MAF

        self.data_dim = features
        self.context_dim = context_features
        self.in_dim = features + context_features
        self.device = device or torch.device("cpu")
        self.model = _MAF(
            features=self.in_dim,
            hidden_features=hidden_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            use_residual_blocks=True,
            use_random_masks=False,
            use_random_permutations=False,
            activation=F.relu,
            dropout_probability=0.0,
            batch_norm_within_layers=False,
            batch_norm_between_layers=False,
        )

    def _pack(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 1:
            context = context.unsqueeze(-1)
        return torch.cat([x, context.float()], dim=-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self._pack(x, context)
        return self.model.log_prob(inputs=z)

    def fit_quick(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: torch.device,
    ) -> None:
        self.to(device)
        self.train()
        y_col = y.float().view(-1, 1)
        Z = torch.cat([X.float(), y_col], dim=-1)
        ds = TensorDataset(Z)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for (zb,) in loader:
                zb = zb.to(device)
                opt.zero_grad()
                ll = self.model.log_prob(inputs=zb)
                loss = -ll.mean()
                loss.backward()
                opt.step()
        self.eval()
