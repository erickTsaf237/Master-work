"""HyperGenerator bruit-only, style TabResNet (blocs résiduels comme HyConEx HyperNet)."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.hidden_state_dropout = nn.Dropout(dropout_rate)
        self.residual_dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_dropout(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.hidden_state_dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = out + residual
        return self.gelu(out)


class HyperGenerator(nn.Module):
    """
    ε ~ N(0, I) -> W_hyper de forme (B, D+1, C).
    MLP + 2-3 blocs résiduels (même esprit que HyConEx/hypernetwork.HyperNet).
    """

    def __init__(
        self,
        noise_dim: int,
        nr_features: int,
        nr_classes: int,
        hidden_size: int = 512,
        n_res_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(noise_dim, hidden_size)
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.act = nn.GELU()
        self.drop_in = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_size, dropout_rate) for _ in range(n_res_blocks)]
        )
        out_dim = (nr_features + 1) * nr_classes
        self.output_layer = nn.Linear(hidden_size, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.blocks:
            if hasattr(m, "bn2") and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        """
        eps: (B, noise_dim)
        retourne W_hyper: (B, D+1, C)
        """
        x = eps.view(-1, self.noise_dim)
        x = self.input_layer(x)
        x = self.bn_in(x)
        x = self.act(x)
        x = self.drop_in(x)
        for blk in self.blocks:
            x = blk(x)
        w = self.output_layer(x)
        w = w.view(-1, self.nr_features + 1, self.nr_classes)
        return w
