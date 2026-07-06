from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloc résiduel tabulaire (style TabResNet). LayerNorm au lieu de BatchNorm pour B=1 en train."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TabResNetBackbone(nn.Module):
    """TabResNet : MLP profond avec connexions résiduelles sur chaque ligne de [B, in_dim].

    LayerNorm (et non BatchNorm1d) pour rester valide avec B=1 en entraînement (dernier batch).
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_blocks: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(x)
        return self.blocks(x)


class HyperWeightGenerator(nn.Module):
    """TabResNet sur x_bin [B, D] ; une tête de poids par échantillon (theta_main, theta_cf) de forme [B, P]."""

    def __init__(
        self,
        input_dim_bin: int,
        num_classes: int,
        num_rules: int,
        cf_hidden_dim: int,
        hidden_dim: int,
        *,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim_bin = input_dim_bin
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim

        self.tab = TabResNetBackbone(input_dim_bin, hidden_dim, n_blocks, dropout)

        main_params = (input_dim_bin * num_rules) + num_rules + (num_rules * num_classes) + num_classes
        cf_in = input_dim_bin + num_classes
        cf_params = (cf_in * cf_hidden_dim) + cf_hidden_dim + (cf_hidden_dim * input_dim_bin) + input_dim_bin

        self.main_head = nn.Linear(hidden_dim, main_params)
        self.cf_head = nn.Linear(hidden_dim, cf_params)

    def forward(self, x_bin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tab(x_bin)
        theta_main = self.main_head(h)
        theta_cf = self.cf_head(h)
        return theta_main, theta_cf
