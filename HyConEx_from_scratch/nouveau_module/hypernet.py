from __future__ import annotations

import torch
import torch.nn as nn


class HyperWeightGenerator(nn.Module):
    """Génère 100% des poids du main network et de la tête CF depuis des stats de X_bin."""

    def __init__(self, input_dim_bin: int, num_classes: int, num_rules: int, cf_hidden_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim_bin = input_dim_bin
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim

        stats_dim = input_dim_bin * 2
        self.backbone = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        main_params = (input_dim_bin * num_rules) + num_rules + (num_rules * num_classes) + num_classes
        cf_in = input_dim_bin + num_classes
        cf_params = (cf_in * cf_hidden_dim) + cf_hidden_dim + (cf_hidden_dim * input_dim_bin) + input_dim_bin

        self.main_head = nn.Linear(hidden_dim, main_params)
        self.cf_head = nn.Linear(hidden_dim, cf_params)

    def _batch_stats(self, x_bin: torch.Tensor) -> torch.Tensor:
        mean = x_bin.mean(dim=0)
        std = x_bin.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)

    def forward(self, x_bin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self._batch_stats(x_bin)
        h = self.backbone(stats)
        theta_main = self.main_head(h)
        theta_cf = self.cf_head(h)
        return theta_main, theta_cf
