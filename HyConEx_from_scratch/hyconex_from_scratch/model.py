from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyConExFromScratch(nn.Module):
    """
    Version from-scratch inspirée du papier HyConEx :
    - encodeur tabulaire -> représentation latente z
    - hypernetwork qui génère un classifieur dynamique dépendant de z
    - générateur de contre-factuels conditionné par la classe cible
    """

    def __init__(self, input_dim: int, num_classes: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.hyper = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * (latent_dim + 1)),
        )

        self.cf_generator = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def dynamic_logits(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        params = self.hyper(z)
        params = params.view(-1, self.num_classes, self.latent_dim + 1)
        w = params[:, :, : self.latent_dim]
        b = params[:, :, self.latent_dim]
        logits = torch.einsum("bcd,bd->bc", w, z) + b
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_logits(x)

    def generate_counterfactual(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        target_onehot = F.one_hot(y_target, num_classes=self.num_classes).float()
        cf_input = torch.cat([z, target_onehot], dim=1)
        delta = self.cf_generator(cf_input)
        x_cf = torch.clamp(x + delta, 0.0, 1.0)
        return x_cf
