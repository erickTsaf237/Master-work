from __future__ import annotations

from hyconex_from_scratch.config import TrainConfig
from hyconex_from_scratch.trainer import HyConExTrainer
from hyconex_pure_local.model import HyConExLocalModel


class HyConExLocalTrainer(HyConExTrainer):
    """Entraîne HyConExLocalModel (même boucle que HyConExTrainer)."""

    def _ensure_model(self, input_dim: int, num_classes: int) -> HyConExLocalModel:
        cfg = self.config
        self._num_classes = num_classes
        self.model = HyConExLocalModel(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        return self.model
