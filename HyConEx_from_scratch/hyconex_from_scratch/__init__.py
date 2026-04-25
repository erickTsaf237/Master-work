"""
HyConEx (implémentation from-scratch) — API simple.

Usage typique::

    from hyconex_from_scratch import HyConExTrainer, TrainConfig

    cfg = TrainConfig(epochs=100, lr=1e-3)
    trainer = HyConExTrainer(config=cfg)
    trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    metrics = trainer.evaluate(X_test, y_test)
"""

from hyconex_from_scratch.config import TrainConfig
from hyconex_from_scratch.model import HyConExFromScratch
from hyconex_from_scratch.trainer import HyConExTrainer, train

__all__ = [
    "TrainConfig",
    "HyConExFromScratch",
    "HyConExTrainer",
    "train",
]
