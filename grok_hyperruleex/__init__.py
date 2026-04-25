"""HyperRuleEx : modèle unifié HyperLogic + contre-factuels (rapport Grok)."""

from grok_hyperruleex.model import HyperRuleEx, HyperRuleExConfig
from grok_hyperruleex.preprocessing import BinarizerConfig, TabularBinarizer
from grok_hyperruleex.train import TrainConfig, train_model

__all__ = [
    "HyperRuleEx",
    "HyperRuleExConfig",
    "TabularBinarizer",
    "BinarizerConfig",
    "TrainConfig",
    "train_model",
]
