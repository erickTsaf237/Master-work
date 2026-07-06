"""RuleConEx : HyConEx + HyperLogic + DLBAC en un seul forward interprétable."""

from ruleconex.config import RuleConExConfig
from ruleconex.model import RuleConExForwardPack, RuleConExModel
from ruleconex.utils import (
    counterfactual_report,
    explain_sample,
    extract_rules_from_pack,
    format_rules_text,
    numpy_to_model_input,
)

__all__ = [
    "RuleConExConfig",
    "RuleConExModel",
    "RuleConExForwardPack",
    "RuleConExTrainer",
    "TrainingResult",
    "explain_sample",
    "extract_rules_from_pack",
    "format_rules_text",
    "counterfactual_report",
    "numpy_to_model_input",
]


def __getattr__(name: str):
    """Import paresseux du trainer (évite cycle utils ↔ trainer au chargement)."""
    if name in ("RuleConExTrainer", "TrainingResult"):
        from ruleconex.trainer import RuleConExTrainer, TrainingResult

        return RuleConExTrainer if name == "RuleConExTrainer" else TrainingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
