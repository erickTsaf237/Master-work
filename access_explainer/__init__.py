"""Dual HyConEx + HyperLogic (DR-Net) explainer for access-control style tabular tasks."""

from .dataset_abac import SyntheticAccessDataset, load_synthetic_access_arrays
from .dual_model import DualExplainModel
from .explain import (
    ExplanationPersona,
    build_explanation_bundle,
    counterfactual_to_target,
    format_for_persona,
)
from .hyperlogic_drnet import HyperLogicBranch, extract_if_then_rules
from .preprocessing import AccessControlPreprocessor, AccessFeatureSpec

__all__ = [
    "DualExplainModel",
    "HyperLogicBranch",
    "AccessControlPreprocessor",
    "AccessFeatureSpec",
    "SyntheticAccessDataset",
    "load_synthetic_access_arrays",
    "extract_if_then_rules",
    "build_explanation_bundle",
    "format_for_persona",
    "ExplanationPersona",
    "counterfactual_to_target",
]
