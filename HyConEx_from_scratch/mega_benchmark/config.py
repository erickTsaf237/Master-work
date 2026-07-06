from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MegaBenchmarkConfig:
    seed: int = 42
    retrain: bool = False
    skip_amazon: bool = False
    verbose: bool = True
    results_dir: str = "results/mega_comparison"

    # Sous-ensembles (vide = tout)
    dataset_sources: list[str] = field(default_factory=lambda: ["dlbac", "hyconex", "hyperlogic", "local"])
    models: list[str] = field(
        default_factory=lambda: [
            "ruleconex",
            "hyconex_local",
            "hyperlogic",
            "hyconex_hyperlogic",
            "dlbac_alpha",
            "tabresnet_dlbac",
            "mlp",
            "rf",
            "decision_tree",
            "svm",
        ]
    )
    focus_dataset: str | None = None

    # Epochs réduits pour benchmark (override par modèle)
    neural_epochs_dlbac: int = 25
    neural_epochs_amazon: int = 15
    neural_epochs_tabular: int = 30
    sklearn_fast: bool = True
    use_gpu_mlp: bool = True
    dlbac_alpha_epochs: int | None = None  # None = défaut papier (60/30) ou neural_epochs_dlbac
