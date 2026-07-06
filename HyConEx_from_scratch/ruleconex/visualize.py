"""Visualisations RuleConEx."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from ruleconex.model import RuleConExModel
from ruleconex.utils import explain_sample, feature_importances_from_pack


def plot_training_history(history: list[dict[str, float]], out_path: Path | str | None = None) -> plt.Figure:
    epochs = [int(h["epoch"]) for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train loss")
    if not np.isnan(history[0].get("val_loss", float("nan"))):
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="val CE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Courbes de perte")

    axes[1].plot(epochs, [h["train_accuracy"] for h in history], label="train acc")
    if not np.isnan(history[0].get("val_accuracy", float("nan"))):
        axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Courbes d'accuracy")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    *,
    title: str = "Matrice de confusion",
    out_path: Path | str | None = None,
) -> plt.Figure:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    tick = np.arange(len(labels))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("Vrai")
    ax.set_xlabel("Prédit")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "Courbes ROC et PR (classe positive = grant)",
    out_path: Path | str | None = None,
) -> plt.Figure | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Faux positifs")
    axes[0].set_ylabel("Vrais positifs")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="steelblue", lw=2, label=f"AUC-PR = {pr_auc:.3f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Rappel")
    axes[1].set_ylabel("Précision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(loc="lower left")

    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_metrics_table(
    metrics_by_split: dict[str, dict[str, float]],
    *,
    title: str = "Métriques RuleConEx par jeu",
    out_path: Path | str | None = None,
) -> plt.Figure:
    keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc"]
    splits = list(metrics_by_split.keys())
    data = np.array([[metrics_by_split[s].get(k, float("nan")) for k in keys] for s in splits])

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(splits) + 2)))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["Accuracy", "F1 macro", "Précision", "Rappel", "AUC"], rotation=25, ha="right")
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(splits)
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "—" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_baseline_comparison(
    rows: list[tuple[str, float, float, float]],
    *,
    title: str = "Comparaison des modèles (jeu de test)",
    out_path: Path | str | None = None,
) -> plt.Figure:
    names = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    f1 = [r[2] for r in rows]
    auc_vals = [r[3] for r in rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 4.5))
    ax.bar(x - w, acc, width=w, label="Accuracy", color="#4c72b0")
    ax.bar(x, f1, width=w, label="F1 macro", color="#55a868")
    ax.bar(x + w, auc_vals, width=w, label="AUC", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "Balayage du seuil grant/deny",
    out_path: Path | str | None = None,
) -> plt.Figure | None:
    from nouveau_module.binary_metrics import predict_with_grant_threshold

    y_true = np.asarray(y_true, dtype=np.int64)
    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        return None

    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    thresholds = np.linspace(0.05, 0.95, 91)
    deny_f1, deny_rec, bal_acc, acc = [], [], [], []
    for t in thresholds:
        pred = predict_with_grant_threshold(y_proba, t)
        deny_f1.append(f1_score(y_true, pred, pos_label=0, zero_division=0))
        deny_rec.append(recall_score(y_true, pred, pos_label=0, zero_division=0))
        bal_acc.append(balanced_accuracy_score(y_true, pred))
        acc.append((pred == y_true).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, deny_f1, label="deny F1", lw=2)
    ax.plot(thresholds, deny_rec, label="deny recall", lw=1.5, ls="--")
    ax.plot(thresholds, bal_acc, label="balanced acc", lw=1.5, ls=":")
    ax.plot(thresholds, acc, label="accuracy", lw=1.5, alpha=0.8)
    ax.set_xlabel("Seuil P(grant)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_counterfactual_metrics(
    cf_metrics: dict[str, float],
    *,
    title: str = "Métriques contrefactuelles",
    out_path: Path | str | None = None,
) -> plt.Figure:
    labels = {
        "validity_cf": "Validité CF",
        "flip_success_rate": "Succès flip",
        "changed_features_mean": "Features modifiées (moy.)",
        "proximity_l1_mean": "Proximité L1 (moy.)",
    }
    keys = [k for k in labels if k in cf_metrics]
    vals = [cf_metrics[k] for k in keys]
    names = [labels[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color=["#4c72b0", "#55a868", "#dd8452", "#8172b3"][: len(vals)])
    ax.set_ylabel("Valeur")
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_importance_heatmap(
    model: RuleConExModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_samples: int = 40,
    device: torch.device | None = None,
    out_path: Path | str | None = None,
) -> plt.Figure:
    device = device or next(model.parameters()).device
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    Xs = torch.tensor(X[idx], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        pack = model.forward_pack(Xs)
        imp = pack.input_importance.mean(dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(imp, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Features (top affichées)")
    ax.set_ylabel("Échantillons")
    ax.set_title("Heatmap des importances locales RuleConEx")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_tsne_embeddings(
    model: RuleConExModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int = 800,
    device: torch.device | None = None,
    out_path: Path | str | None = None,
) -> plt.Figure:
    device = device or next(model.parameters()).device
    idx = np.random.choice(len(X), size=min(max_points, len(X)), replace=False)
    Xs = torch.tensor(X[idx], dtype=torch.float32, device=device)
    ys = y[idx]

    model.eval()
    with torch.no_grad():
        pack = model.forward_pack(Xs)
        z = pack.z.cpu().numpy()

    z2 = TSNE(n_components=2, perplexity=min(30, len(z) - 1), random_state=42).fit_transform(z)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(z2[:, 0], z2[:, 1], c=ys, cmap="tab10", s=12, alpha=0.75)
    ax.set_title("t-SNE des embeddings RuleConEx (z)")
    plt.colorbar(scatter, ax=ax, label="classe")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_rules_bar(rules: list[dict[str, Any]], out_path: Path | str | None = None) -> plt.Figure:
    if not rules:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Aucune règle extraite", ha="center")
        return fig

    labels = [f"R{r['rule_id']}" for r in rules[:10]]
    scores = [r["score"] for r in rules[:10]]
    targets = [r["then_class"] for r in rules[:10]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, scores, color="steelblue")
    ax.set_ylabel("Score")
    ax.set_title("Top règles IF-THEN extraites")
    for bar, t in zip(bars, targets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), t, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def render_explanation_figure(
    report,
    out_path: Path | str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    names = list(report.probabilities.keys())
    vals = list(report.probabilities.values())
    axes[0].bar(names, vals, color=["#c44e52", "#55a868"][: len(names)])
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Prédiction : {report.prediction_label}")

    imp_names = [t[0] for t in report.top_importances[:10]]
    imp_vals = [t[1] for t in report.top_importances[:10]]
    axes[1].barh(imp_names[::-1], imp_vals[::-1], color="coral")
    axes[1].set_title("Top importances")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig
