from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_uid_rid(raw: np.ndarray) -> np.ndarray:
    return raw[:, 2:].astype(np.float32)


def apply_metadata_mask(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = body.shape[1]
    metadata = cols - 4
    x = body[:, :metadata]
    ops = body[:, metadata:cols].astype(np.int64)

    hide_meta_data = max(0, cols - 20)
    umeta_end, rmeta_end = 8, 16
    umeta_hide_end = umeta_end + hide_meta_data
    rmeta_hide_end = rmeta_end + hide_meta_data

    x = np.delete(x, slice(umeta_end, umeta_hide_end), axis=1)
    x = np.delete(x, slice(rmeta_end, rmeta_hide_end), axis=1)
    return x, ops


def joint_op_label(ops: np.ndarray) -> np.ndarray:
    bits = ops.astype(np.int64).clip(0, 1)
    return bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2] + 8 * bits[:, 3]


def load_dlbac_pair(train_path: Path, test_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_raw = np.loadtxt(train_path, dtype=np.float32)
    test_raw = np.loadtxt(test_path, dtype=np.float32)

    train_body = strip_uid_rid(train_raw)
    test_body = strip_uid_rid(test_raw)
    x_tr_raw, ops_tr = apply_metadata_mask(train_body)
    x_te_raw, ops_te = apply_metadata_mask(test_body)

    y_tr = joint_op_label(ops_tr)
    y_te = joint_op_label(ops_te)

    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")

    enc.fit(x_tr_raw)
    x_tr = enc.transform(x_tr_raw).astype(np.float32)
    x_te = enc.transform(x_te_raw).astype(np.float32)
    return x_tr, y_tr.astype(np.int64), x_te, y_te.astype(np.int64)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 128
    hidden_dim: int = 256
    cf_lambda: float = 0.4
    l1_lambda: float = 0.02
    l2_lambda: float = 0.005


class HyConExFromScratch(nn.Module):
    """
    Version from-scratch inspirée du papier:
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


def sample_alternative_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    noise = torch.randint(low=1, high=num_classes, size=y.shape, device=y.device)
    return (y + noise) % num_classes


def evaluate(model: HyConExFromScratch, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(proba, axis=1)

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "classification_report": classification_report(y, y_pred, output_dict=True, digits=4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    try:
        metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    except Exception as exc:
        metrics["auroc_ovr"] = None
        metrics["auroc_error"] = str(exc)
    return metrics


def predict_outputs(model: HyConExFromScratch, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(proba, axis=1)
    return y_pred, proba


def evaluate_counterfactuals(
    model: HyConExFromScratch,
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    max_samples: int = 4000,
) -> dict:
    model.eval()
    idx = np.arange(x.shape[0])
    if x.shape[0] > max_samples:
        idx = np.random.choice(idx, size=max_samples, replace=False)

    x_sub = torch.tensor(x[idx], dtype=torch.float32, device=device)
    y_sub = torch.tensor(y[idx], dtype=torch.long, device=device)
    y_target = sample_alternative_targets(y_sub, num_classes)

    with torch.no_grad():
        x_cf = model.generate_counterfactual(x_sub, y_target)
        y_cf_pred = torch.argmax(model(x_cf), dim=1)

    validity = (y_cf_pred == y_target).float().mean().item()
    l1 = torch.norm(x_cf - x_sub, p=1, dim=1).mean().item()
    changed = ((x_cf - x_sub).abs() > 1e-3).float().sum(dim=1).mean().item()
    return {
        "validity_cf": float(validity),
        "proximity_l1_mean": float(l1),
        "changed_features_mean": float(changed),
        "n_evaluated": int(x_sub.shape[0]),
    }


def collect_counterfactual_examples(
    model: HyConExFromScratch,
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    n_examples: int,
) -> list[dict]:
    model.eval()
    n_examples = min(n_examples, x.shape[0])
    idx = np.random.choice(np.arange(x.shape[0]), size=n_examples, replace=False)

    x_sub = torch.tensor(x[idx], dtype=torch.float32, device=device)
    y_sub = torch.tensor(y[idx], dtype=torch.long, device=device)
    y_target = sample_alternative_targets(y_sub, num_classes)

    with torch.no_grad():
        x_cf = model.generate_counterfactual(x_sub, y_target)
        y_orig_pred = torch.argmax(model(x_sub), dim=1)
        y_cf_pred = torch.argmax(model(x_cf), dim=1)

    x_sub_np = x_sub.cpu().numpy()
    x_cf_np = x_cf.cpu().numpy()
    y_true_np = y_sub.cpu().numpy()
    y_target_np = y_target.cpu().numpy()
    y_orig_pred_np = y_orig_pred.cpu().numpy()
    y_cf_pred_np = y_cf_pred.cpu().numpy()

    rows: list[dict] = []
    for i in range(n_examples):
        delta = x_cf_np[i] - x_sub_np[i]
        changed_idx = np.where(np.abs(delta) > 1e-3)[0].tolist()
        rows.append(
            {
                "sample_index": int(idx[i]),
                "y_true": int(y_true_np[i]),
                "y_pred_orig": int(y_orig_pred_np[i]),
                "y_target_cf": int(y_target_np[i]),
                "y_pred_cf": int(y_cf_pred_np[i]),
                "valid_cf": bool(y_cf_pred_np[i] == y_target_np[i]),
                "l1_distance": float(np.abs(delta).sum()),
                "changed_features_count": int(len(changed_idx)),
                "changed_features_idx": changed_idx,
                "x_orig": x_sub_np[i].astype(float).tolist(),
                "x_cf": x_cf_np[i].astype(float).tolist(),
            }
        )
    return rows


def train(config: TrainConfig, run_name: str, cf_preview_count: int) -> Path:
    dlbac_dir = PROJECT_ROOT / "DlbacAlpha-main" / "dataset" / "synthetic" / "u4k-r4k-auth11k"
    train_path = dlbac_dir / "train_u4k-r4k-auth11k.sample"
    test_path = dlbac_dir / "test_u4k-r4k-auth11k.sample"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Fichiers DLBaC introuvables sous {dlbac_dir}")

    set_seed(config.seed)
    x_train_full, y_train_full, x_test, y_test = load_dlbac_pair(train_path, test_path)
    num_classes = int(np.max(y_train_full) + 1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=config.seed, stratify=y_train_full
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyConExFromScratch(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_target = sample_alternative_targets(yb, num_classes)

            logits = model(xb)
            ce = F.cross_entropy(logits, yb)

            x_cf = model.generate_counterfactual(xb, y_target)
            logits_cf = model(x_cf)
            ce_cf = F.cross_entropy(logits_cf, y_target)
            delta = x_cf - xb
            l1 = delta.abs().mean()
            l2 = (delta ** 2).mean()

            loss = ce + config.cf_lambda * ce_cf + config.l1_lambda * l1 + config.l2_lambda * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.shape[0]

        val_metrics = evaluate(model, x_val, y_val, device)
        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avg_loss = running_loss / len(train_ds)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "val_accuracy": float(val_acc),
                "best_val_accuracy": float(best_val_acc),
            }
        )
        print(
            f"[Epoch {epoch:03d}/{config.epochs}] "
            f"loss={avg_loss:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    y_test_pred, y_test_proba = predict_outputs(model, x_test, device)
    test_metrics = evaluate(model, x_test, y_test, device)
    cf_metrics = evaluate_counterfactuals(model, x_test, y_test, num_classes, device)
    cf_examples = collect_counterfactual_examples(
        model=model,
        x=x_test,
        y=y_test,
        num_classes=num_classes,
        device=device,
        n_examples=cf_preview_count,
    )

    outputs = {
        "dataset": "DLBAC_u4k-r4k-auth11k",
        "num_features": int(x_train.shape[1]),
        "num_classes": int(num_classes),
        "device": str(device),
        "config": config.__dict__,
        "val_best_accuracy": float(best_val_acc),
        "test": test_metrics,
        "counterfactuals_test": cf_metrics,
        "counterfactual_examples_preview": cf_examples,
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{run_name}_metrics.json"
    history_csv_path = out_dir / f"{run_name}_learning_curve.csv"
    test_pred_csv_path = out_dir / f"{run_name}_test_predictions.csv"
    cf_json_path = out_dir / f"{run_name}_counterfactuals_preview.json"
    cf_csv_path = out_dir / f"{run_name}_counterfactuals_preview.csv"
    ckpt_path = out_dir / f"{run_name}_model.pt"
    metrics_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    cf_json_path.write_text(json.dumps(cf_examples, indent=2), encoding="utf-8")
    with cf_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "y_true",
                "y_pred_orig",
                "y_target_cf",
                "y_pred_cf",
                "valid_cf",
                "l1_distance",
                "changed_features_count",
                "changed_features_idx",
            ],
        )
        writer.writeheader()
        for row in cf_examples:
            writer.writerow(
                {
                    "sample_index": row["sample_index"],
                    "y_true": row["y_true"],
                    "y_pred_orig": row["y_pred_orig"],
                    "y_target_cf": row["y_target_cf"],
                    "y_pred_cf": row["y_pred_cf"],
                    "valid_cf": row["valid_cf"],
                    "l1_distance": row["l1_distance"],
                    "changed_features_count": row["changed_features_count"],
                    "changed_features_idx": json.dumps(row["changed_features_idx"]),
                }
            )
    with history_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_accuracy", "best_val_accuracy"],
        )
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)
    with test_pred_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_index", "y_true", "y_pred"] + [f"proba_class_{c}" for c in range(num_classes)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(y_test.shape[0]):
            row = {
                "sample_index": i,
                "y_true": int(y_test[i]),
                "y_pred": int(y_test_pred[i]),
            }
            for c in range(num_classes):
                row[f"proba_class_{c}"] = float(y_test_proba[i, c])
            writer.writerow(row)
    torch.save(model.state_dict(), ckpt_path)

    print("\n=== Resultats HyConEx from scratch / DLBaC ===")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {outputs['test']['accuracy']:.4f}")
    if outputs["test"]["auroc_ovr"] is not None:
        print(f"Test AUROC OvR: {outputs['test']['auroc_ovr']:.4f}")
    print(
        "CF validity: "
        f"{outputs['counterfactuals_test']['validity_cf']:.4f} | "
        "CF proximity L1: "
        f"{outputs['counterfactuals_test']['proximity_l1_mean']:.4f}"
    )
    print(f"CF preview JSON: {cf_json_path}")
    print(f"CF preview CSV: {cf_csv_path}")
    print(f"Learning curve CSV: {history_csv_path}")
    print(f"Test predictions CSV: {test_pred_csv_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {ckpt_path}")

    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Implementation from scratch de HyConEx sur DLBaC.")
    parser.add_argument("--run-name", type=str, default="hyconex_from_scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cf-lambda", type=float, default=0.4)
    parser.add_argument("--l1-lambda", type=float, default=0.02)
    parser.add_argument("--l2-lambda", type=float, default=0.005)
    parser.add_argument("--cf-preview-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cf_lambda=args.cf_lambda,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
    )
    train(config, run_name=args.run_name, cf_preview_count=args.cf_preview_count)


if __name__ == "__main__":
    main()
