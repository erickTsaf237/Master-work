"""
Préparation des jeux de données DLBAC (synthétiques + Amazon réels).

Reproduit le pipeline du papier DLBACα / train_hyconex_from_scratch_dlbac.py :
- suppression uid/rid ;
- masquage des métadonnées au-delà des 8 premières user/resource ;
- encodage one-hot des métadonnées (ajusté sur le train) ;
- étiquettes : jointure des 4 opérations (synthétique) ou accès binaire (Amazon).

Usage :
    python prepare_dlbac_datasets.py --all
    python prepare_dlbac_datasets.py --dataset u4k-r4k-auth11k amazon1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DLBAC_ROOT = PROJECT_ROOT / "DlbacAlpha-main" / "dataset"
DEFAULT_OUT_DIR = ROOT / "data" / "dlbac_prepared"

LabelMode = Literal["joint_ops", "binary_access"]


@dataclass(frozen=True)
class DLBACDatasetSpec:
    name: str
    kind: Literal["synthetic", "real_world"]
    train_path: Path | None
    test_path: Path
    label_mode: LabelMode
    num_ops: int

    @property
    def has_train(self) -> bool:
        return self.train_path is not None and self.train_path.is_file()


@dataclass
class DLBACPreparedSplits:
    name: str
    kind: str
    label_mode: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_features: int
    num_classes: int
    class_names: list[str]

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{self.name}.npz"
        np.savez_compressed(
            npz_path,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            x_test=self.x_test,
            y_test=self.y_test,
        )
        meta = {
            "name": self.name,
            "kind": self.kind,
            "label_mode": self.label_mode,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "shapes": {
                "train": list(self.x_train.shape),
                "val": list(self.x_val.shape),
                "test": list(self.x_test.shape),
            },
        }
        meta_path = out_dir / f"{self.name}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return npz_path


def strip_uid_rid(raw: np.ndarray) -> np.ndarray:
    return raw[:, 2:].astype(np.float32)


def apply_metadata_mask(body: np.ndarray, num_ops: int) -> tuple[np.ndarray, np.ndarray]:
    cols = body.shape[1]
    metadata = cols - num_ops
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


def binary_access_label(ops: np.ndarray) -> np.ndarray:
    return ops[:, 0].astype(np.int64)


def _fit_onehot_encoder(x_train_raw: np.ndarray) -> OneHotEncoder:
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    enc.fit(x_train_raw)
    return enc


def encode_features(enc: OneHotEncoder, x_raw: np.ndarray) -> np.ndarray:
    return enc.transform(x_raw).astype(np.float32)


def load_raw_pair(
    train_path: Path | None,
    test_path: Path,
    label_mode: LabelMode,
    num_ops: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_raw = np.loadtxt(test_path, dtype=np.float32)
    test_body = strip_uid_rid(test_raw)
    x_te_raw, ops_te = apply_metadata_mask(test_body, num_ops=num_ops)

    if label_mode == "joint_ops":
        y_te = joint_op_label(ops_te)
    else:
        y_te = binary_access_label(ops_te)

    if train_path is not None:
        train_raw = np.loadtxt(train_path, dtype=np.float32)
        train_body = strip_uid_rid(train_raw)
        x_tr_raw, ops_tr = apply_metadata_mask(train_body, num_ops=num_ops)
        if label_mode == "joint_ops":
            y_tr = joint_op_label(ops_tr)
        else:
            y_tr = binary_access_label(ops_tr)
    else:
        # Pas de train fourni : on réutilise le test comme placeholder (évaluation seule).
        x_tr_raw, y_tr = x_te_raw.copy(), y_te.copy()

    return x_tr_raw, y_tr.astype(np.int64), x_te_raw, y_te.astype(np.int64)


def class_names_for_labels(y: np.ndarray, label_mode: LabelMode) -> list[str]:
    classes = sorted(int(c) for c in np.unique(y))
    if label_mode == "binary_access":
        mapping = {0: "deny", 1: "grant"}
        return [mapping.get(c, f"class_{c}") for c in classes]
    return [f"ops_pattern_{c}" for c in classes]


def prepare_dataset(
    spec: DLBACDatasetSpec,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DLBACPreparedSplits:
    x_tr_raw, y_tr, x_te_raw, y_te = load_raw_pair(
        spec.train_path,
        spec.test_path,
        spec.label_mode,
        spec.num_ops,
    )

    if spec.has_train:
        enc = _fit_onehot_encoder(x_tr_raw)
        x_train_full = encode_features(enc, x_tr_raw)
        x_test = encode_features(enc, x_te_raw)
        y_train_full, y_test = y_tr, y_te

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_full,
        )
    else:
        enc = _fit_onehot_encoder(x_te_raw)
        x_test = encode_features(enc, x_te_raw)
        y_test = y_te
        x_train = x_test[:0]
        y_train = y_test[:0]
        x_val = x_test[:0]
        y_val = y_test[:0]

    num_classes = int(max(y_test.max(initial=0), y_train.max(initial=0)) + 1)
    names = class_names_for_labels(
        np.concatenate([y_train, y_test]) if y_train.size else y_test,
        spec.label_mode,
    )

    return DLBACPreparedSplits(
        name=spec.name,
        kind=spec.kind,
        label_mode=spec.label_mode,
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        num_features=int(x_test.shape[1]),
        num_classes=num_classes,
        class_names=names,
    )


def discover_dlbac_datasets(dlbac_root: Path | None = None) -> list[DLBACDatasetSpec]:
    root = dlbac_root or DLBAC_ROOT
    specs: list[DLBACDatasetSpec] = []

    synthetic_root = root / "synthetic"
    if synthetic_root.is_dir():
        for folder in sorted(synthetic_root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            train_path = folder / f"train_{name}.sample"
            test_path = folder / f"test_{name}.sample"
            if not test_path.is_file():
                continue
            if not train_path.is_file():
                specs.append(
                    DLBACDatasetSpec(
                        name=name,
                        kind="synthetic",
                        train_path=None,
                        test_path=test_path,
                        label_mode="joint_ops",
                        num_ops=4,
                    )
                )
                continue
            specs.append(
                DLBACDatasetSpec(
                    name=name,
                    kind="synthetic",
                    train_path=train_path,
                    test_path=test_path,
                    label_mode="joint_ops",
                    num_ops=4,
                )
            )

    real_root = root / "real-world"
    if real_root.is_dir():
        for folder in sorted(real_root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            train_path = folder / f"train_{name}.sample"
            test_path = folder / f"test_{name}.sample"
            if not test_path.is_file():
                continue
            specs.append(
                DLBACDatasetSpec(
                    name=name,
                    kind="real_world",
                    train_path=train_path if train_path.is_file() else None,
                    test_path=test_path,
                    label_mode="binary_access",
                    num_ops=1,
                )
            )

    return specs


def load_prepared(name: str, prepared_dir: Path | None = None) -> DLBACPreparedSplits:
    out = prepared_dir or DEFAULT_OUT_DIR
    npz_path = out / f"{name}.npz"
    meta_path = out / f"{name}.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Jeu préparé introuvable: {npz_path}. Lancez prepare_dlbac_datasets.py --all")

    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return DLBACPreparedSplits(
        name=meta["name"],
        kind=meta["kind"],
        label_mode=meta["label_mode"],
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_features=meta["num_features"],
        num_classes=meta["num_classes"],
        class_names=meta["class_names"],
    )


def list_trainable_prepared(prepared_dir: Path | None = None) -> list[str]:
    """Liste les jeux avec train non vide (fichiers *.json par jeu, hors manifest)."""
    out = prepared_dir or DEFAULT_OUT_DIR
    names: list[str] = []
    for meta_path in sorted(out.glob("*.json")):
        if meta_path.name == "manifest.json":
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(meta, dict):
            continue
        if "name" not in meta:
            continue
        shapes = meta.get("shapes", {})
        if not isinstance(shapes, dict):
            continue
        train_shape = shapes.get("train", [0, 0])
        if isinstance(train_shape, (list, tuple)) and len(train_shape) >= 1 and train_shape[0] > 0:
            names.append(str(meta["name"]))
    return sorted(names)


def format_rule(rule: dict) -> str:
    cond = " AND ".join(rule.get("if", []))
    then_cls = rule.get("then_class", "?")
    score = rule.get("score", 0.0)
    return f"IF {cond} THEN {then_cls} (score={score:.3f})"


def explain_counterfactual_flip(
    trainer,
    x_cont: np.ndarray,
    sample_idx: int,
    target_class: int,
    *,
    y_true: int | None = None,
) -> dict:
    """Résumé lisible d'un contrefactuel pour une instance (features continues one-hot)."""
    import torch

    if trainer.model is None:
        raise RuntimeError("Modèle non entraîné.")

    x_row = np.asarray(x_cont[sample_idx : sample_idx + 1], dtype=np.float32)
    x_bin = trainer.binarizer.transform(x_row)

    x_t = torch.tensor(x_bin, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        logits_orig = trainer.model.predict_logits(x_t)
        y_pred = int(torch.argmax(logits_orig, dim=1).item())

        y_tgt = torch.tensor([target_class], dtype=torch.long, device=trainer.device)
        x_cf, logits_cf = trainer.model.generate_counterfactual_binary(x_t, y_tgt)
        y_cf_pred = int(torch.argmax(logits_cf, dim=1).item())

    feat_names = trainer.binarizer.binary_feature_names()
    changed = []
    for j in range(x_bin.shape[1]):
        if x_bin[0, j] != x_cf[0, j].cpu().item():
            changed.append(
                {
                    "feature": feat_names[j],
                    "from": int(x_bin[0, j]),
                    "to": int(x_cf[0, j].cpu().item()),
                }
            )

    class_names = trainer.class_names or [str(i) for i in range(trainer.model.num_classes)]
    return {
        "y_true": y_true,
        "y_pred_orig": y_pred,
        "y_target": target_class,
        "y_target_name": class_names[target_class] if target_class < len(class_names) else str(target_class),
        "y_pred_cf": y_cf_pred,
        "valid": y_cf_pred == target_class,
        "n_flips": len(changed),
        "flips": changed[:12],
    }


def explain_counterfactual_continuous(
    trainer,
    x_cont: np.ndarray,
    sample_idx: int,
    target_class: int,
    *,
    y_true: int | None = None,
    max_flips: int = 12,
) -> dict:
    """Résumé lisible d'un contrefactuel continu (HyConEx, features one-hot [0,1])."""
    import torch

    if trainer.model is None:
        raise RuntimeError("Modèle non entraîné.")

    x_row = np.asarray(x_cont[sample_idx : sample_idx + 1], dtype=np.float32)
    x_t = torch.tensor(x_row, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        logits_orig = trainer.model.predict_logits(x_t)
        y_pred = int(torch.argmax(logits_orig, dim=1).item())
        proba_orig = float(torch.softmax(logits_orig, dim=1)[0, y_pred].item())

        y_tgt = torch.tensor([target_class], dtype=torch.long, device=trainer.device)
        gen = trainer.model.generate_counterfactual(x_t, y_tgt)
        if isinstance(gen, tuple):
            x_cf, logits_cf = gen
        else:
            x_cf = gen
            logits_cf = trainer.model.predict_logits(x_cf)
        y_cf_pred = int(torch.argmax(logits_cf, dim=1).item())
        proba_cf = float(torch.softmax(logits_cf, dim=1)[0, y_cf_pred].item())

    model = trainer.model
    in_embed_space = (
        model is not None
        and getattr(model, "mode", None) == "embed"
        and x_cf.shape[-1] != x_row.shape[-1]
    )
    if in_embed_space:
        from hyperlogic_pure.model import continuous_to_bipolar

        with torch.no_grad():
            emb_orig = continuous_to_bipolar(model.core.encode(x_t)).cpu().numpy()[0]
        x_np = emb_orig
        x_cf_np = x_cf.detach().cpu().numpy()[0]
        if x_cf_np.min() >= -1.01 and x_cf_np.max() <= 1.01:
            x_cf_np = (x_cf_np + 1.0) * 0.5
            x_np = (x_np + 1.0) * 0.5
        feat_names = [f"emb_{i}" for i in range(x_np.shape[0])]
    else:
        feat_names = trainer.feature_names or [f"f{i}" for i in range(x_row.shape[1])]
        x_np = x_row[0]
        x_cf_np = x_cf.detach().cpu().numpy()[0]
        if x_cf_np.min() >= -1.01 and x_cf_np.max() <= 1.01 and x_np.max() <= 1.01:
            x_cf_np = (x_cf_np + 1.0) * 0.5
    changed = []
    for j in range(x_np.shape[0]):
        delta = float(x_cf_np[j] - x_np[j])
        if abs(delta) > 1e-4:
            changed.append(
                {
                    "feature": feat_names[j],
                    "from": float(x_np[j]),
                    "to": float(x_cf_np[j]),
                    "delta": delta,
                }
            )
    changed.sort(key=lambda c: abs(c["delta"]), reverse=True)

    class_names = trainer.class_names or [str(i) for i in range(trainer.model.num_classes)]
    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_pred_orig": y_pred,
        "y_pred_orig_name": class_names[y_pred] if y_pred < len(class_names) else str(y_pred),
        "proba_orig": proba_orig,
        "y_target": target_class,
        "y_target_name": class_names[target_class] if target_class < len(class_names) else str(target_class),
        "y_pred_cf": y_cf_pred,
        "y_pred_cf_name": class_names[y_cf_pred] if y_cf_pred < len(class_names) else str(y_cf_pred),
        "proba_cf": proba_cf,
        "valid": y_cf_pred == target_class,
        "n_changes": len(changed),
        "changes": changed[:max_flips],
    }


def pick_counterfactual_example(
    trainer,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_probe: int = 48,
) -> tuple[int, int] | None:
    """Cherche (index, classe_cible) pour illustrer un flip contrefactuel valide."""
    import torch

    if trainer.model is None:
        return None

    n = min(max_probe, len(y_test))
    x_t = torch.tensor(np.asarray(x_test[:n], dtype=np.float32), device=trainer.device)
    with torch.no_grad():
        preds = trainer.model.predict_logits(x_t).argmax(dim=1).cpu().numpy()

    num_classes = trainer.model.num_classes
    for i in range(n):
        y_i = int(y_test[i])
        candidates = [c for c in range(num_classes) if c != int(preds[i])]
        if not candidates:
            candidates = [c for c in range(num_classes) if c != y_i]
        for target in candidates:
            cf = explain_counterfactual_continuous(trainer, x_test, i, target, y_true=y_i)
            if cf["valid"]:
                return i, target
    return None


def prepare_all(
    specs: list[DLBACDatasetSpec],
    out_dir: Path,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[Path]:
    paths: list[Path] = []
    manifest: list[dict] = []
    for spec in specs:
        print(f"[prepare] {spec.name} ({spec.kind}, train={spec.has_train}) ...")
        splits = prepare_dataset(spec, val_size=val_size, random_state=random_state)
        path = splits.save(out_dir)
        paths.append(path)
        entry = asdict(spec)
        entry["train_path"] = str(spec.train_path) if spec.train_path else None
        entry["test_path"] = str(spec.test_path)
        entry["has_train"] = spec.has_train
        entry["npz"] = str(path)
        entry["shapes"] = {
            "train": list(splits.x_train.shape),
            "val": list(splits.x_val.shape),
            "test": list(splits.x_test.shape),
        }
        manifest.append(entry)
        print(
            f"  -> train {splits.x_train.shape}, val {splits.x_val.shape}, "
            f"test {splits.x_test.shape}, classes={splits.num_classes}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifeste: {manifest_path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Préparer les jeux de données DLBAC.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Préparer tous les jeux détectés (synthétiques + Amazon).",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="Noms de jeux à préparer (ex: u4k-r4k-auth11k amazon1).",
    )
    parser.add_argument(
        "--dlbac-root",
        type=Path,
        default=DLBAC_ROOT,
        help="Racine DlbacAlpha-main/dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Dossier de sortie des .npz/.json.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_specs = discover_dlbac_datasets(args.dlbac_root)
    by_name = {s.name: s for s in all_specs}

    if args.all:
        selected = all_specs
    elif args.dataset:
        missing = [n for n in args.dataset if n not in by_name]
        if missing:
            raise SystemExit(f"Jeux inconnus: {missing}. Disponibles: {sorted(by_name)}")
        selected = [by_name[n] for n in args.dataset]
    else:
        raise SystemExit("Indiquez --all ou --dataset <noms...>")

    prepare_all(selected, args.out_dir, val_size=args.val_size, random_state=args.seed)
    trainable = [s.name for s in selected if s.has_train]
    test_only = [s.name for s in selected if not s.has_train]
    print(f"\nEntraînables ({len(trainable)}): {trainable}")
    if test_only:
        print(f"Test seulement ({len(test_only)}): {test_only}")


if __name__ == "__main__":
    main()
