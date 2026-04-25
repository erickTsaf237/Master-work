"""
Démo Adult : prétrain + fine-tune NoiseHyConEx + MAF.

Depuis la racine du dépôt :

  python -m noise_hyconex.demo
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from noise_hyconex.config import NoiseHyConExConfig
from noise_hyconex.dataset_adapter import fit_layout_from_df
from noise_hyconex.flow_maf import ConditionalMAF
from noise_hyconex.model import NoiseHyConEx
from noise_hyconex.train import TrainLoopConfig, train_noise_hyconex


def main() -> None:
    csv_path = ROOT / "HyConEx" / "data" / "adult.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    numerical_cols = ["age", "hours_per_week"]
    categorical_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "race",
        "gender",
    ]
    raw = pd.read_csv(csv_path)
    if len(raw) > 8000:
        raw = raw.sample(n=8000, random_state=42).reset_index(drop=True)
    df = raw[numerical_cols + categorical_cols].copy()
    y = LabelEncoder().fit_transform(raw["income"].to_numpy())

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df, y_train, test_size=0.25, random_state=0, stratify=y_train
    )

    layout = fit_layout_from_df(X_train_df, numerical_cols, categorical_cols)
    ct = layout.feature_transformer
    X_train = ct.transform(X_train_df).astype("float32")
    X_val = ct.transform(X_val_df).astype("float32")
    X_test = ct.transform(X_test_df).astype("float32")

    d = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()

    flow = ConditionalMAF(
        features=d,
        hidden_features=128,
        context_features=1,
        num_layers=3,
        num_blocks_per_layer=2,
        device=device,
    )
    flow.fit_quick(X_train_t, y_train_t.float(), epochs=25, batch_size=256, device=device)

    cfg = NoiseHyConExConfig(
        nr_features=d,
        nr_classes=2,
        noise_dim=64,
        hidden_size=512,
        n_res_blocks=3,
        use_projection=False,
        pretrain_epochs=15,
        finetune_epochs=25,
        class_start_epoch=0,
        dist_start_epoch=2,
        flow_start_epoch=3,
        class_warm_up_epochs=15,
        dist_warm_up_epochs=15,
        flow_warm_up_epochs=15,
        log_prob_threshold=-25.0,
    )
    model = NoiseHyConEx(cfg)
    train_noise_hyconex(
        model,
        flow,
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        layout,
        cfg,
        loop=TrainLoopConfig(batch_size=256, lr=2e-3, device=str(device), log_every_epoch=5),
        x_target_train=None,
    )

    model.eval()
    with torch.no_grad():
        acc = (
            model(torch.from_numpy(X_test).float().to(device)).argmax(-1)
            == torch.from_numpy(y_test).long().to(device)
        ).float().mean().item()
    print(f"Test accuracy (apres entrainement): {acc:.4f}")


if __name__ == "__main__":
    main()
