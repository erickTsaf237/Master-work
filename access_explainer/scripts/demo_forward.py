"""
Run from project root:

  python -m access_explainer.scripts.demo_forward
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from access_explainer.dual_model import DualExplainModel
from access_explainer.eval_metrics import evaluate_model
from access_explainer.explain import (
    ExplanationPersona,
    build_explanation_bundle,
    format_for_persona,
)
from access_explainer.train import TrainConfig, train_phased

def load_adult_from_hyconex_data(max_rows: int = 6000):
    csv_path = ROOT / "HyConEx" / "data" / "adult.csv"
    raw = pd.read_csv(csv_path)
    feature_columns = [
        "age",
        "hours_per_week",
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "race",
        "gender",
    ]
    if max_rows > 0 and len(raw) > max_rows:
        raw = raw.sample(n=max_rows, random_state=42).reset_index(drop=True)
    X = raw[feature_columns].to_numpy()
    y = raw["income"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0, stratify=y_train
    )
    tf = ColumnTransformer(
        [
            (
                "Num",
                Pipeline(
                    steps=[("min_max", MinMaxScaler(feature_range=(-0.5, 0.5)))]
                ),
                [0, 1],
            ),
            (
                "Cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(range(2, len(feature_columns))),
            ),
        ]
    )
    X_train = tf.fit_transform(X_train).astype("float32")
    X_val = tf.transform(X_val).astype("float32")
    X_test = tf.transform(X_test).astype("float32")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train).astype("int64")
    y_val = le.transform(y_val).astype("int64")
    y_test = le.transform(y_test).astype("int64")
    return X_train, y_train, X_val, y_val, X_test, y_test

def main() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = load_adult_from_hyconex_data()
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    model = DualExplainModel(
        nr_features=X_train.shape[1],
        nr_classes=2,
        n_rules=12,
        hyconex_nr_blocks=3,
        hyconex_hidden=128,
        hyconex_dropout=0.2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(
        epochs_phase1=2,
        epochs_phase2=2,
        epochs_phase3=2,
        batch_size=128,
        lr=2e-3,
        device=device,
    )
    history = train_phased(
        model,
        None,
        X_train,
        y_train,
        X_val,
        y_val,
        cfg,
    )
    print("Last metrics:", history[-1])

    X_test_hc = torch.from_numpy(X_test).float()
    X_test_pm = torch.where(X_test_hc >= 0.0, 1.0, -1.0).float()
    y_test = torch.from_numpy(y_test).long()
    model.eval()
    print(
        "Test hyconex:",
        evaluate_model(
            model,
            X_test_hc.to(device),
            X_test_pm.to(device),
            y_test.to(device),
            head="hyconex",
        ),
    )
    print(
        "Test hyperlogic:",
        evaluate_model(
            model,
            X_test_hc.to(device),
            X_test_pm.to(device),
            y_test.to(device),
            head="hyperlogic",
        ),
    )

    x0_hc = X_test_hc[:1].to(device)
    x0_pm = X_test_pm[:1].to(device)
    with torch.no_grad():
        out = model(x0_hc, x0_pm, return_weights=True)
    bundle = build_explanation_bundle(
        out,
        x0_hc,
        x0_pm,
        feature_names=feature_names,
        use_distance_cf=True,
    )
    for persona in ExplanationPersona:
        print("\n=== Persona:", persona.value, "===")
        print(format_for_persona(bundle, persona))


if __name__ == "__main__":
    main()
