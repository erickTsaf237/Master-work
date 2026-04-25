"""
Démo Adult (HyConEx) : binarisation ±1, entraînement HyperRuleEx, règles et CF.

Exécution depuis la racine du dépôt :

  python -m grok_hyperruleex.demo
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

from grok_hyperruleex.counterfactuals import line_search_alpha
from grok_hyperruleex.explain import global_feature_importance, interpret_counterfactual
from grok_hyperruleex.hyperlogic_core import extract_if_then_rules, local_feature_importance
from grok_hyperruleex.model import HyperRuleEx, HyperRuleExConfig
from grok_hyperruleex.preprocessing import BinarizerConfig, TabularBinarizer
from grok_hyperruleex.train import TrainConfig, train_model


def main() -> None:
    csv_path = ROOT / "HyConEx" / "data" / "adult.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    raw = pd.read_csv(csv_path)
    numerical_cols = ["age", "hours_per_week"]
    categorical_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "race",
        "gender",
    ]
    feature_cols = numerical_cols + categorical_cols
    df = raw[feature_cols].copy()
    y = LabelEncoder().fit_transform(raw["income"].to_numpy())

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df, y_train, test_size=0.25, random_state=0, stratify=y_train
    )

    binarizer = TabularBinarizer(
        BinarizerConfig(numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    )
    X_train = binarizer.fit_transform(X_train_df)
    X_val = binarizer.transform(X_val_df)
    X_test = binarizer.transform(X_test_df)

    dim = X_train.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HyperRuleEx(
        HyperRuleExConfig(dim=dim, n_rules=16, n_classes=2, hidden=256, depth=3, tau=0.1)
    )
    cfg = TrainConfig(
        epochs=50,
        batch_size=256,
        lr=2e-3,
        device=device,
        lambda_sparse=0.05,
        lambda_div=0.02,
        lambda_stab=0.05,
        verbose=True,
        log_every=1,
    )
    history = train_model(
        model,
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
        cfg,
    )
    print("(Historique detaille par epoch dans le tableau ci-dessus.)")

    model.eval()
    x0 = torch.from_numpy(X_test[:1]).float().to(device)
    with torch.no_grad():
        out = model(x0)
        rules = extract_if_then_rules(
            out["w"], out["u"], binarizer.feature_names, u_threshold=1e-6
        )
    print("Exemples de regles IF-THEN (1er point test) :")
    if not rules:
        print("  (aucune regle au seuil courant sur w/u — augmenter epochs ou ajuster seuils)")
    for line in rules[:8]:
        print(" ", line)

    imp = local_feature_importance(out["w"], out["u"])
    top_loc = torch.topk(imp.flatten(), k=min(5, imp.numel()))
    print("Importance locale (top 5):", list(zip(top_loc.indices.tolist(), top_loc.values.tolist())))

    x_full = torch.from_numpy(X_test).float().to(device)
    g_imp = global_feature_importance(model, x_full[:512])
    top_idx = g_imp.argsort()[-5:][::-1]
    names = binarizer.feature_names
    print("Importance globale (moyenne batch, top 5) :")
    for j in top_idx:
        if j < len(names):
            print(f"  {names[j]}: {float(g_imp[j]):.4f}")

    x_np = X_test[0]
    eps = torch.randn_like(x0)
    with torch.no_grad():
        out0 = model(x0, eps)
        m = int(1 - y_test[0])
        alpha, x_cf = line_search_alpha(
            x0.squeeze(0),
            out0["V"].squeeze(0)[m],
            lambda xt, ep: model(xt, ep)["logits"],
            eps.squeeze(0),
            target_class=m,
        )
    delta = interpret_counterfactual(
        x_cf.cpu().numpy(),
        x0.cpu().numpy().flatten(),
        binarizer.feature_names,
    )
    print(f"Contre-factuel (classe cible {m}), alpha={float(alpha):.4f}, dims modifiees: {len(delta)}")


if __name__ == "__main__":
    main()
