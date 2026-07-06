# RuleConEx

**Rule-based Counterfactual eXplainer** — modèle unifié pour le contrôle d'accès basé sur l'apprentissage profond (DLBAC).

Les modèles performants (DLBAC) ne s'expliquent pas ; ceux qui s'expliquent le font souvent d'un seul côté : **règles sans contrefactuels** (HyperLogic) ou **contrefactuels sans règles** (HyConEx). RuleConEx combine les deux en **un seul forward pass**.

Pour chaque requête d'accès (métadonnées utilisateur et ressource encodées), le modèle produit :

1. une **décision** `ŷ` et les probabilités associées ;
2. des **règles IF-THEN** lisibles (`oh_* → ops_pattern_*` ou `grant`/`deny`) ;
3. des **contrefactuels** `x'_t` vers chaque classe alternative `t ≠ ŷ`, avec le minimum de modifications one-hot.

---

## Problème et données

Une requête d'autorisation est modélisée comme un tuple `(uid, rid, m^u, m^r, op)`. RuleConEx prédit la classe `y` à partir des métadonnées encodées `x ∈ R^d` :

- **Jeux synthétiques** (8 jeux) : les 4 opérations binaires forment 16 classes `ops_pattern_0` … `ops_pattern_15`.
- **Jeux Amazon** (3 jeux) : classification binaire `deny` / `grant`.

Le prétraitement est implémenté dans `prepare_dlbac_datasets.py` (protocole DLBACα) :

1. suppression de `uid` / `rid` ;
2. masquage des métadonnées au-delà des 8 premières (user et resource) ;
3. étiquette jointe (synthétique) ou binaire (Amazon) ;
4. encodage one-hot (fit sur le train) ;
5. split 80 % train / 20 % validation + test officiel → `data/dlbac_prepared/*.npz` ;
6. bipolarisation `{-1,+1}^d` pour la branche règles (DR-Net).

---

## Architecture

RuleConEx s'articule autour d'un **encodeur**, d'un **hyperréseau TabResNet** et de **deux branches** fusionnées.

```
Entrée x ∈ [0,1]^d
    │
    ├─► Encodeur MLP + LayerNorm ──► z ∈ R^64
    │
    └─► Hyperréseau H(x) ──► θ_règles, θ_CF, θ_HyC
              │
              ├─► Branche règles (DR-Net, K=48) ──► z^règles
              │       entrée bipolaire, Monte Carlo (M=3 train, M=5 inférence)
              │
              └─► Branche HyConEx (classifieur linéaire local) ──► z^HyC
                        │
                        └─► Contrefactuels : x' = 0.55·x_sub + 0.45·x_flip

Fusion : p̂ = softmax(α_h·z^HyC + α_r·z^règles)   →   ŷ = argmax p̂
```

### Branche règles (HyperLogic)

Réseau de règles différentiable (DR-Net) à **K = 48** neurones-règles. Chaque règle `k` est décodée en clause IF-THEN sur les 4 littéraux `oh_j` les plus importants ; la classe THEN est `argmax_c w^out_{k,c}`.

Exemple :

```
IF oh_8=+1 AND oh_32=+1 AND oh_55=-1 AND oh_62=-1 THEN ops_pattern_6
```

### Branche contrefactuels (HyConEx)

Pour une classe cible `t`, deux candidats sont combinés :

- **sub** : `x_sub = clip(x - 0.35·W_t, 0, 1)` — translation vers la frontière de la classe `t` ;
- **flip** : tête MLP conditionnée sur `t`, modifications one-hot discrètes ;
- **fusion** : `x' = 0.55·x_sub + 0.45·x_flip`.

Un contrefactuel est **valide** si `f(x') = t`.

> Sur les jeux Amazon à haute dimension (`d > 512`), seul le mécanisme **sub** est utilisé ; l'hyperréseau et les règles opèrent dans l'espace latent.

### Hyperparamètres principaux

| Composante | Paramètre | Défaut |
|------------|-----------|--------|
| Encodeur | dimension latente `m` | 64 |
| Hyperréseau | bruit MC `σ` | 0,08 (appris) |
| Branche règles | neurones-règles `K` | 48 |
| Branche règles | température `τ` | 0,7 |
| Branche CF | facteur `λ` (sub) | 0,35 |
| Fusion | `α_h`, `α_r` | 0,45 chacun |

---

## Entraînement

Perte composite optimisée conjointement :

- **CE** sur la décision fusionnée ;
- **CE** sur la branche règles (`λ_r = 0,08`) ;
- **CE** sur les contrefactuels vers une classe alternative `t` (`λ_cf = 0,12`) ;
- proximité L1 / L2 des CF (`λ_1 = λ_2 = 0,04`) ;
- sparsité des règles (`λ_sp = 0,002`) ;
- diversité Monte Carlo via KL symétrique (`λ_kl = 0,05`).

| Réglage | Valeur |
|---------|--------|
| Optimiseur | AdamW, `lr = 1e-3`, `weight_decay = 1e-5` |
| Sélection modèle | meilleure accuracy de validation |
| Époques | 40 (synthétiques), 25–35 (Amazon) |
| Matériel | GPU CUDA requis |

---

## Inférence

En une passe avant, RuleConEx retourne :

| Sortie | Description |
|--------|-------------|
| Décision | `ŷ`, probabilités `p̂` |
| Règles | clauses IF-THEN classées par score |
| Contrefactuels | `x'_t` et liste des attributs modifiés `Δ_t` |
| Audit | logits par branche — divergence = instance proche d'une frontière |

API : `explain_sample()`, `counterfactual_report()`, `extract_rules_from_pack()`.

---

## Installation

```bash
conda create -n hyconex python=3.11 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate hyconex
pip install numpy scikit-learn matplotlib

cd HyConEx_from_scratch
python prepare_dlbac_datasets.py --all
```

Dépendances internes : `nouveau_module/`, `prepare_dlbac_datasets.py`, `train_nouveau_module_dlbac_quantile.py`.

---

## Utilisation

### CLI

```bash
python -m ruleconex.main --dataset u4k-r4k-auth11k
python -m ruleconex.main --dataset amazon1 --epochs 30 --explain
```

### API Python

```python
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_nouveau_module_dlbac_quantile import build_onehot_splits
from ruleconex import RuleConExConfig, RuleConExTrainer, explain_sample

specs = {s.name: s for s in discover_dlbac_datasets()}
splits = build_onehot_splits(specs["u4k-r4k-auth11k"], random_state=42)

trainer = RuleConExTrainer(RuleConExConfig(epochs=30), device="cuda")
trainer.fit(
    splits.x_train, splits.y_train,
    splits.x_val, splits.y_val,
    feature_names=splits.feature_names,
    class_names=splits.class_names,
)

print(explain_sample(
    trainer.model, splits.x_test[0], None,
    feature_names=splits.feature_names,
    class_names=splits.class_names,
    device=trainer.device,
).text_report)
```

### Notebook et tests

```bash
jupyter notebook ruleconex/RuleConEx_Demo.ipynb
python ruleconex/test_ruleconex.py --dataset u4k-r4k-auth11k --epochs 5
```

---

## Structure du module

```
ruleconex/
├── config.py       # RuleConExConfig
├── model.py        # RuleConExModel, hyperréseau, génération CF
├── trainer.py      # entraînement GPU
├── loss.py         # perte composite
├── evaluate.py     # métriques classification + validité CF
├── utils.py        # explications, décodage des règles
├── visualize.py    # courbes et graphiques
├── main.py         # CLI
└── RuleConEx_Demo.ipynb
```

---

## Positionnement

| Approche | Décision | Règles IF-THEN | Contrefactuels | Single forward |
|----------|:--------:|:--------------:|:--------------:|:--------------:|
| DLBACα | ✓ | ✗ | ✗ | ✓ |
| HyperLogic | ✓ | ✓ | ✗ | ✓ |
| HyConEx | ✓ | ✗ | ✓ | ✓ |
| **RuleConEx** | **✓** | **✓** | **✓** | **✓** |

---

## Références

- Nobi et al., *DLBAC*, 2022
- Yang et al., *HyperLogic*, 2024
- Marszalek et al., *HyConEx*, 2026

---

## Auteur

**TSAFACK NTEUDEM ERICK** — Université de Dschang, Master 2 Recherche.
