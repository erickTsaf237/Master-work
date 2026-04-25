# HyperRuleEx (implémentation du rapport `grok_report (7).pdf`)

Modèle hybride tabulaire unifié :

- **Entrée** : `x` binarisé (`±1`) et bruit gaussien `ε` (même dimension), concaténés en `[x ; ε]`.
- **Hyperréseau** : produit en un passage `(w, u)` (HyperLogic) et `V ∈ ℝ^{C×D}` (une direction par classe, contre-factuels type HyConEx).
- **Branche HyperLogic** : seule source de la **prédiction** (`f(x) = Σ u_k h(·)`, puis sigmoid / softmax).
- **Branche contre-factuels** : `x'_m = x - α_m V_m` avec recherche linéaire sur `α` ; la perte de plausibilité MAF du rapport n’est pas encore branchée (extension possible avec `nflows`).
- **Interprétation** : extraction de règles IF-THEN, importances locales/globales, `binary_to_original` pour retraduire.

## Installation

```bash
pip install torch numpy pandas scikit-learn
# optionnel pour plausibilité type HyConEx
pip install nflows
```

## Démo rapide

```bash
cd projet
python -m grok_hyperruleex.demo
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `preprocessing.py` | Binarisation + dictionnaire `binary_to_original` |
| `model.py` | `HyperRuleEx` : hyperréseau + têtes |
| `hyperlogic_core.py` | `f(x)`, `h` lisse, extraction de règles |
| `counterfactuals.py` | Génération CF + recherche linéaire `α` |
| `explain.py` | `interpret_rule`, `interpret_counterfactual`, importances |
| `losses.py` | `L_hyperlogic`, `L_conex`, `L_stability` |
| `train.py` | Boucle d’entraînement minimale |
| `demo.py` | Jeu de données type Adult (CSV HyConEx) |
