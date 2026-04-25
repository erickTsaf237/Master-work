# HyConEx from scratch

Ce dossier contient une implémentation from-scratch d'un modèle de type HyConEx pour données tabulaires, avec:

- classification supervisée;
- génération de contre-factuels (CF) conditionnés par une classe cible;
- API Python simple (`import -> config -> train/fit -> evaluate`).

L'objectif est d'avoir un socle unique pour tester rapidement sur différents datasets (DLBaC, Iris, WDBC, etc.).

## 1) Architecture du modèle

Le coeur du modèle est défini dans `hyconex_from_scratch/model.py` (`HyConExFromScratch`).

### 1.0 Vue d'ensemble (schéma)

```text
Entrée tabulaire x
      |
      v
  Encoder E(.)
      |
      v
 latent z -----------------------> Hypernetwork H(.) -----------------> (W(z), b(z))
      |                                                                  |
      |                                                                  v
      |----------------------------------------------------------> logits = W(z) z + b(z)
      |
      +--> concat(z, one_hot(y_target)) --> CF Generator G(.) --> delta --> x_cf = clamp(x + delta, 0, 1)
                                                                                      |
                                                                                      v
                                                                              model(x_cf) -> classe cible
```

Ce flux réalise simultanément:

- une classification dynamique (dépendante de l'échantillon);
- une génération de contre-factuels dirigée vers une classe cible.

### 1.1 Encodeur tabulaire

Pour une entrée `x` (features tabulaires), l'encodeur apprend une représentation latente:

- `z = Encoder(x)`, avec MLP + ReLU;
- dimension latente contrôlée par `latent_dim`.

Intuition: condenser l'information discriminante dans un espace latent plus compact et manipulable pour la classification et les CF.

### 1.2 Hypernetwork (classifieur dynamique)

Au lieu d'utiliser un classifieur fixe, le modèle génère des paramètres dépendants de `z`:

- `params = Hyper(z)` puis reshape en `(num_classes, latent_dim + 1)`;
- on obtient `W(z)` et `b(z)`;
- logits dynamiques: `logits = W(z) · z + b(z)`.

Conséquence: la frontière de décision peut s'adapter localement à chaque échantillon.

### 1.3 Générateur contre-factuel conditionné

Le générateur CF prend le latent `z` et une classe cible `y_target` (one-hot):

- concat `[z, one_hot(y_target)]`;
- MLP pour générer un delta `delta`;
- `x_cf = clamp(x + delta, 0, 1)`.

Le `clamp(0,1)` suppose des features normalisées dans `[0,1]` (d'où le `MinMaxScaler` dans les pipelines de données).

## 2) Formulation mathématique (version compacte)

Notons:

- `x in R^d` une entrée;
- `y in {0, ..., C-1}` sa classe;
- `t` une classe cible contre-factuelle (`t != y`);
- `E`, `H`, `G` respectivement encodeur, hypernetwork, générateur CF.

### 2.1 Représentation et classification dynamique

1. **Encodage latent**

`z = E(x), z in R^k`

2. **Paramètres du classifieur générés par l'hypernetwork**

`theta(x) = H(z) -> (W(z), b(z))`

avec `W(z) in R^(C x k)` et `b(z) in R^C`.

3. **Logits dynamiques**

`f(x) = W(z) z + b(z)` puis `p(y|x) = softmax(f(x))`.

### 2.2 Génération de contre-factuel

1. Encodage conditionné:

`u = [z ; one_hot(t)]`

2. Perturbation:

`delta = G(u)`

3. Contre-factuel:

`x_cf = clip(x + delta, 0, 1)`

Le contre-factuel est valide si `argmax f(x_cf) = t`.

## 3) Fonction de coût (entraînement)

Définie dans `hyconex_from_scratch/trainer.py`, la loss totale est:

- `CE(x, y)` : classification standard sur l'entrée originale;
- `CE(model(x_cf), y_target)` : contrainte de validité CF;
- `L1(delta)` : parcimonie/proximité (petits changements);
- `L2(delta)` : régularisation lisse.

Forme:

`loss = CE + cf_lambda * CE_cf + l1_lambda * L1 + l2_lambda * L2`

Version explicite:

`L = CE(f(x), y) + lambda_cf CE(f(x_cf), t) + lambda_1 ||x_cf - x||_1 + lambda_2 ||x_cf - x||_2^2`

Les hyperparamètres `cf_lambda`, `l1_lambda`, `l2_lambda` pilotent le compromis:

- plus `cf_lambda` est élevé, plus on force l'atteinte de la classe cible;
- plus `l1_lambda`/`l2_lambda` sont élevés, plus on pénalise les grands changements.

## 4) Pipeline d'entraînement complet

### 4.1 Préparation des données

Le pipeline standard:

1. split stratifié train/val/test;
2. normalisation MinMax (`fit` sur train, `transform` sur val/test);
3. conversion en `float32` pour `X` et `int64` pour `y`.

### 4.2 Boucle d'optimisation

`HyConExTrainer.fit(...)`:

1. crée le modèle avec `input_dim` + `num_classes`;
2. échantillonne pour chaque batch une classe cible alternative `y_target`;
3. calcule la loss composite;
4. met à jour avec `AdamW`;
5. suit les métriques par époque (`train_loss`, `val_accuracy`, `best_val_accuracy`).

Le meilleur état (selon accuracy validation) est restauré en fin d'entraînement.

### 4.3 Évaluation

`HyConExTrainer.evaluate(...)` retourne:

- `accuracy`;
- `classification_report`;
- `confusion_matrix`;
- `auroc_ovr` (si calculable);
- métriques CF (si `counterfactuals=True`):
  - `validity_cf`;
  - `proximity_l1_mean`;
  - `changed_features_mean`.

## 5) Interprétation scientifique des composantes

### 5.1 Pourquoi un hypernetwork?

Un classifieur linéaire fixe en latent est parfois trop rigide.  
Ici, `H(z)` produit un classifieur local, ce qui revient à adapter la géométrie de décision à la région de l'espace latent où se trouve l'échantillon.

### 5.2 Pourquoi contraindre les CF par L1/L2?

- `L1` favorise des modifications concentrées sur peu de features (sparsité relative);
- `L2` évite des perturbations extrêmes sur une seule feature et stabilise l'optimisation.

Le couple `L1+L2` est un compromis standard entre parcimonie et régularité.

### 5.3 Sens des métriques CF

- **Validity** mesure la faisabilité algorithmique: le CF atteint-il vraiment la cible?
- **Proximity L1** mesure le coût de recourse: combien faut-il modifier l'entrée?
- **Changed features** approxime l'effort d'action (nombre de leviers à activer).

## 6) Organisation du code

- `hyconex_from_scratch/config.py`: dataclass `TrainConfig` (hyperparamètres).
- `hyconex_from_scratch/model.py`: architecture du modèle.
- `hyconex_from_scratch/trainer.py`: entraînement, évaluation, API haut niveau.
- `hyconex_from_scratch/__init__.py`: exports publics.
- `feature_engineering_wdbc.py`: feature engineering dédié WDBC.
- `train_iris.py`: script smoke test Iris.
- `train_hyconex_from_scratch_dlbac.py`: script complet DLBaC (avec export de fichiers de résultats).
- `notebooks/`: notebooks d'analyse (Iris, WDBC, DLBaC report).

## 7) API recommandée (simple)

### Option A: API fonctionnelle

```python
from hyconex_from_scratch import TrainConfig, train

cfg = TrainConfig(epochs=80, lr=2e-3, latent_dim=32, hidden_dim=64)
res = train(
    X_train,
    y_train,
    config=cfg,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    verbose=True,
)

print(res.best_val_accuracy)
print(res.test_metrics["accuracy"])
```

### Option B: API objet

```python
from hyconex_from_scratch import TrainConfig, HyConExTrainer

trainer = HyConExTrainer(config=TrainConfig())
result = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
metrics = trainer.evaluate(X_test, y_test, counterfactuals=True)
```

## 8) Jeux de données et scripts

### DLBaC (script historique)

```bash
python train_hyconex_from_scratch_dlbac.py --epochs 20 --run-name hyconex_from_scratch
```

Sorties principales:

- `results/<run-name>_metrics.json`
- `results/<run-name>_learning_curve.csv`
- `results/<run-name>_test_predictions.csv`
- `results/<run-name>_counterfactuals_preview.json`
- `results/<run-name>_counterfactuals_preview.csv`
- `results/<run-name>_model.pt`

### Iris (smoke test rapide)

```bash
python train_iris.py
```

### WDBC (notebook)

Notebook dédié:

- `notebooks/hyconex_from_scratch_wdbc_counterfactuals.ipynb`

avec feature engineering via:

- `feature_engineering_wdbc.py`

## 9) Interprétation des métriques contre-factuelles

- `validity_cf`: proportion de CF qui atteignent bien la classe cible.
- `proximity_l1_mean`: taille moyenne de la modification (plus bas = mieux, toutes choses égales par ailleurs).
- `changed_features_mean`: nombre moyen de features modifiées (proxy de parcimonie).

En pratique, un bon modèle CF cherche un compromis:

- validité élevée;
- proximité élevée au sens "petit changement" (donc `L1` faible);
- peu de features modifiées.

## 10) Bonnes pratiques

- Toujours normaliser les features en `[0,1]` pour la stabilité du générateur CF.
- Commencer avec `epochs` modéré (50-120) et ajuster selon la convergence.
- Ajuster en priorité `cf_lambda`, `l1_lambda`, `l2_lambda` pour le compromis classif/CF.
- Fixer la seed (`TrainConfig.seed`) pour des comparaisons reproductibles.
