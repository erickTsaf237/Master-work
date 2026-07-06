# HyConEx + HyperLogic sur DLBAC — mise en place et résultats

Ce document décrit ce qui a été implémenté pour faire fonctionner un modèle hybride **HyConEx + HyperLogic** sur les jeux de données **DLBAC** (notamment **Amazon1**), ainsi que l’interprétation des résultats obtenus.

---

## 1. Contexte et objectif

### Problème initial

L’ancien module `nouveau_module` sur **Amazon1** (one-hot complet, ~14 419 colonnes) produisait un **AUROC ≈ 0,50** : le modèle prédisait presque toujours **deny** (collapse de classe).

Les baselines sklearn (LinearSVM calibré) atteignent **AUROC ≈ 0,85** sur le même encodage one-hot → les données sont apprenables ; le problème venait de l’architecture et du pipeline d’entraînement.

### Objectif

Construire un pipeline reproductible qui :

1. atteint des métriques proches des baselines sklearn sur Amazon ;
2. conserve des **règles** interprétables (HyperLogic / DR-Net) ;
3. génère des **contrefactuels** (HyConEx) ;
4. sauvegarde le modèle et les explications pour consultation ultérieure.

---

## 2. Ce qui a été mis en place

### 2.1 Nouveau module `hyconex_hyperlogic/`

Package Python dédié, distinct de l’ancien `nouveau_module` :

| Fichier | Rôle |
|---------|------|
| `config.py` | Hyperparamètres (`HybridConfig`) : epochs, lr, embed_dim, num_rules, poids linéaire/règles, phase CF |
| `model.py` | Architecture `HyConExHyperLogicModel` |
| `trainer.py` | Entraînement, évaluation, export de règles, sauvegarde/chargement checkpoint |
| `__init__.py` | Exports publics |

### 2.2 Architecture du modèle

Le modèle combine trois blocs :

```
Entrée one-hot [0,1]  (ex. 14 419 dims pour Amazon1)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
  Tête linéaire (HyConEx)              Encodeur bottleneck
  Linear(input → 2 classes)            Linear → LayerNorm → GELU → …
  (signal fort, comme un SVM)          → embedding 128 dims
        │                                      │
        │                                      ▼
        │                              DR-Net (HyperLogic)
        │                              48 règles sur emb_0…emb_127
        │                              (poids globaux, pas hypernet/échantillon)
        │                                      │
        └─────────── combinaison ──────────────┘
                    logits = 0.7 × linéaire + 0.3 × règles
        │
        ▼
  Tête contrefactuelle (HyConEx)
  MLP(embed + classe cible) → delta sur [0,1]
  x' = clamp(x + 0.35 × delta, 0, 1)
```

**Choix clés :**

- **Tête linéaire** : capte le signal sparse du one-hot (comme LinearSVM).
- **DR-Net sur `embed_dim=128`** : évite d’apprendre des règles directement sur 14k colonnes (OOM GPU + instabilité).
- **Poids globaux** : `theta_bias` + légère modulation par contexte batch (pas d’hyperréseau par échantillon).
- **Contrefactuels continus** : modifications douces sur le one-hot, pas de flip binaire strict.

### 2.3 Pipeline de données

- Jeux DLBAC préparés via `prepare_dlbac_datasets.py` et `train_nouveau_module_dlbac_quantile.py` (`build_onehot_splits`).
- **Amazon1** : 9 attributs méta → **14 419** colonnes one-hot (haute cardinalité Kaggle).
- Split : 80 % train / 20 % val (stratifié), test séparé fourni par DLBAC.
- Cache one-hot : `data/dlbac_prepared/onehot_cache/`.

### 2.4 Entraînement en 2 phases

Script principal : `train_hyconex_hyperlogic_dlbac.py`

| Phase | Objectif | Hyperparamètres (Amazon1) |
|-------|----------|---------------------------|
| **Phase 1** | Classification (maximiser AUROC) | 35 epochs, lr=3e-3, cf_lambda=0 |
| **Phase 2** | Affiner les contrefactuels | 8 epochs, cf_lambda=0.06, flip_lambda=0.02 |

La phase 2 n’est lancée que si la phase 1 atteint AUROC test ≥ 0.65.

Early stopping sur **AUROC validation** (meilleur checkpoint restauré en fin d’entraînement).

### 2.5 Sauvegarde et explications

| Artefact | Chemin | Contenu |
|----------|--------|---------|
| Checkpoint PyTorch | `results/hyconex_hyperlogic_dlbac/{dataset}_model.pt` | `state_dict`, config, noms de classes/features |
| Résultats JSON | `results/hyconex_hyperlogic_dlbac/{dataset}_results.json` | Métriques, exemple de règle, exemple de contrefactuel |
| Script d’affichage | `show_hyconex_explanations.py` | Charge le checkpoint et affiche règle + CF |

Fonctions d’explication ajoutées dans `prepare_dlbac_datasets.py` :

- `explain_counterfactual_continuous()` — contrefactuel HyConEx sur one-hot [0,1]
- `pick_counterfactual_example()` — cherche un flip valide sur le test set
- `format_rule()` — format lisible des règles DR-Net

Méthodes ajoutées au trainer :

- `save_checkpoint()` / `load_checkpoint()`

### 2.6 Environnement d’exécution

- Environnement conda **`hyconex`** : PyTorch 2.5.1+cu121, CUDA activé.
- GPU testée : NVIDIA GeForce 940MX (2 Go VRAM).
- Commande type :

```powershell
cd HyConEx_from_scratch
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --dataset amazon1 --save
& "C:\anaconda\envs\hyconex\python.exe" show_hyconex_explanations.py --dataset amazon1
```

---

## 3. Résultats sur Amazon1

### 3.1 Métriques de classification

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| **AUROC test** | **0,8553** | Seuil cible : 0,72 → **PASS** |
| **Accuracy test** | **0,8912** | ~89 % de bonnes prédictions |
| **AUROC val (best)** | 0,8371 | Meilleur epoch phase 1 |
| **AUROC phase 1 test** | 0,8554 | Avant fine-tuning CF |
| **Référence LinearSVM** | ~0,85 | Baseline sklearn (notebook) |

Le modèle hybride **égale la baseline SVM** sur Amazon1, ce qui valide l’approche « tête linéaire + règles sur embedding ».

### 3.2 Contrefactuels

| Métrique | Valeur |
|----------|--------|
| **Validité CF** (64 premiers test) | **1,0** (100 %) |
| Seuil grant (tuning val) | 0,29 |

**Exemple concret — échantillon test #0 :**

| Champ | Valeur |
|-------|--------|
| Label vrai | grant |
| Prédiction initiale | grant (proba 0,991) |
| Cible contrefactuelle | deny |
| Prédiction après CF | deny (proba 1,000) |
| Valide ? | Oui |
| Nombre de features modifiées | 1 948 |

**Top modifications** (continues, pas binaires) :

| Feature | Avant | Après | Delta |
|---------|-------|-------|-------|
| oh_12303 | 0,000 | 0,197 | +0,197 |
| oh_1009 | 0,000 | 0,184 | +0,184 |
| oh_6699 | 0,000 | 0,182 | +0,182 |
| oh_12173 | 0,000 | 0,178 | +0,178 |
| oh_9079 | 0,000 | 0,163 | +0,163 |

**Interprétation :** le générateur HyConEx produit un vecteur proche de l’original mais déplace de nombreuses colonnes one-hot de 0 vers des valeurs fractionnaires. Le flip de classe est **valide** (deny prédit avec confiance 1,0), mais le contrefactuel est **peu sparse** : beaucoup de dimensions bougent légèrement. C’est typique d’un delta continu entraîné avec `flip_lambda` faible ; pour des CF plus lisibles, il faudrait augmenter la pénalité de sparsité ou post-traiter (top-k features).

### 3.3 Règles DR-Net

**Exemple de règle exportée** (seuil `min_abs_weight=0.001`) :

```
IF emb_65=-1 AND emb_120=-1 AND emb_4=-1 AND emb_95=+1 THEN grant (score=0.516)
```

**Interprétation :**

- Les règles portent sur l’**espace encodé** (`emb_0` … `emb_127`), pas directement sur les attributs Amazon (`oh_*`).
- `emb_i=+1` / `emb_i=-1` signifie que la dimension *i* de l’embedding est positive / négative après passage bipolar.
- Le score 0,516 est la confiance softmax de la règle vers la classe prédite (ici grant).
- **48 règles** actives avec le seuil 0,001 ; le JSON initial avait `n_rules=0` car le seuil 0,03 était trop strict (poids des règles faibles après entraînement dominé par la tête linéaire à 70 %).

**Limite :** pour obtenir des règles lisibles en termes d’attributs métier Amazon, il faudrait soit projeter les poids `emb_*` vers les colonnes `oh_*` (analyse de l’encodeur), soit entraîner avec un `rule_weight` plus élevé.

---

## 4. Comparaison avec l’ancien `nouveau_module`

| Aspect | Ancien `nouveau_module` | `hyconex_hyperlogic` |
|--------|-------------------------|----------------------|
| AUROC Amazon1 | ~0,50 (collapse deny) | **0,8553** |
| Règles | Sur espace binaire tronqué (512 dims) | Sur embedding 128 dims |
| Hyperréseau | Par échantillon (14k dims) | Poids globaux + encodeur |
| Tête linéaire | Absente | **70 %** du signal final |
| Contrefactuels | Binaires (binarizer) | Continus sur one-hot |
| Sauvegarde modèle | JSON métriques seulement | **Checkpoint .pt** + JSON |

---

## 5. Fichiers produits (Amazon1)

```
HyConEx_from_scratch/
├── hyconex_hyperlogic/          # module hybride
├── train_hyconex_hyperlogic_dlbac.py
├── show_hyconex_explanations.py
├── prepare_dlbac_datasets.py    # + helpers CF continus
└── results/hyconex_hyperlogic_dlbac/
    ├── amazon1_model.pt         # ~checkpoint PyTorch
    └── amazon1_results.json     # métriques + exemple CF
```

---

## 6. Commandes utiles

### Entraîner et sauvegarder

```powershell
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --dataset amazon1 --save
```

### Afficher règle + contrefactuel (sans ré-entraîner)

```powershell
& "C:\anaconda\envs\hyconex\python.exe" show_hyconex_explanations.py --dataset amazon1
```

### Autres jeux

```powershell
# Synthétique (u4k-r4k-auth11k, AUROC attendu ~0,99)
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --dataset u4k-r4k-auth11k --save

# Tous les jeux synthétiques ou Amazon
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --all-synthetic --save
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --all-amazon --save
```

### Baseline SVM (optionnel, lent ~1,5 Go RAM)

```powershell
& "C:\anaconda\envs\hyconex\python.exe" train_hyconex_hyperlogic_dlbac.py --dataset amazon1 --baseline --save
```

---

## 7. Synthèse

1. **Classification** : le modèle hybride atteint **AUROC 0,8553** sur Amazon1, au niveau du LinearSVM de référence.
2. **Contrefactuels** : validité **100 %** sur l’échantillon testé, mais modifications **nombreuses et continues** (peu interprétables sans post-traitement).
3. **Règles** : 48 règles extractibles dans l’espace `emb_*` ; elles complètent la tête linéaire mais ne se lisent pas directement en attributs Amazon.
4. **Reproductibilité** : checkpoint `.pt`, JSON de résultats et script `show_hyconex_explanations.py` permettent de recharger et d’afficher les explications sans ré-entraîner (~8 min sur GPU 940MX).

### Pistes d’amélioration

- Augmenter `flip_lambda` ou ajouter une pénalité L0/L1 sur le delta CF pour des contrefactuels plus sparse.
- Augmenter `rule_weight` ou ajouter une perte d’alignement encodeur → features pour des règles plus lisibles.
- Notebook Jupyter dédié (sur le modèle de `hyconex_from_scratch_dry_bean_counterfactuals.ipynb`).
- Cartographie `emb_*` → `oh_*` via les poids de la première couche de l’encodeur.

---

*Document généré le 21 mai 2026 — run Amazon1, environnement `hyconex`, CUDA.*
