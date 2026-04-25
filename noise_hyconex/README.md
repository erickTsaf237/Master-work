# Noise-only HyConEx

Variante HyConEx où les poids dynamiques proviennent d’un **hyperréseau bruit-only** `HyperGenerator(ε)` fusionné avec des poids principaux `W_main` :

`W_final = σ(α) · W_hyper(ε) + (1 − σ(α)) · W_main`

## Composants

- **TabResNet-like** : blocs résiduels (même esprit que `HyConEx/hyconex/hypernetwork.py`).
- **L_conEx** : contre-factuels, CE sur prédictions CF, proximité num/cat, perte de plausibilité via **MAF** (`nflows`). Sur les versions sans `context_features` sur `MaskedAutoregressiveFlow`, le flux est entraîné sur **z = concat(x, c)** (densité jointe, équivalent pratique à un score sur x pour la classe c).
- **`use_projection`** : `True` = déplacement le long de la normale (HyConEx `use_distance`), `False` = soustraction directe **x − w** (spec « x' = x − W_m »).
- **Même bruit ε** : en fine-tune, les forwards sur les contre-factuels réutilisent **ε** répliqué par exemple d’origine (voir `losses.compute_finetune_loss`).
- **`freeze_generator_during_pretrain`** : option sur `NoiseHyConExConfig` pour ne pas mettre à jour le `HyperGenerator` pendant la phase prétrain.
- **Lissage catégoriel** : softmax par bloc one-hot, `T=0.01` (`categorical_smooth.py`).
- **Entraînement** : phase **prétrain** (surclassification + cluster optionnel) puis **fine-tune** avec ramp-up des lambdas (`train.py`).

## Installation

```bash
pip install -r noise_hyconex/requirements.txt
```

## Démo

```bash
cd projet
python -m noise_hyconex.demo
```

Le flux est d’abord entraîné rapidement sur `(x, y)` pour estimer la densité, puis le modèle joint est optimisé.

## API

- `NoiseHyConEx`, `NoiseHyConExConfig` depuis `noise_hyconex`.
- `train_noise_hyconex(...)` pour enchaîner prétrain + fine-tune.
