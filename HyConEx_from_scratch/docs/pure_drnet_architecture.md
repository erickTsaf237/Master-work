# Pure DR-Net — architecture fidèle HyperLogic

## Objectif

Modèle où **100 % de la décision** vient du DR-Net (règles IF-THEN), sans tête linéaire.
Entraînement en **2 phases** :
1. **Phase 1** : DR-Net seul (`cf_lambda=0`)
2. **Phase 2** : greffe CF, classifieur DR-Net gelé

## Pipeline données (DLBACα)

Identique à `prepare_dlbac_datasets.py` :
- suppression uid/rid
- masquage métadonnées
- one-hot sklearn sur train
- labels Amazon : deny/grant

## Deux régimes

### Basse dimension (≤ 512 features) — synthétique

**Fidèle HyperLogic** (`InstanceDRNetCore`) :
- TabResNet par échantillon → `theta_main`, `theta_cf`
- DR-Net : `u_k = x^T w_k - ||w_k||_1 + b_k`, `o_k = exp(-u_k²/τ)`
- CF **binaire** HyperLogic (`generate_cf_binary`, straight-through)
- CF réévalué par le **même** DR-Net

→ On peut affirmer : le modèle **est** un DR-Net ; les règles expliquent **100 %** des logits.

### Haute dimension (Amazon ~14k) — embed

**DR-Net pur sur embedding** (`EmbedDRNetCore`) :
- Encodeur one-hot → 128 dims (sans tête linéaire)
- DR-Net global sur `emb_*` bipolar
- **100 % des logits = règles** (pas de fusion linéaire)
- Règles exportées en `emb_0 … emb_127`

**Phase 2 CF** : CF **binaire HyperLogic** sur l'embedding (128 dims, faisable en VRAM).
- Validé par le même DR-Net gelé
- Pas de greffe HyConEx sur 14k (trop coûteux)

→ Classification : **100 % DR-Net** (sur espace encodé).
→ On peut affirmer : modèle à règles pur ; les littéraux sont `emb_*`, pas `oh_*` directement.

## Fichiers

```
hyperlogic_pure/
  config.py
  model.py      # PureDRNetModel
  trainer.py    # PureDRNetTrainer
train_pure_drnet_dlbac.py
```

## Commandes

```powershell
& "C:\anaconda\envs\hyconex\python.exe" train_pure_drnet_dlbac.py --dataset u4k-r4k-auth11k --save
& "C:\anaconda\envs\hyconex\python.exe" train_pure_drnet_dlbac.py --dataset amazon1 --save
```
