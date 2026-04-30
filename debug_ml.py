import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, '.')

# Manually replicate what step04 does for imatinib
drug = {
    "name": "Imatinib",
    "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "molecular_weight": 493.6,
    "logP": 3.74,
    "hbd": 3, "hba": 9, "psa": 86.3,
    "charge_at_pH74": 0.0,
    "rotatable_bonds": 6,
}

# Load ensemble for ABL1
ens_path = Path('data/sim/ensembles/P00519_ensemble.json')
ens = json.loads(ens_path.read_text()) if ens_path.exists() else {}
pocket_vol  = ens.get('mean_pocket_volume', 500)
pocket_drug = ens.get('mean_druggability', 0.5)

# Simulate the feature building from step04
dG_physics  = -23.0  # approximate from last run
fill_ratio  = 1308 / max(pocket_vol, 1)

cell_env = json.loads(Path('data/sim/cell_environment.json').read_text())['cell_environment']
comp = cell_env.get('nucleus', cell_env.get('cytoplasm', {}))

features = np.array([
    max(-100, min(0, dG_physics)) / -100,
    (drug['logP'] + 5) / 10,
    drug['molecular_weight'] / 500,
    drug['psa'] / 200,
    drug['hbd'] / 10,
    drug['hba'] / 15,
    min(fill_ratio, 2.0) / 2.0,
    0.0,  # pred_norm placeholder
], dtype=np.float64)

# Apply model
model = json.loads(Path('data/sim/ml_correction_model.json').read_text())
weights = np.array(model['weights'], dtype=np.float64)
bias    = float(model['bias'])
correction = float(np.dot(weights, features) + bias)

print('Features:')
names = ['dG_norm','logP_norm','MW_norm','PSA_norm','HBD_norm','HBA_norm','fill_ratio','pred_norm']
for n, v in zip(names, features):
    print(f'  {n:<15} {v:.4f}')
print(f'Correction: {correction:.3f}')
print(f'Expected:   ~+3.0')
print()
print(f'Key issue: pred_norm=0.0 but should be actual_pred/10')
print(f'If pred=4.81: pred_norm={4.81/10:.3f}')
features[-1] = 4.81/10
correction2 = float(np.dot(weights, features) + bias)
print(f'Correction with correct pred_norm: {correction2:.3f}')
