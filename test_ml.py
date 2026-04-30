import json, numpy as np, math, sys
from pathlib import Path
sys.path.insert(0, '.')

# Force import fresh
import importlib
import sim.step04_binding_probability as m4
importlib.reload(m4)

model_path = Path('data/sim/ml_correction_model.json')
print(f'Model exists: {model_path.exists()}')

# Test apply_ml_correction directly
features = np.array([0.23, 0.874, 0.987, 0.432, 0.3, 0.6, 1.0, 0.481], dtype=np.float32)
result = m4.apply_ml_correction(features)
print(f'ML correction result: {result}')
print(f'Expected: ~3.4')

# Test full score_binding for imatinib vs ABL1
drug = {
    "name": "Imatinib",
    "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "molecular_weight": 493.6, "logP": 3.74,
    "hbd": 3, "hba": 9, "psa": 86.3,
    "charge_at_pH74": 0.0, "rotatable_bonds": 6,
}
ens_path = Path('data/sim/ensembles/P00519_ensemble.json')
if ens_path.exists():
    ens = json.loads(ens_path.read_text())
    cell_env = json.loads(Path('data/sim/cell_environment.json').read_text())['cell_environment']
    score = m4.score_binding(drug, 'P00519', ens, cell_env, 1.0, verbose=False)
    print(f'Physics dG:   {score.dG_total_kJ:.2f}')
    print(f'ML correction:{score.ml_correction:.2f}')
    print(f'Corrected dG: {score.dG_corrected_kJ:.2f}')
    print(f'Corrected Kd: {score.Kd_corrected_uM:.3f} uM')
    print(f'pKi:          {score.pKi:.2f}  (exp=8.1)')
    print(f'P(bind):      {score.p_binding:.4f}')
