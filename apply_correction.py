import json, numpy as np, math
from pathlib import Path

ROOT = Path('.')
model = json.loads((ROOT / 'data/sim/ml_correction_model.json').read_text())
weights = np.array(model['weights'], dtype=np.float64)
bias    = float(model['bias'])

val = json.loads((ROOT / 'data/sim/validation_results.json').read_text())
results = val['results']

print('Applying ML correction directly...')
print()

exp_list, pred_list = [], []

for r in results:
    pred = r.get('predicted_pKi')
    exp  = r.get('exp_pKi')
    dG   = r.get('dG_corrected', -30) or -30
    fill = r.get('fill_ratio', 0.5) or 0.5

    if pred is None:
        continue

    features = np.array([
        max(-100, min(0, dG)) / -100,
        (r.get('logP', 2) + 5) / 10,
        r.get('mw', 400) / 600,
        r.get('psa', 80) / 200,
        r.get('hbd', 2) / 10,
        r.get('hba', 5) / 15,
        min(fill, 2.0) / 2.0,
        float(pred) / 10,
    ], dtype=np.float64)

    correction  = float(np.dot(weights, features) + bias)
    final       = pred + correction
    err         = final - exp
    exp_list.append(exp)
    pred_list.append(final)
    print(f"  {r['drug_name']:<18} exp={exp:.1f} raw={pred:.2f} corr={correction:+.2f} final={final:.2f} err={err:+.2f}")

# Statistics
n = len(exp_list)
mx = sum(exp_list)/n
my = sum(pred_list)/n
r  = sum((x-mx)*(y-my) for x,y in zip(exp_list,pred_list))
r /= (sum((x-mx)**2 for x in exp_list) * sum((y-my)**2 for y in pred_list))**0.5
rmse = (sum((p-e)**2 for p,e in zip(pred_list,exp_list))/n)**0.5
mae  = sum(abs(p-e) for p,e in zip(pred_list,exp_list))/n
bias_val = sum(p-e for p,e in zip(pred_list,exp_list))/n

print(f'\n  Pearson r : {r:.3f}')
print(f'  RMSE      : {rmse:.3f}')
print(f'  MAE       : {mae:.3f}')
print(f'  Bias      : {bias_val:+.3f}')
