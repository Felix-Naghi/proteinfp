import json
from pathlib import Path

f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')

# Find and fix the build_ml_features function signature and pred_norm
old = '''def build_ml_features(
    drug:     dict,
    pocket:   dict,
    cell_env: dict,
    comp:     str,
    dG_physics: float,
) -> np.ndarray:'''

new = '''def build_ml_features(
    drug:     dict,
    pocket:   dict,
    cell_env: dict,
    comp:     str,
    dG_physics: float,
    current_pred_pKi: float = 0.0,
) -> np.ndarray:'''

if old in text:
    text = text.replace(old, new)
    print('Fixed: added current_pred_pKi parameter')
else:
    print('Signature not found')

# Fix pred_norm to use actual prediction
old2 = '        float(pred) / 10,                      # current prediction'
new2 = '        float(current_pred_pKi) / 10,          # current prediction'

if old2 in text:
    text = text.replace(old2, new2)
    print('Fixed: pred_norm now uses actual pKi')
else:
    print('pred_norm line not found')

# Fix the call site to pass current pKi
old3 = '''    features      = build_ml_features(drug, pocket, cell_env,
                                       compartment, dG_physics)
    ml_correction = apply_ml_correction(features)'''

new3 = '''    # Compute physics pKi first so we can pass it as a feature
    dG_J_physics   = dG_physics * 1000
    Kd_phys_tmp    = math.exp(dG_J_physics / (R * T)) * 1e6
    pKi_physics    = -math.log10(max(Kd_phys_tmp * 1e-6, 1e-15))
    features      = build_ml_features(drug, pocket, cell_env,
                                       compartment, dG_physics,
                                       current_pred_pKi=pKi_physics)
    ml_correction = apply_ml_correction(features)'''

if old3 in text:
    text = text.replace(old3, new3)
    print('Fixed: physics pKi passed to ML features')
else:
    print('Call site not found')

f.write_text(text, encoding='utf-8')
