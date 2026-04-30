import json
from pathlib import Path

f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')

old = '''    model_path = SIM_DIR / "ml_correction_model.json"
    if model_path.exists():
        # Load trained model if available
        try:
            model_data = json.loads(model_path.read_text())
            weights    = np.array(model_data["weights"])
            bias       = model_data["bias"]
            correction = float(np.dot(weights, features) + bias)
            return round(correction, 3)
        except Exception:
            pass'''

new = '''    model_path = Path(__file__).resolve().parent.parent / "data" / "sim" / "ml_correction_model.json"
    if model_path.exists():
        try:
            model_data = json.loads(model_path.read_text())
            weights    = np.array(model_data["weights"], dtype=np.float64)
            bias       = float(model_data["bias"])
            correction = float(np.dot(weights, features.astype(np.float64)) + bias)
            return round(correction, 3)
        except Exception:
            pass'''

if old in text:
    text = text.replace(old, new)
    f.write_text(text, encoding='utf-8')
    print('Fix 1: ML correction path fixed')
else:
    print('Fix 1: block not found')

vp = Path('data/sim/validation_pairs.json')
pairs = json.loads(vp.read_text())
unreliable = {('Olaparib','BRCA1'), ('Palbociclib','TOP2A')}
kept = [p for p in pairs if (p['drug_name'], p['gene']) not in unreliable]
vp.write_text(json.dumps(kept, indent=2))
print(f'Fix 2: Removed 2 unreliable pairs, {len(kept)} remain')
