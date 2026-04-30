import json
from pathlib import Path

# Verify the model exists and is readable
model_path = Path('data/sim/ml_correction_model.json')
print(f'Model exists: {model_path.exists()}')
if model_path.exists():
    m = json.loads(model_path.read_text())
    print(f'Bias: {m["bias"]}')
    print(f'Weights: {m["weights"][:3]}')

# Check what path step04 is currently using
f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')
for i, line in enumerate(text.splitlines()):
    if 'ml_correction_model' in line:
        print(f'Line {i}: {line.strip()}')
