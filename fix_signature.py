from pathlib import Path

f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')

# Find and fix the broken function signature
old = 'def compute_hydrophobic_score(\n    logP:             float,\n    pocket_volume_A3: float,\n    pocket_shape:     float,\n) -> float:'

new = 'def compute_hydrophobic_score(\n    logP:             float,\n    pocket_volume_A3: float,\n    pocket_shape:     float,\n    n_aromatic_rings: int = 0,\n) -> float:'

if old in text:
    text = text.replace(old, new)
    f.write_text(text, encoding='utf-8')
    print('Fixed function signature')
else:
    print('Signature not found - checking current state...')
    for i, line in enumerate(text.splitlines()):
        if 'def compute_hydrophobic' in line:
            print(f'  Found at line {i}: {line}')
            # Print next 5 lines
            lines = text.splitlines()
            for j in range(i, min(i+6, len(lines))):
                print(f'  {j}: {lines[j]}')
