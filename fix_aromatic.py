from pathlib import Path

f = Path('sim/step04_binding_probability.py')
text = f.read_text(encoding='utf-8')

# Add aromatic stacking before the return statement in hydrophobic function
old = '    dG_hydro = DG_HYDROPHOBIC_A2 * buried_SA * logP_factor * pocket_shape\n\n    return round(dG_hydro, 3)'

new = '''    dG_hydro = DG_HYDROPHOBIC_A2 * buried_SA * logP_factor * pocket_shape

    # Aromatic ring stacking bonus (pi-pi interactions)
    if n_aromatic_rings > 0:
        dG_stack = -3.5 * n_aromatic_rings * pocket_shape * 0.6
        dG_hydro += dG_stack

    return round(dG_hydro, 3)'''

if old in text:
    text = text.replace(old, new)
    f.write_text(text, encoding='utf-8')
    print('Fixed: aromatic stacking added')
else:
    print('Block not found')
    # Show what the return line looks like
    for i, line in enumerate(text.splitlines()):
        if 'DG_HYDROPHOBIC_A2' in line:
            print(f'  Line {i}: {line}')
