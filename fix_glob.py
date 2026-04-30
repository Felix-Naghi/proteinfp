from pathlib import Path
f = Path('sim/step05_network_perturbation.py')
t = f.read_text(encoding='utf-8')
t = t.replace(
    'list((ROOT / "data" / "grn" / "input").glob("*human*umifm*.csv"))',
    'list((ROOT / "data" / "grn" / "input").glob("*human*/*human*umifm*.csv")) or list((ROOT / "data" / "grn" / "input").glob("*human*umifm*.csv"))'
)
f.write_text(t, encoding='utf-8')
print('Fixed')
