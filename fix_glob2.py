from pathlib import Path
f = Path('sim/step05_network_perturbation.py')
t = f.read_text(encoding='utf-8')

old = '''        normal_files = list(
            (ROOT / "data" / "grn" / "input").glob("*human*umifm*.csv")
        )'''

new = '''        base = ROOT / "data" / "grn" / "input"
        normal_files = (
            list(base.glob("*human*/*human*umifm*.csv")) or
            list(base.glob("*human*umifm*.csv"))
        )'''

if old in t:
    t = t.replace(old, new)
    f.write_text(t, encoding='utf-8')
    print('Fixed nested glob')
else:
    print('Pattern not found - printing current glob line:')
    for i, line in enumerate(t.splitlines()):
        if 'human' in line and 'glob' in line:
            print(f'  Line {i}: {line}')
