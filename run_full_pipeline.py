import subprocess, os
from pathlib import Path

env = os.environ.copy()
env['PYTHONPATH'] = 'C:\\Users\\adria\\Documents\\proteinFP'
env['PYTHONIOENCODING'] = 'utf-8'

targets = [
    'Q6PL18',  # ATAD2
    'P11388',  # TOP2A
    'P16949',  # STMN1
    'P11166',  # SLC2A1
    'Q9HAW4',  # CLSPN
    'Q9NRZ9',  # HELLS
    'P80188',  # LCN2
    'P40199',  # CEACAM6
]

# Modules to run in order
modules = [
    'pipeline/physicochemical.py',
    'pipeline/active_sites.py',
    'pipeline/binding_pockets.py',
    'pipeline/allosteric.py',
    'pipeline/homology.py',
    'pipeline/deepfri_go.py',
    'pipeline/clean_ec.py',
    'pipeline/consensus.py',
]

for uid in targets:
    print(f'\n{"="*50}')
    print(f'  {uid}')
    print(f'{"="*50}')
    for module in modules:
        mod_name = Path(module).stem
        r = subprocess.run(
            ['.venv/Scripts/python.exe', module, '--uniprot', uid],
            capture_output=True, text=True, env=env,
            encoding='utf-8', errors='replace'
        )
        status = 'OK' if r.returncode == 0 else 'FAIL'
        print(f'  {mod_name:<25} {status}')
        if r.returncode != 0:
            print(f'    {r.stderr[-80:].strip()}')
