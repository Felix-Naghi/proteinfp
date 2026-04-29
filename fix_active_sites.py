import subprocess, os, sys
from pathlib import Path

env = os.environ.copy()
env['PYTHONPATH'] = 'C:\\Users\\adria\\Documents\\proteinFP'
env['PYTHONIOENCODING'] = 'utf-8'

long_uids = [
    'P02751','Q14524','P38398','P01031','P02452',
    'P08581','P06213','P08603','P00533','P35228','P00519'
]

for i, uid in enumerate(long_uids):
    print(f'[{i+1}/{len(long_uids)}] Running active sites for {uid}...')
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/active_sites.py', '--uniprot', uid],
        capture_output=True, text=True, env=env, encoding='utf-8', errors='replace'
    )
    if r.returncode == 0:
        print(f'  OK')
    else:
        print(f'  FAIL: {r.stderr[-150:].strip()}')

print('Done. Regenerating consensus reports...')
for uid in long_uids:
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/consensus.py', '--uniprot', uid],
        capture_output=True, text=True, env=env, encoding='utf-8', errors='replace'
    )
    print(f'  {uid}: {"OK" if r.returncode == 0 else "FAIL"}')

print('All done.')
