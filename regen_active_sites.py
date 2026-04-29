import subprocess, os
from pathlib import Path

env = os.environ.copy()
env['PYTHONPATH'] = 'C:\\Users\\adria\\Documents\\proteinFP'
env['PYTHONIOENCODING'] = 'utf-8'

uids = [f.stem.replace('_report','') for f in Path('data/reports').glob('*_report.json')]
print(f'Regenerating active sites for {len(uids)} proteins...')

ok, fail = 0, 0
for i, uid in enumerate(uids):
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/active_sites.py', '--uniprot', uid],
        capture_output=True, text=True, env=env, encoding='utf-8', errors='replace'
    )
    if r.returncode == 0:
        ok += 1
        print(f'  [{i+1}/{len(uids)}] {uid} OK')
    else:
        fail += 1
        print(f'  [{i+1}/{len(uids)}] {uid} FAIL — {r.stderr[-100:].strip()}')

print(f'Done. {ok} OK, {fail} failed.')
