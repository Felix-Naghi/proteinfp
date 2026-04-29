import subprocess, os, sys
from pathlib import Path

env = os.environ.copy()
env['PYTHONPATH'] = 'C:\\Users\\adria\\Documents\\proteinFP'
env['PYTHONIOENCODING'] = 'utf-8'

# Priority targets from GRN — selected for druggability potential
targets = [
    ('ATAD2',   'Q6PL18'),   # ATPase — ATP binding pocket
    ('TOP2A',   'P11388'),   # Topoisomerase — already drugged, validate pipeline
    ('STMN1',   'P16949'),   # Microtubule — small but known binding site
    ('SLC2A1',  'P11166'),   # Glucose transporter — already in your validation set!
    ('CLSPN',   'Q9HAW4'),   # DNA replication — large but important
    ('HELLS',   'Q9NRZ9'),   # Helicase — ATP binding pocket
    ('LCN2',    'P80188'),   # Lipocalin — known small molecule binder
    ('CEACAM6', 'P40199'),   # Cell adhesion — antibody target
]

print(f'Running ProteinFP on {len(targets)} GRN top regulators...')
print()

for gene, uid in targets:
    print(f'[{gene} / {uid}]')
    
    # Check if structure already exists
    pdb = Path(f'data/structures/{uid}.pdb')
    if not pdb.exists():
        print(f'  Fetching structure...')
        r = subprocess.run(
            ['.venv/Scripts/python.exe', 'pipeline/fetch_structure.py',
             '--uniprot', uid],
            capture_output=True, text=True, env=env, encoding='utf-8',
            errors='replace'
        )
        if r.returncode != 0:
            print(f'  FAIL: {r.stderr[-100:].strip()}')
            continue
    else:
        print(f'  Structure already exists')

    # Run ESM-2
    print(f'  Running ESM-2...')
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/esm2_embeddings.py',
         '--uniprot', uid],
        capture_output=True, text=True, env=env, encoding='utf-8',
        errors='replace'
    )
    if r.returncode != 0:
        print(f'  ESM-2 FAIL: {r.stderr[-100:].strip()}')
        continue

    # Run category classifier
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/category_classifier.py', uid],
        capture_output=True, text=True, env=env, encoding='utf-8',
        errors='replace'
    )

    # Run consensus
    print(f'  Running consensus...')
    r = subprocess.run(
        ['.venv/Scripts/python.exe', 'pipeline/consensus.py', '--uniprot', uid],
        capture_output=True, text=True, env=env, encoding='utf-8',
        errors='replace'
    )
    if r.returncode == 0:
        print(f'  OK — report saved')
    else:
        print(f'  Consensus FAIL: {r.stderr[-100:].strip()}')
    print()

print('Done. Check data/reports/ for full reports.')
