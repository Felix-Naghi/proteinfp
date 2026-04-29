import subprocess, os, sys
from pathlib import Path

env = os.environ.copy()
env['PYTHONPATH'] = 'C:\\Users\\adria\\Documents\\proteinFP'
env['PYTHONIOENCODING'] = 'utf-8'

SMILES = "O=C1C2CN3CC(NC4=NN5N=C(C)C=CC5=N4)CCC3CC2N(C(=O)c2cc3n(CC(F)(C)C)ncc3nc2)C1"
NAME   = "AZ13824374"

steps = [
    ["sim/step03_drug_distribution.py",
     "--smiles", SMILES, "--name", NAME, "--dose", "1.0"],
    ["sim/step04_binding_probability.py",
     "--smiles", SMILES, "--name", NAME, "--dose", "1.0"],
    ["sim/step05_network_perturbation.py",
     "--drug", NAME, "--dose", "1.0", "--scan"],
    ["sim/step06_pharmacological_scoring.py",
     "--drug", NAME, "--dose", "1.0"],
]

for step in steps:
    script = step[0]
    args   = step[1:]
    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print(f"{'='*50}")
    r = subprocess.run(
        [sys.executable, script] + args,
        env=env, encoding='utf-8', errors='replace'
    )
    if r.returncode != 0:
        print(f"FAILED: {script}")
        break
    print(f"DONE: {script}")

print("\nPipeline complete.")
