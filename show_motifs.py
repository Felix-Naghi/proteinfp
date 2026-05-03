"""
fix_pdbqt_line.py
Fixes the HETATM line format in _smiles_to_ligand_pdbqt.

Vina requires EXACTLY this format (verified against Vina source):
cols  1- 6: record type  "HETATM"
cols  7-11: serial       right-justified integer
col  12:    space
cols 13-16: atom name    left-justified, element only, no numbers
cols 17-20: " LIG"
cols 21-22: " A"
cols 23-26: "   1"
cols 27-30: "    "
cols 31-38: X coordinate 8.3f
cols 39-46: Y coordinate 8.3f
cols 47-54: Z coordinate 8.3f
cols 55-60: occupancy    6.2f  "  1.00"
cols 61-66: B-factor     6.2f  "  0.00"
cols 67-76: 10 spaces
cols 77-82: charge       6.3f  (e.g. "-0.062")
col  83:    space
cols 84-85: AD4 type     2 chars left-justified

Run: python fix_pdbqt_line.py
"""
from pathlib import Path
import ast, sys

DENOVO = Path("pipeline/denovo_design.py")
src = DENOVO.read_text(encoding="utf-8")

# Find and replace the line format in _smiles_to_ligand_pdbqt
# There may be several variants — replace all of them

REPLACEMENTS = [
    # variant 1: 8.4f charge with 4 spaces before
    (
        '            line = (\n'
        '                f"HETATM{i+1:5d}  {aname} LIG A   1    "\n'
        '                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"\n'
        '                f"  1.00  0.00"\n'
        '                f"    {q:8.4f} {ad4}"\n'
        '            )',
        '            # Vina PDBQT: cols 1-6 record, 7-11 serial, 13-16 name,\n'
        '            # 31-54 XYZ, 55-66 occ/bfac, 67-76 spaces, 77-82 charge, 84-85 type\n'
        '            line = (\n'
        '                f"HETATM{i+1:5d}  {aname} LIG A   1    "\n'
        '                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"\n'
        '                f"  1.00  0.00          "\n'
        '                f"{q:6.3f} {ad4}"\n'
        '            )',
    ),
    # variant 2: 6.3f charge with 2 spaces before
    (
        '            line = (\n'
        '                f"HETATM{i+1:5d}  {aname} LIG A   1    "\n'
        '                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"\n'
        '                f"  1.00  0.00"\n'
        '                f"  {q:6.3f} {ad4}"\n'
        '            )',
        '            # Vina PDBQT: cols 1-6 record, 7-11 serial, 13-16 name,\n'
        '            # 31-54 XYZ, 55-66 occ/bfac, 67-76 spaces, 77-82 charge, 84-85 type\n'
        '            line = (\n'
        '                f"HETATM{i+1:5d}  {aname} LIG A   1    "\n'
        '                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"\n'
        '                f"  1.00  0.00          "\n'
        '                f"{q:6.3f} {ad4}"\n'
        '            )',
    ),
    # variant 3: original with {q:6.3f} space {ad4} all in one f-string
    (
        '                f"  1.00  0.00"\n'
        '                f"          {q:6.3f} {ad4}"\n',
        '                f"  1.00  0.00          "\n'
        '                f"{q:6.3f} {ad4}"\n',
    ),
]

changed = False
for old, new in REPLACEMENTS:
    if old in src:
        src = src.replace(old, new, 1)
        print(f"Applied replacement variant")
        changed = True
        break

if not changed:
    # Find the line block manually
    print("Standard patterns not found. Searching for the line= block...")
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if 'HETATM' in l and 'aname' in l and 'LIG A' in l:
            print(f"Found at line {i+1}: {repr(l)}")
            # Show surrounding lines
            for j in range(max(0,i-2), min(len(lines), i+8)):
                print(f"  {j+1}: {repr(lines[j])}")
    sys.exit(1)

# Verify with a test
try:
    ast.parse(src)
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax FAILED: {e}")
    sys.exit(1)

DENOVO.write_text(src, encoding="utf-8")
print("Saved.")

# Test the output format
print("\nTesting output format...")
import importlib, sys as _sys
if 'pipeline.denovo_design' in _sys.modules:
    del _sys.modules['pipeline.denovo_design']

_sys.path.insert(0, '.')
from pipeline.denovo_design import _smiles_to_ligand_pdbqt
import tempfile
tmp = Path(tempfile.mktemp(suffix='.pdbqt'))
ok = _smiles_to_ligand_pdbqt('NCCc1cccc2nccnc12', tmp)
if ok and tmp.exists():
    lines = tmp.read_text().splitlines()
    atom_line = next((l for l in lines if l.startswith('HETATM')), None)
    if atom_line:
        print(f"Sample line ({len(atom_line)} chars):")
        print(f"  {repr(atom_line)}")
        # Check charge and type are separate
        parts = atom_line.split()
        print(f"  Last two tokens: {parts[-2]!r} {parts[-1]!r}")
        # Verify type is not numeric
        last = parts[-1]
        if any(c.isdigit() for c in last):
            print(f"  ERROR: atom type contains digit: {last!r}")
        else:
            print(f"  OK: atom type is clean: {last!r}")
        # Verify charge is numeric
        charge_tok = parts[-2]
        try:
            float(charge_tok)
            print(f"  OK: charge is numeric: {charge_tok!r}")
        except ValueError:
            print(f"  ERROR: charge not numeric: {charge_tok!r}")
    tmp.unlink()
else:
    print("Conversion failed")
    
print("\nDone. Now run:")
print("  del data\\intermediate\\P28593_dock_cache.json")
print("  proteinfp --uniprot P28593 --therapy --denovo --vina pipeline/vina.exe")