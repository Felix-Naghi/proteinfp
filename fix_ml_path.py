"""
fix_obabel_precise.py
----------------------
Precisely replaces the obabel ligand conversion in pipeline/denovo_design.py.
Only touches the obabel subprocess.run call and its immediate error checks.
Does NOT touch surrounding code.

Run: python fix_obabel_precise.py
"""
from pathlib import Path
import ast, sys

DENOVO = Path("pipeline/denovo_design.py")
lines  = DENOVO.read_text(encoding="utf-8").splitlines()
print(f"File: {len(lines)} lines")

# Find the obabel line
obabel_idx = None
for i, l in enumerate(lines):
    if 'obabel' in l and 'partialcharge' in l:
        obabel_idx = i
        print(f"Found obabel at line {i+1}: {l.strip()}")
        break

if obabel_idx is None:
    print("obabel line not found — may already be fixed")
    sys.exit(0)

# Show context around it (10 lines before and after)
print("\nContext (lines around obabel call):")
for i in range(max(0, obabel_idx-8), min(len(lines), obabel_idx+8)):
    marker = " >>>" if i == obabel_idx else "    "
    print(f"{marker} {i+1}: {repr(lines[i])}")

# Find the start of the try: block that contains the obabel call
# Scan backwards to find the matching try:
try_idx = obabel_idx
while try_idx > 0:
    stripped = lines[try_idx].strip()
    if stripped == 'try:':
        break
    try_idx -= 1

print(f"\ntry: at line {try_idx+1}")

# Find the end: look for the except/finally that closes this try
# The try block ends when we hit an except at the same indent level
try_indent = len(lines[try_idx]) - len(lines[try_idx].lstrip())
end_idx = try_idx + 1
while end_idx < len(lines):
    l = lines[end_idx]
    stripped = l.strip()
    if stripped:
        indent = len(l) - len(l.lstrip())
        if indent == try_indent and (stripped.startswith('except') or 
                                      stripped.startswith('finally')):
            # Found the except — now find end of except block
            end_idx += 1
            while end_idx < len(lines):
                l2 = lines[end_idx]
                stripped2 = l2.strip()
                if stripped2:
                    indent2 = len(l2) - len(l2.lstrip())
                    if indent2 <= try_indent:
                        break
                end_idx += 1
            break
    end_idx += 1

print(f"Block to replace: lines {try_idx+1} to {end_idx}")
print("Lines being replaced:")
for i in range(try_idx, end_idx):
    print(f"  {i+1}: {repr(lines[i])}")

# Check that we're only replacing an obabel-related block
block_text = "\n".join(lines[try_idx:end_idx])
if 'obabel' not in block_text and 'partialcharge' not in block_text:
    print("\nERROR: Block doesn't contain obabel — aborting to be safe")
    sys.exit(1)

# Build replacement with same indentation as try:
ind = " " * try_indent
replacement = [
    f"{ind}# Convert SMILES to ligand PDBQT using RDKit (replaces obabel)",
    f"{ind}if not _smiles_to_ligand_pdbqt(smiles, Path(pdbqt_path)):",
    f'{ind}    res["error"] = "Ligand PDBQT conversion failed"; return res',
]

print(f"\nReplacement ({len(replacement)} lines):")
for l in replacement:
    print(f"  {repr(l)}")

new_lines = lines[:try_idx] + replacement + lines[end_idx:]
print(f"\nFile after edit: {len(new_lines)} lines")

# Syntax check
content = "\n".join(new_lines)
try:
    ast.parse(content)
    print("Syntax check: PASSED")
    DENOVO.write_text(content + "\n", encoding="utf-8")
    print("Saved.")
    print()
    print("Now run:")
    print("  del data\\intermediate\\P28593_dock_cache.json")
    print("  proteinfp --uniprot P28593 --therapy --denovo --vina pipeline/vina.exe")
except SyntaxError as e:
    print(f"\nSyntax FAILED at line {e.lineno}: {e.msg}")
    err_lines = content.splitlines()
    lo = max(0, e.lineno - 4)
    hi = min(len(err_lines), e.lineno + 4)
    for i, l in enumerate(err_lines[lo:hi], lo+1):
        marker = " >>>" if i == e.lineno else "    "
        print(f"{marker} {i}: {l}")
    sys.exit(1)