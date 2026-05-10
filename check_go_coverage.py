"""
fix_uniprot_varname.py
"""
import shutil, py_compile, sys
from pathlib import Path

f = Path("pipeline/clean_ec.py")
src = f.read_text(encoding="utf-8")
shutil.copy2(f, f.with_suffix(".py.bak_varname"))

OLD = '    _hom_path = Path(cfg.paths["intermediate"]) / f"{uniprot}_homology.json"'
NEW = '    _hom_path = Path(cfg.paths["intermediate"]) / f"{uniprot_id}_homology.json"'

if OLD in src:
    src = src.replace(OLD, NEW)
    f.write_text(src, encoding="utf-8")
    print("Fixed: uniprot -> uniprot_id")
else:
    print("Not found — checking what variable name is used nearby...")
    for i, line in enumerate(src.split("\n")):
        if "_hom_path" in line or "homology.json" in line:
            print(f"  {i}: {line}")
    sys.exit(1)

try:
    py_compile.compile(str(f), doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Quick test
import subprocess, os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
r = subprocess.run(
    [sys.executable, "pipeline/clean_ec.py", "--uniprot", "P00533"],
    capture_output=True, text=True, timeout=60, env=env
)
print("P00533 returncode:", r.returncode)
if r.returncode != 0:
    print("STDERR:", r.stderr[-300:])
else:
    print("OK")