"""
check_esm2_coverage.py
Run this before any validation or EC classifier run to ensure all
proteins have ESM-2 embeddings computed.

    python validation/check_esm2_coverage.py [list_of_uniprot_ids...]
"""
import sys, subprocess
from pathlib import Path

INTER = Path(__file__).parent.parent / "data" / "intermediate"

def ensure_esm2(uniprot_ids):
    missing = [uid for uid in uniprot_ids
               if not (INTER / f"{uid}_esm2.json").exists()
               and (INTER / f"{uid}_structure.json").exists()]
    if not missing:
        print(f"ESM-2 coverage: all {len(uniprot_ids)} proteins OK")
        return []
    print(f"ESM-2 missing for {len(missing)} proteins — computing now...")
    failed = []
    for i, uid in enumerate(missing):
        print(f"  [{i+1}/{len(missing)}] {uid}", end=" ", flush=True)
        r = subprocess.run(
            [sys.executable, "pipeline/esm2_embeddings.py", "--uniprot", uid],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0:
            print("OK")
        else:
            print("FAILED")
            failed.append(uid)
    return failed

if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else []
    if not ids:
        # Default: check all proteins that have a structure
        ids = [p.stem.replace("_structure", "")
               for p in INTER.glob("*_structure.json")]
    ensure_esm2(ids)
