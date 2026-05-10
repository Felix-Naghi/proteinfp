"""
rerun_ec_and_score.py
─────────────────────
Re-runs Module 10 (EC prediction) and Module 13 (consensus) for all
100 proteins using the newly trained ec_ensemble_v5 model, then
re-scores validation.

Run from project root:
    python rerun_ec_and_score.py

Add --dry-run to see what would run without doing it.
"""

import subprocess
import sys
import time
from pathlib import Path

import click

PROTEINS = [
    "P00533","P06213","P15056","P00519","P27361","Q02750","P45983","Q13153",
    "P06239","P08581","P04049","P00734","P42574","P55210","P29466","P08246",
    "P07711","P07858","P04637","P01106","P10275","P03372","Q01094","P17275",
    "P05412","P01100","P14921","P00441","P16083","P00387","P00352","P14550",
    "P11473","Q07869","P37231","P10827","P20393","P07900","P08238","P11142",
    "P04792","P02511","Q14524","P35561","P01116","P62834","P84095","P61586",
    "P60953","Q00987","O15151","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P68871","P69905","P02768","P11166","P38398","P51587","P12004","P02452",
    "P68032","P02144","Q9BYF1","P00918","P08473","P15144","P00488","P04180",
    "P01375","P05156","P46108","Q13480",
    "P06493","P24941","P11802","P36507","P45985",
    "P07339","P43235","P08311",
    "P28562",
    "P15923","P17542","P22415",
    "P00491","P15531","P07737",
    "P08107","P38646",
    "P20585","P52701",
    "P62070","P55197",
    "Q07812","P10415","Q16611",
]

# Deduplicate preserving order
seen = set()
PROTEINS_UNIQUE = []
for uid in PROTEINS:
    if uid not in seen:
        seen.add(uid)
        PROTEINS_UNIQUE.append(uid)


def run_module(script: str, uid: str, dry_run: bool) -> bool:
    cmd = [sys.executable, f"pipeline/{script}.py", "--uniprot", uid]
    if dry_run:
        print(f"    [dry] {' '.join(cmd)}")
        return True
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=120,
        env=env,
    )
    if result.returncode != 0:
        # Print last 3 lines of stderr for quick diagnosis
        lines = result.stderr.strip().splitlines()
        snippet = "\n    ".join(lines[-3:]) if lines else "(no stderr)"
        print(f"    FAILED ({script}): {snippet}")
        return False
    return True


@click.command()
@click.option("--dry-run", is_flag=True, default=False,
              help="Print commands without running them")
@click.option("--skip-missing", is_flag=True, default=False,
              help="Skip proteins whose intermediate files are missing")
def main(dry_run: bool, skip_missing: bool):
    """Re-run EC + consensus for all proteins, then validate."""

    inter_dir = Path("data/intermediate")
    failed = []
    skipped = []
    t_start = time.time()

    print(f"\nRe-running EC + consensus for {len(PROTEINS_UNIQUE)} proteins")
    print(f"Model: models/ec_ensemble_v5\n")

    for i, uid in enumerate(PROTEINS_UNIQUE):
        # Check intermediate files exist (structure + homology are needed by clean_ec)
        structure_ok = (inter_dir / f"{uid}_structure.json").exists()
        homology_ok  = (inter_dir / f"{uid}_homology.json").exists()

        if skip_missing and not (structure_ok and homology_ok):
            print(f"[{i+1:3d}/{len(PROTEINS_UNIQUE)}] {uid}  SKIP (missing intermediates)")
            skipped.append(uid)
            continue

        print(f"[{i+1:3d}/{len(PROTEINS_UNIQUE)}] {uid}", end="  ", flush=True)

        ok_ec = run_module("clean_ec", uid, dry_run)
        if not ok_ec:
            failed.append((uid, "clean_ec"))
            print()
            continue

        ok_con = run_module("consensus", uid, dry_run)
        if not ok_con:
            failed.append((uid, "consensus"))
            print()
            continue

        print("OK")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  OK      : {len(PROTEINS_UNIQUE) - len(failed) - len(skipped)}")
    print(f"  Skipped : {len(skipped)}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"  Failures: {[uid for uid, _ in failed]}")
    print(f"{'='*60}")

    if dry_run:
        print("\n[dry-run] Skipping validation step.")
        return

    # Re-score
    print("\nRunning validation --score-only ...\n")
    subprocess.run(
        [sys.executable, "validation/run_validation.py", "--score-only"],
        check=False,
    )


if __name__ == "__main__":
    main()