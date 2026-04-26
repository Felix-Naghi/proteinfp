"""
regenerate_reports.py
─────────────────────
Re-runs only the consensus module (Module 13) for every UniProt ID that
has a report file. This is used after patching consensus.py thresholds
or caps, so that already-cached intermediate results get re-aggregated
into a fresh report with the new settings.

This does NOT re-run BLAST, ESM-2, Foldseek, etc. — it only re-aggregates
the existing intermediate JSONs into a new consensus report.

Usage:
    python regenerate_reports.py
    python regenerate_reports.py --root C:\\Users\\adria\\Documents\\proteinFP
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only regenerate the first N (default: all)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reports_dir = root / "data" / "reports"
    consensus_script = root / "pipeline" / "consensus.py"

    if not consensus_script.exists():
        print(f"{RED}  ✗ Not found: {consensus_script}{RESET}")
        sys.exit(1)

    # Find every existing report (one per UniProt ID)
    report_files = sorted(reports_dir.glob("*_report.json"))
    uniprot_ids = [p.stem.replace("_report", "") for p in report_files]

    if args.limit > 0:
        uniprot_ids = uniprot_ids[:args.limit]

    print(f"{CYAN}Regenerating consensus reports for {len(uniprot_ids)} proteins...{RESET}")
    print(f"  (this only re-runs Module 13; intermediate JSONs are reused)")
    print()

    env = {"PYTHONPATH": str(root)}
    import os
    env.update(os.environ)
    # Force UTF-8 for stdout/stderr so the consensus module's Unicode
    # characters (✓, ─, etc.) don't trip Windows' default cp1252 codec.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    n_ok = 0
    n_fail = 0
    for i, uid in enumerate(uniprot_ids, 1):
        try:
            result = subprocess.run(
                [sys.executable, str(consensus_script), "--uniprot", uid],
                capture_output=True, text=True, timeout=120,
                env=env, cwd=str(root),
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                print(f"  {GREEN}✓{RESET} [{i:>2}/{len(uniprot_ids)}] {uid}")
                n_ok += 1
            else:
                err_tail = (result.stderr or "")[-150:].replace("\n", " ")
                print(f"  {RED}✗{RESET} [{i:>2}/{len(uniprot_ids)}] {uid}  {err_tail}")
                n_fail += 1
        except subprocess.TimeoutExpired:
            print(f"  {YELLOW}!{RESET} [{i:>2}/{len(uniprot_ids)}] {uid}  (timeout)")
            n_fail += 1
        except Exception as e:
            print(f"  {RED}✗{RESET} [{i:>2}/{len(uniprot_ids)}] {uid}  {e}")
            n_fail += 1

    print()
    print(f"  {GREEN}OK: {n_ok}{RESET}, {RED}FAIL: {n_fail}{RESET}")
    print()
    print("Next:  python validation/run_validation.py --score-only")


if __name__ == "__main__":
    main()