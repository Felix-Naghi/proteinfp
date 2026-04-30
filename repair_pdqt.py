"""
repair_pdbqt.py
────────────────
Finds all .pdbqt files with 0 ATOM records and regenerates them
from their source .pdb files.

Run from project root:
    python repair_pdbqt.py

What happened:
    _fast_pdb_to_pdbqt() in denovo_design.py writes ATOM/HETATM records
    from the PDB into PDBQT format. For some proteins the conversion
    succeeded (file was created) but produced 0 ATOM lines — likely
    because the PDB used non-standard record formatting or the function
    hit an exception mid-write and left an empty file.

    Vina then docks against an empty receptor and returns score=0.00,
    causing the selectivity optimizer to see SI=0.00 for all off-targets.

This script:
    1. Scans data/structures/*.pdbqt for files with 0 ATOM records
    2. For each broken pdbqt, finds the matching .pdb file
    3. Re-runs conversion using a robust fallback that handles
       non-standard PDB formatting
    4. Reports which files were fixed and which still failed
"""

import sys
from pathlib import Path

ROOT          = Path(__file__).resolve().parent
STRUCTURES    = ROOT / "data" / "structures"

# ── Add pipeline to path ──────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))


def _count_atoms(pdbqt_path: Path) -> int:
    """Count ATOM/HETATM records in a pdbqt file."""
    try:
        lines = pdbqt_path.read_text(errors="replace").splitlines()
        return sum(1 for l in lines if l.startswith(("ATOM", "HETATM")))
    except Exception:
        return 0


def _robust_pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """
    Robust PDB → PDBQT converter.
    Handles non-standard AlphaFold PDB formatting.

    AutoDock Vina PDBQT format:
      Cols 1-4:   Record type (ATOM/HETATM)
      Cols 5-11:  Atom serial
      Cols 13-16: Atom name
      Cols 18-20: Residue name
      Col  22:    Chain ID
      Cols 23-26: Residue sequence number
      Cols 31-54: X, Y, Z coordinates
      Cols 55-60: Occupancy
      Cols 61-66: B-factor (= pLDDT for AlphaFold)
      Cols 77-78: Partial charge (for PDBQT)
      Cols 79-80: AutoDock atom type

    AutoDock4 atom type mapping by element:
      C  → C  (aliphatic) or A (aromatic)
      N  → N  (H-bond acceptor) or NA (donor)
      O  → OA (acceptor)
      S  → SA (acceptor)
      H  → HD (donor) or H
      P  → P
      F  → F
      Cl → Cl
      Br → Br
      I  → I
      All others → element symbol
    """
    AD4_TYPES = {
        "C": "C", "N": "N", "O": "OA", "S": "SA",
        "H": "HD", "P": "P", "F": "F", "CL": "Cl",
        "BR": "Br", "I": "I", "FE": "Fe", "ZN": "Zn",
        "MG": "Mg", "CA": "Ca", "MN": "Mn", "NA": "N",
    }

    try:
        raw = pdb_path.read_text(errors="replace")
    except Exception as e:
        print(f"    Could not read {pdb_path.name}: {e}")
        return False

    lines_out = []
    n_atoms   = 0

    for line in raw.splitlines():
        rec = line[:6].strip()

        # Copy TER and END records unchanged
        if rec in ("TER", "END", "ENDMDL"):
            lines_out.append(line.rstrip())
            continue

        if rec not in ("ATOM", "HETATM"):
            continue

        # Parse the element from cols 76-78 (standard PDB) or atom name
        element = ""
        if len(line) > 76:
            element = line[76:78].strip().upper()
        if not element or element == "":
            # Fall back to atom name (cols 12-16)
            atom_name = line[12:16].strip() if len(line) > 16 else ""
            # Strip digit prefixes (1HB → H, 2CA → C)
            element = "".join(c for c in atom_name if c.isalpha())
            element = element[:2].upper() if element else "C"

        ad4_type = AD4_TYPES.get(element[:2], element[:1] if element else "C")

        # Ensure line is at least 60 chars for coordinate fields
        line_padded = line.rstrip().ljust(80)

        # Build PDBQT record:
        # Keep cols 1-66 from PDB, add charge=0.000 and AD4 type
        pdb_part = line_padded[:66]
        pdbqt_line = f"{pdb_part}    0.000    {ad4_type:<2s}"
        lines_out.append(pdbqt_line.rstrip())
        n_atoms += 1

    if n_atoms == 0:
        print(f"    WARNING: No ATOM records found in {pdb_path.name}")
        # Write a minimal placeholder so vina at least doesn't error
        # (vina will just return very poor scores for empty receptors)
        return False

    lines_out.append("END")
    pdbqt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return True


def main():
    print("=" * 60)
    print("  PDBQT Repair Tool")
    print("=" * 60)

    # ── Find all pdbqt files ───────────────────────────────────────────────
    all_pdbqt = sorted(STRUCTURES.glob("*.pdbqt"))
    print(f"\n  Found {len(all_pdbqt)} .pdbqt files in {STRUCTURES}")

    # ── Identify broken ones ───────────────────────────────────────────────
    broken = []
    ok     = []
    for pdbqt in all_pdbqt:
        n = _count_atoms(pdbqt)
        if n == 0:
            broken.append(pdbqt)
        else:
            ok.append((pdbqt, n))

    print(f"  OK (have ATOM records)  : {len(ok)}")
    print(f"  Broken (0 ATOM records) : {len(broken)}")

    if not broken:
        print("\n  All pdbqt files look healthy. Nothing to repair.")
        return

    print(f"\n  Broken files:")
    for p in broken:
        print(f"    {p.name}")

    # ── Repair each broken file ────────────────────────────────────────────
    print(f"\n  Repairing {len(broken)} files...")
    fixed   = []
    failed  = []

    for pdbqt_path in broken:
        uid      = pdbqt_path.stem              # e.g. Q14524
        pdb_path = STRUCTURES / f"{uid}.pdb"

        if not pdb_path.exists():
            print(f"  ✗ {uid}: source .pdb not found — skipping")
            failed.append(uid)
            continue

        print(f"  → Repairing {uid}...", end=" ")
        pdb_atoms = sum(
            1 for l in pdb_path.read_text(errors="replace").splitlines()
            if l.startswith(("ATOM", "HETATM"))
        )
        print(f"(source .pdb has {pdb_atoms} ATOM records) ...", end=" ")

        if pdb_atoms == 0:
            print(f"✗ source PDB also empty — cannot repair")
            failed.append(uid)
            continue

        success = _robust_pdb_to_pdbqt(pdb_path, pdbqt_path)

        if success:
            n_fixed = _count_atoms(pdbqt_path)
            print(f"✓ {n_fixed} ATOM records written")
            fixed.append(uid)
        else:
            print(f"✗ conversion failed")
            failed.append(uid)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Repair complete:")
    print(f"    Fixed   : {len(fixed)}")
    print(f"    Failed  : {len(failed)}")
    print(f"    Already OK: {len(ok)}")

    if fixed:
        print(f"\n  Fixed: {', '.join(fixed)}")

    if failed:
        print(f"\n  Still broken: {', '.join(failed)}")
        print(f"  For these proteins, re-run Module 01:")
        for uid in failed:
            print(f"    python pipeline\\01_fetch_structure.py --uniprot {uid}")

    if fixed:
        print(f"\n  Re-run the de novo design to get real selectivity scores:")
        print(f"    python pipeline\\denovo_design.py --uniprot P04637 \\")
        print(f"        --vina pipeline\\vina.exe --generations 30")


if __name__ == "__main__":
    main()