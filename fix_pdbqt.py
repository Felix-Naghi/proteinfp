"""
fix_pdbqt.py
─────────────
One-shot repair script for malformed PDBQT receptor files.

The file in data/structures/P04637.pdbqt has its charge field
left-aligned with a trailing space ("0.000 ") in cols 71-76.
Vina's strict parser rejects this with:
    PDBQT parsing error: Charge "0.000 " is not valid.

This script reads the existing PDBQT and rewrites every ATOM/HETATM
line with the charge right-aligned in cols 71-76 ("  0.000" pattern),
which is what Vina expects.

Usage:
    python fix_pdbqt.py data/structures/P04637.pdbqt

Or for a whole directory:
    python fix_pdbqt.py data/structures/

The original file is backed up to <name>.pdbqt.bak before being overwritten.
"""

import sys
from pathlib import Path


def fix_pdbqt_line(line: str) -> str:
    """
    Re-emit a single ATOM/HETATM PDBQT line with the charge right-aligned
    in cols 71-76 and the AD4 atom type left-aligned in cols 77-78.

    Reads the existing fields rather than recomputing — we only fix layout.
    """
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line.rstrip("\n")

    # Pad to at least 80 chars so col-slicing is safe
    line = line.rstrip("\n").ljust(80)

    head = line[:54]                      # cols 1-54
    occ_str  = line[54:60].strip()        # cols 55-60
    temp_str = line[60:66].strip()        # cols 61-66

    try:    occ = float(occ_str)  if occ_str  else 1.00
    except: occ = 1.00
    try:    temp = float(temp_str) if temp_str else 0.00
    except: temp = 0.00

    # The remainder (cols 67+) holds spaces, charge, and AD4 type in some
    # order depending on the broken writer.  Pull out the charge and type
    # by scanning for a float and the trailing letters.
    tail = line[66:].strip()              # everything after temp factor

    # Tail looks like "0.000    N" (broken) or "  0.000 N" (correct) or
    # "0.000 N" with various spacing.  Split on whitespace.
    parts = tail.split()
    charge = 0.000
    ad4    = "C"
    if parts:
        # First token that parses as a float = charge
        for i, tok in enumerate(parts):
            try:
                charge = float(tok)
                # Whatever comes after the charge token is the AD4 type
                if i + 1 < len(parts):
                    ad4 = parts[i + 1]
                break
            except ValueError:
                continue
        else:
            # No float found — the last 1-2 chars must be the type
            ad4 = parts[-1]

    # AD4 types are 1-2 chars: C, N, OA, SA, HD, NA, etc.
    ad4 = ad4[:2]

    # Emit the canonical AutoDock4 PDBQT layout, exactly like obabel:
    #   cols 1-54: head
    #   cols 55-60: occupancy (6.2f)
    #   cols 61-66: temp factor (6.2f)
    #   cols 67-70: 4 spaces
    #   cols 71-76: charge (6.3f, right-aligned)
    #   col 77:     space
    #   cols 78-79: AD4 atom type (left-aligned in 2 chars)
    return f"{head:<54}{occ:6.2f}{temp:6.2f}    {charge:6.3f} {ad4:<2s}"


def fix_file(path: Path) -> bool:
    """Repair one PDBQT file. Returns True if changes were made."""
    if not path.exists():
        print(f"  [SKIP] {path} — does not exist")
        return False
    if path.suffix.lower() != ".pdbqt":
        print(f"  [SKIP] {path} — not a .pdbqt file")
        return False

    original = path.read_text().splitlines()
    repaired = [fix_pdbqt_line(ln) for ln in original]

    if original == repaired:
        print(f"  [OK]   {path} — already valid")
        return False

    # Back up the original
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text("\n".join(original) + "\n")
    path.write_text("\n".join(repaired) + "\n")
    n_atoms = sum(1 for ln in repaired if ln.startswith(("ATOM", "HETATM")))
    print(f"  [FIX]  {path} — {n_atoms} atom lines rewritten (backup: {bak.name})")
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = list(target.glob("*.pdbqt"))
        if not files:
            print(f"No .pdbqt files found in {target}")
            sys.exit(1)
        print(f"Found {len(files)} PDBQT file(s) in {target}")
        n_fixed = sum(1 for f in files if fix_file(f))
        print(f"\nDone. Fixed {n_fixed}/{len(files)} files.")
    elif target.is_file():
        fix_file(target)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()