"""
pipeline/02_physicochemical.py
───────────────────────────────
Module 02 — Physicochemical surface analysis.

Takes the .pdb from Module 01 and computes the full chemical surface profile:
  1. Solvent-Accessible Surface Area (SASA) per residue via freesasa
  2. Per-residue charge state at physiological pH (7.4)
  3. Hydrophobicity map (Kyte-Doolittle scale)
  4. Secondary structure assignment (DSSP via BioPython)
  5. Hydrophobic patch detection (clusters of exposed hydrophobic residues)
  6. Charged surface patch detection (positive / negative clusters)
  7. Per-region summary for active site / binding pocket modules

Modules 03, 04, and 06 all consume the output of this module.

Usage (standalone):
    python pipeline/02_physicochemical.py --uniprot P04637
    python pipeline/02_physicochemical.py --pdb data/structures/P04637.pdb --uniprot P04637

Usage (from orchestrator):
    from pipeline.physicochemical import compute_physicochemical
    result = compute_physicochemical(structure_result)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import freesasa
import numpy as np
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

from utils.config import cfg, get_logger
from utils.pdb_parser import (
    ParsedStructure, ResidueInfo,
    AA3TO1, HYDROPHOBICITY,
    parse_pdb,
)

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Charge of each amino acid at pH 7.4
# Positive = basic, Negative = acidic, 0 = neutral
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}

# DSSP secondary structure codes → readable labels
DSSP_LABELS: dict[str, str] = {
    "H": "alpha_helix",
    "B": "beta_bridge",
    "E": "beta_strand",
    "G": "helix_3_10",
    "I": "pi_helix",
    "T": "turn",
    "S": "bend",
    "-": "coil",
    " ": "coil",
}

# Minimum cluster size for patch detection
MIN_PATCH_SIZE = 3

# SASA threshold — residue considered "exposed" if SASA > this (Å²)
EXPOSED_SASA_THRESHOLD = 20.0


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ResiduePhysChem:
    """Physicochemical annotation for a single residue."""
    residue_number: int
    one_letter:     str
    chain_id:       str
    sasa:           float        # Å² solvent-accessible surface area
    sasa_fraction:  float        # fraction of max theoretical SASA (0–1)
    is_exposed:     bool         # True if sasa > threshold
    charge:         float        # charge at pH 7.4
    hydrophobicity: float        # Kyte-Doolittle
    secondary_structure: str     # alpha_helix / beta_strand / coil / etc.
    plddt:          float        # carried over from Module 01

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SurfacePatch:
    """A cluster of residues with shared physicochemical character."""
    patch_type:     str          # "hydrophobic" / "positive" / "negative"
    residue_numbers: list[int]
    size:           int
    mean_sasa:      float
    total_sasa:     float
    centroid:       list[float]  # [x, y, z] geometric centre
    mean_property:  float        # mean hydrophobicity or charge

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhysicochemResult:
    """Full physicochemical surface profile. Output of Module 02."""
    uniprot_id:          str
    pdb_path:            str
    length:              int
    residues:            list[ResiduePhysChem] = field(default_factory=list)

    # Global surface stats
    total_sasa:          float = 0.0
    mean_sasa_per_res:   float = 0.0
    n_exposed:           int   = 0
    exposed_fraction:    float = 0.0

    # Charge summary
    total_charge:        float = 0.0
    n_positive:          int   = 0
    n_negative:          int   = 0
    charge_asymmetry:    str   = ""   # "positive" / "negative" / "neutral"

    # Hydrophobicity summary
    mean_hydrophobicity: float = 0.0
    hydrophobic_fraction: float = 0.0

    # Secondary structure fractions
    helix_fraction:      float = 0.0
    strand_fraction:     float = 0.0
    coil_fraction:       float = 0.0

    # Surface patches
    hydrophobic_patches: list[SurfacePatch] = field(default_factory=list)
    positive_patches:    list[SurfacePatch] = field(default_factory=list)
    negative_patches:    list[SurfacePatch] = field(default_factory=list)

    # DSSP availability flag
    dssp_available:      bool = False

    def summary(self) -> str:
        return (
            f"\n{'─'*60}\n"
            f"  Physicochemical profile: {self.uniprot_id}\n"
            f"  Total SASA      : {self.total_sasa:.1f} Å²\n"
            f"  Exposed residues: {self.n_exposed}/{self.length} "
            f"({self.exposed_fraction*100:.1f}%)\n"
            f"  Net charge (pH7): {self.total_charge:+.1f} "
            f"({self.charge_asymmetry})\n"
            f"  Mean hydrophob. : {self.mean_hydrophobicity:+.2f}\n"
            f"  2° structure    : "
            f"helix {self.helix_fraction*100:.0f}% | "
            f"strand {self.strand_fraction*100:.0f}% | "
            f"coil {self.coil_fraction*100:.0f}%\n"
            f"  Hydrophob. patches: {len(self.hydrophobic_patches)}\n"
            f"  Positive patches  : {len(self.positive_patches)}\n"
            f"  Negative patches  : {len(self.negative_patches)}\n"
            f"{'─'*60}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved physicochemical JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def compute_physicochemical(
    structure: ParsedStructure,
) -> PhysicochemResult:
    """
    Compute full physicochemical surface profile from a ParsedStructure.

    Args:
        structure: Output of Module 01 parse_pdb()

    Returns:
        PhysicochemResult with per-residue and patch-level annotations.
    """
    log.info(f"── Module 02: Physicochemical analysis for {structure.uniprot_id} ──")

    # ── Step 1: SASA via freesasa ─────────────────────────────────────────────
    log.info("  [1/4] Computing SASA (freesasa)...")
    sasa_map = _compute_sasa(structure.pdb_path)

    # ── Step 2: Secondary structure via DSSP ─────────────────────────────────
    log.info("  [2/4] Assigning secondary structure (DSSP)...")
    ss_map, dssp_ok = _compute_secondary_structure(structure.pdb_path)

    # ── Step 3: Build per-residue records ─────────────────────────────────────
    log.info("  [3/4] Building per-residue physicochemical profile...")
    res_records = _build_residue_records(structure.residues, sasa_map, ss_map)

    # ── Step 4: Detect surface patches ────────────────────────────────────────
    log.info("  [4/4] Detecting surface patches...")
    hydrophobic_patches = _find_patches(res_records, structure.residues,
                                        "hydrophobic", min_size=MIN_PATCH_SIZE)
    positive_patches    = _find_patches(res_records, structure.residues,
                                        "positive",   min_size=MIN_PATCH_SIZE)
    negative_patches    = _find_patches(res_records, structure.residues,
                                        "negative",   min_size=MIN_PATCH_SIZE)

    # ── Compute global stats ──────────────────────────────────────────────────
    sasas        = np.array([r.sasa for r in res_records])
    charges      = np.array([r.charge for r in res_records])
    hydrophobs   = np.array([r.hydrophobicity for r in res_records])
    ss_labels    = [r.secondary_structure for r in res_records]

    total_sasa        = float(np.sum(sasas))
    mean_sasa         = float(np.mean(sasas))
    n_exposed         = int(np.sum([r.is_exposed for r in res_records]))
    exposed_fraction  = n_exposed / max(len(res_records), 1)
    total_charge      = float(np.sum(charges))
    n_positive        = int(np.sum(charges > 0))
    n_negative        = int(np.sum(charges < 0))
    mean_hydrophob    = float(np.mean(hydrophobs))
    hydrophob_frac    = float(np.mean(hydrophobs > 0))

    if total_charge > 1:
        charge_asymmetry = "positive"
    elif total_charge < -1:
        charge_asymmetry = "negative"
    else:
        charge_asymmetry = "neutral"

    helix_frac  = ss_labels.count("alpha_helix") / max(len(ss_labels), 1)
    strand_frac = ss_labels.count("beta_strand") / max(len(ss_labels), 1)
    coil_frac   = 1.0 - helix_frac - strand_frac

    result = PhysicochemResult(
        uniprot_id=structure.uniprot_id,
        pdb_path=structure.pdb_path,
        length=structure.length,
        residues=res_records,
        total_sasa=round(total_sasa, 2),
        mean_sasa_per_res=round(mean_sasa, 2),
        n_exposed=n_exposed,
        exposed_fraction=round(exposed_fraction, 3),
        total_charge=round(total_charge, 1),
        n_positive=n_positive,
        n_negative=n_negative,
        charge_asymmetry=charge_asymmetry,
        mean_hydrophobicity=round(mean_hydrophob, 3),
        hydrophobic_fraction=round(hydrophob_frac, 3),
        helix_fraction=round(helix_frac, 3),
        strand_fraction=round(strand_frac, 3),
        coil_fraction=round(coil_frac, 3),
        hydrophobic_patches=hydrophobic_patches,
        positive_patches=positive_patches,
        negative_patches=negative_patches,
        dssp_available=dssp_ok,
    )

    log.info(result.summary())
    return result


# ── SASA computation ───────────────────────────────────────────────────────────

def _compute_sasa(pdb_path: str) -> dict[tuple[str, int], float]:
    """
    Run freesasa on the .pdb file.
    Returns a dict keyed by (chain_id, residue_number) → SASA in Å².
    """
    try:
        structure = freesasa.Structure(str(pdb_path))
        result    = freesasa.calc(structure)
        residue_areas = result.residueAreas()

        sasa_map: dict[tuple[str, int], float] = {}
        for chain_id, residues in residue_areas.items():
            for res_num_str, area in residues.items():
                try:
                    res_num = int(res_num_str.strip())
                    sasa_map[(chain_id, res_num)] = round(area.total, 3)
                except ValueError:
                    continue

        log.debug(f"    SASA computed for {len(sasa_map)} residues")
        return sasa_map

    except Exception as e:
        log.warning(f"    freesasa failed: {e} — using zero SASA fallback")
        return {}


# ── Secondary structure via DSSP ───────────────────────────────────────────────

def _compute_secondary_structure(
    pdb_path: str,
) -> tuple[dict[tuple[str, int], str], bool]:
    """
    Assign secondary structure using BioPython's DSSP wrapper.
    Falls back to all-coil if DSSP binary not available.

    Returns:
        ss_map : dict (chain_id, res_num) → ss_label string
        ok     : True if DSSP ran successfully
    """
    ss_map: dict[tuple[str, int], str] = {}

    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_path))
        model     = structure[0]

        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")

        for key in dssp.keys():
            chain_id = key[0]
            res_num  = key[1][1]
            ss_code  = dssp[key][2]
            ss_map[(chain_id, res_num)] = DSSP_LABELS.get(ss_code, "coil")

        log.debug(f"    DSSP assigned for {len(ss_map)} residues")
        return ss_map, True

    except Exception as e:
        log.warning(
            f"    DSSP not available ({e}).\n"
            f"    Install mkdssp: https://github.com/PDB-REDO/dssp/releases\n"
            f"    Falling back to coil for all residues."
        )
        return {}, False


# ── Per-residue record builder ─────────────────────────────────────────────────

# Maximum theoretical SASA per residue (Gly-X-Gly extended, Miller et al.)
MAX_SASA: dict[str, float] = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G":  97.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def _build_residue_records(
    residues: list[ResidueInfo],
    sasa_map: dict[tuple[str, int], float],
    ss_map:   dict[tuple[str, int], str],
) -> list[ResiduePhysChem]:
    records = []
    for res in residues:
        key  = (res.chain_id, res.residue_number)
        sasa = sasa_map.get(key, 0.0)
        max_s = MAX_SASA.get(res.one_letter, 200.0)
        sasa_frac = min(sasa / max_s, 1.0) if max_s > 0 else 0.0

        records.append(ResiduePhysChem(
            residue_number=res.residue_number,
            one_letter=res.one_letter,
            chain_id=res.chain_id,
            sasa=sasa,
            sasa_fraction=round(sasa_frac, 3),
            is_exposed=(sasa > EXPOSED_SASA_THRESHOLD),
            charge=CHARGE_AT_PH7.get(res.one_letter, 0.0),
            hydrophobicity=HYDROPHOBICITY.get(res.one_letter, 0.0),
            secondary_structure=ss_map.get(key, "coil"),
            plddt=res.plddt,
        ))
    return records


# ── Surface patch detection ────────────────────────────────────────────────────

def _find_patches(
    records:   list[ResiduePhysChem],
    residues:  list[ResidueInfo],
    patch_type: str,
    min_size:  int = 3,
    distance_cutoff: float = 8.0,
) -> list[SurfacePatch]:
    """
    Find spatial clusters of exposed residues sharing a property.

    patch_type: "hydrophobic" | "positive" | "negative"
    Uses a simple distance-based clustering on CA coordinates.
    """
    # Select candidate residues
    candidates = []
    coord_map  = {r.residue_number: r.coords for r in residues}

    for rec in records:
        if not rec.is_exposed:
            continue
        if patch_type == "hydrophobic" and rec.hydrophobicity > 0:
            candidates.append(rec)
        elif patch_type == "positive" and rec.charge > 0:
            candidates.append(rec)
        elif patch_type == "negative" and rec.charge < 0:
            candidates.append(rec)

    if len(candidates) < min_size:
        return []

    # Build distance matrix and cluster greedily
    res_nums = [r.residue_number for r in candidates]
    coords   = np.array([coord_map.get(n, [0.0, 0.0, 0.0]) for n in res_nums])

    visited  = set()
    patches  = []

    for i, rec in enumerate(candidates):
        if rec.residue_number in visited:
            continue

        cluster = [rec.residue_number]
        visited.add(rec.residue_number)

        for j, other in enumerate(candidates):
            if other.residue_number in visited:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist <= distance_cutoff:
                cluster.append(other.residue_number)
                visited.add(other.residue_number)

        if len(cluster) < min_size:
            continue

        # Compute patch stats
        patch_coords = np.array([coord_map.get(n, [0.0, 0.0, 0.0])
                                  for n in cluster])
        centroid     = patch_coords.mean(axis=0).tolist()
        patch_recs   = [r for r in candidates if r.residue_number in cluster]
        mean_sasa    = float(np.mean([r.sasa for r in patch_recs]))
        total_sasa   = float(np.sum([r.sasa for r in patch_recs]))

        if patch_type == "hydrophobic":
            mean_prop = float(np.mean([r.hydrophobicity for r in patch_recs]))
        else:
            mean_prop = float(np.mean([r.charge for r in patch_recs]))

        patches.append(SurfacePatch(
            patch_type=patch_type,
            residue_numbers=sorted(cluster),
            size=len(cluster),
            mean_sasa=round(mean_sasa, 2),
            total_sasa=round(total_sasa, 2),
            centroid=[round(c, 2) for c in centroid],
            mean_property=round(mean_prop, 3),
        ))

    patches.sort(key=lambda p: p.total_sasa, reverse=True)
    log.debug(f"    {patch_type} patches found: {len(patches)}")
    return patches


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637). Looks for existing .pdb in data/structures/")
@click.option("--pdb", "-p", default=None,
              help="Override: path to a specific .pdb file")
def main(uniprot: str, pdb: Optional[str]) -> None:
    """
    Module 02 — Physicochemical surface analysis.

    Example:
        python pipeline/02_physicochemical.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uniprot}_physicochemical.json"

    # Locate the .pdb
    if pdb:
        pdb_path = Path(pdb)
    else:
        pdb_path = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"

    if not pdb_path.exists():
        log.error(
            f".pdb not found: {pdb_path}\n"
            f"  Run Module 01 first:  python pipeline/fetch_structure.py --uniprot {uniprot}"
        )
        raise SystemExit(1)

    # Parse the structure (re-use Module 01's parser)
    parsed = parse_pdb(pdb_path, uniprot)

    # Run physicochemical analysis
    result = compute_physicochemical(parsed)

    # Save output
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()