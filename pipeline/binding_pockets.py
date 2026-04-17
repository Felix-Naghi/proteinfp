"""
pipeline/04_binding_pockets.py
───────────────────────────────
Module 04 — Binding pocket detection.

Uses a distance-based pocket detection approach that works reliably on
AlphaFold structures (which lack hydrogen atoms and have flexible regions).

Algorithm (simplified alpha-sphere method):
  1. For each residue CA, find all neighbouring CA atoms within a shell
  2. A point surrounded by atoms on multiple sides = pocket centre candidate
  3. Score each candidate by: burial depth, neighbour count, residue chemistry
  4. Cluster nearby candidates into discrete pockets
  5. Score pockets for druggability

This approach is more robust than grid flood-fill for AlphaFold structures
because it doesn't require fully enclosed cavities.

Usage (standalone):
    python pipeline/04_binding_pockets.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.binding_pockets import detect_binding_pockets
    result = detect_binding_pockets(parsed_structure, active_site_residues, sasa_map)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np
from scipy.spatial import cKDTree

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, HYDROPHOBICITY

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# A residue is considered a pocket lining residue if its CA is within this
# distance of the pocket centre
POCKET_LINING_CUTOFF   = 8.0   # Å

# Minimum number of surrounding atoms to consider a point "buried"
MIN_BURIAL_COUNT       = 8

# Shell for counting surrounding atoms: inner and outer radius
BURIAL_INNER           = 4.0   # Å  — atoms closer than this don't count
BURIAL_OUTER           = 10.0  # Å  — atoms further than this don't count

# Cluster nearby pocket centres within this distance into one pocket
CLUSTER_MERGE_DIST     = 6.0   # Å

# Minimum pocket score to report
MIN_POCKET_SCORE       = 0.15

# Maximum pockets to return
MAX_POCKETS            = 10

# Charge lookup (same as physicochemical module)
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class BindingPocket:
    pocket_id:            str
    volume_A3:            float
    center:               list[float]
    n_lining:             int
    lining_residues:      list[int]
    lining_letters:       list[str]
    mean_hydrophobicity:  float
    net_charge:           float
    burial_score:         float
    near_active_site:     bool
    active_site_residues: list[int]
    druggability_score:   float
    druggability_class:   str
    enclosure:            float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BindingPocketResult:
    uniprot_id:        str
    length:            int
    n_pockets:         int                   = 0
    pockets:           list[BindingPocket]   = field(default_factory=list)
    top_pocket_id:     str                   = ""
    top_pocket_volume: float                 = 0.0
    n_druggable:       int                   = 0

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Binding pocket detection: {self.uniprot_id}",
            f"  Pockets found    : {self.n_pockets}",
            f"  Druggable (>0.5) : {self.n_druggable}",
        ]
        for p in self.pockets[:MAX_POCKETS]:
            lines.append(
                f"  [{p.pocket_id}] vol≈{p.volume_A3:.0f}Å³  "
                f"drug={p.druggability_score:.2f} ({p.druggability_class})  "
                f"lining={p.n_lining} residues  "
                f"{'*active site*' if p.near_active_site else ''}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved binding pockets JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def detect_binding_pockets(
    structure:            ParsedStructure,
    active_site_residues: Optional[set[int]] = None,
    sasa_map:             Optional[dict[tuple[str, int], float]] = None,
) -> BindingPocketResult:
    """
    Detect and score binding pockets using alpha-sphere-inspired method.
    """
    log.info(f"── Module 04: Binding pocket detection for {structure.uniprot_id} ──")

    active_set = active_site_residues or set()
    sasa       = sasa_map or {}

    # Build CA coordinate array and residue lookup
    res_nums  = [r.residue_number for r in structure.residues]
    res_aas   = [r.one_letter     for r in structure.residues]
    res_chain = [r.chain_id       for r in structure.residues]
    ca_coords = np.array([r.coords for r in structure.residues], dtype=np.float64)

    if len(ca_coords) == 0:
        log.warning("  No residues found")
        return BindingPocketResult(uniprot_id=structure.uniprot_id, length=0)

    log.info(f"  [1/3] Finding buried pocket candidate points "
             f"({len(ca_coords)} residues)...")

    # Build KD-tree for fast neighbour queries
    tree = cKDTree(ca_coords)

    # ── Step 1: Generate candidate pocket centre points ───────────────────────
    # For every pair of residues within BURIAL_OUTER of each other,
    # test the midpoint between them. If that midpoint is surrounded by
    # MIN_BURIAL_COUNT+ atoms, it's a pocket candidate.
    candidates = []

    pairs = tree.query_pairs(r=BURIAL_OUTER)
    log.info(f"    Testing {len(pairs):,} residue pairs...")

    for i, j in pairs:
        midpoint = (ca_coords[i] + ca_coords[j]) / 2.0

        # Count atoms in the burial shell around the midpoint
        nearby = tree.query_ball_point(midpoint, r=BURIAL_OUTER)
        shell  = [
            k for k in nearby
            if BURIAL_INNER <= np.linalg.norm(ca_coords[k] - midpoint) <= BURIAL_OUTER
        ]

        if len(shell) >= MIN_BURIAL_COUNT:
            burial_score = len(shell) / 20.0  # normalise to ~0-1
            candidates.append((midpoint, burial_score, shell))

    log.info(f"    Candidate pocket points: {len(candidates)}")

    if not candidates:
        log.warning("  No buried points found — protein may be too disordered")
        return BindingPocketResult(
            uniprot_id=structure.uniprot_id,
            length=structure.length,
        )

    # ── Step 2: Cluster candidates into discrete pockets ─────────────────────
    log.info("  [2/3] Clustering candidates into pockets...")
    pocket_centres = _cluster_candidates(candidates)
    log.info(f"    Pocket clusters found: {len(pocket_centres)}")

    # ── Step 3: Score each pocket ─────────────────────────────────────────────
    log.info("  [3/3] Scoring pockets...")
    pockets = []

    for idx, (centre, burial, shell_indices) in enumerate(pocket_centres):
        # Find all residues within lining cutoff
        lining_idx = tree.query_ball_point(centre, r=POCKET_LINING_CUTOFF)
        lining_nums    = [res_nums[k]  for k in lining_idx]
        lining_letters = [res_aas[k]   for k in lining_idx]
        lining_chains  = [res_chain[k] for k in lining_idx]

        if not lining_nums:
            continue

        # Chemistry of lining
        hydrophobes = [HYDROPHOBICITY.get(aa, 0.0) for aa in lining_letters]
        charges     = [CHARGE_AT_PH7.get(aa, 0.0)  for aa in lining_letters]
        mean_hydro  = float(np.mean(hydrophobes))
        net_charge  = float(sum(charges))

        # SASA-based enclosure: buried lining residues = more enclosed pocket
        lining_sasas = [
            sasa.get((lining_chains[k], lining_nums[k]), 50.0)
            for k in range(len(lining_nums))
        ]
        mean_lining_sasa = float(np.mean(lining_sasas)) if lining_sasas else 50.0
        enclosure = max(0.0, 1.0 - mean_lining_sasa / 100.0)

        # Volume estimate: based on number of buried shell atoms
        volume_est = len(shell_indices) * 20.0

        # Active site overlap
        active_in_pocket = [n for n in lining_nums if n in active_set]
        near_active      = len(active_in_pocket) > 0

        # Druggability score
        drug_score = _druggability_score(
            burial, enclosure, mean_hydro, net_charge,
            len(lining_nums), near_active, volume_est
        )

        if drug_score < MIN_POCKET_SCORE:
            continue

        if drug_score >= 0.6:
            drug_class = "high"
        elif drug_score >= 0.35:
            drug_class = "medium"
        else:
            drug_class = "low"

        pockets.append(BindingPocket(
            pocket_id=f"P{idx+1}",
            volume_A3=round(volume_est, 1),
            center=[round(float(c), 2) for c in centre],
            n_lining=len(lining_nums),
            lining_residues=sorted(lining_nums),
            lining_letters=lining_letters,
            mean_hydrophobicity=round(mean_hydro, 3),
            net_charge=round(net_charge, 1),
            burial_score=round(float(burial), 3),
            near_active_site=near_active,
            active_site_residues=sorted(active_in_pocket),
            druggability_score=round(drug_score, 3),
            druggability_class=drug_class,
            enclosure=round(enclosure, 3),
        ))

    # Sort by druggability and re-label
    pockets.sort(key=lambda p: p.druggability_score, reverse=True)
    pockets = pockets[:MAX_POCKETS]
    for i, p in enumerate(pockets):
        p.pocket_id = f"P{i+1}"

    n_druggable = sum(1 for p in pockets if p.druggability_score > 0.5)
    top = pockets[0] if pockets else None

    result = BindingPocketResult(
        uniprot_id=structure.uniprot_id,
        length=structure.length,
        n_pockets=len(pockets),
        pockets=pockets,
        top_pocket_id=top.pocket_id if top else "",
        top_pocket_volume=top.volume_A3 if top else 0.0,
        n_druggable=n_druggable,
    )

    log.info(result.summary())
    return result


# ── Candidate clustering ───────────────────────────────────────────────────────

def _cluster_candidates(
    candidates: list[tuple],
) -> list[tuple]:
    """
    Merge nearby candidate pocket points into discrete pockets.
    Returns list of (centre, max_burial_score, shell_indices) per cluster.
    Uses greedy distance-based merging.
    """
    if not candidates:
        return []

    points = np.array([c[0] for c in candidates])
    scores = np.array([c[1] for c in candidates])
    shells = [c[2] for c in candidates]

    tree    = cKDTree(points)
    visited = np.zeros(len(candidates), dtype=bool)
    clusters = []

    # Process in order of decreasing burial score (best centres first)
    order = np.argsort(-scores)

    for idx in order:
        if visited[idx]:
            continue

        # Find all unvisited candidates within merge distance
        nearby = tree.query_ball_point(points[idx], r=CLUSTER_MERGE_DIST)
        members = [k for k in nearby if not visited[k]]

        if not members:
            continue

        # Cluster centre = weighted mean by burial score
        member_pts    = points[members]
        member_scores = scores[members]
        centre = np.average(member_pts, weights=member_scores, axis=0)
        best_score = float(member_scores.max())

        # Combine all shell indices
        combined_shell = set()
        for k in members:
            combined_shell.update(shells[k])

        clusters.append((centre, best_score, list(combined_shell)))

        for k in members:
            visited[k] = True

    return clusters


# ── Druggability scoring ───────────────────────────────────────────────────────

def _druggability_score(
    burial:      float,
    enclosure:   float,
    hydro:       float,
    charge:      float,
    n_lining:    int,
    near_active: bool,
    volume:      float,
) -> float:
    score = 0.0

    # Burial depth (how surrounded the pocket is)
    score += 0.25 * min(burial, 1.0)

    # Enclosure (how buried the lining residues are)
    score += 0.20 * enclosure

    # Lining size: 8-20 residues is ideal
    if 8 <= n_lining <= 20:
        score += 0.20
    elif 5 <= n_lining <= 30:
        score += 0.10
    elif n_lining > 0:
        score += 0.05

    # Hydrophobicity: mixed preferred
    if -1.0 <= hydro <= 2.0:
        score += 0.15
    elif hydro > 0:
        score += 0.07

    # Active site proximity
    if near_active:
        score += 0.20

    return min(score, 1.0)


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 04 — Binding pocket detection.

    Example:
        python pipeline/binding_pockets.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_binding_pockets.json"

    if not pdb_path.exists():
        log.error(f".pdb not found: {pdb_path}\n"
                  f"  Run Module 01 first: python pipeline/fetch_structure.py "
                  f"--uniprot {uniprot}")
        raise SystemExit(1)

    parsed = parse_pdb(pdb_path, uniprot)

    # Load SASA from Module 02
    sasa_map: dict[tuple[str, int], float] = {}
    phys_path = inter_dir / f"{uniprot}_physicochemical.json"
    if phys_path.exists():
        with open(phys_path) as f:
            phys_data = json.load(f)
        for rec in phys_data.get("residues", []):
            sasa_map[(rec["chain_id"], rec["residue_number"])] = rec["sasa"]
        log.info(f"  Loaded SASA for {len(sasa_map)} residues from Module 02")

    # Load active site residues from Module 03
    active_set: set[int] = set()
    active_path = inter_dir / f"{uniprot}_active_sites.json"
    if active_path.exists():
        with open(active_path) as f:
            active_data = json.load(f)
        for rec in active_data.get("active_residues", []):
            if rec.get("confidence") in {"HIGH", "MEDIUM"}:
                active_set.add(rec["residue_number"])
        log.info(f"  Loaded {len(active_set)} active site residues from Module 03")

    result = detect_binding_pockets(parsed, active_set, sasa_map)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()