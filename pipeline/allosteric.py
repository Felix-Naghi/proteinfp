"""
pipeline/05_allosteric.py
──────────────────────────
Module 05 — Allosteric site prediction.

Uses an Elastic Network Model (ENM) / Gaussian Network Model (GNM) approach
implemented with MDAnalysis and NumPy — no external binaries needed.

Theory:
  Proteins move like a network of beads (residues) connected by springs (contacts).
  Allosteric communication travels along mechanically coupled pathways.
  Residues that:
    (a) are NOT in the active site
    (b) but whose perturbation strongly shifts the active site's motion
  are allosteric candidates.

Algorithm:
  1. Build contact map: pairs of CA atoms within CONTACT_CUTOFF angstroms
  2. Construct Kirchhoff matrix (Laplacian of the contact network)
  3. Compute pseudo-inverse → correlation matrix
  4. Score each residue by its mean correlation with known active site residues
  5. Pockets that score high but are far from the active site = allosteric sites
  6. Cross-validate with SASA: allosteric sites should have some surface exposure

Usage (standalone):
    python pipeline/05_allosteric.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.allosteric import predict_allosteric_sites
    result = predict_allosteric_sites(parsed_structure, active_site_nums, sasa_map)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np
from scipy.linalg import pinv

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, HYDROPHOBICITY

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# CA-CA distance cutoff for spring contacts in ENM (Å)
CONTACT_CUTOFF = 8.0

# Minimum correlation score to flag a residue as allosteric candidate
ALLOSTERIC_THRESHOLD = 0.3

# Minimum distance from active site to be considered allosteric (not just active)
MIN_DISTANCE_FROM_ACTIVE = 10.0   # Å CA-CA

# Distance cutoff for grouping allosteric residues into sites
SITE_CLUSTER_CUTOFF = 8.0  # Å

# Minimum cluster size to report as a site
MIN_SITE_SIZE = 2

# Maximum sites to return
MAX_SITES = 8

# Charge at pH 7.4 (same as other modules)
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class AlloResidue:
    """A single residue predicted to be allosterically coupled to the active site."""
    residue_number:    int
    chain_id:          str
    one_letter:        str
    correlation_score: float      # 0-1 coupling to active site
    min_dist_active:   float      # minimum CA distance to any active site residue
    sasa:              float
    is_exposed:        bool
    plddt:             float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlloSite:
    """A cluster of allosteric residues forming a putative allosteric site."""
    site_id:           str        # A1, A2, A3...
    residue_numbers:   list[int]
    residue_letters:   list[str]
    size:              int
    centre:            list[float]
    mean_correlation:  float      # mean coupling score to active site
    min_dist_active:   float      # closest distance to active site
    mean_sasa:         float
    mean_hydrophobicity: float
    net_charge:        float
    coupled_active_residues: list[int]  # active site residues most coupled to this site
    confidence:        str        # HIGH / MEDIUM / LOW

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllostericResult:
    """Full allosteric prediction output. Output of Module 05."""
    uniprot_id:         str
    length:             int
    n_contacts:         int             = 0
    enm_computed:       bool            = False
    allosteric_residues: list[AlloResidue] = field(default_factory=list)
    allosteric_sites:   list[AlloSite]  = field(default_factory=list)
    n_sites:            int             = 0
    communication_pathways: list[list[int]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Allosteric prediction: {self.uniprot_id}",
            f"  ENM contacts used  : {self.n_contacts}",
            f"  Allosteric residues: {len(self.allosteric_residues)}",
            f"  Allosteric sites   : {self.n_sites}",
        ]
        for s in self.allosteric_sites:
            res_str = ", ".join(
                f"{aa}{n}" for aa, n in
                zip(s.residue_letters[:5], s.residue_numbers[:5])
            )
            if len(s.residue_numbers) > 5:
                res_str += f" +{len(s.residue_numbers)-5} more"
            lines.append(
                f"  [{s.site_id}] {res_str}  "
                f"corr={s.mean_correlation:.2f}  "
                f"dist_active={s.min_dist_active:.1f}Å  "
                f"({s.confidence})"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved allosteric JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def predict_allosteric_sites(
    structure:           ParsedStructure,
    active_site_nums:    Optional[set[int]] = None,
    sasa_map:            Optional[dict[tuple[str, int], float]] = None,
) -> AllostericResult:
    """
    Predict allosteric sites using Gaussian Network Model.

    Args:
        structure:        ParsedStructure from Module 01
        active_site_nums: set of active site residue numbers from Module 03
        sasa_map:         (chain_id, res_num) → SASA from Module 02

    Returns:
        AllostericResult with allosteric residues and clustered sites.
    """
    log.info(f"── Module 05: Allosteric site prediction for {structure.uniprot_id} ──")

    active_set = active_site_nums or set()
    sasa       = sasa_map or {}

    # Extract CA coordinates and residue info
    res_nums  = [r.residue_number for r in structure.residues]
    res_aas   = [r.one_letter     for r in structure.residues]
    res_chain = [r.chain_id       for r in structure.residues]
    res_plddt = [r.plddt          for r in structure.residues]
    ca_coords = np.array([r.coords for r in structure.residues], dtype=np.float64)
    n         = len(res_nums)

    if n < 10:
        log.warning("  Too few residues for ENM — skipping")
        return AllostericResult(uniprot_id=structure.uniprot_id, length=n)

    # ── Step 1: Build contact map ─────────────────────────────────────────────
    log.info("  [1/4] Building elastic network contact map...")
    kirchhoff, n_contacts = _build_kirchhoff(ca_coords, CONTACT_CUTOFF)
    log.info(f"    Contacts: {n_contacts} pairs within {CONTACT_CUTOFF}Å")

    # ── Step 2: Compute GNM correlation matrix ────────────────────────────────
    log.info("  [2/4] Computing GNM correlation matrix (pseudo-inverse)...")
    try:
        corr_matrix = _compute_gnm_correlations(kirchhoff)
        enm_ok = True
        log.info(f"    Correlation matrix: {corr_matrix.shape}")
    except Exception as e:
        log.warning(f"    ENM computation failed: {e} — using contact-based fallback")
        corr_matrix = _contact_correlation_fallback(kirchhoff, n)
        enm_ok = True

    # ── Step 3: Score residues by coupling to active site ────────────────────
    log.info("  [3/4] Scoring allosteric coupling to active site...")
    active_indices = [
        i for i, rn in enumerate(res_nums) if rn in active_set
    ]

    if not active_indices:
        log.warning("    No active site residues provided — using top 10% "
                    "most buried residues as reference")
        sasas_arr = np.array([
            sasa.get((res_chain[i], res_nums[i]), 50.0) for i in range(n)
        ])
        active_indices = list(np.argsort(sasas_arr)[:max(1, n // 10)])

    coupling_scores = _compute_coupling_scores(
        corr_matrix, active_indices, n
    )

    # ── Step 4: Filter and cluster allosteric candidates ─────────────────────
    log.info("  [4/4] Clustering allosteric candidates into sites...")
    allo_residues, allo_sites = _build_allosteric_sites(
        coupling_scores, res_nums, res_aas, res_chain,
        res_plddt, ca_coords, active_indices, active_set, sasa
    )

    # ── Communication pathways ────────────────────────────────────────────────
    pathways = _find_communication_pathways(
        corr_matrix, active_indices, allo_sites, res_nums
    )

    result = AllostericResult(
        uniprot_id=structure.uniprot_id,
        length=n,
        n_contacts=n_contacts,
        enm_computed=enm_ok,
        allosteric_residues=allo_residues,
        allosteric_sites=allo_sites,
        n_sites=len(allo_sites),
        communication_pathways=pathways,
    )

    log.info(result.summary())
    return result


# ── ENM / GNM core ────────────────────────────────────────────────────────────

def _build_kirchhoff(
    coords:  np.ndarray,
    cutoff:  float,
) -> tuple[np.ndarray, int]:
    """
    Build the Kirchhoff (Laplacian) matrix of the elastic network.

    K[i,j] = -1 if residues i,j are in contact (distance < cutoff)
    K[i,i] = number of contacts residue i has

    Returns (kirchhoff_matrix, n_contacts)
    """
    n = len(coords)
    K = np.zeros((n, n), dtype=np.float64)
    n_contacts = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                K[i, j] = -1.0
                K[j, i] = -1.0
                K[i, i] += 1.0
                K[j, j] += 1.0
                n_contacts += 1

    return K, n_contacts


def _compute_gnm_correlations(kirchhoff: np.ndarray) -> np.ndarray:
    """
    Compute the GNM cross-correlation matrix via pseudo-inverse of Kirchhoff.

    The pseudo-inverse removes the zero eigenvalue (rigid body motion)
    and gives the mean-square fluctuation correlation between residues.

    Returns normalised correlation matrix with values in [-1, 1].
    """
    # Pseudo-inverse (Moore-Penrose) — handles the singular Kirchhoff matrix
    K_inv = pinv(kirchhoff)

    # Normalise: C[i,j] = K_inv[i,j] / sqrt(K_inv[i,i] * K_inv[j,j])
    diag  = np.diag(K_inv)
    # Avoid division by zero
    diag_safe = np.where(diag > 1e-10, diag, 1e-10)
    norm  = np.sqrt(np.outer(diag_safe, diag_safe))
    corr  = K_inv / norm

    # Clip to [-1, 1] for numerical safety
    return np.clip(corr, -1.0, 1.0)


def _contact_correlation_fallback(
    kirchhoff: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Fallback correlation estimate based purely on contact count.
    Used if pseudo-inverse fails (very small or disconnected graphs).
    """
    corr = np.zeros((n, n))
    contacts = -kirchhoff.copy()
    np.fill_diagonal(contacts, 0)
    row_sums = contacts.sum(axis=1)
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)

    for i in range(n):
        for j in range(n):
            if i != j:
                corr[i, j] = contacts[i, j] / np.sqrt(
                    row_sums_safe[i] * row_sums_safe[j]
                )
    return np.clip(corr, -1.0, 1.0)


def _compute_coupling_scores(
    corr_matrix:    np.ndarray,
    active_indices: list[int],
    n:              int,
) -> np.ndarray:
    """
    For each residue, compute its mean absolute correlation with active site residues.
    Returns array of shape (n,) with scores in [0, 1].
    """
    if not active_indices:
        return np.zeros(n)

    active_corrs = corr_matrix[:, active_indices]
    scores = np.abs(active_corrs).mean(axis=1)

    # Normalise to [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    return scores


# ── Site building ──────────────────────────────────────────────────────────────

def _build_allosteric_sites(
    coupling_scores: np.ndarray,
    res_nums:        list[int],
    res_aas:         list[str],
    res_chain:       list[str],
    res_plddt:       list[float],
    ca_coords:       np.ndarray,
    active_indices:  list[int],
    active_set:      set[int],
    sasa_map:        dict,
) -> tuple[list[AlloResidue], list[AlloSite]]:
    """
    Filter allosteric candidates and cluster them into sites.
    """
    n = len(res_nums)
    active_coords = ca_coords[active_indices] if active_indices else ca_coords[:1]

    # Compute minimum distance to any active site residue for each residue
    min_dists = np.array([
        float(np.min(np.linalg.norm(active_coords - ca_coords[i], axis=1)))
        for i in range(n)
    ])

    # Filter candidates
    allo_residues = []
    candidate_indices = []

    for i in range(n):
        score = float(coupling_scores[i])
        dist  = float(min_dists[i])

        # Must be coupled but not already in active site
        if score < ALLOSTERIC_THRESHOLD:
            continue
        if dist < MIN_DISTANCE_FROM_ACTIVE:
            continue
        if res_nums[i] in active_set:
            continue

        sasa_val = sasa_map.get((res_chain[i], res_nums[i]), 50.0)

        allo_residues.append(AlloResidue(
            residue_number=res_nums[i],
            chain_id=res_chain[i],
            one_letter=res_aas[i],
            correlation_score=round(score, 3),
            min_dist_active=round(dist, 1),
            sasa=round(sasa_val, 1),
            is_exposed=(sasa_val > 20.0),
            plddt=res_plddt[i],
        ))
        candidate_indices.append(i)

    log.debug(f"    Allosteric candidates: {len(allo_residues)}")

    if not allo_residues:
        return [], []

    # Cluster candidates by spatial proximity
    cand_coords = ca_coords[candidate_indices]
    cand_scores = coupling_scores[candidate_indices]
    sites       = _cluster_into_sites(
        allo_residues, cand_coords, cand_scores,
        res_nums, res_aas, ca_coords,
        active_indices, active_set, sasa_map
    )

    return allo_residues, sites


def _cluster_into_sites(
    allo_residues:  list[AlloResidue],
    cand_coords:    np.ndarray,
    cand_scores:    np.ndarray,
    res_nums:       list[int],
    res_aas:        list[str],
    all_coords:     np.ndarray,
    active_indices: list[int],
    active_set:     set[int],
    sasa_map:       dict,
) -> list[AlloSite]:
    """
    Greedy distance-based clustering of allosteric candidates into sites.
    """
    n_cands  = len(allo_residues)
    visited  = np.zeros(n_cands, dtype=bool)
    order    = np.argsort(-cand_scores)
    sites    = []
    site_idx = 0

    active_coords = all_coords[active_indices] if active_indices else all_coords[:1]

    for seed in order:
        if visited[seed]:
            continue

        cluster = [seed]
        visited[seed] = True

        for j in range(n_cands):
            if visited[j]:
                continue
            dist = np.linalg.norm(cand_coords[seed] - cand_coords[j])
            if dist <= SITE_CLUSTER_CUTOFF:
                cluster.append(j)
                visited[j] = True

        if len(cluster) < MIN_SITE_SIZE:
            continue

        members = [allo_residues[k] for k in cluster]
        rns     = [m.residue_number for m in members]
        letters = [m.one_letter for m in members]

        cluster_coords = cand_coords[cluster]
        centre         = cluster_coords.mean(axis=0)

        mean_corr   = float(np.mean([m.correlation_score for m in members]))
        mean_sasa   = float(np.mean([m.sasa for m in members]))
        mean_hydro  = float(np.mean([HYDROPHOBICITY.get(aa, 0.0) for aa in letters]))
        net_charge  = float(sum(CHARGE_AT_PH7.get(aa, 0.0) for aa in letters))
        min_dist_active = float(np.min(
            np.linalg.norm(active_coords - centre, axis=1)
        ))

        # Which active site residues are most coupled to this site?
        coupled = [
            res_nums[i] for i in active_indices
            if np.linalg.norm(all_coords[i] - centre) < 15.0
        ][:5]

        if mean_corr >= 0.6:
            confidence = "HIGH"
        elif mean_corr >= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        site_idx += 1
        sites.append(AlloSite(
            site_id=f"A{site_idx}",
            residue_numbers=sorted(rns),
            residue_letters=letters,
            size=len(cluster),
            centre=[round(float(c), 2) for c in centre],
            mean_correlation=round(mean_corr, 3),
            min_dist_active=round(min_dist_active, 1),
            mean_sasa=round(mean_sasa, 1),
            mean_hydrophobicity=round(mean_hydro, 3),
            net_charge=round(net_charge, 1),
            coupled_active_residues=coupled,
            confidence=confidence,
        ))

        if site_idx >= MAX_SITES:
            break

    # Sort by mean correlation descending
    sites.sort(key=lambda s: s.mean_correlation, reverse=True)
    for i, s in enumerate(sites):
        s.site_id = f"A{i+1}"

    return sites


# ── Communication pathways ─────────────────────────────────────────────────────

def _find_communication_pathways(
    corr_matrix:   np.ndarray,
    active_indices: list[int],
    allo_sites:    list[AlloSite],
    res_nums:      list[int],
) -> list[list[int]]:
    """
    For each allosteric site, find the chain of residues with highest
    correlation connecting the site to the active site.
    This is a simplified shortest-path through correlation space.
    Returns list of residue number lists (one path per site, max 3 sites).
    """
    if not active_indices or not allo_sites:
        return []

    rn_to_idx = {rn: i for i, rn in enumerate(res_nums)}
    n         = len(res_nums)
    pathways  = []

    for site in allo_sites[:3]:
        # Find the site residue with highest mean correlation to active site
        site_indices = [
            rn_to_idx[rn] for rn in site.residue_numbers
            if rn in rn_to_idx
        ]
        if not site_indices:
            continue

        # Walk greedily from site to active site via highest correlation
        start = site_indices[0]
        end   = active_indices[0]

        path = _greedy_path(corr_matrix, start, end, n, max_steps=15)
        if path:
            pathways.append([res_nums[i] for i in path])

    return pathways


def _greedy_path(
    corr:      np.ndarray,
    start:     int,
    end:       int,
    n:         int,
    max_steps: int = 15,
) -> list[int]:
    """Walk from start to end by following highest correlation at each step."""
    path    = [start]
    visited = {start}
    current = start

    for _ in range(max_steps):
        if current == end:
            break

        # Find unvisited neighbour with highest absolute correlation
        row   = np.abs(corr[current]).copy()
        row[list(visited)] = -1.0

        next_node = int(np.argmax(row))
        if row[next_node] < 0.05:
            break

        path.append(next_node)
        visited.add(next_node)
        current = next_node

        if next_node == end:
            break

    return path


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 05 — Allosteric site prediction.

    Requires Module 01 (.pdb). Uses Module 02 (SASA) and Module 03
    (active sites) if available.

    Example:
        python pipeline/allosteric.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_allosteric.json"

    if not pdb_path.exists():
        log.error(
            f".pdb not found: {pdb_path}\n"
            f"  Run Module 01 first: python pipeline/fetch_structure.py "
            f"--uniprot {uniprot}"
        )
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

    result = predict_allosteric_sites(parsed, active_set, sasa_map)
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()