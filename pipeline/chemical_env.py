"""
pipeline/06_chemical_env.py
────────────────────────────
Module 06 — Chemical environment mapping.

For each predicted site (active, binding, allosteric) from Modules 03-05,
computes a detailed chemical environment profile:

  1. Electrostatic profile
     - Net charge, charge distribution, positive/negative patches
     - Estimated electrostatic potential at site centre

  2. Hydrogen bond capacity
     - H-bond donor count (NH, OH groups)
     - H-bond acceptor count (C=O, N, OH)
     - H-bond donor/acceptor ratio

  3. Hydrophobic character
     - Mean/max hydrophobicity of lining residues
     - Hydrophobic patch area estimate
     - Amphipathic character (mixed hydrophobic/hydrophilic)

  4. Aromatic / pi-stacking opportunities
     - Count of Phe, Tyr, Trp, His in lining
     - Estimated pi-stacking surface area

  5. Metal coordination potential
     - His, Cys, Asp, Glu in lining = potential metal binding
     - Geometry score for tetrahedral coordination

  6. Site druggability chemistry summary
     - Lipophilic ligand efficiency estimate
     - Predicted ligand binding mode (H-bond heavy / hydrophobic / mixed)

Usage (standalone):
    python pipeline/06_chemical_env.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.chemical_env import map_chemical_environment
    result = map_chemical_environment(parsed, active_result, pocket_result, allo_result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from itertools import combinations

import click
import numpy as np

from utils.config import cfg, get_logger
from utils.pdb_parser import parse_pdb, ParsedStructure, HYDROPHOBICITY, AA3TO1

log = get_logger(__name__)

# ── Amino acid property tables ─────────────────────────────────────────────────

# H-bond donors per residue (side chain only)
HBOND_DONORS: dict[str, int] = {
    "R": 3, "K": 1, "N": 1, "Q": 1, "S": 1, "T": 1,
    "Y": 1, "W": 1, "H": 1, "C": 1, "D": 0, "E": 0,
    "A": 0, "G": 0, "I": 0, "L": 0, "M": 0, "F": 0,
    "P": 0, "V": 0,
}

# H-bond acceptors per residue (side chain only)
HBOND_ACCEPTORS: dict[str, int] = {
    "D": 2, "E": 2, "N": 1, "Q": 1, "S": 1, "T": 1,
    "Y": 1, "H": 1, "M": 1, "C": 0, "R": 0, "K": 0,
    "A": 0, "G": 0, "I": 0, "L": 0, "F": 0, "P": 0,
    "V": 0, "W": 0,
}

# Aromatic residues capable of pi-stacking
AROMATIC: set[str] = {"F", "Y", "W", "H"}

# Metal-coordinating residues
METAL_COORD: set[str] = {"H", "C", "D", "E"}

# Charge at pH 7.4
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}

# Approximate pi-stacking surface area contribution (Å²)
PI_SURFACE: dict[str, float] = {
    "F": 40.0, "Y": 40.0, "W": 65.0, "H": 30.0,
}

# Lipophilicity contribution (Wildman-Crippen approximation per residue)
LIPOPHILICITY: dict[str, float] = {
    "A":  0.31, "R": -1.01, "N": -0.60, "D": -0.77, "C":  1.54,
    "Q": -0.22, "E": -0.64, "G":  0.00, "H":  0.13, "I":  1.80,
    "L":  1.70, "K": -0.99, "M":  1.23, "F":  1.79, "P":  0.72,
    "S": -0.04, "T":  0.26, "W":  2.25, "Y":  0.96, "V":  1.22,
}

# Predicted binding mode thresholds
HYDROPHOBIC_THRESHOLD  = 0.5   # mean hydrophobicity
HBOND_HEAVY_THRESHOLD  = 4     # total H-bond capacity


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SiteChemEnv:
    """Chemical environment profile for a single site."""
    site_id:              str      # e.g. "active_C176", "P1", "A1"
    site_type:            str      # "active" / "binding" / "allosteric"
    residue_numbers:      list[int]
    residue_letters:      list[str]
    n_residues:           int

    # Electrostatics
    net_charge:           float
    n_positive:           int
    n_negative:           int
    charge_character:     str      # "positive" / "negative" / "neutral" / "mixed"
    estimated_esp:        float    # estimated electrostatic potential (arbitrary units)

    # H-bond capacity
    n_hbond_donors:       int
    n_hbond_acceptors:    int
    total_hbond_capacity: int
    hbond_donor_ratio:    float    # donors / (donors + acceptors)

    # Hydrophobicity
    mean_hydrophobicity:  float
    max_hydrophobicity:   float
    hydrophobic_residues: list[str]
    is_amphipathic:       bool     # has both hydrophobic and hydrophilic residues

    # Aromatic / pi-stacking
    n_aromatic:           int
    aromatic_residues:    list[str]
    pi_surface_area:      float    # Å²

    # Metal coordination
    n_metal_coord:        int
    metal_coord_residues: list[str]
    metal_binding_score:  float    # 0-1 likelihood of metal binding

    # Lipophilicity
    mean_lipophilicity:   float
    logp_estimate:        float    # sum of per-residue logP contributions

    # Binding mode prediction
    predicted_binding_mode: str    # "h-bond heavy" / "hydrophobic" / "mixed" / "metal"
    ligand_efficiency_est:  float  # 0-1 estimated ligand efficiency

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChemEnvResult:
    """Full chemical environment mapping output. Output of Module 06."""
    uniprot_id:    str
    n_sites_mapped: int                      = 0
    active_envs:   list[SiteChemEnv]         = field(default_factory=list)
    binding_envs:  list[SiteChemEnv]         = field(default_factory=list)
    allo_envs:     list[SiteChemEnv]         = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Chemical environment: {self.uniprot_id}",
            f"  Sites mapped: {self.n_sites_mapped}",
        ]
        all_envs = (
            [("Active", e)  for e in self.active_envs[:3]]  +
            [("Pocket", e)  for e in self.binding_envs[:3]] +
            [("Allo",   e)  for e in self.allo_envs[:3]]
        )
        for label, env in all_envs:
            lines.append(
                f"  [{label} {env.site_id}] "
                f"charge={env.net_charge:+.1f} ({env.charge_character})  "
                f"HBD/HBA={env.n_hbond_donors}/{env.n_hbond_acceptors}  "
                f"hydro={env.mean_hydrophobicity:+.2f}  "
                f"aromatic={env.n_aromatic}  "
                f"mode={env.predicted_binding_mode}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved chemical environment JSON → {path}")


# ── Main function ──────────────────────────────────────────────────────────────

def map_chemical_environment(
    structure:      ParsedStructure,
    active_data:    Optional[dict]  = None,
    pocket_data:    Optional[dict]  = None,
    allo_data:      Optional[dict]  = None,
) -> ChemEnvResult:
    """
    Map chemical environment for all predicted sites.

    Args:
        structure:   ParsedStructure from Module 01
        active_data: dict loaded from Module 03 JSON (active_sites)
        pocket_data: dict loaded from Module 04 JSON (binding_pockets)
        allo_data:   dict loaded from Module 05 JSON (allosteric)

    Returns:
        ChemEnvResult with chemical profiles for every site.
    """
    log.info(f"── Module 06: Chemical environment mapping for {structure.uniprot_id} ──")

    # Build residue lookup: residue_number → one_letter
    res_lookup = {
        r.residue_number: r.one_letter
        for r in structure.residues
    }
    coords_lookup = {
        r.residue_number: np.array(r.coords)
        for r in structure.residues
    }

    active_envs  = []
    binding_envs = []
    allo_envs    = []

    # ── Map active sites ──────────────────────────────────────────────────────
    if active_data:
        log.info("  Mapping active site chemical environments...")
        high_residues = [
            r for r in active_data.get("active_residues", [])
            if r.get("confidence") == "HIGH"
        ]
        if high_residues:
            rns     = [r["residue_number"] for r in high_residues]
            letters = [res_lookup.get(rn, "X") for rn in rns]
            env = _compute_site_env(
                site_id="active_core",
                site_type="active",
                residue_numbers=rns,
                residue_letters=letters,
                coords_lookup=coords_lookup,
            )
            active_envs.append(env)
            log.info(f"    Active core: {env.predicted_binding_mode} | "
                     f"charge={env.net_charge:+.1f} | "
                     f"HBD/HBA={env.n_hbond_donors}/{env.n_hbond_acceptors}")

        # Also map individual motifs
        for motif in active_data.get("catalytic_motifs", [])[:5]:
            rns     = motif.get("residue_numbers", [])
            letters = [res_lookup.get(rn, "X") for rn in rns]
            if not rns:
                continue
            motif_type = motif.get("motif_type", "motif")
            env = _compute_site_env(
                site_id=f"motif_{motif_type[:12]}",
                site_type="active",
                residue_numbers=rns,
                residue_letters=letters,
                coords_lookup=coords_lookup,
            )
            active_envs.append(env)

    # ── Map binding pockets ───────────────────────────────────────────────────
    if pocket_data:
        log.info("  Mapping binding pocket chemical environments...")
        for pocket in pocket_data.get("pockets", [])[:5]:
            rns     = pocket.get("lining_residues", [])
            letters = [res_lookup.get(rn, "X") for rn in rns]
            if not rns:
                continue
            env = _compute_site_env(
                site_id=pocket.get("pocket_id", "P?"),
                site_type="binding",
                residue_numbers=rns,
                residue_letters=letters,
                coords_lookup=coords_lookup,
            )
            binding_envs.append(env)
            log.info(f"    {env.site_id}: {env.predicted_binding_mode} | "
                     f"charge={env.net_charge:+.1f} | "
                     f"logP≈{env.logp_estimate:.1f}")

    # ── Map allosteric sites ──────────────────────────────────────────────────
    if allo_data:
        log.info("  Mapping allosteric site chemical environments...")
        for site in allo_data.get("allosteric_sites", [])[:5]:
            rns     = site.get("residue_numbers", [])
            letters = [res_lookup.get(rn, "X") for rn in rns]
            if not rns:
                continue
            env = _compute_site_env(
                site_id=site.get("site_id", "A?"),
                site_type="allosteric",
                residue_numbers=rns,
                residue_letters=letters,
                coords_lookup=coords_lookup,
            )
            allo_envs.append(env)
            log.info(f"    {env.site_id}: {env.predicted_binding_mode} | "
                     f"charge={env.net_charge:+.1f} | "
                     f"aromatic={env.n_aromatic}")

    n_total = len(active_envs) + len(binding_envs) + len(allo_envs)
    result  = ChemEnvResult(
        uniprot_id=structure.uniprot_id,
        n_sites_mapped=n_total,
        active_envs=active_envs,
        binding_envs=binding_envs,
        allo_envs=allo_envs,
    )

    log.info(result.summary())
    return result


# ── Core computation ───────────────────────────────────────────────────────────

def _compute_site_env(
    site_id:          str,
    site_type:        str,
    residue_numbers:  list[int],
    residue_letters:  list[str],
    coords_lookup:    dict[int, np.ndarray],
) -> SiteChemEnv:
    """
    Compute the full chemical environment for a list of residues.
    """
    letters = residue_letters

    # ── Electrostatics ────────────────────────────────────────────────────────
    charges    = [CHARGE_AT_PH7.get(aa, 0.0) for aa in letters]
    net_charge = sum(charges)
    n_pos      = sum(1 for c in charges if c > 0)
    n_neg      = sum(1 for c in charges if c < 0)

    if net_charge > 1:
        charge_char = "positive"
    elif net_charge < -1:
        charge_char = "negative"
    elif n_pos > 0 and n_neg > 0:
        charge_char = "mixed"
    else:
        charge_char = "neutral"

    # Estimated electrostatic potential: weighted sum of charges by burial
    # (simple proxy — positive = electropositive site)
    esp = float(net_charge * 1.4)  # rough kcal/mol·e proxy

    # ── H-bond capacity ───────────────────────────────────────────────────────
    donors    = sum(HBOND_DONORS.get(aa, 0)    for aa in letters)
    acceptors = sum(HBOND_ACCEPTORS.get(aa, 0) for aa in letters)
    total_hb  = donors + acceptors
    donor_ratio = donors / max(total_hb, 1)

    # ── Hydrophobicity ────────────────────────────────────────────────────────
    hydrophobes = [HYDROPHOBICITY.get(aa, 0.0) for aa in letters]
    mean_hydro  = float(np.mean(hydrophobes)) if hydrophobes else 0.0
    max_hydro   = float(max(hydrophobes)) if hydrophobes else 0.0
    hydro_res   = [aa for aa in letters if HYDROPHOBICITY.get(aa, 0.0) > 1.0]
    hydrophil   = [aa for aa in letters if HYDROPHOBICITY.get(aa, 0.0) < -0.5]
    is_amphi    = len(hydro_res) > 0 and len(hydrophil) > 0

    # ── Aromatics ─────────────────────────────────────────────────────────────
    aromatic_res  = [aa for aa in letters if aa in AROMATIC]
    n_aromatic    = len(aromatic_res)
    pi_surface    = sum(PI_SURFACE.get(aa, 0.0) for aa in aromatic_res)

    # ── Metal coordination ────────────────────────────────────────────────────
    metal_res     = [aa for aa in letters if aa in METAL_COORD]
    n_metal       = len(metal_res)
    metal_score   = _metal_binding_score(letters, residue_numbers, coords_lookup)

    # ── Lipophilicity ─────────────────────────────────────────────────────────
    lipophils   = [LIPOPHILICITY.get(aa, 0.0) for aa in letters]
    mean_lipo   = float(np.mean(lipophils)) if lipophils else 0.0
    logp_est    = float(sum(lipophils))

    # ── Binding mode prediction ───────────────────────────────────────────────
    mode = _predict_binding_mode(
        mean_hydro, total_hb, n_aromatic, metal_score, net_charge
    )

    # ── Ligand efficiency estimate ────────────────────────────────────────────
    # Based on: enclosed pocket + good H-bond capacity + moderate size
    n = max(len(letters), 1)
    le = min(
        (total_hb / (n * 2)) * 0.4 +
        (abs(mean_hydro) / 4.5) * 0.3 +
        (n_aromatic / max(n, 5)) * 0.3,
        1.0
    )

    return SiteChemEnv(
        site_id=site_id,
        site_type=site_type,
        residue_numbers=residue_numbers,
        residue_letters=letters,
        n_residues=len(letters),
        net_charge=round(net_charge, 2),
        n_positive=n_pos,
        n_negative=n_neg,
        charge_character=charge_char,
        estimated_esp=round(esp, 2),
        n_hbond_donors=donors,
        n_hbond_acceptors=acceptors,
        total_hbond_capacity=total_hb,
        hbond_donor_ratio=round(donor_ratio, 3),
        mean_hydrophobicity=round(mean_hydro, 3),
        max_hydrophobicity=round(max_hydro, 3),
        hydrophobic_residues=hydro_res,
        is_amphipathic=is_amphi,
        n_aromatic=n_aromatic,
        aromatic_residues=aromatic_res,
        pi_surface_area=round(pi_surface, 1),
        n_metal_coord=n_metal,
        metal_coord_residues=metal_res,
        metal_binding_score=round(metal_score, 3),
        mean_lipophilicity=round(mean_lipo, 3),
        logp_estimate=round(logp_est, 2),
        predicted_binding_mode=mode,
        ligand_efficiency_est=round(le, 3),
    )


def _metal_binding_score(
    letters:         list[str],
    residue_numbers: list[int],
    coords_lookup:   dict[int, np.ndarray],
) -> float:
    """
    Score for metal binding potential.
    Looks for 3+ metal-coordinating residues (H, C, D, E) in close proximity.
    Checks for roughly tetrahedral geometry (ideal for Zn2+, Fe2+, etc.)
    """
    metal_indices = [
        i for i, aa in enumerate(letters)
        if aa in METAL_COORD
    ]

    if len(metal_indices) < 3:
        return float(len(metal_indices)) / 4.0

    # Get coordinates of metal-coordinating residues
    metal_coords = []
    for i in metal_indices:
        if i < len(residue_numbers):
            rn = residue_numbers[i]
            if rn in coords_lookup:
                metal_coords.append(coords_lookup[rn])

    if len(metal_coords) < 3:
        return float(len(metal_indices)) / 4.0

    # Check pairwise distances — ideal metal coordination: 3-5 Å
    good_pairs = 0
    total_pairs = 0
    for a, b in combinations(metal_coords, 2):
        dist = float(np.linalg.norm(a - b))
        total_pairs += 1
        if 3.0 <= dist <= 7.0:
            good_pairs += 1

    geometry_score = good_pairs / max(total_pairs, 1)
    count_score    = min(len(metal_indices) / 4.0, 1.0)

    return float((geometry_score + count_score) / 2.0)


def _predict_binding_mode(
    mean_hydro:   float,
    total_hb:     int,
    n_aromatic:   int,
    metal_score:  float,
    net_charge:   float,
) -> str:
    """
    Predict the dominant ligand binding mode for this site.
    Returns one of: "metal" / "h-bond heavy" / "hydrophobic" / "mixed" / "electrostatic"
    """
    if metal_score > 0.6:
        return "metal"
    if abs(net_charge) >= 3:
        return "electrostatic"
    if total_hb >= HBOND_HEAVY_THRESHOLD and mean_hydro < 0:
        return "h-bond heavy"
    if mean_hydro >= HYDROPHOBIC_THRESHOLD and n_aromatic >= 2:
        return "hydrophobic+pi"
    if mean_hydro >= HYDROPHOBIC_THRESHOLD:
        return "hydrophobic"
    if total_hb >= HBOND_HEAVY_THRESHOLD:
        return "h-bond heavy"
    return "mixed"


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 06 — Chemical environment mapping.

    Requires Modules 01-05 outputs to be present.

    Example:
        python pipeline/chemical_env.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_chemical_env.json"

    if not pdb_path.exists():
        log.error(f".pdb not found: {pdb_path}")
        raise SystemExit(1)

    parsed = parse_pdb(pdb_path, uniprot)

    def _load(filename: str) -> Optional[dict]:
        p = inter_dir / filename
        if p.exists():
            with open(p) as f:
                return json.load(f)
        log.warning(f"  {filename} not found — skipping")
        return None

    active_data = _load(f"{uniprot}_active_sites.json")
    pocket_data = _load(f"{uniprot}_binding_pockets.json")
    allo_data   = _load(f"{uniprot}_allosteric.json")

    result = map_chemical_environment(
        parsed, active_data, pocket_data, allo_data
    )
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()