"""
pipeline/cryptic_pockets.py
─────────────────────────────
Module 18 — Cryptic / Transient Pocket Detection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS MODULE EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 04 (binding_pockets.py) detects pockets from the static AlphaFold
structure. This works well for constitutively open pockets — the ones
visible in the crystal or predicted structure.

However, ~40% of biologically important druggable pockets are "cryptic":
they only open transiently during conformational fluctuations. They are
INVISIBLE in the static structure and completely missed by Module 04.

Famous examples:
  - ABL1 myristoyl pocket  — opens transiently in inactive conformation
  - MDM2 p53-binding cleft — larger in dynamic form than crystal structure
  - HIV protease flap      — covers active site; opens to let substrate in
  - TP53 DNA-binding domain — cryptic groove exploited by APR-246/Eprenetapopt

The key insight: where the protein is MOST FLEXIBLE (high RMSF from MD
Module 14) is where it samples the widest range of conformations. If a
flexible region also has residues that can geometrically ENCLOSE a pocket
in some of those conformations, that's a cryptic pocket.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — RMSF hotspot identification
    Load per-residue RMSF from Module 14 (data/intermediate/{uid}_md.json).
    Identify "flexible hotspots": contiguous segments where RMSF > threshold
    (default 1.5 Å). These are regions that sample significant conformational
    space during MD.

    Fallback: if no MD data, use per-residue pLDDT from AlphaFold as a proxy
    for flexibility (low pLDDT → disordered → flexible). Less accurate but
    still useful.

Step 2 — Geometric pocket opening simulation
    For each flexible hotspot, simulate pocket opening by:
      a) Taking the static structure positions of hotspot residues
      b) Perturbing Cα positions along their principal displacement axes
         (derived from RMSF direction covariance if available, else radially)
      c) Testing if the PERTURBED configuration encloses a pocket using
         the same alpha-sphere burial algorithm from Module 04
      d) Scoring the hypothetical pocket geometry

    This is NOT an MD simulation — it's a geometric approximation of
    "what pocket would exist if these flexible residues moved outward"
    using the RMSF as the displacement magnitude.

Step 3 — Allosteric gate detection
    Cross-reference flexible hotspots with:
      - Module 05 (allosteric sites): flexible allosteric sites can gate
        cryptic pockets at distal locations
      - Module 03 (active sites): flexible loops gating the active site
        (like the ABL1 activation loop or HIV protease flap)

    If a flexible hotspot is within 15 Å of an active/allosteric site
    and has RMSF > 2 Å, flag it as a "gate" that may reveal a cryptic
    pocket when open.

Step 4 — Druggability estimation
    Score each cryptic pocket using:
      - estimated_volume: from geometric perturbation × RMSF magnitude
      - transient_druggability: burial_score × hydrophobicity × enclosure
        (lower than static pockets because only partially open)
      - opening_probability: estimated from Boltzmann factor using RMSF
        P_open ≈ 1 - exp(-RMSF / kT_displacement_units)
      - effective_druggability: transient_druggability × P_open

Step 5 — Integration with upstream modules
    Output feeds into:
      - de_novo_design.py: cryptic pockets as alternative docking targets
      - selectivity_optimizer.py: check if drug exploits cryptic pocket
        (more selective — only opens in target protein)
      - sim/step02_protein_ensemble.py: cryptic pocket state added to ensemble
      - consensus.py: cryptic pockets reported alongside static pockets
      - antibody_design.py: flag flexible surface regions as dynamic epitopes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
    python pipeline/cryptic_pockets.py --uniprot P04637
    python pipeline/cryptic_pockets.py --uniprot P04637 --rmsf-threshold 1.2
    python pipeline/cryptic_pockets.py --uniprot P04637 --no-md-fallback
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import cfg, get_logger
from utils.pdb_parser import HYDROPHOBICITY, parse_pdb

log = get_logger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────

kT_ROOM = 0.593   # kJ/mol at 300 K — used for Boltzmann opening probability

# ── Tuneable thresholds ───────────────────────────────────────────────────────

RMSF_FLEXIBLE_THRESHOLD  = 1.5   # Å — residues above this are "flexible"
RMSF_GATE_THRESHOLD      = 2.0   # Å — residues above this can be pocket gates
RMSF_FALLBACK_PLDDT      = 70.0  # pLDDT below this = flexible (no MD fallback)
MIN_HOTSPOT_LENGTH       = 3     # minimum contiguous flexible residues
POCKET_LINING_CUTOFF     = 8.0   # Å — same as Module 04
BURIAL_INNER             = 4.0
BURIAL_OUTER             = 10.0
MIN_BURIAL_COUNT         = 6     # slightly relaxed vs Module 04 (cryptic = smaller)
PERTURBATION_SCALE       = 1.0   # Å per unit RMSF — displacement magnitude
NEAR_FUNCTIONAL_CUTOFF   = 15.0  # Å — near active/allosteric site
MIN_CRYPTIC_SCORE        = 0.12  # minimum effective_druggability to report

# Charge at pH 7.4 (same as Module 04)
CHARGE_AT_PH7: dict[str, float] = {
    "A":  0.0, "R": +1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H": +0.1, "I":  0.0,
    "L":  0.0, "K": +1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FlexibleHotspot:
    """A contiguous region of high RMSF — potential cryptic pocket region."""
    hotspot_id:        str
    start_residue:     int
    end_residue:       int
    residue_numbers:   list[int]
    mean_rmsf:         float        # Å
    max_rmsf:          float        # Å
    max_rmsf_residue:  int
    rmsf_source:       str          # "md" | "plddt_proxy"

    # Functional context
    near_active_site:    bool  = False
    near_allosteric:     bool  = False
    dist_to_active_A:    float = 999.0
    dist_to_allosteric_A: float = 999.0
    is_gate:             bool  = False   # True if this loop gates a known site

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrypticPocket:
    """A predicted cryptic/transient pocket."""
    pocket_id:           str         # e.g. "CP1", "CP2"

    # Location
    center:              list[float]  # [x, y, z] centroid of the open pocket
    lining_residues:     list[int]   # residue numbers forming the pocket
    lining_letters:      list[str]   # one-letter AAs
    hotspot_id:          str         # which flexible hotspot opens this pocket

    # Geometry in the predicted open state
    estimated_volume_A3: float       # estimated pocket volume when open
    enclosure:           float       # 0–1, how enclosed the pocket is when open
    mean_hydrophobicity: float
    net_charge:          float
    n_lining:            int

    # Dynamics
    mean_rmsf_hotspot:   float       # mean RMSF of the gating hotspot
    max_rmsf_hotspot:    float
    opening_probability: float       # Boltzmann estimate 0–1
    rmsf_source:         str         # "md" | "plddt_proxy"

    # Druggability
    static_structure_present: bool   # Is there already a Module 04 pocket here?
    transient_druggability:  float   # druggability score in open state
    effective_druggability:  float   # transient_druggability × P_open
    druggability_class:      str     # "high" | "medium" | "low"

    # Functional context
    near_active_site:    bool  = False
    near_allosteric:     bool  = False
    gate_type:           str   = ""   # "activation_loop" | "flap" | "lid" | "helix"

    # Integration payloads
    denovo_target:       bool  = False  # flag for de novo design
    selectivity_note:    str   = ""     # note for selectivity optimizer
    sim02_state_name:    str   = ""     # state name for SIM-02 ensemble

    # Confidence
    confidence:          float = 0.0
    evidence:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrypticPocketResult:
    """Full output of Module 18."""
    uniprot_id:          str
    gene_name:           str
    sequence_length:     int
    rmsf_source:         str         # "md" | "plddt_proxy" | "none"

    n_flexible_hotspots: int         = 0
    n_cryptic_pockets:   int         = 0
    n_high_confidence:   int         = 0

    hotspots:            list[FlexibleHotspot] = field(default_factory=list)
    cryptic_pockets:     list[CrypticPocket]   = field(default_factory=list)

    # Module 04 pocket count for comparison
    n_static_pockets:    int         = 0

    # Integration payloads
    sim02_new_states:    list[dict]  = field(default_factory=list)
    denovo_targets:      list[dict]  = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  Cryptic Pocket Detection: {self.gene_name} ({self.uniprot_id})",
            f"{'═'*65}",
            f"  RMSF source        : {self.rmsf_source}",
            f"  Flexible hotspots  : {self.n_flexible_hotspots}",
            f"  Static pockets (M04): {self.n_static_pockets}",
            f"  Cryptic pockets    : {self.n_cryptic_pockets}",
            f"  High confidence    : {self.n_high_confidence}",
        ]

        if self.hotspots:
            lines += ["", "  Flexible hotspots:"]
            for h in self.hotspots:
                gate = " [GATE]" if h.is_gate else ""
                active = " near-active" if h.near_active_site else ""
                allo = " near-allo" if h.near_allosteric else ""
                lines.append(
                    f"    {h.hotspot_id}: res {h.start_residue}–{h.end_residue}"
                    f"  RMSF={h.mean_rmsf:.2f}Å (max {h.max_rmsf:.2f}Å)"
                    f"{gate}{active}{allo}"
                )

        if self.cryptic_pockets:
            lines += ["", "  Cryptic pockets:"]
            lines += [
                f"  {'ID':<6} {'Residues':<20} {'Vol':>7} "
                f"{'P_open':>7} {'Eff.drug':>9} {'Class':<8} {'Note'}",
                f"  {'─'*6} {'─'*20} {'─'*7} "
                f"{'─'*7} {'─'*9} {'─'*8} {'─'*20}",
            ]
            for cp in sorted(self.cryptic_pockets,
                             key=lambda x: -x.effective_druggability):
                res_str = f"{cp.lining_residues[0]}–{cp.lining_residues[-1]}" \
                          if cp.lining_residues else "?"
                note = cp.selectivity_note[:20] if cp.selectivity_note else cp.gate_type
                lines.append(
                    f"  {cp.pocket_id:<6} {res_str:<20} "
                    f"{cp.estimated_volume_A3:>6.0f}Å "
                    f"{cp.opening_probability:>7.2f} "
                    f"{cp.effective_druggability:>9.3f} "
                    f"{cp.druggability_class:<8} {note}"
                )

        lines += [f"{'═'*65}", ""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved cryptic pockets JSON → {path}")


# ── Step 1: RMSF / flexibility loading ───────────────────────────────────────

def _load_rmsf(
    uniprot_id: str,
    inter_dir:  Path,
    residue_numbers: list[int],
    plddt_values:    list[float],
    use_md_fallback: bool = True,
) -> tuple[list[float], str]:
    """
    Load per-residue RMSF values from Module 14 MD output.
    Falls back to pLDDT-derived proxy if MD not available.

    Returns: (rmsf_list, source_string)
    rmsf_list is indexed 0..N-1 matching residue_numbers.
    """
    md_path = inter_dir / f"{uniprot_id}_md.json"

    if md_path.exists():
        try:
            md_data = json.loads(md_path.read_text())
            rmsf_raw = md_data.get("rmsf_per_residue", [])

            if rmsf_raw and len(rmsf_raw) > 0:
                # MD RMSF is indexed by residue order, same as structure residues
                # Pad or trim to match our residue list length
                n = len(residue_numbers)
                if len(rmsf_raw) >= n:
                    rmsf = [float(rmsf_raw[i]) for i in range(n)]
                else:
                    # MD covered fewer residues (truncated model) — pad with mean
                    mean_val = float(np.mean(rmsf_raw))
                    rmsf = [float(rmsf_raw[i]) if i < len(rmsf_raw)
                            else mean_val for i in range(n)]

                log.info(f"  RMSF loaded from MD: {len(rmsf)} residues, "
                         f"mean={float(np.mean(rmsf)):.2f}Å, "
                         f"max={float(np.max(rmsf)):.2f}Å")
                return rmsf, "md"

        except Exception as e:
            log.warning(f"  MD RMSF load failed ({e}), trying fallback")

    if not use_md_fallback:
        log.warning("  No MD data and fallback disabled — using uniform RMSF=1.0")
        return [1.0] * len(residue_numbers), "none"

    # pLDDT proxy: low pLDDT → high predicted flexibility
    # Empirical mapping: pLDDT 90→0.5Å, 70→1.0Å, 50→2.0Å, 30→4.0Å
    # RMSF_proxy = 8.0 × exp(-pLDDT / 30) — gives biologically plausible range
    rmsf_proxy = []
    for plddt in plddt_values:
        # Clamp pLDDT to [10, 100]
        p = max(10.0, min(100.0, float(plddt)))
        rmsf_val = 8.0 * math.exp(-p / 30.0)
        rmsf_proxy.append(round(rmsf_val, 3))

    log.info(f"  RMSF from pLDDT proxy: "
             f"mean={float(np.mean(rmsf_proxy)):.2f}Å, "
             f"max={float(np.max(rmsf_proxy)):.2f}Å")
    return rmsf_proxy, "plddt_proxy"


# ── Step 2: Hotspot identification ────────────────────────────────────────────

def _find_flexible_hotspots(
    residue_numbers: list[int],
    rmsf_values:     list[float],
    rmsf_threshold:  float,
    rmsf_source:     str,
) -> list[FlexibleHotspot]:
    """
    Find contiguous segments of residues with RMSF above threshold.
    Merges segments separated by ≤2 rigid residues (to avoid fragmentation).
    """
    n = len(residue_numbers)
    if n == 0:
        return []

    # Boolean mask: which residues are flexible
    flexible = [rmsf_values[i] >= rmsf_threshold for i in range(n)]

    # Fill small gaps (≤1 rigid residue between flexible segments)
    # Deliberately conservative — we want distinct hotspots, not one merged blob
    for i in range(1, n - 1):
        if not flexible[i]:
            left_flex  = flexible[i-1] if i > 0 else False
            right_flex = flexible[i+1] if i < n-1 else False
            if left_flex and right_flex:
                flexible[i] = True

    # Extract contiguous segments
    segments: list[list[int]] = []
    current: list[int] = []

    for i, is_flex in enumerate(flexible):
        if is_flex:
            current.append(i)
        else:
            if len(current) >= MIN_HOTSPOT_LENGTH:
                segments.append(current)
            current = []
    if len(current) >= MIN_HOTSPOT_LENGTH:
        segments.append(current)

    hotspots = []
    for seg_idx, seg_indices in enumerate(segments):
        rn = [residue_numbers[i] for i in seg_indices]
        seg_rmsf = [rmsf_values[i] for i in seg_indices]
        max_idx  = seg_indices[int(np.argmax(seg_rmsf))]

        hotspots.append(FlexibleHotspot(
            hotspot_id=f"HS{seg_idx + 1}",
            start_residue=rn[0],
            end_residue=rn[-1],
            residue_numbers=rn,
            mean_rmsf=round(float(np.mean(seg_rmsf)), 3),
            max_rmsf=round(float(np.max(seg_rmsf)), 3),
            max_rmsf_residue=residue_numbers[max_idx],
            rmsf_source=rmsf_source,
        ))

    log.info(f"  Flexible hotspots found: {len(hotspots)} "
             f"(threshold={rmsf_threshold}Å, source={rmsf_source})")
    return hotspots


# ── Step 3: Functional context annotation ────────────────────────────────────

def _annotate_hotspot_context(
    hotspot:       FlexibleHotspot,
    coord_map:     dict[int, np.ndarray],
    active_data:   Optional[dict],
    allo_data:     Optional[dict],
) -> FlexibleHotspot:
    """
    Check if hotspot is near active site or allosteric site residues.
    Annotates is_gate if the hotspot could be gating a known functional site.
    """
    if not coord_map:
        return hotspot

    # Get centroid of hotspot
    hotspot_coords = [coord_map[rn] for rn in hotspot.residue_numbers
                      if rn in coord_map]
    if not hotspot_coords:
        return hotspot

    hotspot_centroid = np.mean(hotspot_coords, axis=0)

    # Distance to active site residues
    if active_data:
        min_dist = 999.0
        for ar in active_data.get("active_residues", []):
            rn = ar.get("residue_number")
            if rn and rn in coord_map:
                d = float(np.linalg.norm(hotspot_centroid - coord_map[rn]))
                if d < min_dist:
                    min_dist = d
        hotspot.dist_to_active_A = round(min_dist, 1)
        hotspot.near_active_site = min_dist < NEAR_FUNCTIONAL_CUTOFF

    # Distance to allosteric sites
    if allo_data:
        min_dist = 999.0
        for site in allo_data.get("allosteric_sites", []):
            for rn in site.get("residue_numbers", []):
                if rn in coord_map:
                    d = float(np.linalg.norm(hotspot_centroid - coord_map[rn]))
                    if d < min_dist:
                        min_dist = d
        hotspot.dist_to_allosteric_A = round(min_dist, 1)
        hotspot.near_allosteric = min_dist < NEAR_FUNCTIONAL_CUTOFF

    # Gate detection: flexible loop near functional site with high RMSF
    hotspot.is_gate = (
        hotspot.max_rmsf >= RMSF_GATE_THRESHOLD and
        (hotspot.near_active_site or hotspot.near_allosteric)
    )

    return hotspot


# ── Step 4: Geometric pocket simulation ──────────────────────────────────────

def _simulate_pocket_opening(
    hotspot:         FlexibleHotspot,
    ca_coords:       np.ndarray,    # (N, 3) all residue CA positions
    residue_numbers: list[int],
    residue_aas:     list[str],
    rmsf_values:     list[float],
    static_pocket_centres: list[np.ndarray],
) -> Optional[CrypticPocket]:
    """
    Simulate pocket opening by displacing hotspot residues outward.

    The displacement is proportional to RMSF magnitude and directed
    away from the protein centre of mass (radial outward displacement).
    Then we re-test for pocket burial in the perturbed configuration.

    This is NOT MD — it's a fast geometric proxy for what the protein
    looks like in an open conformation.
    """
    rn_to_idx = {rn: i for i, rn in enumerate(residue_numbers)}
    hotspot_indices = [rn_to_idx[rn] for rn in hotspot.residue_numbers
                       if rn in rn_to_idx]

    if len(hotspot_indices) < MIN_HOTSPOT_LENGTH:
        return None

    # Protein centre of mass (all residues weighted equally)
    com = ca_coords.mean(axis=0)

    # Create perturbed coordinate set
    perturbed = ca_coords.copy()

    for idx in hotspot_indices:
        # Displacement direction: away from centre of mass (radial outward)
        radial = ca_coords[idx] - com
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 0.1:
            radial_unit = np.array([0.0, 0.0, 1.0])
        else:
            radial_unit = radial / radial_norm

        # Displacement magnitude: RMSF × scale factor
        rn = residue_numbers[idx]
        rmsf_i = rmsf_values[idx] if idx < len(rmsf_values) else 1.5
        displacement = rmsf_i * PERTURBATION_SCALE

        # Move residue outward — this opens potential pockets
        perturbed[idx] = ca_coords[idx] + radial_unit * displacement

    # Now find pocket candidates in the PERTURBED structure
    # Same alpha-sphere algorithm as Module 04 but applied to new coords
    tree_p = cKDTree(perturbed)
    candidates = []

    # Test midpoints between hotspot residues and their neighbourhood
    for idx_i in hotspot_indices:
        nearby_all = tree_p.query_ball_point(perturbed[idx_i], r=BURIAL_OUTER)
        for idx_j in nearby_all:
            if idx_j <= idx_i:
                continue
            midpoint = (perturbed[idx_i] + perturbed[idx_j]) / 2.0
            shell_nearby = tree_p.query_ball_point(midpoint, r=BURIAL_OUTER)
            shell = [
                k for k in shell_nearby
                if BURIAL_INNER <= np.linalg.norm(perturbed[k] - midpoint) <= BURIAL_OUTER
            ]
            if len(shell) >= MIN_BURIAL_COUNT:
                burial = len(shell) / 20.0
                candidates.append((midpoint, burial, shell))

    if not candidates:
        return None

    # Pick the best candidate (highest burial)
    best_cand = max(candidates, key=lambda c: c[1])
    pocket_centre, burial_score, shell_indices = best_cand

    # Find lining residues in perturbed structure
    lining_idx = tree_p.query_ball_point(pocket_centre, r=POCKET_LINING_CUTOFF)
    lining_nums    = [residue_numbers[k] for k in lining_idx if k < len(residue_numbers)]
    lining_letters = [residue_aas[k]     for k in lining_idx if k < len(residue_aas)]

    if len(lining_nums) < 4:
        return None

    # Chemistry
    hydrophobes = [HYDROPHOBICITY.get(aa, 0.0) for aa in lining_letters]
    charges     = [CHARGE_AT_PH7.get(aa, 0.0)  for aa in lining_letters]
    mean_hydro  = float(np.mean(hydrophobes))
    net_charge  = float(sum(charges))

    # Enclosure: in perturbed structure (will be less than static pockets)
    # Estimate as burial_score × compaction factor
    enclosure = min(0.85, burial_score * 0.7)

    # Volume estimate: smaller than static pockets (partially open)
    volume_est = len(shell_indices) * 15.0   # 15 Å³ per atom (vs 20 for static)

    # Check if this overlaps an existing Module 04 pocket
    static_overlap = False
    pocket_centre_np = np.array(pocket_centre)
    for sc in static_pocket_centres:
        if np.linalg.norm(pocket_centre_np - sc) < 8.0:
            static_overlap = True
            break

    # Druggability score for the transient pocket
    transient_drug = _transient_druggability(
        burial_score, enclosure, mean_hydro, net_charge,
        len(lining_nums), volume_est
    )

    # Opening probability: Boltzmann estimate
    # P_open ≈ 1 - exp(-mean_rmsf / displacement_scale)
    # Higher RMSF → more likely to sample open conformation
    p_open = float(1.0 - math.exp(-hotspot.mean_rmsf / 2.5))
    p_open = round(min(0.95, max(0.05, p_open)), 3)

    # Effective druggability (the key metric)
    eff_drug = round(transient_drug * p_open, 3)

    if eff_drug < MIN_CRYPTIC_SCORE:
        return None

    # Druggability class
    if eff_drug >= 0.35:
        drug_class = "high"
    elif eff_drug >= 0.20:
        drug_class = "medium"
    else:
        drug_class = "low"

    # Confidence: higher if MD source, high RMSF, large pocket, gate
    confidence = 0.4
    if hotspot.rmsf_source == "md":
        confidence += 0.25
    if hotspot.max_rmsf >= RMSF_GATE_THRESHOLD:
        confidence += 0.15
    if hotspot.is_gate:
        confidence += 0.15
    if volume_est >= 150:
        confidence += 0.05
    confidence = round(min(0.95, confidence), 2)

    # Evidence list
    evidence = [f"RMSF={hotspot.mean_rmsf:.2f}Å ({hotspot.rmsf_source})"]
    if hotspot.is_gate:
        evidence.append("gates functional site")
    if hotspot.near_active_site:
        evidence.append(f"near active site ({hotspot.dist_to_active_A:.0f}Å)")
    if hotspot.near_allosteric:
        evidence.append(f"near allosteric ({hotspot.dist_to_allosteric_A:.0f}Å)")
    if static_overlap:
        evidence.append("partially overlaps static pocket")

    # Gate type classification
    gate_type = _classify_gate(hotspot, lining_letters)

    # Selectivity note
    sel_note = ""
    if hotspot.rmsf_source == "md" and hotspot.mean_rmsf > 2.0:
        sel_note = "conformation-selective target"
    elif hotspot.is_gate:
        sel_note = "gate-dependent binding"

    # SIM-02 state name
    sim02_name = f"cryptic_{hotspot.hotspot_id}_open"

    return CrypticPocket(
        pocket_id=f"CP_{hotspot.hotspot_id}",
        center=[round(float(c), 2) for c in pocket_centre],
        lining_residues=sorted(lining_nums),
        lining_letters=lining_letters,
        hotspot_id=hotspot.hotspot_id,
        estimated_volume_A3=round(volume_est, 1),
        enclosure=round(enclosure, 3),
        mean_hydrophobicity=round(mean_hydro, 3),
        net_charge=round(net_charge, 1),
        n_lining=len(lining_nums),
        mean_rmsf_hotspot=hotspot.mean_rmsf,
        max_rmsf_hotspot=hotspot.max_rmsf,
        opening_probability=p_open,
        rmsf_source=hotspot.rmsf_source,
        static_structure_present=static_overlap,
        transient_druggability=round(transient_drug, 3),
        effective_druggability=eff_drug,
        druggability_class=drug_class,
        near_active_site=hotspot.near_active_site,
        near_allosteric=hotspot.near_allosteric,
        gate_type=gate_type,
        denovo_target=eff_drug >= 0.25 and not static_overlap,
        selectivity_note=sel_note,
        sim02_state_name=sim02_name,
        confidence=confidence,
        evidence=evidence,
    )


def _transient_druggability(
    burial:    float,
    enclosure: float,
    hydro:     float,
    charge:    float,
    n_lining:  int,
    volume:    float,
) -> float:
    """Druggability score for a transient pocket — same logic as Module 04 but tuned down."""
    score = 0.0
    score += 0.22 * min(burial, 1.0)
    score += 0.18 * enclosure
    if 4 <= n_lining <= 15:
        score += 0.18
    elif n_lining > 2:
        score += 0.08
    if -0.5 <= hydro <= 2.5:
        score += 0.15
    elif hydro > 0:
        score += 0.07
    if 100 <= volume <= 600:
        score += 0.10
    elif volume > 50:
        score += 0.05
    if abs(charge) <= 2:
        score += 0.05   # small charge penalty relaxed vs static pockets
    return round(min(score, 0.85), 3)   # cap at 0.85 — transient < static


def _classify_gate(
    hotspot:       FlexibleHotspot,
    lining_letters: list[str],
) -> str:
    """Classify the type of gating motion based on sequence context."""
    # Simple heuristics based on hotspot length and composition
    length = len(hotspot.residue_numbers)
    has_gly = lining_letters.count("G") >= 2
    has_pro = "P" in lining_letters

    if length <= 6 and hotspot.near_active_site:
        return "activation_loop"
    if length <= 8 and has_gly:
        return "flap"
    if has_pro and length <= 10:
        return "lid"
    if length > 10:
        return "helix"
    return "loop"


# ── Step 5: SIM-02 integration payload ───────────────────────────────────────

def _build_sim02_states(
    cryptic_pockets: list[CrypticPocket],
    base_pocket_volume: float,
) -> list[dict]:
    """
    Build new conformational states from cryptic pockets for SIM-02 injection.

    Each cryptic pocket defines a new "open" state with:
    - Different pocket geometry (larger, more accessible)
    - ΔG estimate from RMSF-based Boltzmann factor
    - Probability estimate = opening_probability
    """
    states = []
    for cp in cryptic_pockets:
        # ΔG for the open state relative to apo
        # ΔG = -kT × ln(P_open / P_closed)
        p_open   = max(0.001, cp.opening_probability)
        p_closed = max(0.001, 1.0 - p_open)
        delta_G  = round(-kT_ROOM * math.log(p_open / p_closed), 2)

        states.append({
            "name":              cp.sim02_state_name,
            "delta_G_kJ_mol":    delta_G,
            "pocket_volume_A3":  cp.estimated_volume_A3,
            "pocket_shape":      cp.enclosure,
            "druggability":      cp.transient_druggability,
            "probability":       cp.opening_probability,
            "accessible":        True,
            "cryptic":           True,
            "gate_type":         cp.gate_type,
            "hotspot_rmsf":      cp.mean_rmsf_hotspot,
            "rmsf_source":       cp.rmsf_source,
        })
    return states


def _build_denovo_targets(cryptic_pockets: list[CrypticPocket]) -> list[dict]:
    """Build de novo design target descriptors for each flagged cryptic pocket."""
    targets = []
    for cp in [c for c in cryptic_pockets if c.denovo_target]:
        targets.append({
            "pocket_id":          cp.pocket_id,
            "center":             cp.center,
            "lining_residues":    cp.lining_residues,
            "estimated_volume_A3": cp.estimated_volume_A3,
            "druggability":       cp.effective_druggability,
            "opening_probability": cp.opening_probability,
            "gate_type":          cp.gate_type,
            "selectivity_note":   cp.selectivity_note,
            "rmsf_source":        cp.rmsf_source,
            "confidence":         cp.confidence,
        })
    return targets


# ── Main analysis function ────────────────────────────────────────────────────

def detect_cryptic_pockets(
    uniprot_id:      str,
    structure_data:  dict,
    inter_dir:       Path,
    md_data:         Optional[dict]  = None,
    active_data:     Optional[dict]  = None,
    pocket_data:     Optional[dict]  = None,
    allo_data:       Optional[dict]  = None,
    physico_data:    Optional[dict]  = None,
    rmsf_threshold:  float = RMSF_FLEXIBLE_THRESHOLD,
    use_md_fallback: bool  = True,
) -> CrypticPocketResult:
    """
    Full cryptic pocket detection pipeline.

    Args:
        uniprot_id:     UniProt accession
        structure_data: Module 01 output (residues + coords)
        inter_dir:      Path to intermediate data directory
        md_data:        Module 14 output (optional, for RMSF)
        active_data:    Module 03 output (for context)
        pocket_data:    Module 04 output (static pockets for comparison)
        allo_data:      Module 05 output (allosteric sites)
        physico_data:   Module 02 output (SASA data)
        rmsf_threshold: Å threshold for "flexible" (default 1.5)
        use_md_fallback: Use pLDDT proxy if no MD data

    Returns:
        CrypticPocketResult with all detected cryptic pockets
    """
    log.info(f"── Module 18: Cryptic pocket detection for {uniprot_id} ──")

    gene_name = structure_data.get("gene_name", uniprot_id)

    # ── Load residues from PDB file ──────────────────────────────────────────
    # StructureResult.to_dict() pops the 'parsed' field, so _structure.json
    # has no 'residues' key. We must re-parse the .pdb file directly.
    residues_raw = structure_data.get("residues", [])

    if not residues_raw:
        # No residues in JSON — re-parse from .pdb file
        pdb_path = structure_data.get("pdb_path", "")
        if not pdb_path:
            # Try canonical location
            structures_dir = inter_dir.parent / "structures"
            pdb_path = str(structures_dir / f"{uniprot_id}.pdb")

        from pathlib import Path as _Path
        pdb_file = _Path(pdb_path)
        if not pdb_file.exists():
            log.error(f"  PDB file not found: {pdb_file}")
            log.error(f"  Run Module 01 first: python pipeline/01_fetch_structure.py --uniprot {uniprot_id}")
            return CrypticPocketResult(
                uniprot_id=uniprot_id, gene_name=gene_name,
                sequence_length=0, rmsf_source="none"
            )

        log.info(f"  Re-parsing PDB: {pdb_file.name}")
        try:
            parsed = parse_pdb(pdb_file, uniprot_id)
            residues_raw = [r.to_dict() for r in parsed.residues]
        except Exception as e:
            log.error(f"  PDB parse failed: {e}")
            return CrypticPocketResult(
                uniprot_id=uniprot_id, gene_name=gene_name,
                sequence_length=0, rmsf_source="none"
            )

    if not residues_raw:
        log.error("  No residues found even after PDB re-parse")
        return CrypticPocketResult(
            uniprot_id=uniprot_id, gene_name=gene_name,
            sequence_length=0, rmsf_source="none"
        )

    residues = residues_raw

    # Extract arrays from structure
    residue_numbers = [r.get("residue_number", 0) for r in residues]
    residue_aas     = [r.get("one_letter", "A")   for r in residues]
    plddt_values    = [r.get("plddt", 70.0)        for r in residues]
    ca_coords_raw   = [r.get("coords", [0.0, 0.0, 0.0]) for r in residues]
    ca_coords       = np.array(ca_coords_raw, dtype=np.float64)

    # Coord map for distance calculations
    coord_map: dict[int, np.ndarray] = {
        rn: ca_coords[i] for i, rn in enumerate(residue_numbers)
    }

    n_static = pocket_data.get("n_pockets", 0) if pocket_data else 0
    static_centres: list[np.ndarray] = []
    if pocket_data:
        for p in pocket_data.get("pockets", []):
            c = p.get("center") or p.get("centroid")
            if c and len(c) == 3:
                static_centres.append(np.array(c))

    # ── Step 1: Load RMSF ────────────────────────────────────────────────────
    log.info("  [1/5] Loading RMSF data...")

    # If MD data passed directly, extract RMSF from it
    if md_data and md_data.get("rmsf_per_residue"):
        rmsf_raw = md_data["rmsf_per_residue"]
        n = len(residue_numbers)
        mean_val = float(np.mean(rmsf_raw)) if rmsf_raw else 1.0
        rmsf = [float(rmsf_raw[i]) if i < len(rmsf_raw) else mean_val
                for i in range(n)]
        rmsf_source = "md"
        log.info(f"  RMSF from passed MD data: mean={np.mean(rmsf):.2f}Å")
    else:
        rmsf, rmsf_source = _load_rmsf(
            uniprot_id, inter_dir,
            residue_numbers, plddt_values,
            use_md_fallback
        )

    # ── Step 2: Find flexible hotspots ───────────────────────────────────────
    log.info(f"  [2/5] Finding flexible hotspots (threshold={rmsf_threshold}Å)...")

    # Adaptive threshold: if mean RMSF >> threshold (highly mobile MD run),
    # switch to relative mode using the 70th percentile as threshold.
    # This ensures we identify the MOST flexible regions rather than flagging
    # the entire protein as one giant hotspot.
    rmsf_array = np.array(rmsf)
    mean_rmsf  = float(rmsf_array.mean())
    if mean_rmsf > rmsf_threshold * 2.0:
        # Use 70th percentile — top 30% most flexible residues
        adaptive_threshold = float(np.percentile(rmsf_array, 70))
        log.info(
            f"  High-mobility MD (mean={mean_rmsf:.2f}Å >> threshold={rmsf_threshold}Å). "
            f"Switching to adaptive threshold: 70th percentile = {adaptive_threshold:.2f}Å"
        )
        effective_threshold = adaptive_threshold
    else:
        effective_threshold = rmsf_threshold

    hotspots = _find_flexible_hotspots(
        residue_numbers, rmsf, effective_threshold, rmsf_source
    )

    # ── Step 3: Annotate functional context ──────────────────────────────────
    log.info("  [3/5] Annotating functional context...")
    for i, hs in enumerate(hotspots):
        hotspots[i] = _annotate_hotspot_context(hs, coord_map, active_data, allo_data)

    n_gates = sum(1 for h in hotspots if h.is_gate)
    log.info(f"  → {len(hotspots)} hotspots, {n_gates} gate-type")

    # ── Step 4: Simulate pocket opening for each hotspot ─────────────────────
    log.info("  [4/5] Simulating pocket opening...")
    cryptic_pockets: list[CrypticPocket] = []

    for hs in hotspots:
        cp = _simulate_pocket_opening(
            hs, ca_coords, residue_numbers, residue_aas,
            rmsf, static_centres
        )
        if cp is not None:
            cryptic_pockets.append(cp)

    # Sort by effective druggability
    cryptic_pockets.sort(key=lambda c: -c.effective_druggability)

    # Re-number pocket IDs after sorting
    for i, cp in enumerate(cryptic_pockets):
        cp.pocket_id = f"CP{i + 1}"

    n_high = sum(1 for cp in cryptic_pockets if cp.confidence >= 0.70)
    log.info(f"  → {len(cryptic_pockets)} cryptic pockets found "
             f"({n_high} high confidence)")

    # ── Step 5: Build integration payloads ───────────────────────────────────
    log.info("  [5/5] Building integration payloads...")
    base_vol = (static_centres[0].mean() if static_centres else 300.0)
    sim02_states  = _build_sim02_states(cryptic_pockets, base_vol)
    denovo_targets = _build_denovo_targets(cryptic_pockets)

    result = CrypticPocketResult(
        uniprot_id=uniprot_id,
        gene_name=gene_name,
        sequence_length=len(residues),
        rmsf_source=rmsf_source,
        n_flexible_hotspots=len(hotspots),
        n_cryptic_pockets=len(cryptic_pockets),
        n_high_confidence=n_high,
        n_static_pockets=n_static,
        hotspots=hotspots,
        cryptic_pockets=cryptic_pockets,
        sim02_new_states=sim02_states,
        denovo_targets=denovo_targets,
    )

    log.info(result.summary())
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--rmsf-threshold", "-r", default=RMSF_FLEXIBLE_THRESHOLD,
              type=float,
              help=f"RMSF threshold for flexible residues in Å "
                   f"(default: {RMSF_FLEXIBLE_THRESHOLD})")
@click.option("--no-md-fallback", "no_fallback", is_flag=True, default=False,
              help="Disable pLDDT proxy fallback if MD not available")
def main(uniprot: str, rmsf_threshold: float, no_fallback: bool) -> None:
    """
    Module 18 — Cryptic / transient pocket detection.

    Correlates high-RMSF flexible regions (from Module 14 MD or pLDDT proxy)
    with geometric pocket opening simulation to find pockets invisible
    in the static AlphaFold structure.

    Requires: Module 01 (structure). Benefits from: Module 14 (MD), 03, 04, 05.

    Example:
        python pipeline/cryptic_pockets.py --uniprot P04637
        python pipeline/cryptic_pockets.py --uniprot P04637 --rmsf-threshold 1.2
    """
    uid       = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    def _load(fname: str) -> Optional[dict]:
        p = inter_dir / fname
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"  Could not load {fname}: {e}")
        return None

    struct_data  = _load(f"{uid}_structure.json")
    if not struct_data:
        log.error(
            f"Structure JSON not found for {uid}.\n"
            f"  Run Module 01 first: python pipeline/01_fetch_structure.py --uniprot {uid}"
        )
        raise SystemExit(1)

    md_data      = _load(f"{uid}_md.json")
    active_data  = _load(f"{uid}_active_sites.json")
    pocket_data  = _load(f"{uid}_binding_pockets.json")
    allo_data    = _load(f"{uid}_allosteric.json")
    physico_data = _load(f"{uid}_physicochemical.json")

    # Log what's available
    available = []
    if md_data:      available.append("MD (RMSF)")
    if active_data:  available.append("active sites")
    if pocket_data:  available.append(f"static pockets ({pocket_data.get('n_pockets', 0)})")
    if allo_data:    available.append("allosteric")
    log.info(f"  Upstream modules: {', '.join(available) or 'structure only'}")

    result = detect_cryptic_pockets(
        uniprot_id=uid,
        structure_data=struct_data,
        inter_dir=inter_dir,
        md_data=md_data,
        active_data=active_data,
        pocket_data=pocket_data,
        allo_data=allo_data,
        physico_data=physico_data,
        rmsf_threshold=rmsf_threshold,
        use_md_fallback=not no_fallback,
    )

    # Save output
    out_path = inter_dir / f"{uid}_cryptic_pockets.json"
    result.to_json(out_path)

    click.echo(result.summary())

    if result.cryptic_pockets:
        click.echo(f"  Top cryptic pockets for de novo design:")
        for cp in result.cryptic_pockets[:5]:
            click.echo(
                f"    {cp.pocket_id}: {cp.estimated_volume_A3:.0f}Å³  "
                f"P_open={cp.opening_probability:.2f}  "
                f"eff.drug={cp.effective_druggability:.3f}  "
                f"[{cp.druggability_class}]  {cp.gate_type}"
            )
        if result.denovo_targets:
            click.echo(
                f"\n  {len(result.denovo_targets)} cryptic pockets flagged "
                f"for de novo design (see _cryptic_pockets.json → denovo_targets)"
            )

    click.echo(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()