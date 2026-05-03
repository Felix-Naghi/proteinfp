"""
pipeline/denovo_design_context.py
──────────────────────────────────
Gap 2 + Gap 3 fix: bridges the consensus report and MD flexibility
data into the de novo molecular designer.

This module is the only file that needs to change on the context-loading
side.  denovo_design.py gets two additional optional arguments:

    run_denovo_design(
        ...,
        consensus_data = load_consensus_context(uniprot),   # Gap 2
        md_data        = load_md_context(uniprot),          # Gap 3
    )

WHAT EACH FUNCTION DOES
───────────────────────
load_consensus_context()
    Reads the Module 13 consensus report and extracts the pocket that
    the consensus scoring ranked #1 by druggability, cross-referenced
    with active site evidence.  Returns a ConsensusContext dataclass.

    The designer uses this to:
      - Override pocket selection: use the consensus-ranked pocket
        instead of the raw Module 04 order.
      - Use the consensus-weighted druggability score (which fuses
        evidence from ESM-2, homology, active site motifs) rather
        than the geometry-only score from Module 04.

load_md_context()
    Reads the Module 14 MD output and extracts per-residue RMSF,
    flexible residue list, and per-site stability scores.
    Returns an MDContext dataclass.

    The designer uses this to:
      - Adjust fragment size: rigid pockets (low RMSF) → larger,
        more rigid scaffolds; flexible pockets (high RMSF) → smaller,
        more conformationally adaptable fragments.
      - Bias mutation operators: avoid mutating residue contacts at
        flexible sites (they move too much to optimise against the
        static structure).
      - Set box size: flexible loops near the pocket → larger docking
        box to capture alternative conformations.
      - Log a flexibility warning if the target pocket RMSF is so high
        that static docking results may be unreliable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Thresholds ─────────────────────────────────────────────────────────────────

# Pocket mean RMSF above this → "flexible", use adaptive strategy
FLEXIBLE_RMSF_THRESHOLD   = 1.5   # Å  (same as MD module)
# Pocket mean RMSF above this → static docking is unreliable, warn loudly
VERY_FLEXIBLE_RMSF_WARN   = 3.0   # Å
# Druggability score required for consensus to override raw pocket order
MIN_CONSENSUS_DRUG_SCORE  = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsensusPocket:
    """
    A single pocket as ranked by the consensus report.
    Fields mirror the BindingPocket dataclass in binding_pockets.py
    so existing code can consume them unchanged.
    """
    pocket_id:           str
    center:              list[float]
    volume_A3:           float
    druggability_score:  float
    druggability_class:  str
    lining_residues:     list[int]
    lining_letters:      list[str]
    near_active_site:    bool
    active_site_residues: list[int]
    net_charge:          float
    mean_hydrophobicity: float
    # Consensus-specific additions
    consensus_rank:      int    = 0    # 1 = top consensus pick
    evidence_score:      float  = 0.0  # Module 13 weighted evidence score
    n_evidence_sources:  int    = 0


@dataclass
class ConsensusContext:
    """
    Everything the de novo designer needs from the consensus report.
    """
    uniprot_id:        str
    overall_confidence: str                    # HIGH / MEDIUM / LOW
    n_evidence_sources: int
    top_pocket:        Optional[ConsensusPocket]   = None
    ranked_pockets:    list[ConsensusPocket]        = field(default_factory=list)
    top_active_site_residues: list[int]             = field(default_factory=list)
    is_enzyme:         bool                    = False
    ec_number:         str                     = ""
    # Raw consensus report for downstream use
    raw_report:        dict                    = field(default_factory=dict)

    @property
    def has_good_pocket(self) -> bool:
        return (
            self.top_pocket is not None
            and self.top_pocket.druggability_score >= MIN_CONSENSUS_DRUG_SCORE
        )

    def pocket_summary(self) -> str:
        if not self.top_pocket:
            return "  Consensus: no pocket data"
        p = self.top_pocket
        return (
            f"  Consensus pocket {p.pocket_id}: "
            f"druggability={p.druggability_score:.2f} ({p.druggability_class})  "
            f"vol={p.volume_A3:.0f}Å³  "
            f"evidence={p.evidence_score:.1f} ({p.n_evidence_sources} sources)"
        )


@dataclass
class SiteFlexibility:
    """
    Flexibility profile for one binding site (pocket / active / allosteric).
    """
    site_id:          str
    site_type:        str          # "pocket" | "active" | "allosteric"
    mean_rmsf:        float        # Å — mean over lining residues
    max_rmsf:         float        # Å — worst residue
    flexible_fraction: float       # fraction of lining residues above threshold
    stability_score:  float        # 0–1, from MD module
    is_flexible:      bool         # mean_rmsf > FLEXIBLE_RMSF_THRESHOLD
    is_very_flexible: bool         # mean_rmsf > VERY_FLEXIBLE_RMSF_WARN


@dataclass
class MDContext:
    """
    Everything the de novo designer needs from the MD simulation.
    """
    uniprot_id:       str
    source:           str          # "md" | "plddt_proxy" | "none"
    mean_rmsf:        float        # global mean RMSF (Å)
    flexible_residues: list[int]   # residue numbers with RMSF > threshold
    rmsf_per_residue: list[float]  # full per-residue array (same order as sequence)
    site_flexibility: list[SiteFlexibility] = field(default_factory=list)

    def get_site_flex(self, site_id: str) -> Optional[SiteFlexibility]:
        for sf in self.site_flexibility:
            if sf.site_id == site_id:
                return sf
        return None

    def pocket_rmsf(self, lining_residues: list[int]) -> float:
        """Mean RMSF for a set of pocket lining residues."""
        if not self.rmsf_per_residue or not lining_residues:
            return 0.0
        # Build a residue-number → index map using positional fallback
        # rmsf_per_residue is indexed 0..N-1 matching sequence order
        # We approximate by treating residue numbers as 1-based indices
        vals = []
        for rn in lining_residues:
            idx = rn - 1  # most proteins start at residue 1
            if 0 <= idx < len(self.rmsf_per_residue):
                vals.append(self.rmsf_per_residue[idx])
        return float(sum(vals) / len(vals)) if vals else 0.0

    def flexibility_strategy(self, mean_pocket_rmsf: float) -> dict:
        """
        Return a strategy dict that denovo_design.py uses to adapt
        its molecular evolution parameters.

        Rigid pocket   (RMSF < 1.5Å): larger scaffolds, more precise fit
        Flexible pocket (RMSF 1.5–3Å): medium scaffolds, more mutations
        Very flexible  (RMSF > 3.0Å): small, adaptable fragments; warn user

        Returns dict with keys:
          max_build_steps   int    max fragment growth steps
          mutation_rate     float  relative mutation rate multiplier
          box_padding       float  extra Å added to docking box
          scaffold_bias     str    "rigid" | "flexible" | "small"
          warning           str    "" or a warning message
        """
        if mean_pocket_rmsf < FLEXIBLE_RMSF_THRESHOLD:
            return {
                "max_build_steps": 5,
                "mutation_rate":   1.0,
                "box_padding":     0.0,
                "scaffold_bias":   "rigid",
                "warning":         "",
            }
        elif mean_pocket_rmsf < VERY_FLEXIBLE_RMSF_WARN:
            return {
                "max_build_steps": 3,
                "mutation_rate":   1.4,
                "box_padding":     3.0,
                "scaffold_bias":   "flexible",
                "warning":         (
                    f"  Pocket RMSF={mean_pocket_rmsf:.1f}Å — site is flexible. "
                    f"Using shorter scaffolds and wider docking box."
                ),
            }
        else:
            return {
                "max_build_steps": 2,
                "mutation_rate":   2.0,
                "box_padding":     6.0,
                "scaffold_bias":   "small",
                "warning":         (
                    f"  WARNING: Pocket RMSF={mean_pocket_rmsf:.1f}Å — "
                    f"site is highly flexible. Static docking scores may be "
                    f"unreliable. Consider running MD-ensemble docking."
                ),
            }


# ══════════════════════════════════════════════════════════════════════════════
# GAP 2 FIX: load_consensus_context()
# ══════════════════════════════════════════════════════════════════════════════

def load_consensus_context(
    uniprot_id: str,
    inter_dir:  Optional[Path] = None,
) -> Optional[ConsensusContext]:
    """
    Load and parse the Module 13 consensus report for a protein.

    Returns ConsensusContext with ranked pockets, or None if the
    consensus report doesn't exist yet.

    The returned pocket ranking is based on the consensus report's
    druggability_score, which is the Module 13 fused score (geometry
    + active site evidence + homology), not the raw Module 04 score.
    This is the key difference from what denovo_design.py was doing
    before — it was using Module 04's raw order, not Module 13's.
    """
    if inter_dir is None:
        from utils.config import cfg
        inter_dir = Path(cfg.paths["intermediate"])

    report_dir  = Path(str(inter_dir)).parent / "reports"
    report_path = report_dir / f"{uniprot_id}_report.json"

    if not report_path.exists():
        return None

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # ── Extract ranked pockets from consensus ──────────────────────────────────
    raw_pockets = report.get("binding_pockets", [])
    n_sources   = report.get("n_evidence_sources", 0)
    overall_conf = report.get("overall_confidence", "LOW")

    # The consensus report already has pockets sorted by druggability_score
    # (done in Module 13 build_consensus_report → pockets[:10]).
    # We wrap each pocket in ConsensusPocket and add consensus-specific fields.
    ranked = []
    for i, p in enumerate(raw_pockets):
        drug_score = float(p.get("druggability_score", 0.0))
        if drug_score < MIN_CONSENSUS_DRUG_SCORE:
            continue

        # Estimate evidence score: druggability × evidence sources × active bonus
        near_active = p.get("near_active_site", False)
        evidence_boost = 1.5 if near_active else 1.0
        evidence_score = drug_score * n_sources * evidence_boost

        cp = ConsensusPocket(
            pocket_id            = p.get("pocket_id", f"P{i+1}"),
            center               = p.get("center", [0.0, 0.0, 0.0]),
            volume_A3            = float(p.get("volume_A3", 0.0)),
            druggability_score   = drug_score,
            druggability_class   = p.get("druggability_class", "low"),
            lining_residues      = p.get("lining_residues", []),
            lining_letters       = p.get("lining_letters", []),
            near_active_site     = near_active,
            active_site_residues = p.get("active_site_residues", []),
            net_charge           = float(p.get("net_charge", 0.0)),
            mean_hydrophobicity  = float(p.get("mean_hydrophobicity", 0.0)),
            consensus_rank       = i + 1,
            evidence_score       = round(evidence_score, 2),
            n_evidence_sources   = n_sources,
        )
        ranked.append(cp)

    # Re-sort by evidence_score (fuses druggability + active site + sources)
    ranked.sort(key=lambda p: p.evidence_score, reverse=True)
    for i, p in enumerate(ranked):
        p.consensus_rank = i + 1

    # ── Extract top active site residue numbers ────────────────────────────────
    active_sites = report.get("active_sites", [])
    top_active_res = [
        s["residue_number"] for s in active_sites
        if s.get("confidence") in ("HIGH", "MEDIUM")
        and "residue_number" in s
    ]

    return ConsensusContext(
        uniprot_id            = uniprot_id,
        overall_confidence    = overall_conf,
        n_evidence_sources    = n_sources,
        top_pocket            = ranked[0] if ranked else None,
        ranked_pockets        = ranked,
        top_active_site_residues = top_active_res,
        is_enzyme             = report.get("is_enzyme", False),
        ec_number             = report.get("ec_number", ""),
        raw_report            = report,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GAP 3 FIX: load_md_context()
# ══════════════════════════════════════════════════════════════════════════════

def load_md_context(
    uniprot_id:   str,
    inter_dir:    Optional[Path] = None,
    pocket_data:  Optional[dict] = None,
    active_data:  Optional[dict] = None,
) -> MDContext:
    """
    Load Module 14 MD output and build an MDContext for the designer.

    Falls back gracefully — if no MD data exists, returns an MDContext
    with source="none" and all RMSF values set to 0.0 (no effect on
    the designer's behaviour).

    pocket_data / active_data are passed in so we can compute per-pocket
    RMSF from the raw lining residue lists.
    """
    if inter_dir is None:
        from utils.config import cfg
        inter_dir = Path(cfg.paths["intermediate"])

    md_path = Path(str(inter_dir)) / f"{uniprot_id}_md.json"

    if not md_path.exists():
        return MDContext(
            uniprot_id        = uniprot_id,
            source            = "none",
            mean_rmsf         = 0.0,
            flexible_residues = [],
            rmsf_per_residue  = [],
        )

    try:
        md = json.loads(md_path.read_text(encoding="utf-8"))
    except Exception:
        return MDContext(
            uniprot_id        = uniprot_id,
            source            = "none",
            mean_rmsf         = 0.0,
            flexible_residues = [],
            rmsf_per_residue  = [],
        )

    rmsf_raw          = md.get("rmsf_per_residue", [])
    flexible_residues = md.get("flexible_residues", [])
    mean_rmsf         = float(sum(rmsf_raw) / len(rmsf_raw)) if rmsf_raw else 0.0

    # ── Build per-site flexibility from MD site_dynamics ─────────────────────
    site_flex: list[SiteFlexibility] = []
    for sd in md.get("site_dynamics", []):
        m_rmsf  = float(sd.get("mean_rmsf", 0.0))
        mx_rmsf = float(sd.get("max_rmsf", 0.0))
        site_flex.append(SiteFlexibility(
            site_id           = sd.get("site_id", "?"),
            site_type         = sd.get("site_type", "pocket"),
            mean_rmsf         = m_rmsf,
            max_rmsf          = mx_rmsf,
            flexible_fraction = float(sd.get("flexible_fraction", 0.0)),
            stability_score   = float(sd.get("stability_score", 1.0)),
            is_flexible       = m_rmsf > FLEXIBLE_RMSF_THRESHOLD,
            is_very_flexible  = m_rmsf > VERY_FLEXIBLE_RMSF_WARN,
        ))

    # ── Add per-pocket RMSF for any pocket not already in site_dynamics ───────
    if pocket_data and rmsf_raw:
        existing_ids = {sf.site_id for sf in site_flex}
        for p in pocket_data.get("pockets", []):
            pid = p.get("pocket_id", "P?")
            if pid in existing_ids:
                continue
            lining = p.get("lining_residues", [])
            vals   = []
            for rn in lining:
                idx = rn - 1
                if 0 <= idx < len(rmsf_raw):
                    vals.append(rmsf_raw[idx])
            if vals:
                m  = float(sum(vals) / len(vals))
                mx = float(max(vals))
                ff = sum(1 for v in vals if v > FLEXIBLE_RMSF_THRESHOLD) / len(vals)
                stab = 1.0 / (1.0 + m)
                site_flex.append(SiteFlexibility(
                    site_id           = pid,
                    site_type         = "pocket",
                    mean_rmsf         = m,
                    max_rmsf          = mx,
                    flexible_fraction = ff,
                    stability_score   = stab,
                    is_flexible       = m > FLEXIBLE_RMSF_THRESHOLD,
                    is_very_flexible  = m > VERY_FLEXIBLE_RMSF_WARN,
                ))

    return MDContext(
        uniprot_id        = uniprot_id,
        source            = "md",
        mean_rmsf         = mean_rmsf,
        flexible_residues = flexible_residues,
        rmsf_per_residue  = [float(v) for v in rmsf_raw],
        site_flexibility  = site_flex,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FRAGMENT BIAS TABLES
# Used by denovo_design.py to bias the fragment pool based on site properties.
# ══════════════════════════════════════════════════════════════════════════════

# Rigid pocket fragments: larger, more rigid scaffolds with defined geometry
RIGID_SITE_FRAGMENTS = [
    "c1ccc2ccccc2c1",   # naphthalene — bicyclic, flat, fills rigid groove
    "C1CCCCC1",         # cyclohexane — rigid saturated ring
    "c1cccc2ccccc12",   # azulene — rigid bicyclic
    "C1CC2CCCC2CC1",    # decalin — rigid bicyclic saturated
    "c1cnc2ccccc2n1",   # quinazoline — planar, good for kinase hinge
    "C1CCC(CC1)C(=O)",  # cyclohexyl carbonyl — rigid with H-bond acceptor
    "c1ccc(cc1)C(F)(F)F",  # trifluoromethyl phenyl — rigid + metabolically stable
]

# Flexible pocket fragments: smaller, more conformationally adaptable
FLEXIBLE_SITE_FRAGMENTS = [
    "CC(C)C",           # isobutyl — small, flexible, adapts to moving pocket
    "CCC",              # propyl — minimal
    "CC(=O)N",          # acetamide — small H-bond donor
    "CCO",              # ethanol — small, polar
    "c1ccncc1",         # pyridine — small aromatic, H-bond acceptor
    "CC(C)O",           # isopropanol — small, branched
    "CN(C)C",           # trimethylamine — small, flexible
]

# Very flexible / highly dynamic pockets: very small fragments
SMALL_ADAPTABLE_FRAGMENTS = [
    "CC",               # ethyl — minimal
    "CO",               # methanol
    "CN",               # methylamine
    "c1ccncc1",         # pyridine
    "CCN",              # ethylamine
]


def get_flexibility_fragments(scaffold_bias: str) -> list[str]:
    """Return the appropriate fragment list for a given scaffold bias."""
    return {
        "rigid":    RIGID_SITE_FRAGMENTS,
        "flexible": FLEXIBLE_SITE_FRAGMENTS,
        "small":    SMALL_ADAPTABLE_FRAGMENTS,
    }.get(scaffold_bias, FLEXIBLE_SITE_FRAGMENTS)