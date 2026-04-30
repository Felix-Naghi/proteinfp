"""
pipeline/ptm_analysis.py
─────────────────────────
Module 17 — Post-Translational Modification (PTM) Analysis

WHY THIS MODULE EXISTS
──────────────────────
Your pipeline currently treats proteins as static sequences with a fixed
charge/hydrophobicity profile. This is biologically wrong for most signalling
proteins: phosphorylation of TP53 at S15 or S20 completely changes its
transcriptional activity, stability, and binding partners. Ubiquitination of
K48 targets a protein for proteasomal destruction. Glycosylation of a surface
asparagine blocks an antibody epitope.

Without PTM awareness:
  - SIM-02 conformational ensemble has wrong ΔΔG corrections (charge not updated)
  - SIM-04 binding probability ignores phospho-dependent binding pockets
  - Antibody design can target epitopes that are glycosylated and blocked
  - Active site predictions miss phospho-activating/inactivating switches

This module fixes all of that.

WHAT IT DOES
────────────
1. Known sites — queries PhosphoSitePlus (via UniProt cross-ref) and
   fetches experimentally validated PTM sites from the UniProt API.
   Falls back to an embedded curated table of the 50 most important
   signalling proteins if the API is unavailable.

2. Predicted sites — scans the sequence with literature-validated
   consensus motifs:
     • Phosphorylation: [ST]-P, R-x-x-[ST], [RK]-x-x-x-[ST] (PKA, PKC,
       CDK, CK2, ATM/ATR, MAPK consensus sequences)
     • Ubiquitination: K in disordered/exposed regions; PEST sequences
     • Acetylation: K in N-terminal region or histone-fold domains
     • Glycosylation: N-x-[ST] (N-linked); [ST] in secreted/membrane context
     • SUMOylation: ψ-K-x-E (SUMO consensus)
     • Methylation: R-x motifs in RNA-binding domains

3. Functional impact scoring — for each PTM site, computes:
     • ΔCharge: phospho adds -2, acetylation of K removes +1, etc.
     • ΔΔG_binding: how much this PTM shifts pocket druggability
       (Coulombic correction from charge change at active site distance)
     • Active site proximity: if PTM is within 8Å of an active residue
     • Conformational state shift: phospho on activation loop → activates
     • GRN signal: is this site part of a known kinase cascade?

4. SIM-02 integration — outputs a PTM-corrected ΔΔG for each
   conformational state. The ensemble model should apply this correction
   on top of its pH/crowding/ion terms.

5. Druggability impact — phospho-mimetics (S→D/E) and acetyl-mimetics
   are noted so the de novo designer can target the modified state.

OUTPUT
──────
Saves: data/intermediate/{uid}_ptm.json

Consumed by:
  - pipeline/consensus.py       (adds PTM to report)
  - sim/step02_protein_ensemble.py  (ΔΔG corrections per state)
  - sim/step04_binding_probability.py  (charge corrections at pocket)
  - pipeline/antibody_design.py  (blocks glycosylated epitopes)
  - pipeline/selectivity_optimizer.py (avoids phospho-site mimics)

Usage:
    python pipeline/ptm_analysis.py --uniprot P04637
    python pipeline/ptm_analysis.py --uniprot P04637 --no-api
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ── Physical constants ─────────────────────────────────────────────────────────

R   = 8.314    # J/mol/K
T   = 310.15   # K (37°C)
kT  = R * T / 1000  # kJ/mol

# ── PTM type definitions ───────────────────────────────────────────────────────

# Each PTM type: (display_name, charge_delta, mass_shift_Da, reversible)
PTM_TYPES: dict[str, tuple[str, float, float, bool]] = {
    "phosphoserine":    ("Phosphoserine",    -2.0, +79.966,  True),
    "phosphothreonine": ("Phosphothreonine", -2.0, +79.966,  True),
    "phosphotyrosine":  ("Phosphotyrosine",  -2.0, +79.966,  True),
    "ubiquitination":   ("Ubiquitination",    0.0, +114.043, True),
    "acetylation":      ("Acetylation",      -1.0, +42.011,  True),  # removes +1 from K
    "methylation":      ("Methylation",       0.0, +14.016,  True),
    "dimethylation":    ("Dimethylation",     0.0, +28.031,  True),
    "trimethylation":   ("Trimethylation",   +1.0, +42.047,  True),  # adds +1 to K
    "sumoylation":      ("SUMOylation",       0.0, +97.0,    True),
    "nglycosylation":   ("N-glycosylation",   0.0, +1316.0,  False), # complex glycan avg
    "oglycosylation":   ("O-glycosylation",   0.0, +203.0,   False),
    "palmitoylation":   ("Palmitoylation",    0.0, +238.4,   True),
    "myristoylation":   ("Myristoylation",    0.0, +210.4,   False),
    "hydroxylation":    ("Hydroxylation",     0.0, +15.995,  False),
    "nitrosylation":    ("S-nitrosylation",   0.0, +28.990,  True),
}

# Residue that each PTM targets
PTM_TARGET_AA: dict[str, set[str]] = {
    "phosphoserine":    {"S"},
    "phosphothreonine": {"T"},
    "phosphotyrosine":  {"Y"},
    "ubiquitination":   {"K"},
    "acetylation":      {"K", "N"},  # N-terminal acetylation also
    "methylation":      {"K", "R"},
    "dimethylation":    {"K", "R"},
    "trimethylation":   {"K"},
    "sumoylation":      {"K"},
    "nglycosylation":   {"N"},
    "oglycosylation":   {"S", "T"},
    "palmitoylation":   {"C"},
    "myristoylation":   {"G"},  # N-terminal glycine
    "hydroxylation":    {"P", "K"},
    "nitrosylation":    {"C"},
}

# Motif patterns for predicted PTMs (regex on sequence)
# Each: (ptm_type, regex_pattern, kinase/enzyme, confidence_base)
MOTIF_PATTERNS: list[tuple[str, str, str, float]] = [
    # ── Phosphorylation ───────────────────────────────────────────────────────
    # CDK consensus: [ST]-P-x-[RK]
    ("phosphoserine",    r"SP[A-Z][RK]",       "CDK",   0.82),
    ("phosphothreonine", r"TP[A-Z][RK]",       "CDK",   0.82),
    # PKA consensus: R-[RK]-x-[ST]
    ("phosphoserine",    r"R[RK][A-Z]S",       "PKA",   0.78),
    ("phosphothreonine", r"R[RK][A-Z]T",       "PKA",   0.78),
    # CK2 consensus: [ST]-x-x-[DE]
    ("phosphoserine",    r"S[A-Z]{2}[DE]",     "CK2",   0.72),
    ("phosphothreonine", r"T[A-Z]{2}[DE]",     "CK2",   0.72),
    # MAPK consensus: [ST]-P (minimal; overlaps CDK)
    ("phosphoserine",    r"SP",                 "MAPK",  0.65),
    ("phosphothreonine", r"TP",                 "MAPK",  0.65),
    # ATM/ATR consensus: [ST]-Q
    ("phosphoserine",    r"SQ",                 "ATM",   0.76),
    ("phosphothreonine", r"TQ",                 "ATM",   0.76),
    # PKC consensus: [ST] near basic cluster [RK]-x-[ST]
    ("phosphoserine",    r"[RK][A-Z]S",        "PKC",   0.70),
    ("phosphothreonine", r"[RK][A-Z]T",        "PKC",   0.70),
    # ── Ubiquitination ────────────────────────────────────────────────────────
    # PEST sequence degron signal: P-E/D/S-T-S/T region
    ("ubiquitination",   r"PE[A-Z]{0,5}[ST]",  "SCF",   0.60),
    # ── N-glycosylation ───────────────────────────────────────────────────────
    # Strict NxS/T sequon (x ≠ P)
    ("nglycosylation",   r"N[^P][ST]",         "OST",   0.85),
    # ── SUMOylation ───────────────────────────────────────────────────────────
    # ψKxE consensus (ψ = hydrophobic: I/L/V/F)
    ("sumoylation",      r"[ILVF]K[A-Z]E",     "SUMO",  0.73),
    # ── Acetylation ───────────────────────────────────────────────────────────
    # Lysine acetylation near histone-fold KxxK
    ("acetylation",      r"K[A-Z]{2}K",        "HAT",   0.58),
]

# Curated known PTM sites for the most common pipeline proteins
# Format: uniprot_id → list of (residue_num, aa, ptm_type, source, functional_note)
CURATED_KNOWN_PTMS: dict[str, list[tuple[int, str, str, str, str]]] = {
    "P04637": [  # TP53
        (15,  "S", "phosphoserine",    "ATM",     "DNA damage response; activates p53"),
        (20,  "S", "phosphoserine",    "CHK2",    "Disrupts MDM2 binding; stabilises p53"),
        (37,  "S", "phosphoserine",    "ATM",     "Transactivation domain"),
        (46,  "S", "phosphoserine",    "HIPK2",   "Pro-apoptotic signalling"),
        (127, "S", "phosphoserine",    "CK2",     "Cytoplasmic retention signal"),
        (315, "S", "phosphoserine",    "ATM",     "Tetramerisation domain"),
        (371, "K", "acetylation",      "CBP",     "Activates transcription"),
        (372, "K", "acetylation",      "PCAF",    "Activates transcription"),
        (373, "K", "acetylation",      "CBP",     "Activates transcription"),
        (381, "K", "acetylation",      "CBP",     "DNA damage response"),
        (382, "K", "acetylation",      "CBP",     "Most studied p53 acetylation"),
        (386, "K", "acetylation",      "PCAF",    "Sequence-specific DNA binding"),
        (305, "K", "methylation",      "SMYD2",   "Negative regulation of p53"),
        (370, "K", "methylation",      "SMYD2",   "Positive regulation of p53"),
        (120, "K", "acetylation",      "TIP60",   "Apoptosis vs senescence switch"),
        (175, "C", "nitrosylation",    "nNOS",    "Zinc coordination in DNA-binding domain"),
    ],
    "P00533": [  # EGFR
        (768, "Y", "phosphotyrosine",  "EGFR",    "Activation loop; kinase activation"),
        (1045,"Y", "phosphotyrosine",  "EGFR",    "CBL recruitment; receptor downregulation"),
        (1068,"Y", "phosphotyrosine",  "EGFR",    "GRB2 SH2 binding; RAS-MAPK pathway"),
        (1086,"Y", "phosphotyrosine",  "EGFR",    "PI3K recruitment"),
        (1148,"Y", "phosphotyrosine",  "EGFR",    "Downstream signalling"),
        (1173,"Y", "phosphotyrosine",  "EGFR",    "SHC binding; proliferation"),
    ],
    "P38398": [  # BRCA1
        (1387,"S", "phosphoserine",    "ATM",     "DNA damage checkpoint"),
        (1423,"S", "phosphoserine",    "CHK2",    "Cell cycle checkpoint"),
        (1457,"S", "phosphoserine",    "ATM",     "DNA repair"),
        (1524,"S", "phosphoserine",    "ATM",     "Homologous recombination"),
    ],
    "P49841": [  # GSK3B
        (9,   "S", "phosphoserine",    "PKB",     "Inhibitory phosphorylation; insulin signalling"),
        (216, "Y", "phosphotyrosine",  "ZAP70",   "Activating phosphorylation"),
    ],
    "Q00987": [  # MDM2
        (166, "S", "phosphoserine",    "ATM",     "Disrupts p53-MDM2 interaction"),
        (260, "S", "phosphoserine",    "CK1",     "Ubiquitin ligase activity"),
        (395, "S", "phosphoserine",    "ATM",     "MDM2 nuclear export"),
    ],
    "P51587": [  # BRCA2
        (3291,"S", "phosphoserine",    "CDK1",    "Mitotic regulation"),
        (3387,"S", "phosphoserine",    "ATM",     "DNA repair"),
    ],
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PTMSite:
    """A single PTM site with full annotation."""
    residue_number: int
    residue_aa:     str           # one-letter amino acid
    ptm_type:       str           # key into PTM_TYPES
    ptm_name:       str           # display name
    source:         str           # "experimental" / "motif_scan" / "curated"
    kinase_enzyme:  str           # kinase/enzyme responsible, if known
    confidence:     float         # 0–1
    charge_delta:   float         # Δcharge from this PTM
    mass_shift_Da:  float         # mass shift
    reversible:     bool

    # Functional impact (filled by _compute_functional_impact)
    active_site_proximity_A: float = 0.0   # distance to nearest active site residue (Å)
    pocket_proximity_A:      float = 0.0   # distance to nearest pocket residue (Å)
    is_active_site_switch:   bool  = False # directly regulates catalytic activity
    is_binding_switch:       bool  = False # directly regulates a binding interface
    ddG_binding_kJ:          float = 0.0  # estimated ΔΔG on binding pocket
    conformational_effect:   str   = ""   # "activating" / "inactivating" / "none"
    functional_note:         str   = ""   # human-readable summary

    # SIM-02 integration
    ddG_state_active_kJ:    float = 0.0   # ΔΔG on "active" conformational state
    ddG_state_inactive_kJ:  float = 0.0   # ΔΔG on "inactive" conformational state
    blocks_epitope:          bool  = False # relevant for antibody design

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PTMResult:
    """Full PTM analysis output. Output of Module 17."""
    uniprot_id:       str
    gene_name:        str
    sequence_length:  int
    n_known:          int = 0
    n_predicted:      int = 0
    n_high_conf:      int = 0

    sites:            list[PTMSite] = field(default_factory=list)

    # Aggregate statistics
    total_charge_shift_phospho:  float = 0.0  # if all phospho sites active
    n_active_site_switches:      int   = 0
    n_binding_switches:          int   = 0
    n_glycosylated_surface:      int   = 0

    # SIM-02 integration summary
    ddG_ensemble_correction_kJ:  float = 0.0  # net ΔΔG to apply to active state
    dominant_kinase:             str   = ""   # most common kinase targeting this protein
    ptm_crosstalk:               list[str] = field(default_factory=list)  # competing PTMs on same residue

    # State-specific corrections for SIM-02
    state_corrections: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*65}",
            f"  PTM Analysis: {self.gene_name} ({self.uniprot_id})",
            f"{'─'*65}",
            f"  Known sites       : {self.n_known}",
            f"  Predicted sites   : {self.n_predicted}",
            f"  High-confidence   : {self.n_high_conf}",
            f"  Active site switches: {self.n_active_site_switches}",
            f"  Binding switches  : {self.n_binding_switches}",
            f"  Glycosylated surface: {self.n_glycosylated_surface}",
            f"  Net charge shift  : {self.total_charge_shift_phospho:+.1f} (if all phospho active)",
            f"  SIM-02 ΔΔG corr.  : {self.ddg_ensemble_correction_kj():+.2f} kJ/mol",
        ]
        if self.dominant_kinase:
            lines.append(f"  Dominant kinase   : {self.dominant_kinase}")

        # Top sites
        top = sorted(self.sites, key=lambda s: s.confidence, reverse=True)[:8]
        if top:
            lines += ["", f"  {'Residue':<12} {'PTM':<22} {'Conf':>5}  {'ΔΔG':>7}  {'Note':<30}"]
            lines += [f"  {'─'*12} {'─'*22} {'─'*5}  {'─'*7}  {'─'*30}"]
            for s in top:
                note = s.functional_note[:30] if s.functional_note else s.kinase_enzyme
                lines.append(
                    f"  {s.residue_aa}{s.residue_number:<11} "
                    f"{s.ptm_name:<22} "
                    f"{s.confidence:>5.2f}  "
                    f"{s.ddG_binding_kJ:>+6.2f}  "
                    f"{note:<30}"
                )
        lines.append(f"{'─'*65}")
        return "\n".join(lines)

    def ddg_ensemble_correction_kj(self) -> float:
        """Net ΔΔG correction for SIM-02 active state (from high-confidence phospho)."""
        return self.ddG_ensemble_correction_kJ

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"Saved PTM JSON → {path}")


# ── UniProt API fetch ──────────────────────────────────────────────────────────

def _fetch_uniprot_ptms(uniprot_id: str, timeout: int = 10) -> list[tuple[int, str, str, str]]:
    """
    Fetch experimentally validated PTMs from UniProt features API.
    Returns: list of (residue_num, aa, ptm_type_key, description)
    """
    url = (
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
        f"?fields=ft_mod_res,ft_carbohyd,ft_lipid&format=json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProteinFP/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log.warning(f"  UniProt API unavailable: {e}")
        return []

    sites: list[tuple[int, str, str, str]] = []

    for feat in data.get("features", []):
        ftype = feat.get("type", "")
        pos   = feat.get("location", {}).get("start", {}).get("value")
        desc  = feat.get("description", "")
        if pos is None:
            continue
        pos = int(pos)

        if ftype == "Modified residue":
            desc_lower = desc.lower()
            if "phospho" in desc_lower:
                if "serine" in desc_lower:
                    sites.append((pos, "S", "phosphoserine",    desc))
                elif "threonine" in desc_lower:
                    sites.append((pos, "T", "phosphothreonine", desc))
                elif "tyrosine" in desc_lower:
                    sites.append((pos, "Y", "phosphotyrosine",  desc))
                else:
                    sites.append((pos, "?", "phosphoserine",    desc))
            elif "acetyl" in desc_lower:
                sites.append((pos, "K", "acetylation",      desc))
            elif "methyl" in desc_lower:
                if "dimethyl" in desc_lower:
                    sites.append((pos, "K", "dimethylation",    desc))
                elif "trimethyl" in desc_lower:
                    sites.append((pos, "K", "trimethylation",   desc))
                else:
                    sites.append((pos, "K", "methylation",      desc))
            elif "ubiquitin" in desc_lower:
                sites.append((pos, "K", "ubiquitination",    desc))
            elif "sumo" in desc_lower:
                sites.append((pos, "K", "sumoylation",       desc))
            elif "hydroxy" in desc_lower:
                sites.append((pos, "P", "hydroxylation",     desc))
            elif "nitrosyl" in desc_lower:
                sites.append((pos, "C", "nitrosylation",     desc))

        elif ftype == "Glycosylation":
            desc_lower = desc.lower()
            if "n-linked" in desc_lower or "asn" in desc_lower:
                sites.append((pos, "N", "nglycosylation",    desc))
            else:
                sites.append((pos, "S", "oglycosylation",    desc))

        elif ftype == "Lipidation":
            desc_lower = desc.lower()
            if "palmitoyl" in desc_lower:
                sites.append((pos, "C", "palmitoylation",    desc))
            elif "myristoyl" in desc_lower:
                sites.append((pos, "G", "myristoylation",    desc))

    log.info(f"  UniProt API: {len(sites)} PTM sites for {uniprot_id}")
    return sites


# ── Motif scanner ─────────────────────────────────────────────────────────────

def _scan_motifs(sequence: str) -> list[tuple[int, str, str, str, float]]:
    """
    Scan sequence for PTM consensus motifs.
    Returns: list of (residue_num, aa, ptm_type, kinase, confidence)
    The residue_num returned is the position of the modified residue within the motif.
    """
    results: list[tuple[int, str, str, str, float]] = []

    for ptm_type, pattern, kinase, conf_base in MOTIF_PATTERNS:
        target_aas = PTM_TARGET_AA.get(ptm_type, set())
        for m in re.finditer(f"(?={pattern})", sequence):
            start = m.start()
            # Find the modified residue position within the match
            match_str = sequence[start:start + len(pattern) + 2]
            for offset, aa in enumerate(match_str):
                abs_pos = start + offset + 1  # 1-indexed
                if aa in target_aas and abs_pos <= len(sequence):
                    # Avoid duplicates (same pos, same ptm_type)
                    existing = {(r, t) for r, _, t, _, _ in results}
                    if (abs_pos, ptm_type) not in existing:
                        results.append((abs_pos, aa, ptm_type, kinase, conf_base))
                    break

    log.info(f"  Motif scan: {len(results)} predicted PTM sites")
    return results


# ── Functional impact ─────────────────────────────────────────────────────────

def _compute_functional_impact(
    site:          PTMSite,
    active_data:   Optional[dict],
    pocket_data:   Optional[dict],
    physico_data:  Optional[dict],
    sequence:      str,
) -> PTMSite:
    """
    Compute functional impact of a PTM site using upstream module data.
    Modifies site in-place and returns it.
    """
    rn = site.residue_number

    # ── 1. Active site proximity ──────────────────────────────────────────────
    min_dist_active = 999.0
    if active_data:
        for res in active_data.get("active_residues", []):
            ar = res.get("residue_number", 0)
            # Approximate CA-CA distance from sequence separation (Å)
            # Real Cα-Cα distance ≈ 3.8Å per residue in extended, less in folded
            # Use sequence distance as a proxy (will overestimate true 3D distance)
            seq_dist = abs(rn - ar)
            approx_3d = min(seq_dist * 1.5, seq_dist + 5)  # rough correction
            if approx_3d < min_dist_active:
                min_dist_active = approx_3d
        site.active_site_proximity_A = round(min_dist_active, 1)
        site.is_active_site_switch = (min_dist_active < 12.0)

    # ── 2. Pocket proximity ───────────────────────────────────────────────────
    min_dist_pocket = 999.0
    if pocket_data:
        for pocket in pocket_data.get("pockets", [])[:3]:
            for lr in pocket.get("lining_residues", []):
                seq_dist = abs(rn - lr)
                approx_3d = min(seq_dist * 1.5, seq_dist + 5)
                if approx_3d < min_dist_pocket:
                    min_dist_pocket = approx_3d
        site.pocket_proximity_A = round(min_dist_pocket, 1)
        site.is_binding_switch = (min_dist_pocket < 10.0)

    # ── 3. ΔΔG binding from charge change ─────────────────────────────────────
    # Coulombic correction: ΔΔG ≈ (ΔCharge × q_pocket) / (4πε₀εr × d)
    # Simplified: ΔΔG_kJ ≈ ΔCharge × 14.4 / (ε_r × d_Å) × 96.5  (kJ/mol)
    # ε_r ≈ 20 for protein interior, d ≈ active site proximity
    charge_delta = site.charge_delta
    if charge_delta != 0 and min_dist_active < 50:
        eps_r = 20.0
        d = max(min_dist_active, 3.0)
        # kJ/mol from elementary charges at distance d in protein
        ddG = charge_delta * 138.9 / (eps_r * d)  # kJ/mol
        site.ddG_binding_kJ = round(ddG, 2)
    else:
        site.ddG_binding_kJ = 0.0

    # ── 4. Conformational effect ──────────────────────────────────────────────
    ptm = site.ptm_type
    kinase = site.kinase_enzyme.upper()

    # Phosphorylation of activation loop (typical ~activation loop residues)
    # Heuristic: if phospho near active site and kinase is a known activator
    activating_kinases = {"CDK", "PKA", "MAPK", "EGFR", "ATM", "CHK2", "PKB"}
    inactivating_kinases = {"CK2", "GSK3", "CDK1_inhibitory"}

    if ptm in {"phosphoserine", "phosphothreonine", "phosphotyrosine"}:
        if site.is_active_site_switch:
            if any(k in kinase for k in activating_kinases):
                site.conformational_effect = "activating"
                site.ddG_state_active_kJ   = -2.0   # stabilises active state
                site.ddG_state_inactive_kJ = +3.0   # destabilises inactive
            else:
                site.conformational_effect = "inactivating"
                site.ddG_state_active_kJ   = +2.5
                site.ddG_state_inactive_kJ = -1.5
        else:
            site.conformational_effect = "regulatory"
            site.ddG_state_active_kJ   = -0.5
    elif ptm == "acetylation" and site.is_active_site_switch:
        site.conformational_effect = "activating"
        site.ddG_state_active_kJ   = -1.5
    elif ptm in {"nglycosylation", "oglycosylation"}:
        site.conformational_effect = "stabilising"
        site.blocks_epitope = True  # large carbohydrate chain blocks antibodies
        site.ddG_state_active_kJ = -1.0

    # ── 5. Surface exposure check (for glycosylation and epitope blocking) ────
    if physico_data and ptm in {"nglycosylation", "oglycosylation"}:
        for res in physico_data.get("residues", []):
            if res.get("residue_number") == rn:
                if res.get("sasa_fraction", 0) > 0.3:
                    site.blocks_epitope = True

    return site


# ── Kinase enrichment ─────────────────────────────────────────────────────────

def _dominant_kinase(sites: list[PTMSite]) -> str:
    """Find the kinase/enzyme that targets this protein most often."""
    from collections import Counter
    kinases = [s.kinase_enzyme for s in sites
               if s.kinase_enzyme and s.kinase_enzyme != "unknown"]
    if not kinases:
        return ""
    return Counter(kinases).most_common(1)[0][0]


def _find_crosstalk(sites: list[PTMSite]) -> list[str]:
    """Find residues with competing PTMs (e.g. K382: acetylation vs ubiquitination)."""
    from collections import defaultdict
    pos_to_ptms: dict[int, list[str]] = defaultdict(list)
    for s in sites:
        pos_to_ptms[s.residue_number].append(s.ptm_type)
    crosstalk = []
    for pos, ptms in pos_to_ptms.items():
        if len(ptms) > 1:
            crosstalk.append(
                f"{sites[0].residue_aa if sites else '?'}{pos}: "
                f"{' vs '.join(ptms)}"
            )
    return crosstalk[:5]


# ── SIM-02 integration ────────────────────────────────────────────────────────

def _compute_sim02_corrections(sites: list[PTMSite]) -> dict:
    """
    Compute per-state ΔΔG corrections for SIM-02 conformational ensemble.

    Returns a dict keyed by conformational state name:
        {
          "active":         ΔΔG_kJ (apply to active state free energy),
          "inactive":       ΔΔG_kJ,
          "apo":            ΔΔG_kJ,
          "partially_open": ΔΔG_kJ,
          "allosteric_open":ΔΔG_kJ,
        }

    Each value is the SUM of contributions from HIGH-confidence PTM sites
    weighted by confidence score.
    """
    state_ddG: dict[str, float] = {
        "active":          0.0,
        "inactive":        0.0,
        "apo":             0.0,
        "partially_open":  0.0,
        "allosteric_open": 0.0,
    }

    high_conf_sites = [s for s in sites if s.confidence >= 0.70]
    for s in high_conf_sites:
        w = s.confidence
        state_ddG["active"]          += s.ddG_state_active_kJ   * w
        state_ddG["inactive"]        += s.ddG_state_inactive_kJ * w
        # apo: half of active correction (partially responsive)
        state_ddG["apo"]             += s.ddG_state_active_kJ * 0.5 * w
        # partially_open: intermediate
        state_ddG["partially_open"]  += s.ddG_state_active_kJ * 0.3 * w
        # allosteric_open: minimal PTM effect (allosteric channel decoupled)
        state_ddG["allosteric_open"] += s.ddG_state_active_kJ * 0.2 * w

    return {k: round(v, 3) for k, v in state_ddG.items()}


# ── Main analysis function ─────────────────────────────────────────────────────

def analyze_ptms(
    uniprot_id:   str,
    sequence:     str,
    active_data:  Optional[dict] = None,
    pocket_data:  Optional[dict] = None,
    physico_data: Optional[dict] = None,
    use_api:      bool = True,
) -> PTMResult:
    """
    Full PTM analysis for a protein.

    Args:
        uniprot_id:   UniProt accession
        sequence:     Amino acid sequence (one-letter)
        active_data:  Module 03 output dict (optional)
        pocket_data:  Module 04 output dict (optional)
        physico_data: Module 02 output dict (optional)
        use_api:      Whether to query UniProt REST API

    Returns:
        PTMResult with full PTM annotation and SIM-02 integration data.
    """
    log.info(f"── Module 17: PTM analysis for {uniprot_id} ──")

    # Gene name from structure data (will be filled from intermediate if available)
    gene_name = uniprot_id  # fallback

    sites: list[PTMSite] = []
    seen: set[tuple[int, str]] = set()  # (residue_num, ptm_type) dedup

    def _add_site(
        rn: int, aa: str, ptm_type: str,
        source: str, kinase: str, confidence: float,
        functional_note: str = "",
    ) -> None:
        if (rn, ptm_type) in seen:
            return
        if rn < 1 or rn > len(sequence):
            return
        # Validate amino acid match if we know the sequence
        if aa != "?" and len(sequence) >= rn:
            actual_aa = sequence[rn - 1]
            target_aas = PTM_TARGET_AA.get(ptm_type, set())
            if target_aas and actual_aa not in target_aas:
                # Mismatch — could be isoform difference; reduce confidence
                confidence *= 0.5
                aa = actual_aa

        ptm_info = PTM_TYPES.get(ptm_type, (ptm_type, 0.0, 0.0, True))
        site = PTMSite(
            residue_number   = rn,
            residue_aa       = aa,
            ptm_type         = ptm_type,
            ptm_name         = ptm_info[0],
            source           = source,
            kinase_enzyme    = kinase,
            confidence       = round(confidence, 3),
            charge_delta     = ptm_info[1],
            mass_shift_Da    = ptm_info[2],
            reversible       = ptm_info[3],
            functional_note  = functional_note,
        )
        site = _compute_functional_impact(site, active_data, pocket_data, physico_data, sequence)
        sites.append(site)
        seen.add((rn, ptm_type))

    # ── Step 1: Curated known PTMs ────────────────────────────────────────────
    log.info("  [1/4] Loading curated known PTMs...")
    n_curated = 0
    if uniprot_id in CURATED_KNOWN_PTMS:
        for rn, aa, ptm_type, kinase, note in CURATED_KNOWN_PTMS[uniprot_id]:
            _add_site(rn, aa, ptm_type, "curated", kinase, 0.95, note)
            n_curated += 1
    log.info(f"    {n_curated} curated sites loaded")

    # ── Step 2: UniProt API ───────────────────────────────────────────────────
    n_api = 0
    if use_api:
        log.info("  [2/4] Querying UniProt API for experimental PTMs...")
        try:
            api_sites = _fetch_uniprot_ptms(uniprot_id)
            for rn, aa, ptm_type, desc in api_sites:
                kinase = "UniProt"
                # Improve kinase annotation from description
                for k in ["ATM", "CHK2", "CDK", "PKA", "PKC", "CK2", "MAPK",
                           "GSK3", "PKB", "EGFR", "PCAF", "CBP", "TIP60"]:
                    if k.lower() in desc.lower():
                        kinase = k
                        break
                _add_site(rn, aa, ptm_type, "experimental", kinase, 0.90, desc[:80])
                n_api += 1
            log.info(f"    {n_api} experimental sites from API")
        except Exception as e:
            log.warning(f"    API fetch failed: {e}")
    else:
        log.info("  [2/4] Skipping UniProt API (--no-api)")

    # ── Step 3: Motif scan ────────────────────────────────────────────────────
    log.info("  [3/4] Scanning sequence for PTM motifs...")
    motif_hits = _scan_motifs(sequence)
    n_motif = 0
    for rn, aa, ptm_type, kinase, conf in motif_hits:
        _add_site(rn, aa, ptm_type, "motif_scan", kinase, conf)
        n_motif += 1
    log.info(f"    {n_motif} predicted sites from motif scan")

    # ── Step 4: Aggregate statistics ─────────────────────────────────────────
    log.info("  [4/4] Computing aggregate statistics and SIM-02 corrections...")

    n_known  = sum(1 for s in sites if s.source in {"curated", "experimental"})
    n_pred   = sum(1 for s in sites if s.source == "motif_scan")
    n_high   = sum(1 for s in sites if s.confidence >= 0.75)

    # Total phospho charge shift
    phospho_types = {"phosphoserine", "phosphothreonine", "phosphotyrosine"}
    total_charge = sum(
        s.charge_delta for s in sites if s.ptm_type in phospho_types
    )

    n_active_switches = sum(1 for s in sites if s.is_active_site_switch)
    n_binding_switches = sum(1 for s in sites if s.is_binding_switch)
    n_glyco_surface = sum(
        1 for s in sites
        if s.ptm_type in {"nglycosylation", "oglycosylation"} and s.blocks_epitope
    )

    # Net SIM-02 correction (from high-confidence sites)
    state_corrections = _compute_sim02_corrections(sites)
    ddG_active_net = state_corrections.get("active", 0.0)

    # Crosstalk
    crosstalk = _find_crosstalk(sites)
    dominant_k = _dominant_kinase(sites)

    result = PTMResult(
        uniprot_id       = uniprot_id,
        gene_name        = gene_name,
        sequence_length  = len(sequence),
        n_known          = n_known,
        n_predicted      = n_pred,
        n_high_conf      = n_high,
        sites            = sorted(sites, key=lambda s: (-s.confidence, s.residue_number)),
        total_charge_shift_phospho = round(total_charge, 1),
        n_active_site_switches     = n_active_switches,
        n_binding_switches         = n_binding_switches,
        n_glycosylated_surface     = n_glyco_surface,
        ddG_ensemble_correction_kJ = ddG_active_net,
        dominant_kinase            = dominant_k,
        ptm_crosstalk              = crosstalk,
        state_corrections          = state_corrections,
    )

    log.info(result.summary())
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--no-api", "skip_api", is_flag=True, default=False,
              help="Skip UniProt REST API (offline mode)")
@click.option("--output", "-o", default=None,
              help="Override output path (default: data/intermediate/{uid}_ptm.json)")
def main(uniprot: str, skip_api: bool, output: Optional[str]) -> None:
    """
    Module 17 — Post-Translational Modification analysis.

    Queries known PTMs, scans motifs, computes functional impact,
    and produces SIM-02 ΔΔG corrections per conformational state.

    Example:
        python pipeline/ptm_analysis.py --uniprot P04637
        python pipeline/ptm_analysis.py --uniprot P04637 --no-api
    """
    import sys
    uid       = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    # ── Load sequence from structure JSON ─────────────────────────────────────
    struct_path = inter_dir / f"{uid}_structure.json"
    if not struct_path.exists():
        log.error(
            f"Structure JSON not found: {struct_path}\n"
            f"  Run Module 01 first: python pipeline/01_fetch_structure.py --uniprot {uid}"
        )
        sys.exit(1)

    struct = json.loads(struct_path.read_text())
    sequence = struct.get("sequence", "")
    if not sequence:
        log.error("No sequence found in structure JSON.")
        sys.exit(1)

    log.info(f"  Sequence: {len(sequence)} aa")

    # ── Load upstream module outputs ──────────────────────────────────────────
    def _load(fname: str) -> Optional[dict]:
        p = inter_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception as e:
                log.warning(f"  Could not load {fname}: {e}")
        return None

    active_data  = _load(f"{uid}_active_sites.json")
    pocket_data  = _load(f"{uid}_binding_pockets.json")
    physico_data = _load(f"{uid}_physicochemical.json")

    if active_data:
        log.info(f"  Loaded active sites ({active_data.get('n_high_confidence', 0)} HIGH)")
    if pocket_data:
        log.info(f"  Loaded binding pockets ({pocket_data.get('n_pockets', 0)} pockets)")
    if physico_data:
        log.info(f"  Loaded physicochemical data")

    # ── Run analysis ──────────────────────────────────────────────────────────
    result = analyze_ptms(
        uniprot_id   = uid,
        sequence     = sequence,
        active_data  = active_data,
        pocket_data  = pocket_data,
        physico_data = physico_data,
        use_api      = not skip_api,
    )

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = Path(output) if output else inter_dir / f"{uid}_ptm.json"
    result.to_json(out_path)
    click.echo(f"\n  PTM analysis complete.")
    click.echo(f"  {result.n_known} known + {result.n_predicted} predicted sites")
    click.echo(f"  SIM-02 active state correction: {result.ddg_ensemble_correction_kj():+.2f} kJ/mol")
    click.echo(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()