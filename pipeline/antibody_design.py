"""
pipeline/antibody_design.py
────────────────────────────
Module 16 — De Novo Antibody Design

Evolutionary CDR loop designer that generates therapeutic antibody sequences
targeting a specific epitope on the antigen surface.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS IS FUNDAMENTALLY DIFFERENT FROM SMALL-MOLECULE DE NOVO DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Small molecules (denovo_design.py):
  - 200–600 Da, bind buried hydrophobic pockets
  - Fitness = Vina docking score (kcal/mol) in a pocket
  - Evolved by fragment growing / SMILES mutation
  - ADMET = Lipinski rules, hERG, CYP450

Antibodies (this module):
  - ~150 kDa, bind SURFACE-EXPOSED epitopes (not pockets)
  - Six hypervariable CDR loops (H1/H2/H3, L1/L2/L3) contact the antigen
  - CDR H3 is the primary determinant of specificity (~60% of contacts)
  - Fitness = predicted affinity (ΔG) + developability score + specificity
  - Evolved by CDR sequence mutation (amino acid substitutions, loop length)
  - "ADMET equivalent" = immunogenicity, aggregation, stability, expression

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIOLOGICAL FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EPITOPE SELECTION
   Sources (priority order, mirrors denovo_design.py site selection):
   a) Active site / functional surface from Module 03 — functional blocking
   b) PPI interface residues from Module 12 — disrupt a known interaction
   c) Most exposed surface patch from Module 02 — accessibility
   d) Allosteric surface from Module 05 — conformational modulation

   Epitopes must be SURFACE-EXPOSED (SASA > threshold). We never target
   buried residues — antibodies cannot access them.

2. ANTIBODY FRAMEWORK
   We use a human IgG1 framework (most common therapeutic format):
   - Heavy chain: VH domain (113 aa) with CDR-H1, CDR-H2, CDR-H3
   - Light chain: Vκ domain (107 aa) with CDR-L1, CDR-L2, CDR-L3
   - Framework regions (FR) are fixed — only CDR loops evolve
   - Kabat numbering used throughout

   CDR length constraints (Kabat):
   - CDR-H1:  5 residues  (Kabat pos 31–35)
   - CDR-H2: 17 residues  (Kabat pos 50–65, highly variable)
   - CDR-H3:  3–25 residues (Kabat pos 95–102, most variable)
   - CDR-L1: 10–17 residues (Kabat pos 24–34)
   - CDR-L2:  7 residues  (Kabat pos 50–56)
   - CDR-L3:  7–11 residues (Kabat pos 89–97)

3. FITNESS FUNCTION (multi-objective, no Vina)
   Antibodies cannot be docked with Vina (they are proteins, not small
   molecules). Instead we use:

   a) SEQUENCE-BASED AFFINITY PROXY
      - Physicochemical complementarity to the epitope
      - Charge complementarity (opposite charges attract)
      - Hydrophobic burial estimation (buried surface area proxy)
      - H-bond capacity between CDR residues and epitope
      - Shape complementarity index (CDR loop composition vs epitope)

   b) ROSETTA-STYLE ENERGY (if PyRosetta available, otherwise analytical)
      - van der Waals packing (buried hydrophobic contacts)
      - Electrostatic interactions
      - Solvation penalty
      - Backbone torsion validity

   c) DEVELOPABILITY SCORE (critical for therapeutic antibodies)
      Combines 7 empirically validated metrics:
      1. Hydrophobicity of CDR-H3 (high → aggregation prone)
      2. Net charge of VH domain (outside ±5 → poor expression)
      3. Isoelectric point (pI 6–9 optimal for serum stability)
      4. CDR-H3 length (optimal 12–16 aa; very long loops aggregate)
      5. Sequence liability motifs (deamidation NG/NS, oxidation W/M,
         isomerization DG/DS, glycosylation NxS/T)
      6. Human germline similarity (< 85% → immunogenicity risk)
      7. Predicted aggregation propensity (TANGO-like score)

   d) SPECIFICITY SCORE
      - Absence of polyreactivity motifs (basic charged patches)
      - Low predicted non-specific binding (sticky patches)

4. MUTATION OPERATORS
   All biologically grounded:
   - Single amino acid substitution (most common in affinity maturation)
   - CDR-H3 length extension/contraction (adds/removes residues at loop tip)
   - Conservative substitution (within biochemical group, e.g. K→R, D→E)
   - Hot-spot focused mutation (positions 31, 52, 96, 100 are highest impact)
   - Charge swap at CDR positions complementary to epitope
   - Somatic hypermutation-like: bias mutations toward positions 31–35,
     50–65, 95–102 (true CDR hot spots from SHM data)

5. EVOLUTION STRATEGY
   - Population: 30 antibody sequences per generation
   - Selection: fitness-proportionate with elitism (top 5 always survive)
   - Diversity: novelty archive based on CDR sequence similarity
   - Surrogate: ridge regression on sequence features (same as denovo)
   - Convergence: stagnation detection + CDR loop restart injection
   - Final validation: top 8 sequences rescored with higher-fidelity model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (standalone):
    python pipeline/antibody_design.py --uniprot P04637
    python pipeline/antibody_design.py --uniprot P04637 --generations 40
    python pipeline/antibody_design.py --uniprot P04637 --epitope-mode ppi

Usage (from orchestrator):
    from pipeline.antibody_design import run_antibody_design
    result = run_antibody_design("P04637", active_data, physico_data, ppi_data)

Output:
    data/intermediate/{uniprot}_antibody.json
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from utils.config import cfg, get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MAX_GENERATIONS      = 50
POP_SIZE             = 30
ELITISM              = 5
PARALLEL_WORKERS     = min(8, os.cpu_count() or 2)

SURROGATE_MIN_DATA   = 20
SURROGATE_CANDIDATES = 50

NOVELTY_K            = 8
NOVELTY_ARCHIVE_MAX  = 400
NOVELTY_ADD_THRESH   = 0.15   # CDR sequence novelty threshold

STAGNATION_LIMIT     = 4
STAGNATION_HARD      = 8
DIVERSITY_MIN        = 0.20

TOP_FOR_FINAL        = 8

# Fitness weights (start → end over generations)
W_AFFINITY_START = 0.30;  W_AFFINITY_END = 0.65
W_DEVELOP_START  = 0.45;  W_DEVELOP_END  = 0.20
W_NOVELTY_START  = 0.25;  W_NOVELTY_END  = 0.15

# Developability thresholds
MAX_CDR_H3_LENGTH    = 22    # >22 aa → severe aggregation risk
MIN_CDR_H3_LENGTH    = 6     # <6 aa → weak binding
OPTIMAL_H3_MIN       = 10
OPTIMAL_H3_MAX       = 16
MAX_HYDROPHOBIC_PATCH = 5    # consecutive hydrophobic residues in CDR

NET_CHARGE_MIN       = -5.0  # VH domain net charge range
NET_CHARGE_MAX       = +5.0
OPTIMAL_PI_MIN       = 6.0
OPTIMAL_PI_MAX       = 9.0

HUMAN_GERMLINE_SIM_MIN = 0.80  # fraction — below this = immunogenic

# Surface exposure threshold for epitope residues (Å²)
EPITOPE_SASA_MIN     = 25.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTIBODY BIOLOGY CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# 20 canonical amino acids
AAs = list("ACDEFGHIKLMNPQRSTVWY")

# Biochemical groups for conservative substitution
BIOCHEM_GROUPS = [
    list("GAVLIP"),    # aliphatic/nonpolar
    list("FYW"),       # aromatic
    list("ST"),        # small hydroxyl
    list("NQ"),        # amide
    list("DE"),        # acidic
    list("KRH"),       # basic
    list("CM"),        # sulfur-containing
]
# Reverse map: aa → group members
AA_GROUP: Dict[str, List[str]] = {}
for grp in BIOCHEM_GROUPS:
    for aa in grp:
        AA_GROUP[aa] = grp

# Hydrophobicity (Kyte-Doolittle)
HYDRO: Dict[str, float] = {
    "A": 1.8, "R":-4.5, "N":-3.5, "D":-3.5, "C": 2.5,
    "Q":-3.5, "E":-3.5, "G":-0.4, "H":-3.2, "I": 4.5,
    "L": 3.8, "K":-3.9, "M": 1.9, "F": 2.8, "P":-1.6,
    "S":-0.8, "T":-0.7, "W":-0.9, "Y":-1.3, "V": 4.2,
}

# Charge at pH 7.4
CHARGE_PH7: Dict[str, float] = {
    "A": 0, "R":+1, "N": 0, "D":-1, "C": 0,
    "Q": 0, "E":-1, "G": 0, "H":+0.1,"I": 0,
    "L": 0, "K":+1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# pKa for pI calculation
PKA_VALUES = {
    "D": 3.9, "E": 4.1, "H": 6.0,
    "C": 8.3, "Y": 10.1, "K": 10.5,
    "R": 12.5,
}
N_TERM_PKA = 8.0
C_TERM_PKA = 3.1

# Sequence liability motifs (regex → description)
LIABILITY_MOTIFS = {
    r"N[^P][ST]":  "N-glycosylation (NxS/T)",
    r"NG":         "deamidation (NG)",
    r"NS":         "deamidation (NS)",
    r"DG":         "Asp isomerization (DG)",
    r"DS":         "Asp isomerization (DS)",
    r"[MW]":       "oxidation-prone (M/W)",
    r"CP":         "cis-Pro near Cys",
    r"[KR]{3,}":   "basic patch (polyreactivity risk)",
}

# Human IgG1 framework sequences (VH and Vκ)
# Source: IMGT germline IGHV1-69 (heavy) and IGKV1-39 (light)
# These are the most common human germlines in therapeutic antibodies
# CDR positions marked as XXXXX (will be replaced by evolved sequences)
VH_FRAMEWORK = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"   # FR1 (1-30)
    "{CDR_H1}"                          # CDR-H1 (31-35, 5 aa)
    "WVRQAPGQGLEWMG"                    # FR2 (36-49)
    "{CDR_H2}"                          # CDR-H2 (50-65, up to 17 aa)
    "RVTITADESTSTAYMELSSLRSEDTAVYYCAR"  # FR3 (66-94)
    "{CDR_H3}"                          # CDR-H3 (95-102, variable)
    "WGQGTLVTVSS"                       # FR4 (103-113)
)

VL_FRAMEWORK = (
    "DIQMTQSPSSLSASVGDRVTITC"           # FR1 (1-23)
    "{CDR_L1}"                          # CDR-L1 (24-34, ~11 aa)
    "WYQQKPGKAPKLLIY"                   # FR2 (35-49)
    "{CDR_L2}"                          # CDR-L2 (50-56, 7 aa)
    "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC"  # FR3 (57-88)
    "{CDR_L3}"                          # CDR-L3 (89-97, ~9 aa)
    "FGQGTKVEIK"                        # FR4 (98-107)
)

# Canonical CDR length ranges (from Kabat/IMGT analysis of therapeutic abs)
CDR_LENGTHS = {
    "CDR_H1": (5,  5),    # nearly fixed at 5
    "CDR_H2": (6, 17),    # 6-17, mode ~17
    "CDR_H3": (6, 22),    # most variable, mode ~12
    "CDR_L1": (10, 17),   # 10-17, mode ~11
    "CDR_L2": (7,  7),    # nearly fixed at 7
    "CDR_L3": (7, 11),    # 7-11, mode ~9
}

# SHM hot spot positions within each CDR (1-indexed from CDR start)
# Based on codon usage bias in germinal center B cells
SHM_HOTSPOTS = {
    "CDR_H1": [1, 2, 3, 4, 5],        # all positions hot
    "CDR_H2": [1, 3, 7, 14, 17],      # key specificity positions
    "CDR_H3": [1, 3, 5, -3, -1],      # N-terminus and C-terminus of loop
    "CDR_L1": [1, 4, 5, 8, 11],
    "CDR_L2": [1, 3, 7],
    "CDR_L3": [1, 3, 6, 9],
}

# Seed CDR sequences — derived from real therapeutic antibodies
# These are starting points for evolution, NOT used as-is
CDR_SEEDS = {
    "CDR_H1": [
        "GYTFT",   # trastuzumab-like
        "NYGMH",   # bevacizumab-like
        "SYAMS",   # adalimumab-like
        "GYSFN",   # generic
        "DYGVH",   # generic
    ],
    "CDR_H2": [
        "WINPNSGGTNYAQKFQG",   # trastuzumab-like
        "IIWYDGSKKYYVDSVKG",   # bevacizumab-like
        "VIWYDGSNKYYADSVKG",   # adalimumab-like
        "RIYPGDGDTNYNG",       # shorter variant
        "AISGSGGSTYYADSVKG",   # generic
    ],
    "CDR_H3": [
        "DNYGSSPY",            # short loop (8 aa)
        "RFPYYYYGMDV",         # medium loop (11 aa)
        "DRYYGNSGFAY",         # medium loop (11 aa)
        "SRWGGDGFYAMDY",       # long loop (13 aa)
        "GGLYRSGWYFDL",        # long loop (12 aa)
    ],
    "CDR_L1": [
        "RASQDVNTAVA",         # 11 aa
        "KASQSVDYDGDSYMN",     # 15 aa
        "RASESVDNYGISFMN",     # 15 aa
        "SGSSSNIGNNFVS",       # 13 aa
        "RASQSISSYLA",         # 11 aa
    ],
    "CDR_L2": [
        "AASSLQS",             # 7 aa (fixed length)
        "YASSLQS",
        "DASNRAT",
        "GASSRAT",
        "YTSSLHS",
    ],
    "CDR_L3": [
        "QQYSTVPWT",           # 9 aa
        "QQHYTTPPT",           # 9 aa
        "QQDYNLPWT",           # 9 aa
        "LQHNSYPWT",           # 9 aa
        "QQGNTLPWT",           # 9 aa
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AntibodySequence:
    """
    A complete antibody variable domain (VH + VL) defined by its 6 CDR loops.
    The framework regions are fixed human germline sequences.
    """
    CDR_H1: str = ""
    CDR_H2: str = ""
    CDR_H3: str = ""
    CDR_L1: str = ""
    CDR_L2: str = ""
    CDR_L3: str = ""

    def vh_sequence(self) -> str:
        """Full VH domain sequence."""
        return VH_FRAMEWORK.format(
            CDR_H1=self.CDR_H1,
            CDR_H2=self.CDR_H2,
            CDR_H3=self.CDR_H3,
        )

    def vl_sequence(self) -> str:
        """Full VL domain sequence."""
        return VL_FRAMEWORK.format(
            CDR_L1=self.CDR_L1,
            CDR_L2=self.CDR_L2,
            CDR_L3=self.CDR_L3,
        )

    def cdr_string(self) -> str:
        """Concatenated CDR sequence for comparison/hashing."""
        return f"{self.CDR_H1}|{self.CDR_H2}|{self.CDR_H3}|{self.CDR_L1}|{self.CDR_L2}|{self.CDR_L3}"

    def cdr_dict(self) -> Dict[str, str]:
        return {
            "CDR_H1": self.CDR_H1,
            "CDR_H2": self.CDR_H2,
            "CDR_H3": self.CDR_H3,
            "CDR_L1": self.CDR_L1,
            "CDR_L2": self.CDR_L2,
            "CDR_L3": self.CDR_L3,
        }

    def total_cdr_length(self) -> int:
        return sum(len(s) for s in [
            self.CDR_H1, self.CDR_H2, self.CDR_H3,
            self.CDR_L1, self.CDR_L2, self.CDR_L3,
        ])

    def is_valid(self) -> bool:
        """Check all CDRs are non-empty, correct amino acids, within length bounds."""
        for cdr_name, seq in self.cdr_dict().items():
            lo, hi = CDR_LENGTHS[cdr_name]
            if not (lo <= len(seq) <= hi):
                return False
            if not all(aa in AAs for aa in seq):
                return False
        return True

    def __hash__(self):
        return hash(self.cdr_string())

    def __eq__(self, other):
        return isinstance(other, AntibodySequence) and self.cdr_string() == other.cdr_string()


@dataclass
class EpitopeTarget:
    """The antigen surface region targeted by the antibody."""
    source:           str           # "active_site" / "ppi_interface" / "surface_patch" / "allosteric"
    residue_numbers:  List[int]
    residue_letters:  List[str]
    center_coords:    List[float]   # [x, y, z] geometric center
    mean_sasa:        float
    net_charge:       float
    mean_hydrophobicity: float
    n_aromatic:       int
    n_hbond_donors:   int
    n_hbond_acceptors: int
    surface_area:     float         # total SASA of epitope (Å²)
    description:      str           = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AntibodyCandidate:
    """A single evaluated antibody with all scores."""
    cdr_h1:             str
    cdr_h2:             str
    cdr_h3:             str
    cdr_l1:             str
    cdr_l2:             str
    cdr_l3:             str
    generation:         int
    affinity_score:     float     # predicted binding affinity (higher = better)
    developability:     float     # 0-1 (1 = excellent developability)
    specificity:        float     # 0-1 (1 = highly specific)
    novelty:            float     # 0-1 (1 = highly novel)
    fitness:            float     # composite fitness
    # Developability sub-scores
    net_charge_vh:      float
    pi_vh:              float
    cdr_h3_length:      int
    hydrophobic_score:  float     # lower = better
    liability_count:    int       # sequence liability motifs found
    germline_sim:       float     # human germline similarity (0-1)
    aggregation_score:  float     # TANGO-like (lower = better)
    # Predicted binding
    charge_comp:        float     # charge complementarity to epitope
    hydrophobic_burial: float     # estimated buried hydrophobic surface
    hbond_capacity:     int       # H-bonds predicted at interface
    shape_score:        float     # shape complementarity index
    origin:             str       # how it was generated

    def to_dict(self) -> dict:
        return asdict(self)

    def vh_sequence(self) -> str:
        return AntibodySequence(
            self.cdr_h1, self.cdr_h2, self.cdr_h3,
            self.cdr_l1, self.cdr_l2, self.cdr_l3,
        ).vh_sequence()

    def vl_sequence(self) -> str:
        return AntibodySequence(
            self.cdr_h1, self.cdr_h2, self.cdr_h3,
            self.cdr_l1, self.cdr_l2, self.cdr_l3,
        ).vl_sequence()


@dataclass
class AntibodyResult:
    """Full output of Module 16."""
    uniprot_id:         str
    target_gene:        str           = ""
    epitope:            Optional[EpitopeTarget] = None
    epitope_source:     str           = ""
    n_generations:      int           = 0
    n_evaluated:        int           = 0
    n_unique:           int           = 0
    top_candidates:     List[AntibodyCandidate] = field(default_factory=list)
    best_affinity:      float         = 0.0
    best_fitness:       float         = 0.0
    best_cdr_h3:        str           = ""
    generation_stats:   List[dict]    = field(default_factory=list)
    notes:              str           = ""

    def summary(self) -> str:
        lines = [
            f"\n{'═'*72}",
            f"  MODULE 16 — Antibody Design: {self.uniprot_id} ({self.target_gene})",
            f"{'═'*72}",
            f"  Epitope source   : {self.epitope_source}",
            f"  Generations run  : {self.n_generations}",
            f"  Sequences eval.  : {self.n_evaluated}",
            f"  Best affinity    : {self.best_affinity:.3f}",
            f"  Best CDR-H3      : {self.best_cdr_h3}",
            f"{'─'*72}",
            f"  TOP CANDIDATES:",
        ]
        for i, c in enumerate(self.top_candidates[:8], 1):
            dev_grade = "✓✓" if c.developability > 0.75 else ("✓" if c.developability > 0.5 else "✗")
            lines.append(
                f"  #{i:2d}: Aff={c.affinity_score:.3f}  "
                f"Dev={c.developability:.2f}{dev_grade}  "
                f"Spc={c.specificity:.2f}  "
                f"pI={c.pi_vh:.1f}  "
                f"Chg={c.net_charge_vh:+.1f}  "
                f"Liab={c.liability_count}  "
                f"H3={c.cdr_h3} ({len(c.cdr_h3)}aa)"
            )
        lines.append(f"{'═'*72}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.epitope:
            d["epitope"] = self.epitope.to_dict()
        return d

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# EPITOPE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_epitope(
    uniprot_id:     str,
    active_data:    Optional[dict],
    physico_data:   Optional[dict],
    ppi_data:       Optional[dict],
    allosteric_data: Optional[dict],
    epitope_mode:   str = "auto",
) -> Optional[EpitopeTarget]:
    """
    Select the best epitope for antibody targeting.

    Priority (mirrors denovo_design.py site selection logic):
    1. active / functional surface (mode="active" or "auto")
    2. PPI interface (mode="ppi" or "auto" fallback)
    3. Largest surface patch (mode="surface" or final fallback)
    4. Allosteric surface (mode="allosteric")

    Key constraint: epitope residues MUST be surface-exposed (SASA > threshold).
    Buried residues are inaccessible to antibodies.
    """
    log.info("  [Epitope] Selecting antibody target epitope...")

    # Build SASA lookup from physicochemical data
    sasa_lookup: Dict[int, float] = {}
    coord_lookup: Dict[int, List[float]] = {}
    if physico_data:
        for rec in physico_data.get("residues", []):
            rn = rec.get("residue_number", 0)
            sasa_lookup[rn]  = rec.get("sasa", 0.0)
            coord_lookup[rn] = rec.get("coords", [0, 0, 0])

    def _build_epitope_from_residues(
        res_numbers: List[int],
        res_letters:  List[str],
        source:       str,
        description:  str,
    ) -> Optional[EpitopeTarget]:
        """Filter to surface-exposed residues and compute epitope properties."""
        exposed_nums = []
        exposed_lets = []
        for rn, rl in zip(res_numbers, res_letters):
            sasa = sasa_lookup.get(rn, 100.0)  # default: assume exposed
            if sasa >= EPITOPE_SASA_MIN:
                exposed_nums.append(rn)
                exposed_lets.append(rl)

        if len(exposed_nums) < 4:
            log.debug(f"    {source}: only {len(exposed_nums)} exposed residues — skipping")
            return None

        # Compute epitope physicochemistry
        charges    = [CHARGE_PH7.get(aa, 0.0)  for aa in exposed_lets]
        hydros     = [HYDRO.get(aa, 0.0)        for aa in exposed_lets]
        aromatics  = sum(1 for aa in exposed_lets if aa in "FYW")
        hbd_map    = {"S":1,"T":1,"N":1,"Q":1,"K":1,"R":2,"H":1,"W":1,"Y":1}
        hba_map    = {"D":2,"E":2,"N":1,"Q":1,"S":1,"T":1,"H":1,"Y":1}
        hbd  = sum(hbd_map.get(aa, 0) for aa in exposed_lets)
        hba  = sum(hba_map.get(aa, 0) for aa in exposed_lets)

        coords = [coord_lookup.get(rn, [0, 0, 0]) for rn in exposed_nums
                  if coord_lookup.get(rn)]
        center = list(np.mean(coords, axis=0).tolist()) if coords else [0, 0, 0]
        total_sasa = sum(sasa_lookup.get(rn, 0) for rn in exposed_nums)

        log.info(f"    {source}: {len(exposed_nums)} exposed epitope residues  "
                 f"charge={sum(charges):+.1f}  hydro={np.mean(hydros):.2f}  "
                 f"SASA={total_sasa:.0f}Å²")

        return EpitopeTarget(
            source=source,
            residue_numbers=exposed_nums,
            residue_letters=exposed_lets,
            center_coords=[round(c, 3) for c in center],
            mean_sasa=round(total_sasa / len(exposed_nums), 1),
            net_charge=round(sum(charges), 2),
            mean_hydrophobicity=round(float(np.mean(hydros)), 3),
            n_aromatic=aromatics,
            n_hbond_donors=hbd,
            n_hbond_acceptors=hba,
            surface_area=round(total_sasa, 1),
            description=description,
        )

    # ── Priority 1: Active / functional surface ───────────────────────────────
    if epitope_mode in ("auto", "active") and active_data:
        residues = active_data.get("active_residues", [])
        # HIGH-confidence residues first, then all
        high = [r for r in residues if r.get("confidence") == "HIGH" and r.get("coords")]
        pool = high if high else [r for r in residues if r.get("coords")]
        if pool:
            nums  = [r["residue_number"] for r in pool]
            lets  = [r.get("one_letter", "A") for r in pool]
            epi   = _build_epitope_from_residues(
                nums, lets, "active_site",
                f"Active/functional surface ({len(pool)} residues, "
                f"{len(high)} high-confidence)"
            )
            if epi:
                log.info(f"  [Epitope] Selected: ACTIVE SITE ({len(epi.residue_numbers)} residues)")
                return epi

    # ── Priority 2: PPI interface ─────────────────────────────────────────────
    if epitope_mode in ("auto", "ppi") and ppi_data:
        partners = ppi_data.get("partners", [])
        # Use the highest-confidence partner's interface
        high_conf = [p for p in partners if p.get("combined_score", 0) >= 700]
        if high_conf:
            top_partner = high_conf[0]
            nums  = top_partner.get("interface_residues", [])
            lets  = top_partner.get("interface_letters", [])
            if nums:
                epi = _build_epitope_from_residues(
                    nums, lets, "ppi_interface",
                    f"PPI interface with {top_partner.get('partner_name','?')} "
                    f"(score={top_partner.get('combined_score',0)})"
                )
                if epi:
                    log.info(f"  [Epitope] Selected: PPI INTERFACE with "
                             f"{top_partner.get('partner_name','?')} "
                             f"({len(epi.residue_numbers)} residues)")
                    return epi

    # ── Priority 3: Largest surface-exposed hydrophobic patch ─────────────────
    if epitope_mode in ("auto", "surface") and physico_data:
        patches = (physico_data.get("hydrophobic_patches", []) +
                   physico_data.get("positive_patches", []) +
                   physico_data.get("negative_patches", []))
        if patches:
            # Sort by total SASA (largest exposed area first)
            patches_sorted = sorted(patches, key=lambda p: p.get("total_sasa", 0), reverse=True)
            top_patch = patches_sorted[0]
            nums = top_patch.get("residue_numbers", [])
            # Build letter list from physico data
            res_map = {r["residue_number"]: r.get("one_letter", "A")
                       for r in physico_data.get("residues", [])}
            lets = [res_map.get(n, "A") for n in nums]
            if nums:
                epi = _build_epitope_from_residues(
                    nums, lets, "surface_patch",
                    f"Largest surface patch ({top_patch.get('patch_type','?')}, "
                    f"SASA={top_patch.get('total_sasa',0):.0f}Å²)"
                )
                if epi:
                    log.info(f"  [Epitope] Selected: SURFACE PATCH "
                             f"({len(epi.residue_numbers)} residues, "
                             f"SASA={epi.surface_area:.0f}Å²)")
                    return epi

    # ── Priority 4: Allosteric surface ───────────────────────────────────────
    if epitope_mode in ("auto", "allosteric") and allosteric_data:
        sites = allosteric_data.get("allosteric_sites", [])
        if sites:
            site = sites[0]
            residues = site.get("residues", [])
            if residues:
                nums = [r.get("residue_number", 0) for r in residues if isinstance(r, dict)]
                lets = [r.get("one_letter", "A") for r in residues if isinstance(r, dict)]
                if not nums:
                    nums = [int(r) for r in residues if isinstance(r, (int, float))]
                    lets = ["A"] * len(nums)
                epi = _build_epitope_from_residues(
                    nums, lets, "allosteric",
                    f"Allosteric site {site.get('site_id','A1')}"
                )
                if epi:
                    log.info(f"  [Epitope] Selected: ALLOSTERIC SITE "
                             f"({len(epi.residue_numbers)} residues)")
                    return epi

    log.warning("  [Epitope] No suitable epitope found. Using generic surface targeting.")
    # Minimal fallback — create a generic epitope marker so evolution can proceed
    return EpitopeTarget(
        source="generic",
        residue_numbers=[],
        residue_letters=[],
        center_coords=[0.0, 0.0, 0.0],
        mean_sasa=100.0,
        net_charge=0.0,
        mean_hydrophobicity=0.0,
        n_aromatic=0,
        n_hbond_donors=5,
        n_hbond_acceptors=5,
        surface_area=1000.0,
        description="Generic surface (no epitope data available)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _random_cdr(cdr_name: str, rng: random.Random) -> str:
    """Generate a random CDR sequence within valid length bounds."""
    lo, hi = CDR_LENGTHS[cdr_name]
    length = rng.randint(lo, hi)
    return "".join(rng.choices(AAs, k=length))


def _seed_cdr(cdr_name: str, rng: random.Random) -> str:
    """Pick a random seed CDR from the therapeutic antibody library."""
    seeds = CDR_SEEDS.get(cdr_name, [])
    if seeds:
        base = rng.choice(seeds)
        # Light random perturbation (0-2 mutations from the seed)
        n_mut = rng.randint(0, min(2, len(base)))
        seq   = list(base)
        for _ in range(n_mut):
            pos = rng.randrange(len(seq))
            seq[pos] = rng.choice(AAs)
        return "".join(seq)
    return _random_cdr(cdr_name, rng)


def build_initial_population(
    size: int,
    epitope: Optional[EpitopeTarget],
    rng: random.Random,
) -> List[AntibodySequence]:
    """
    Build starting population biased toward epitope complementarity.

    Biases:
    - If epitope is charged positive → CDR-H3 enriched in acidic residues
    - If epitope is hydrophobic → CDR-H3 enriched in hydrophobic residues
    - If epitope has many H-bond donors → CDR-H3 enriched in acceptors
    Half the population from seeds, half random.
    """
    population: List[AntibodySequence] = []
    seen: set = set()
    attempts = 0

    # Compute complementarity bias
    charge_bias: List[str] = []
    hydro_bias: List[str]  = []

    if epitope:
        # Charge complementarity: if epitope is positive, CDR should be negative
        if epitope.net_charge > 1.0:
            charge_bias = list("DEED" * 3)   # acidic bias
        elif epitope.net_charge < -1.0:
            charge_bias = list("KRKR" * 3)   # basic bias

        # Hydrophobic complementarity
        if epitope.mean_hydrophobicity > 1.0:
            hydro_bias = list("VILMF" * 3)
        elif epitope.mean_hydrophobicity < -1.0:
            hydro_bias = list("STNQD" * 3)

    while len(population) < size and attempts < size * 50:
        attempts += 1

        if len(population) < size // 2:
            # First half: from seeds
            ab = AntibodySequence(
                CDR_H1=_seed_cdr("CDR_H1", rng),
                CDR_H2=_seed_cdr("CDR_H2", rng),
                CDR_H3=_seed_cdr("CDR_H3", rng),
                CDR_L1=_seed_cdr("CDR_L1", rng),
                CDR_L2=_seed_cdr("CDR_L2", rng),
                CDR_L3=_seed_cdr("CDR_L3", rng),
            )
        else:
            # Second half: biased random
            h3_len = rng.randint(OPTIMAL_H3_MIN, OPTIMAL_H3_MAX)
            h3_pool = AAs + (charge_bias if charge_bias else []) + (hydro_bias if hydro_bias else [])
            h3 = "".join(rng.choices(h3_pool, k=h3_len))

            ab = AntibodySequence(
                CDR_H1=_random_cdr("CDR_H1", rng),
                CDR_H2=_random_cdr("CDR_H2", rng),
                CDR_H3=h3,
                CDR_L1=_random_cdr("CDR_L1", rng),
                CDR_L2=_random_cdr("CDR_L2", rng),
                CDR_L3=_random_cdr("CDR_L3", rng),
            )

        if ab.is_valid() and ab.cdr_string() not in seen:
            population.append(ab)
            seen.add(ab.cdr_string())

    log.info(f"  Initial population: {len(population)} antibody sequences")
    return population


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pi(sequence: str) -> float:
    """
    Compute isoelectric point (pI) of a protein sequence.
    Uses Henderson-Hasselbalch iterative method.
    Biologically validated against known antibody pI values.
    """
    def _net_charge_at_pH(seq: str, pH: float) -> float:
        charge = 0.0
        # N-terminus
        charge += 1.0 / (1.0 + 10 ** (pH - N_TERM_PKA))
        # C-terminus
        charge -= 1.0 / (1.0 + 10 ** (C_TERM_PKA - pH))
        # Side chains
        for aa in seq:
            if aa == "D": charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES["D"] - pH))
            elif aa == "E": charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES["E"] - pH))
            elif aa == "C": charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES["C"] - pH))
            elif aa == "Y": charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES["Y"] - pH))
            elif aa == "H": charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES["H"]))
            elif aa == "K": charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES["K"]))
            elif aa == "R": charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES["R"]))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        nc  = _net_charge_at_pH(sequence, mid)
        if abs(nc) < 1e-4:
            return round(mid, 2)
        if nc > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)


def _count_liabilities(sequence: str) -> Tuple[int, List[str]]:
    """
    Count sequence liability motifs.
    Returns (count, list_of_descriptions).
    These are empirically validated causes of antibody failure in development.
    """
    found = []
    for pattern, description in LIABILITY_MOTIFS.items():
        if re.search(pattern, sequence):
            found.append(description)
    return len(found), found


def _aggregation_score(cdr_h3: str, vh_seq: str) -> float:
    """
    Predict aggregation propensity.
    Based on TANGO algorithm principles:
    - Consecutive hydrophobic residues in CDR-H3 are the main driver
    - Basic patch in VH also contributes
    Returns 0 (no risk) to 1 (high risk).
    """
    score = 0.0

    # CDR-H3 hydrophobic stretches
    h3_hydros = [HYDRO.get(aa, 0) > 1.5 for aa in cdr_h3]
    max_run   = 0
    current   = 0
    for h in h3_hydros:
        if h:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    score += min(1.0, max_run / MAX_HYDROPHOBIC_PATCH)

    # Overall hydrophobicity of CDR-H3
    if cdr_h3:
        mean_h = sum(HYDRO.get(aa, 0) for aa in cdr_h3) / len(cdr_h3)
        if mean_h > 2.0:
            score += 0.3
        elif mean_h > 1.0:
            score += 0.1

    return round(min(1.0, score / 2.0), 3)


def _germline_similarity(vh_seq: str) -> float:
    """
    Estimate human germline similarity of VH sequence.
    Uses simplified k-mer comparison against IGHV1-69 germline framework.
    Real implementation would use IMGT/V-QUEST.
    """
    # IGHV1-69 framework region sequence (the fixed parts)
    germline_kmer_set = set()
    germline_fr = (
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
        "WVRQAPGQGLEWMG"
        "RVTITADESTSTAYMELSSLRSEDTAVYYCAR"
        "WGQGTLVTVSS"
    )
    k = 5
    for i in range(len(germline_fr) - k + 1):
        germline_kmer_set.add(germline_fr[i:i+k])

    # Count matching k-mers in the framework regions of vh_seq
    # (CDRs are not compared since they should differ from germline)
    query_kmers = set()
    for i in range(len(vh_seq) - k + 1):
        query_kmers.add(vh_seq[i:i+k])

    if not query_kmers:
        return 0.85  # default: assume reasonable germline similarity
    overlap = len(germline_kmer_set & query_kmers)
    return round(min(1.0, overlap / len(germline_kmer_set)), 3)


def _affinity_proxy(ab: AntibodySequence, epitope: EpitopeTarget) -> Tuple[float, float, float, float]:
    """
    Compute sequence-based affinity proxy between antibody CDRs and epitope.

    Returns (charge_comp, hydrophobic_burial, hbond_score, shape_score).
    All components are 0-1 floats so they combine cleanly.

    The four components:
    1. Charge complementarity: CDR charge should oppose epitope charge
    2. Hydrophobic complementarity: works for BOTH hydrophobic AND polar epitopes
       - Hydrophobic epitope → reward hydrophobic CDRs
       - Polar/charged epitope → reward polar CDRs (amphipathic complementarity)
    3. H-bond score: normalised by what is achievable given CDR length, not
       by epitope residue count (which is tiny and creates a false ceiling)
    4. Shape complementarity: H3 length vs total epitope surface area directly
       (not divided by n_residues, which causes the ideal_len to collapse)
    """
    cdr_seq = (ab.CDR_H1 + ab.CDR_H2 + ab.CDR_H3 +
               ab.CDR_L1 + ab.CDR_L2 + ab.CDR_L3)
    n_cdr = max(1, len(cdr_seq))

    # ── 1. Charge complementarity ────────────────────────────────────────────
    # CDR net charge should be opposite in sign to epitope net charge.
    # Optimal: cdr_charge ≈ -epitope.net_charge, deviation penalised smoothly.
    cdr_charge  = sum(CHARGE_PH7.get(aa, 0) for aa in cdr_seq)
    ideal_cdr_charge = -epitope.net_charge                 # perfect complement
    charge_dev  = abs(cdr_charge - ideal_cdr_charge)
    charge_comp = max(0.0, 1.0 - charge_dev / 12.0)       # 12 = full VH charge range

    # ── 2. Hydrophobic complementarity ──────────────────────────────────────
    # Fix: we score complementarity against the ABSOLUTE hydrophobicity of the
    # epitope — polar epitopes are complemented by polar CDRs, hydrophobic by
    # hydrophobic. Both directions now give a positive gradient.
    cdr_hydro_mean = sum(HYDRO.get(aa, 0) for aa in cdr_seq) / n_cdr
    epi_hydro      = epitope.mean_hydrophobicity            # can be negative
    # Ideal CDR hydrophobicity mirrors epitope (complementarity = similarity here,
    # because the CDR must bury against whatever surface is presented)
    hydro_dev      = abs(cdr_hydro_mean - epi_hydro)
    # Hydrophobicity range is roughly -4.5 to +4.5 (Kyte-Doolittle)
    hydrophobic_burial = max(0.0, 1.0 - hydro_dev / 6.0)

    # ── 3. H-bond score ───────────────────────────────────────────────────────
    # Normalise by CDR donor/acceptor capacity, not by epitope residue count.
    # A CDR with many donors pairs better with an H-bond-rich epitope.
    hbd_map = {"S":1,"T":1,"N":1,"Q":1,"K":1,"R":2,"H":1,"W":1,"Y":1}
    hba_map = {"D":2,"E":2,"N":1,"Q":1,"S":1,"T":1,"H":1,"Y":1}
    cdr_donors    = sum(hbd_map.get(aa, 0) for aa in cdr_seq)
    cdr_acceptors = sum(hba_map.get(aa, 0) for aa in cdr_seq)
    # Maximum achievable H-bonds given both CDR and epitope capacities
    max_possible_hb = min(cdr_donors + cdr_acceptors,
                          epitope.n_hbond_donors + epitope.n_hbond_acceptors)
    actual_hb = (min(cdr_donors,    epitope.n_hbond_acceptors) +
                 min(cdr_acceptors, epitope.n_hbond_donors))
    hbond_score = actual_hb / max(1, max_possible_hb)

    # ── 4. Shape complementarity ─────────────────────────────────────────────
    # Ideal CDR-H3 length scales with total epitope surface area.
    # Rule of thumb: ~1 CDR-H3 residue covers ~40-50 Å² of epitope SASA.
    # Flat epitope (like TP53 surface): fewer, shorter contacts → lower ideal len.
    # Concave epitope (enzyme active site): longer loop that inserts → higher ideal.
    ideal_h3_len = float(np.clip(epitope.surface_area / 50.0, OPTIMAL_H3_MIN, OPTIMAL_H3_MAX))
    h3_len       = len(ab.CDR_H3)
    # Gaussian-like penalty centred on ideal length, ±3 aa = half score
    shape_score  = math.exp(-0.5 * ((h3_len - ideal_h3_len) / 3.0) ** 2)

    return (
        round(charge_comp,       3),
        round(hydrophobic_burial, 3),
        round(hbond_score,       3),
        round(shape_score,       3),
    )


def _compute_affinity_score(
    charge_comp: float,
    hydrophobic_burial: float,
    hbond_score: float,
    shape_score: float,
    n_epitope_res: int,   # kept for API compatibility, no longer used for normalisation
) -> float:
    """
    Combine affinity components into a single score (0-1, higher = better).

    Weights from structural analysis of antibody-antigen complexes:
      H-bonds:     ~35% (Lo Conte et al. 1999, JMB)
      Hydrophobic: ~40% (Jones & Thornton 1996)
      Shape:       ~15%
      Charge:      ~10%
    All four components are now proper 0-1 gradients, so the score spans
    the full 0-1 range and selection pressure is maintained throughout.
    """
    score = (0.35 * hbond_score         +
             0.40 * hydrophobic_burial   +
             0.15 * shape_score          +
             0.10 * charge_comp)
    return round(min(1.0, score), 4)


def _compute_developability(ab: AntibodySequence) -> Tuple[float, dict]:
    """
    Compute antibody developability score (0-1, 1 = excellent).

    7 empirically validated developability metrics used in industry:
    1. Net charge of VH (optimal: -3 to +3)
    2. pI of VH (optimal: 6.0-9.0 for serum stability)
    3. CDR-H3 length (optimal: 10-16 aa)
    4. Sequence liabilities (deamidation, oxidation, glycosylation, etc.)
    5. Aggregation propensity (TANGO-like, based on hydrophobic CDR-H3)
    6. Human germline similarity (>80% for low immunogenicity)
    7. Specificity / polyreactivity (no long basic patches)

    Each metric scored 0-1, then weighted into final score.
    """
    vh_seq = ab.vh_sequence()

    # 1. Net charge
    net_charge = sum(CHARGE_PH7.get(aa, 0) for aa in vh_seq)
    if NET_CHARGE_MIN <= net_charge <= NET_CHARGE_MAX:
        charge_score = 1.0
    else:
        charge_score = max(0.0, 1.0 - abs(net_charge - 0) / 10.0)

    # 2. pI
    pi = _compute_pi(vh_seq)
    if OPTIMAL_PI_MIN <= pi <= OPTIMAL_PI_MAX:
        pi_score = 1.0
    else:
        pi_score = max(0.0, 1.0 - min(abs(pi - OPTIMAL_PI_MIN),
                                      abs(pi - OPTIMAL_PI_MAX)) / 3.0)

    # 3. CDR-H3 length
    h3_len = len(ab.CDR_H3)
    if OPTIMAL_H3_MIN <= h3_len <= OPTIMAL_H3_MAX:
        h3_score = 1.0
    elif h3_len < MIN_CDR_H3_LENGTH or h3_len > MAX_CDR_H3_LENGTH:
        h3_score = 0.0
    else:
        h3_score = 0.5

    # 4. Sequence liabilities
    full_cdr = (ab.CDR_H1 + ab.CDR_H2 + ab.CDR_H3 +
                ab.CDR_L1 + ab.CDR_L2 + ab.CDR_L3)
    n_liab, liab_list = _count_liabilities(full_cdr)
    liability_score   = max(0.0, 1.0 - n_liab * 0.2)

    # 5. Aggregation propensity
    agg_score_raw = _aggregation_score(ab.CDR_H3, vh_seq)
    agg_score     = 1.0 - agg_score_raw

    # 6. Human germline similarity
    germline_sim = _germline_similarity(vh_seq)
    germ_score   = 1.0 if germline_sim >= HUMAN_GERMLINE_SIM_MIN else \
                   germline_sim / HUMAN_GERMLINE_SIM_MIN

    # 7. Specificity proxy: penalize long polybasic patches
    spec_score = 1.0
    if re.search(r"[KR]{4,}", full_cdr):
        spec_score = 0.3   # severe polyreactivity risk
    elif re.search(r"[KR]{3}", full_cdr):
        spec_score = 0.7

    # Weighted developability score
    dev = (0.15 * charge_score +
           0.15 * pi_score     +
           0.15 * h3_score     +
           0.20 * liability_score +
           0.15 * agg_score    +
           0.10 * germ_score   +
           0.10 * spec_score)

    sub_scores = {
        "net_charge":     round(net_charge, 2),
        "pi":             pi,
        "h3_length":      h3_len,
        "liability_count": n_liab,
        "liabilities":    liab_list,
        "aggregation":    round(agg_score_raw, 3),
        "germline_sim":   round(germline_sim, 3),
        "charge_score":   round(charge_score, 3),
        "pi_score":       round(pi_score, 3),
        "h3_score":       round(h3_score, 3),
        "liability_score": round(liability_score, 3),
        "agg_score":      round(agg_score, 3),
        "germ_score":     round(germ_score, 3),
        "spec_score":     round(spec_score, 3),
    }

    return round(dev, 4), sub_scores


def evaluate_antibody(
    ab:      AntibodySequence,
    epitope: EpitopeTarget,
    gen:     int,
    origin:  str = "evolved",
) -> AntibodyCandidate:
    """
    Full evaluation of one antibody sequence.
    Combines affinity proxy + developability into a scored candidate.
    """
    if not ab.is_valid():
        # Return a zero-score candidate for invalid sequences
        return AntibodyCandidate(
            cdr_h1=ab.CDR_H1, cdr_h2=ab.CDR_H2, cdr_h3=ab.CDR_H3,
            cdr_l1=ab.CDR_L1, cdr_l2=ab.CDR_L2, cdr_l3=ab.CDR_L3,
            generation=gen, affinity_score=0.0, developability=0.0,
            specificity=0.0, novelty=0.0, fitness=0.0,
            net_charge_vh=0.0, pi_vh=0.0, cdr_h3_length=0,
            hydrophobic_score=0.0, liability_count=99, germline_sim=0.0,
            aggregation_score=1.0, charge_comp=0.0, hydrophobic_burial=0.0,
            hbond_capacity=0, shape_score=0.0, origin=origin,
        )

    # Affinity components
    charge_comp, hydro_burial, hbond_score, shape = _affinity_proxy(ab, epitope)
    n_epi = max(1, len(epitope.residue_letters))
    affinity = _compute_affinity_score(charge_comp, hydro_burial, hbond_score, shape, n_epi)

    # Developability
    dev_score, dev_sub = _compute_developability(ab)

    # Specificity: absence of polyreactivity motifs
    spec = dev_sub["spec_score"]

    return AntibodyCandidate(
        cdr_h1=ab.CDR_H1,
        cdr_h2=ab.CDR_H2,
        cdr_h3=ab.CDR_H3,
        cdr_l1=ab.CDR_L1,
        cdr_l2=ab.CDR_L2,
        cdr_l3=ab.CDR_L3,
        generation=gen,
        affinity_score=affinity,
        developability=dev_score,
        specificity=spec,
        novelty=0.0,  # filled in by NoveltyArchive
        fitness=0.0,  # filled in by fitness function
        net_charge_vh=dev_sub["net_charge"],
        pi_vh=dev_sub["pi"],
        cdr_h3_length=dev_sub["h3_length"],
        hydrophobic_score=dev_sub["aggregation"],
        liability_count=dev_sub["liability_count"],
        germline_sim=dev_sub["germline_sim"],
        aggregation_score=dev_sub["aggregation"],
        charge_comp=charge_comp,
        hydrophobic_burial=hydro_burial,
        hbond_capacity=int(round(hbond_score * 10)),  # store as approx count for display
        shape_score=shape,
        origin=origin,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════

class CDRNoveltyArchive:
    """
    Novelty archive based on CDR sequence similarity.
    Uses Hamming distance on aligned CDR strings (not Tanimoto — these are
    sequences, not fingerprints).
    """
    def __init__(self, max_size: int = NOVELTY_ARCHIVE_MAX, k: int = NOVELTY_K):
        self.archive: List[str] = []  # list of cdr_strings
        self.max_size = max_size
        self.k = k

    def _cdr_distance(self, s1: str, s2: str) -> float:
        """Normalized edit distance between CDR strings."""
        if not s1 or not s2:
            return 1.0
        # Simple: count differing positions on aligned parts
        min_len = min(len(s1), len(s2))
        max_len = max(len(s1), len(s2))
        matches = sum(a == b for a, b in zip(s1, s2))
        return round(1.0 - matches / max_len, 4)

    def score(self, cdr_str: str) -> float:
        """Novelty score: mean distance to k nearest neighbors in archive."""
        if len(self.archive) < self.k:
            return 1.0
        dists = sorted(
            [self._cdr_distance(cdr_str, a) for a in self.archive],
            reverse=True,
        )
        return round(statistics.mean(dists[:self.k]), 4)

    def try_add(self, cdr_str: str) -> bool:
        if self.score(cdr_str) >= NOVELTY_ADD_THRESH or len(self.archive) < self.k * 2:
            self.archive.append(cdr_str)
            if len(self.archive) > self.max_size:
                self._prune()
            return True
        return False

    def _prune(self) -> None:
        """Remove the most similar (least novel) member."""
        if len(self.archive) <= 1:
            return
        sims = []
        for i, a in enumerate(self.archive):
            avg_sim = statistics.mean(
                1 - self._cdr_distance(a, self.archive[j])
                for j in range(len(self.archive)) if j != i
            )
            sims.append(avg_sim)
        self.archive.pop(sims.index(max(sims)))


# ══════════════════════════════════════════════════════════════════════════════
# SURROGATE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AntibodySurrogate:
    """
    Ridge regression surrogate for antibody affinity prediction.
    Features: amino acid composition + charge + hydrophobicity + length
    of each CDR loop independently, plus interaction terms.
    """
    def __init__(self):
        self.trained  = False
        self.w: Optional[np.ndarray] = None
        self.bias: float             = 0.0
        self.n_obs:  int             = 0

    def _featurize(self, cdr_str: str) -> np.ndarray:
        """Extract feature vector from CDR string."""
        parts = cdr_str.split("|")
        feats = []
        for part in parts:
            if not part:
                feats.extend([0.0] * 24)
                continue
            # AA composition (20 features)
            aac = [part.count(aa) / max(1, len(part)) for aa in AAs]
            # Summary (4 features)
            charge = sum(CHARGE_PH7.get(aa, 0) for aa in part)
            hydro  = sum(HYDRO.get(aa, 0) for aa in part) / max(1, len(part))
            length = len(part) / 25.0  # normalized
            n_arom = sum(1 for aa in part if aa in "FYW") / max(1, len(part))
            feats.extend(aac + [charge, hydro, length, n_arom])
        return np.array(feats, dtype=float)

    def update(self, history: List[dict]) -> None:
        """Retrain on observed data."""
        if len(history) < SURROGATE_MIN_DATA:
            return
        try:
            X = np.array([self._featurize(h["cdr_string"]) for h in history])
            y = np.array([h["affinity"] for h in history])
            # Ridge regression: (XᵀX + λI)⁻¹ Xᵀy
            lam = 0.01
            A   = X.T @ X + lam * np.eye(X.shape[1])
            b   = X.T @ y
            self.w    = np.linalg.solve(A, b)
            self.bias = y.mean() - X.mean(axis=0) @ self.w
            self.trained = True
            self.n_obs   = len(history)
        except Exception:
            self.trained = False

    def predict(self, cdr_str: str) -> float:
        """Predict affinity score for a CDR string."""
        if not self.trained or self.w is None:
            return 0.5
        x = self._featurize(cdr_str)
        return float(np.clip(x @ self.w + self.bias, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# MUTATION OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _mutate_cdr(seq: str, cdr_name: str, rng: random.Random,
                epitope: EpitopeTarget, temperature: float = 1.0) -> str:
    """
    Apply one biologically grounded mutation to a CDR sequence.

    Mutation types (weighted by biological occurrence during SHM):
    1. Point substitution at SHM hotspot position (45%)
    2. Conservative substitution (same biochemical group) (25%)
    3. Charge-targeted substitution (complementary to epitope) (15%)
    4. Length change: ±1 residue at loop tip (CDR-H3/L1/L3 only) (10%)
    5. Full random substitution at random position (5%)
    """
    if not seq:
        return seq

    lo, hi = CDR_LENGTHS[cdr_name]
    hotspots = SHM_HOTSPOTS.get(cdr_name, list(range(len(seq))))
    seq_list = list(seq)

    # Resolve negative hotspot indices
    hotspots = [h if h >= 0 else len(seq) + h for h in hotspots]
    hotspots = [h for h in hotspots if 0 <= h < len(seq)]

    r = rng.random()

    if r < 0.45 and hotspots:
        # SHM hotspot substitution
        pos = rng.choice(hotspots)
        # Bias toward charged AAs if epitope has complementary charge
        if epitope.net_charge > 1.5:
            pool = list("DEEE") + AAs
        elif epitope.net_charge < -1.5:
            pool = list("KRRH") + AAs
        else:
            pool = AAs
        new_aa = rng.choice(pool)
        seq_list[pos] = new_aa

    elif r < 0.70:
        # Conservative substitution
        pos    = rng.randrange(len(seq_list))
        current = seq_list[pos]
        group  = AA_GROUP.get(current, AAs)
        others = [a for a in group if a != current]
        if others:
            seq_list[pos] = rng.choice(others)

    elif r < 0.85:
        # Charge-targeted substitution
        # Find position with wrong charge relative to epitope
        pos = rng.randrange(len(seq_list))
        if epitope.net_charge > 0.5:
            # Need negative CDR to complement positive epitope
            seq_list[pos] = rng.choice(list("DEED"))
        elif epitope.net_charge < -0.5:
            seq_list[pos] = rng.choice(list("KRRH"))
        else:
            seq_list[pos] = rng.choice(AAs)

    elif r < 0.95 and cdr_name in ("CDR_H3", "CDR_L1", "CDR_L3"):
        # Length change (only in variable-length loops)
        action = rng.choice(["extend", "shrink"])
        if action == "extend" and len(seq_list) < hi:
            ins_pos = len(seq_list) // 2  # insert at loop tip
            seq_list.insert(ins_pos, rng.choice(AAs))
        elif action == "shrink" and len(seq_list) > lo:
            del_pos = len(seq_list) // 2
            seq_list.pop(del_pos)

    else:
        # Random substitution
        pos = rng.randrange(len(seq_list))
        seq_list[pos] = rng.choice(AAs)

    return "".join(seq_list)


def mutate(ab: AntibodySequence, epitope: EpitopeTarget,
           rng: random.Random, temperature: float = 1.0) -> AntibodySequence:
    """
    Mutate an antibody by applying CDR mutations.
    Higher temperature → more mutations, more drastic changes.

    CDR-H3 is mutated with 2x probability (it dominates specificity).
    CDR-H2 with 1.5x probability.
    Others with 1x probability.
    """
    n_mutations = max(1, int(temperature * 3))

    # Weighted CDR selection (H3 most likely, others less)
    cdr_weights = {
        "CDR_H3": 3.0,   # primary determinant of specificity
        "CDR_H2": 1.5,   # secondary
        "CDR_H1": 1.0,
        "CDR_L3": 1.5,   # light chain specificity
        "CDR_L1": 1.0,
        "CDR_L2": 0.5,   # nearly framework-like, mutate rarely
    }
    cdr_names = list(cdr_weights.keys())
    weights   = [cdr_weights[c] for c in cdr_names]

    new_cdrs = ab.cdr_dict().copy()
    for _ in range(n_mutations):
        cdr_name = rng.choices(cdr_names, weights=weights, k=1)[0]
        new_cdrs[cdr_name] = _mutate_cdr(
            new_cdrs[cdr_name], cdr_name, rng, epitope, temperature
        )

    child = AntibodySequence(**new_cdrs)
    return child if child.is_valid() else ab


def crossover(ab1: AntibodySequence, ab2: AntibodySequence,
              rng: random.Random) -> AntibodySequence:
    """
    CDR-level crossover: each CDR loop independently chosen from either parent.
    Biologically analogous to V(D)J recombination / combinatorial diversity.
    """
    new_cdrs = {}
    for cdr_name in ["CDR_H1", "CDR_H2", "CDR_H3", "CDR_L1", "CDR_L2", "CDR_L3"]:
        new_cdrs[cdr_name] = (ab1.cdr_dict()[cdr_name]
                              if rng.random() < 0.5
                              else ab2.cdr_dict()[cdr_name])
    child = AntibodySequence(**new_cdrs)
    return child if child.is_valid() else ab1


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVOLUTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_antibody_design(
    uniprot_id:       str,
    active_data:      Optional[dict] = None,
    physico_data:     Optional[dict] = None,
    ppi_data:         Optional[dict] = None,
    allosteric_data:  Optional[dict] = None,
    n_generations:    int            = MAX_GENERATIONS,
    epitope_mode:     str            = "auto",
    rng_seed:         Optional[int]  = None,
) -> AntibodyResult:
    """
    Run evolutionary antibody CDR design against the target protein.

    Args:
        uniprot_id:      UniProt accession of the target antigen
        active_data:     Module 03 output (active_sites JSON)
        physico_data:    Module 02 output (physicochemical JSON)
        ppi_data:        Module 12 output (ppi_network JSON)
        allosteric_data: Module 05 output (allosteric JSON)
        n_generations:   Number of evolution generations
        epitope_mode:    "auto" / "active" / "ppi" / "surface" / "allosteric"
        rng_seed:        For reproducibility

    Returns:
        AntibodyResult with top CDR sequences and all scores.
    """
    t0   = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    rng  = random.Random(seed)
    np.random.seed(seed % (2**32))

    uid = uniprot_id.strip().upper()
    log.info(f"══ Module 16: Antibody Design: {uid} [seed={seed}] ══")

    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uid}_antibody.json"

    # Load structure metadata for gene name
    target_gene = uid
    struct_json = inter_dir / f"{uid}_structure.json"
    if struct_json.exists():
        try:
            target_gene = json.loads(struct_json.read_text()).get("gene_name", uid)
        except Exception:
            pass

    log.info(f"  Target: {uid} ({target_gene})")
    log.info(f"  Epitope mode: {epitope_mode}")

    # ── Select epitope ────────────────────────────────────────────────────────
    epitope = select_epitope(
        uid, active_data, physico_data, ppi_data, allosteric_data, epitope_mode
    )

    if epitope:
        log.info(f"  Epitope: {len(epitope.residue_numbers)} residues  "
                 f"charge={epitope.net_charge:+.1f}  "
                 f"hydro={epitope.mean_hydrophobicity:.2f}  "
                 f"SASA={epitope.surface_area:.0f}Å²  "
                 f"({epitope.source})")
    else:
        log.warning("  No epitope found — using generic surface")
        epitope = EpitopeTarget(
            source="generic", residue_numbers=[], residue_letters=[],
            center_coords=[0,0,0], mean_sasa=100.0, net_charge=0.0,
            mean_hydrophobicity=0.0, n_aromatic=2, n_hbond_donors=5,
            n_hbond_acceptors=5, surface_area=800.0,
        )

    # ── Build initial population ──────────────────────────────────────────────
    population = build_initial_population(POP_SIZE, epitope, rng)

    # ── Evolution infrastructure ──────────────────────────────────────────────
    novelty_archive = CDRNoveltyArchive()
    surrogate       = AntibodySurrogate()
    history: List[dict] = []
    hall_of_fame: List[AntibodyCandidate] = []
    gen_stats: List[dict] = []

    best_fitness   = 0.0
    best_affinity  = 0.0
    best_cdr_h3    = ""
    stagnation_cnt = 0
    n_evaluated    = 0

    log.info(f"  Starting evolution: {n_generations} generations × {POP_SIZE} sequences")

    for gen in range(1, n_generations + 1):
        # Adaptive weight schedule
        t         = min(1.0, (gen - 1) / max(1, n_generations - 1))
        w_aff     = W_AFFINITY_START + t * (W_AFFINITY_END - W_AFFINITY_START)
        w_dev     = W_DEVELOP_START  + t * (W_DEVELOP_END  - W_DEVELOP_START)
        w_nov     = W_NOVELTY_START  + t * (W_NOVELTY_END  - W_NOVELTY_START)
        total_w   = w_aff + w_dev + w_nov
        w_aff /= total_w; w_dev /= total_w; w_nov /= total_w

        temperature = max(0.1, 1.0 - 0.8 * t)

        # Update surrogate
        if len(history) >= SURROGATE_MIN_DATA:
            surrogate.update(history)

        # ── Evaluate population ───────────────────────────────────────────────
        def _eval_one(ab_idx):
            ab, idx = ab_idx
            cand = evaluate_antibody(ab, epitope, gen, origin="evolved")
            cand.novelty = novelty_archive.score(ab.cdr_string())
            cand.fitness = (w_aff * cand.affinity_score +
                            w_dev * cand.developability +
                            w_nov * cand.novelty)
            return cand

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            results = list(ex.map(_eval_one, [(ab, i) for i, ab in enumerate(population)]))

        results = [r for r in results if r.affinity_score > 0]
        if not results:
            continue

        n_evaluated += len(results)

        # Update novelty archive and history
        # Hall of fame tracks by AFFINITY only (not fitness) so novelty decay
        # doesn't corrupt it over time.
        for cand in results:
            ab_str = cand.cdr_h1 + "|" + cand.cdr_h2 + "|" + cand.cdr_h3 + "|" + cand.cdr_l1 + "|" + cand.cdr_l2 + "|" + cand.cdr_l3
            novelty_archive.try_add(ab_str)
            history.append({
                "cdr_string": ab_str,
                "affinity":   cand.affinity_score,
                "fitness":    cand.fitness,
                "generation": gen,
            })
            # HoF sorted by affinity_score, not fitness — immune to weight schedule drift
            if cand.affinity_score > min((h.affinity_score for h in hall_of_fame), default=0) \
                    or len(hall_of_fame) < 10:
                hall_of_fame.append(cand)
                hall_of_fame = sorted(hall_of_fame, key=lambda h: h.affinity_score, reverse=True)[:10]

        # Sort
        by_fitness  = sorted(results, key=lambda c: c.fitness, reverse=True)
        by_affinity = sorted(results, key=lambda c: c.affinity_score, reverse=True)

        gen_best = by_fitness[0]
        # Use affinity (not composite fitness) to detect real improvement.
        # Composite fitness decays over time as novelty weight drops, which
        # would otherwise make every generation look like stagnation.
        improved = gen_best.affinity_score > best_affinity

        if improved:
            best_fitness  = gen_best.fitness
            best_affinity = gen_best.affinity_score
            best_cdr_h3   = gen_best.cdr_h3
            stagnation_cnt = 0
        else:
            stagnation_cnt += 1

        # Diversity (mean pairwise CDR-H3 identity difference)
        h3s = [c.cdr_h3 for c in results]
        if len(h3s) > 1:
            pairs = [(h3s[i], h3s[j]) for i in range(len(h3s)) for j in range(i+1, min(i+5, len(h3s)))]
            div = statistics.mean(
                sum(a != b for a, b in zip(s1, s2)) / max(len(s1), len(s2), 1)
                for s1, s2 in pairs
            ) if pairs else 0.5
        else:
            div = 0.5

        flag = "↑ IMPROVED" if improved else f"stagnant {stagnation_cnt}"
        log.info(
            f"  Gen {gen:3d}: fit={gen_best.fitness:.3f}  "
            f"aff={gen_best.affinity_score:.3f}  "
            f"dev={gen_best.developability:.2f}  "
            f"H3={gen_best.cdr_h3[:15]}  "
            f"pI={gen_best.pi_vh:.1f}  "
            f"chg={gen_best.net_charge_vh:+.1f}  "
            f"div={div:.2f}  {flag}"
        )

        gen_stats.append({
            "generation":   gen,
            "best_fitness": gen_best.fitness,
            "best_affinity": gen_best.affinity_score,
            "best_dev":     gen_best.developability,
            "best_h3":      gen_best.cdr_h3,
            "mean_fitness": round(statistics.mean(c.fitness for c in results), 4),
            "diversity":    round(div, 3),
            "stagnation":   stagnation_cnt,
            "n_surrogate":  surrogate.n_obs,
        })

        # ── Build next generation ─────────────────────────────────────────────
        next_pop: List[AntibodySequence] = []
        seen_cdrs: set = set()

        def _add(ab: AntibodySequence) -> bool:
            cs = ab.cdr_string()
            if ab.is_valid() and cs not in seen_cdrs:
                next_pop.append(ab); seen_cdrs.add(cs); return True
            return False

        # Elitism: top fitness survives
        for cand in by_fitness[:ELITISM]:
            ab = AntibodySequence(cand.cdr_h1, cand.cdr_h2, cand.cdr_h3,
                                  cand.cdr_l1, cand.cdr_l2, cand.cdr_l3)
            _add(ab)

        # Hall of fame injection — skipped when stagnation is severe to avoid
        # re-injecting the same stuck sequences and undoing the diversity push
        if stagnation_cnt < STAGNATION_HARD - 1:
            for cand in hall_of_fame[:3]:
                ab = AntibodySequence(cand.cdr_h1, cand.cdr_h2, cand.cdr_h3,
                                      cand.cdr_l1, cand.cdr_l2, cand.cdr_l3)
                _add(ab)

        # Crossover between top performers
        top_abs = [AntibodySequence(c.cdr_h1, c.cdr_h2, c.cdr_h3,
                                    c.cdr_l1, c.cdr_l2, c.cdr_l3)
                   for c in by_fitness[:10]]
        for _ in range(POP_SIZE // 3):
            if len(top_abs) >= 2:
                p1, p2 = rng.sample(top_abs, 2)
                child  = crossover(p1, p2, rng)
                _add(child)

        # Mutation of survivors
        for cand in by_fitness[:POP_SIZE // 2]:
            parent = AntibodySequence(cand.cdr_h1, cand.cdr_h2, cand.cdr_h3,
                                      cand.cdr_l1, cand.cdr_l2, cand.cdr_l3)
            child  = mutate(parent, epitope, rng, temperature)
            _add(child)

        # Diversity injection
        if div < DIVERSITY_MIN or stagnation_cnt >= STAGNATION_LIMIT:
            n_new = POP_SIZE // 4
            for _ in range(n_new * 20):
                if sum(1 for ab in next_pop if ab not in top_abs[:ELITISM]) >= POP_SIZE - ELITISM:
                    break
                ab = AntibodySequence(
                    CDR_H1=_seed_cdr("CDR_H1", rng),
                    CDR_H2=_seed_cdr("CDR_H2", rng),
                    CDR_H3=_seed_cdr("CDR_H3", rng),
                    CDR_L1=_seed_cdr("CDR_L1", rng),
                    CDR_L2=_seed_cdr("CDR_L2", rng),
                    CDR_L3=_seed_cdr("CDR_L3", rng),
                )
                _add(ab)

        # Hard stagnation reset: rebuild majority of population from scratch
        # Only keep the single all-time best sequence; everything else is replaced.
        # Also resets stagnation_cnt so we don't fire every generation.
        if stagnation_cnt >= STAGNATION_HARD:
            n_keep   = 1   # keep only the absolute best
            n_reset  = POP_SIZE - n_keep
            log.info(f"  HARD RESET gen {gen}: replacing {n_reset} sequences with fresh seeds")
            best_ab  = next_pop[:n_keep]   # the single elite
            fresh    = []
            attempts = 0
            while len(fresh) < n_reset and attempts < n_reset * 30:
                attempts += 1
                ab = AntibodySequence(
                    CDR_H1=_seed_cdr("CDR_H1", rng),
                    CDR_H2=_seed_cdr("CDR_H2", rng),
                    CDR_H3=_seed_cdr("CDR_H3", rng),
                    CDR_L1=_seed_cdr("CDR_L1", rng),
                    CDR_L2=_seed_cdr("CDR_L2", rng),
                    CDR_L3=_seed_cdr("CDR_L3", rng),
                )
                cs = ab.cdr_string()
                if ab.is_valid() and cs not in seen_cdrs:
                    fresh.append(ab)
                    seen_cdrs.add(cs)
            next_pop = best_ab + fresh
            stagnation_cnt = 0   # ← THE critical fix: reset after hard reset

        # Surrogate-guided screening: generate extra candidates and pre-screen
        if surrogate.trained and len(next_pop) < POP_SIZE:
            candidates_screened = []
            for _ in range(SURROGATE_CANDIDATES):
                parent = rng.choice(top_abs) if top_abs else AntibodySequence(
                    **{k: _seed_cdr(k, rng) for k in CDR_LENGTHS})
                child = mutate(parent, epitope, rng, temperature * 0.5)
                if child.is_valid() and child.cdr_string() not in seen_cdrs:
                    pred = surrogate.predict(child.cdr_string())
                    candidates_screened.append((child, pred))
            candidates_screened.sort(key=lambda x: x[1], reverse=True)
            for child, _ in candidates_screened[:POP_SIZE - len(next_pop)]:
                _add(child)

        # Fill remaining slots
        attempts = 0
        while len(next_pop) < POP_SIZE and attempts < POP_SIZE * 20:
            attempts += 1
            ab = AntibodySequence(
                **{k: _seed_cdr(k, rng) for k in CDR_LENGTHS}
            )
            _add(ab)

        population = next_pop[:POP_SIZE]

    # ── Final validation of top candidates ────────────────────────────────────
    log.info(f"\n  ── Final validation of top {TOP_FOR_FINAL} candidates ──")

    # Collect best unique sequences from history
    seen_h3: set = set()
    final_candidates: List[AntibodyCandidate] = []

    for cand in sorted(hall_of_fame, key=lambda c: c.fitness, reverse=True):
        if cand.cdr_h3 not in seen_h3 and cand.affinity_score > 0:
            final_candidates.append(cand)
            seen_h3.add(cand.cdr_h3)
        if len(final_candidates) >= TOP_FOR_FINAL:
            break

    # Re-score top candidates with novelty relative to final pool
    for cand in final_candidates:
        ab_str = f"{cand.cdr_h1}|{cand.cdr_h2}|{cand.cdr_h3}|{cand.cdr_l1}|{cand.cdr_l2}|{cand.cdr_l3}"
        cand.novelty = novelty_archive.score(ab_str)
        log.info(
            f"  Final #{final_candidates.index(cand)+1}: "
            f"aff={cand.affinity_score:.3f}  dev={cand.developability:.2f}  "
            f"pI={cand.pi_vh:.1f}  H3={cand.cdr_h3}"
        )

    top_candidates = sorted(final_candidates, key=lambda c: c.fitness, reverse=True)

    result = AntibodyResult(
        uniprot_id=uid,
        target_gene=target_gene,
        epitope=epitope,
        epitope_source=epitope.source,
        n_generations=n_generations,
        n_evaluated=n_evaluated,
        n_unique=len(set(h["cdr_string"] for h in history)),
        top_candidates=top_candidates,
        best_affinity=best_affinity,
        best_fitness=best_fitness,
        best_cdr_h3=best_cdr_h3,
        generation_stats=gen_stats,
        notes=f"seed={seed}  epitope={epitope.source}  "
              f"runtime={time.time()-t0:.1f}s",
    )

    result.to_json(out_path)
    log.info(result.summary())
    log.info(f"\n  Results saved: {out_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

import click

@click.command()
@click.option("--uniprot",       "-u", required=True,
              help="UniProt ID of the target antigen (e.g. P04637)")
@click.option("--generations",   "-g", default=MAX_GENERATIONS, type=int,
              help=f"Number of evolution generations (default: {MAX_GENERATIONS})")
@click.option("--epitope-mode",  "-e",
              default="auto",
              type=click.Choice(["auto", "active", "ppi", "surface", "allosteric"]),
              help="Epitope selection strategy (default: auto)")
@click.option("--seed",          "-s", default=None, type=int,
              help="Random seed for reproducibility")
def main(uniprot: str, generations: int, epitope_mode: str, seed: Optional[int]) -> None:
    """
    Module 16 — De Novo Antibody CDR Design.

    Evolves antibody CDR loop sequences to bind a surface epitope on the
    target protein. Requires Modules 02, 03, 05, 12 to have run first
    for best epitope selection, but will fall back gracefully.

    Example:
        python pipeline\\antibody_design.py --uniprot P04637
        python pipeline\\antibody_design.py --uniprot P04637 --epitope-mode ppi
        python pipeline\\antibody_design.py --uniprot P04637 --generations 60
    """
    uid       = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    def _load(fname):
        p = inter_dir / fname
        return json.loads(p.read_text()) if p.exists() else None

    active_data     = _load(f"{uid}_active_sites.json")
    physico_data    = _load(f"{uid}_physicochemical.json")
    ppi_data        = _load(f"{uid}_ppi.json")
    allosteric_data = _load(f"{uid}_allosteric.json")

    loaded = []
    if active_data:     loaded.append(f"active_sites ({sum(1 for r in active_data.get('active_residues',[]) if r.get('confidence')=='HIGH')} HIGH)")
    if physico_data:    loaded.append("physicochemical")
    if ppi_data:        loaded.append(f"ppi ({ppi_data.get('n_partners',0)} partners)")
    if allosteric_data: loaded.append(f"allosteric ({allosteric_data.get('n_sites',0)} sites)")
    log.info(f"  Loaded modules: {', '.join(loaded) if loaded else 'none — running with generic epitope'}")

    result = run_antibody_design(
        uniprot_id=uid,
        active_data=active_data,
        physico_data=physico_data,
        ppi_data=ppi_data,
        allosteric_data=allosteric_data,
        n_generations=generations,
        epitope_mode=epitope_mode,
        rng_seed=seed,
    )

    click.echo(result.summary())


if __name__ == "__main__":
    main()