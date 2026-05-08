"""
pipeline/adc_design.py
───────────────────────
Module 18 — De Novo Antibody-Drug Conjugate (ADC) Design

Evolves complete ADC candidates: antibody CDR sequences + warhead SMILES +
linker chemistry. Skips if data/intermediate/{uid}_adc.json already exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AN ADC?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An ADC = antibody (targeting) + linker (release control) + warhead (cytotoxin).

  [Antibody (CDR sequences)]──[Linker]──[Warhead (SMILES)]
        ↓                          ↓              ↓
  surface epitope binding    cleavability    cell-killing payload

This module co-evolves all three components simultaneously:
  - Antibody CDRs: evolved from Module 16 (antibody_design.py) logic
  - Linker: selected from a validated linker library (cleavable/non-cleavable)
  - Warhead: evolved from a cytotoxin fragment library (MMAE-like, DM1-like)

FITNESS FUNCTION (multi-objective):
  affinity_score     — CDR complementarity to epitope (from antibody_design)
  developability     — antibody therapeutic properties (pI, charge, aggregation)
  warhead_potency    — estimated IC50 proxy from warhead scaffold class
  dac_ratio          — drug-antibody ratio compatibility (DAR 2-8 preferred)
  linker_stability   — plasma stability + tumour cleavability balance
  off_target_score   — penalty for predicted normal tissue expression exposure
  composite_fitness  — weighted sum, adaptive weights over generations

SKIP LOGIC:
  If data/intermediate/{uid}_adc.json already exists and --force is not set,
  the module exits immediately with the cached result.

OUTPUT:
  data/intermediate/{uid}_adc.json
  Top ADC candidates ranked by composite fitness, each with:
    - CDR sequences (H1/H2/H3/L1/L2/L3)
    - Linker type + SMILES
    - Warhead SMILES + class + estimated potency
    - DAR recommendation
    - Full score breakdown

Usage (standalone):
    python pipeline/adc_design.py --uniprot P04637
    python pipeline/adc_design.py --uniprot P04637 --generations 60
    python pipeline/adc_design.py --uniprot P04637 --epitope-mode ppi --warhead mmae

Usage (from orchestrator):
    from pipeline.adc_design import run_adc_design
    result = run_adc_design("P04637", active_data, physico_data, ppi_data, allosteric_data)
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MAX_GENERATIONS   = 50
POP_SIZE          = 30
ELITISM           = 5
TOP_FOR_FINAL     = 8
STAGNATION_LIMIT  = 5
STAGNATION_HARD   = 10
DIVERSITY_MIN     = 0.20

# Fitness weight schedule (start → end)
W_AFF_START = 0.40;  W_AFF_END = 0.55
W_DEV_START = 0.20;  W_DEV_END = 0.20
W_WAR_START = 0.25;  W_WAR_END = 0.15
W_LNK_START = 0.15;  W_LNK_END = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

AAs = list("ACDEFGHIKLMNPQRSTVWY")

CDR_LENGTHS: Dict[str, Tuple[int, int]] = {
    "CDR_H1": (5, 12),   # Kabat H1: 5 residues core, real antibodies 5-12
    "CDR_H2": (10, 19),  # Kabat H2: 17 aa standard but 10-19 observed
    "CDR_H3": (3, 25),   # H3 most variable: 3-25 aa
    "CDR_L1": (9, 17),   # L1: 9-17 aa
    "CDR_L2": (7, 7),    # L2: fixed at 7
    "CDR_L3": (7, 11),   # L3: 7-11 aa
}

# Validated ADC warhead classes with representative SMILES and estimated potency tier
WARHEAD_LIBRARY = [
    # class, smiles, potency (0-1), DAR_preference, notes
    ("MMAE",    "CCOC(=O)[C@@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H]"
                "(CC(C)C)NC(=O)CNC(=O)[C@@H](NC(=O)c1ccc(N)cc1)CC(C)C)CC(C)C)"
                "C(C)C", 0.95, (2, 4), "Auristatin — microtubule inhibitor, most common ADC warhead"),
    ("DM1",     "COC1=CC2=C(C=C1)[C@H]1CC[C@]3(OC3=O)N(C)[C@@H]([C@@H]"
                "(OC(=O)N2)SC)C1=O", 0.90, (3, 4), "Maytansinoid — inhibits microtubule polymerisation"),
    ("DM4",     "COC1=CC2=C(C=C1)[C@H]1CC[C@]3(OC3=O)N(C)[C@@H]([C@@H]"
                "(OC(=O)N2)SC(C)(C)C)C1=O", 0.90, (3, 4), "DM4 variant — improved DAR flexibility"),
    ("SN38",    "OCC1=C2CC3=C(CC(=O)OCC)C=CC=C3C(=O)N2C=C1", 0.80, (4, 8),
                "Camptothecin analogue — topoisomerase I inhibitor"),
    ("Dxd",     "OCC1=C2CC3=C(CC(=O)O)C=CC=C3C(=O)N2C=C1", 0.82, (4, 8),
                "Exatecan derivative — used in DS-8201 (T-DXd)"),
    ("CalicheA","OC1OC(C(=O)C2=CC=CC=C2)C(N)C(O)C1OC1OC(CO)C(O)C(O)C1O",
                0.98, (1, 2), "Calicheamicin — DNA strand-breaking, ultra-potent (Mylotarg)"),
    ("PBD",     "O=C1NC2=CC=CC=C2CC2=CC(OC)=C(OC)C=C12", 0.97, (2, 4),
                "Pyrrolobenzodiazepine — DNA cross-linker"),
    ("MMAF",    "CCC(CC(C(=O)NC(CC(=O)O)C(=O)NC(Cc1ccccc1)C(=O)NC(C)C(=O)N"
                "C(CC(C)C)C(=O)O)OC)NC(=O)CNC(=O)C(NC(=O)C(C)NC(=O)C(NC(=O)"
                "c1ccc(N)cc1)CC(C)C)C(C)C", 0.88, (2, 4),
                "Auristatin F — charged variant, reduced bystander effect"),
]

# Linker library: (name, smiles_fragment, cleavable, plasma_stability, tumour_release)
LINKER_LIBRARY = [
    ("mc-VC-PABC",  "CC(C)(CS)C(=O)NCC(=O)NC(CC(=O)O)C(=O)NCC(=O)NCCOC",
                    True,  0.92, 0.90, "Protease-cleavable (cathepsin B) — most common"),
    ("SMCC",        "O=C1CC=CC(=O)NCCCCCC(=O)O",
                    False, 0.95, 0.50, "Thioether, stable — used with maytansinoids"),
    ("SPDB",        "CCOC(=O)CCNC(=O)CCSSC",
                    True,  0.88, 0.85, "Disulfide — cleaved in tumour reductive environment"),
    ("CL2A",        "OC(=O)CCNC(=O)OCC1=CC=CC=C1",
                    True,  0.90, 0.80, "Carbonate — pH-sensitive release"),
    ("Glucuronide",  "OC1OC(C(=O)O)C(O)C(O)C1O",
                    True,  0.94, 0.88, "β-glucuronidase cleavable — high tumour specificity"),
    ("PEG4-VC",     "CCCCOC(=O)NCC(=O)NCCOCCOCCOCCOCCC(=O)O",
                    True,  0.91, 0.87, "PEGylated protease-cleavable — improved PK"),
    ("Acid-labile",  "CC(C)(C)OC(=O)NCC(=O)O",
                    True,  0.82, 0.92, "pH-sensitive — endosomal release"),
    ("Non-cleavable","CCCC(=O)NCCCC(=O)O",
                    False, 0.97, 0.40, "Stable thioether — lysosomal degradation releases warhead"),
]

# CDR seeds from therapeutic antibodies (same as antibody_design.py)
CDR_SEEDS: Dict[str, List[str]] = {
    "CDR_H3": ["ARDYYGSGSYYFDY", "AKDSSSWYFDY", "ARGLGLVRGAMDY",
               "AKDRWGGDAFDM", "ARDQRSGYYFDY", "ARYGDYYGFAY"],
    "CDR_H1": ["GYTFTDYY", "GFSLTNYG", "GYTFTSYW", "GFTFSSYW", "GYTFTGYY"],
    "CDR_H2": ["INTYTGEPTYADSVKG", "IYPGDGDTRYSPSFQG", "INPYNDGTKYDPKFQG"],
    "CDR_L1": ["SSSVSSYLY", "RASESVDNYGISFMN", "QASQDISNYLN"],
    "CDR_L2": ["DTSNLAS", "GASNRAT", "AASTLQS"],
    "CDR_L3": ["QQSYSTPLT", "QQRSNWPYT", "QQYYSYPLT"],
}

CHARGE_AT_PH7 = {
    "R": +1.0, "K": +1.0, "H": +0.1, "D": -1.0, "E": -1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ADCEpitope:
    """Epitope targeted by the ADC antibody component."""
    source:               str
    residue_numbers:      List[int]
    residue_letters:      List[str]
    mean_sasa:            float
    net_charge:           float
    mean_hydrophobicity:  float
    surface_area:         float
    n_hbond_donors:       int
    n_hbond_acceptors:    int
    description:          str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ADCCandidate:
    """A single evaluated ADC candidate — antibody + linker + warhead."""
    # Antibody component
    cdr_h1:             str
    cdr_h2:             str
    cdr_h3:             str
    cdr_l1:             str
    cdr_l2:             str
    cdr_l3:             str
    generation:         int
    # Payload component
    warhead_class:      str
    warhead_smiles:     str
    linker_name:        str
    linker_smiles:      str
    linker_cleavable:   bool
    dar_min:            int
    dar_max:            int
    # Scores
    affinity_score:     float   # antibody-epitope complementarity
    developability:     float   # antibody druggability
    warhead_potency:    float   # 0-1 estimated payload potency
    linker_score:       float   # stability × tumour release balance
    off_target_penalty: float   # 0-1, lower is better
    fitness:            float   # composite
    # Sub-scores
    net_charge_vh:      float
    pi_vh:              float
    charge_comp:        float
    hbond_capacity:     int
    origin:             str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, rank: int) -> str:
        return (
            f"  #{rank:<2}  aff={self.affinity_score:.3f}  "
            f"dev={self.developability:.2f}  "
            f"warhead={self.warhead_class}  "
            f"linker={self.linker_name}  "
            f"DAR={self.dar_min}-{self.dar_max}  "
            f"H3={self.cdr_h3}"
        )


@dataclass
class ADCResult:
    """Full output of the ADC design module."""
    uniprot_id:         str
    target_gene:        str          = ""
    epitope:            Optional[ADCEpitope] = None
    epitope_source:     str          = ""
    n_generations:      int          = 0
    n_evaluated:        int          = 0
    top_candidates:     List[ADCCandidate] = field(default_factory=list)
    best_fitness:       float        = 0.0
    best_cdr_h3:        str          = ""
    best_warhead:       str          = ""
    best_linker:        str          = ""
    generation_stats:   List[dict]   = field(default_factory=list)
    notes:              str          = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self

    def summary(self) -> str:
        lines = [
            f"\n{'═'*72}",
            f"  MODULE 18 — ADC Design: {self.uniprot_id} ({self.target_gene})",
            f"{'═'*72}",
            f"  Epitope source   : {self.epitope_source}",
            f"  Generations run  : {self.n_generations}",
            f"  Candidates eval. : {self.n_evaluated}",
            f"  Best fitness     : {self.best_fitness:.4f}",
            f"  Best warhead     : {self.best_warhead}",
            f"  Best linker      : {self.best_linker}",
            f"  Best CDR-H3      : {self.best_cdr_h3}",
            f"\n  Top ADC Candidates:",
        ]
        for i, c in enumerate(self.top_candidates[:5], 1):
            lines.append(c.summary_line(i))
        lines.append(f"{'═'*72}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EPITOPE SELECTION (reuses antibody_design logic via import)
# ══════════════════════════════════════════════════════════════════════════════

def _select_adc_epitope(
    uid: str,
    active_data: Optional[dict],
    physico_data: Optional[dict],
    ppi_data: Optional[dict],
    allosteric_data: Optional[dict],
    epitope_mode: str,
) -> Optional[ADCEpitope]:
    """
    Select epitope using the same priority order as antibody_design.
    Returns an ADCEpitope (slimmer dataclass than antibody_design.EpitopeTarget).
    """
    try:
        from pipeline.antibody_design import select_epitope
        epi = select_epitope(uid, active_data, physico_data, ppi_data,
                             allosteric_data, epitope_mode)
        if epi is None:
            return None
        return ADCEpitope(
            source=epi.source,
            residue_numbers=epi.residue_numbers,
            residue_letters=epi.residue_letters,
            mean_sasa=epi.mean_sasa,
            net_charge=epi.net_charge,
            mean_hydrophobicity=epi.mean_hydrophobicity,
            surface_area=epi.surface_area,
            n_hbond_donors=epi.n_hbond_donors,
            n_hbond_acceptors=epi.n_hbond_acceptors,
            description=epi.description,
        )
    except Exception as e:
        log.warning(f"  Epitope selection failed: {e} — using generic fallback")
        return ADCEpitope(
            source="generic", residue_numbers=[], residue_letters=[],
            mean_sasa=100.0, net_charge=0.0, mean_hydrophobicity=0.0,
            surface_area=800.0, n_hbond_donors=5, n_hbond_acceptors=5,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CDR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _random_cdr(cdr_name: str, rng: random.Random) -> str:
    lo, hi = CDR_LENGTHS[cdr_name]
    return "".join(rng.choices(AAs, k=rng.randint(lo, hi)))


def _seed_cdr(cdr_name: str, rng: random.Random) -> str:
    seeds = CDR_SEEDS.get(cdr_name, [])
    if seeds:
        base = rng.choice(seeds)
        n_mut = rng.randint(0, min(2, len(base)))
        seq = list(base)
        for _ in range(n_mut):
            seq[rng.randrange(len(seq))] = rng.choice(AAs)
        return "".join(seq)
    return _random_cdr(cdr_name, rng)


def _cdr_string(h1, h2, h3, l1, l2, l3) -> str:
    return f"{h1}|{h2}|{h3}|{l1}|{l2}|{l3}"


def _is_valid_cdrs(h1, h2, h3, l1, l2, l3) -> bool:
    for name, seq in zip(["CDR_H1","CDR_H2","CDR_H3","CDR_L1","CDR_L2","CDR_L3"],
                         [h1, h2, h3, l1, l2, l3]):
        lo, hi = CDR_LENGTHS[name]
        if not (lo <= len(seq) <= hi):
            return False
        if not all(a in AAs for a in seq):
            return False
    return True


def _mutate_cdr(seq: str, cdr_name: str, rng: random.Random) -> str:
    lo, hi = CDR_LENGTHS[cdr_name]
    seq = list(seq)
    op = rng.random()
    if op < 0.60 and seq:                      # point mutation
        seq[rng.randrange(len(seq))] = rng.choice(AAs)
    elif op < 0.80 and len(seq) < hi:           # insertion
        pos = rng.randrange(len(seq) + 1)
        seq.insert(pos, rng.choice(AAs))
    elif op < 0.95 and len(seq) > lo:           # deletion
        seq.pop(rng.randrange(len(seq)))
    else:                                       # conservative swap
        conservative = {"R": "K", "K": "R", "D": "E", "E": "D",
                        "I": "V", "V": "I", "L": "I", "F": "Y", "Y": "F"}
        pos = rng.randrange(len(seq))
        seq[pos] = conservative.get(seq[pos], rng.choice(AAs))
    return "".join(seq)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_antibody(h1, h2, h3, l1, l2, l3,
                    epitope: ADCEpitope) -> Tuple[float, float, float, float, float, int]:
    """Returns (affinity, developability, charge_vh, pi_vh, charge_comp, hbond_cap)."""
    all_cdr = h1 + h2 + h3 + l1 + l2 + l3

    # Charge complementarity to epitope
    cdr_charge = sum(CHARGE_AT_PH7.get(a, 0.0) for a in all_cdr)
    charge_comp = max(0.0, min(1.0, 0.5 - 0.1 * (cdr_charge + epitope.net_charge) ** 2))

    # H-bond capacity
    hbond_donors    = sum(1 for a in all_cdr if a in "NQSTHY")
    hbond_acceptors = sum(1 for a in all_cdr if a in "DEQNSTY")
    hbond_cap       = min(hbond_donors + hbond_acceptors, epitope.n_hbond_donors + epitope.n_hbond_acceptors)

    # Hydrophobic burial at interface
    cdr_hydro = sum({"F": 1, "W": 1, "Y": 0.8, "I": 0.7, "L": 0.7,
                     "V": 0.5, "M": 0.5, "A": 0.3}.get(a, 0) for a in h3)
    epi_hydro = epitope.mean_hydrophobicity
    hydro_match = max(0.0, min(1.0, (cdr_hydro / max(1, len(h3))) * epi_hydro))

    # CDR-H3 length bonus (longer H3 = better epitope burial for ADC)
    len_bonus = min(1.0, len(h3) / 15.0)

    affinity = (0.35 * charge_comp + 0.25 * hydro_match +
                0.20 * (hbond_cap / max(1, epitope.n_hbond_donors + epitope.n_hbond_acceptors)) +
                0.20 * len_bonus)
    affinity = max(0.0, min(1.0, affinity + random.gauss(0, 0.03)))

    # Developability
    vh_charge = sum(CHARGE_AT_PH7.get(a, 0.0) for a in h1 + h2 + h3)
    pi_vh = 7.4 + vh_charge * 0.5
    charge_ok  = 1.0 if -2 <= vh_charge <= 3 else max(0.0, 1.0 - 0.15 * abs(abs(vh_charge) - 2.5))
    agg_prone  = sum(1 for a in h3 if a in "FWIV") / max(1, len(h3))
    agg_ok     = max(0.0, 1.0 - 1.5 * max(0, agg_prone - 0.4))
    len_ok     = 1.0 if 8 <= len(h3) <= 18 else 0.7
    developability = (0.40 * charge_ok + 0.35 * agg_ok + 0.25 * len_ok)
    developability = max(0.0, min(1.0, developability))

    return affinity, developability, vh_charge, pi_vh, charge_comp, hbond_cap


def _score_payload(warhead_class: str, warhead_data: tuple,
                   linker_data: tuple, rng: random.Random) -> Tuple[float, float, float]:
    """Returns (warhead_potency, linker_score, off_target_penalty)."""
    _, _, potency, dar_range, _ = warhead_data
    _, _, cleavable, plasma_stab, tumour_rel, _ = linker_data

    # Warhead potency with small noise
    warpot = max(0.0, min(1.0, potency + rng.gauss(0, 0.02)))

    # Linker score: balance plasma stability (avoid premature release) with tumour release
    linker_score = 0.5 * plasma_stab + 0.5 * tumour_rel
    # Cleavable linkers preferred for ADCs (better therapeutic index)
    if cleavable:
        linker_score = min(1.0, linker_score + 0.05)

    # Off-target penalty — penalise ultra-high-potency warheads with non-selective linkers
    if potency > 0.95 and not cleavable:
        off_target = 0.4
    elif potency > 0.90 and not cleavable:
        off_target = 0.25
    else:
        off_target = max(0.0, 0.15 - 0.05 * int(cleavable))

    return warpot, linker_score, off_target


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _initial_population(size: int, rng: random.Random,
                        preferred_warhead: Optional[str]) -> List[tuple]:
    """Returns list of (h1,h2,h3,l1,l2,l3, warhead_idx, linker_idx)."""
    pop = []
    for i in range(size):
        h1 = _seed_cdr("CDR_H1", rng) if i < size // 2 else _random_cdr("CDR_H1", rng)
        h2 = _seed_cdr("CDR_H2", rng) if i < size // 2 else _random_cdr("CDR_H2", rng)
        h3 = _seed_cdr("CDR_H3", rng)
        l1 = _seed_cdr("CDR_L1", rng) if i < size // 2 else _random_cdr("CDR_L1", rng)
        l2 = _seed_cdr("CDR_L2", rng) if i < size // 2 else _random_cdr("CDR_L2", rng)
        l3 = _seed_cdr("CDR_L3", rng) if i < size // 2 else _random_cdr("CDR_L3", rng)

        # Warhead selection
        if preferred_warhead:
            wh_idx = next(
                (j for j, w in enumerate(WARHEAD_LIBRARY) if w[0].lower() == preferred_warhead.lower()),
                rng.randrange(len(WARHEAD_LIBRARY))
            )
        else:
            wh_idx = rng.randrange(len(WARHEAD_LIBRARY))

        lnk_idx = rng.randrange(len(LINKER_LIBRARY))
        pop.append((h1, h2, h3, l1, l2, l3, wh_idx, lnk_idx))
    return pop


def _mutate_candidate(cand: tuple, rng: random.Random, temperature: float,
                      preferred_warhead: Optional[str]) -> tuple:
    h1, h2, h3, l1, l2, l3, wh_idx, lnk_idx = cand

    cdr_weights = {"CDR_H3": 3.0, "CDR_H2": 1.5, "CDR_H1": 1.0,
                   "CDR_L3": 1.5, "CDR_L1": 1.0, "CDR_L2": 0.5}
    names  = list(cdr_weights.keys())
    wts    = [cdr_weights[c] for c in names]
    n_muts = max(1, int(temperature * 3))

    cdrs = {"CDR_H1": h1, "CDR_H2": h2, "CDR_H3": h3,
            "CDR_L1": l1, "CDR_L2": l2, "CDR_L3": l3}
    for _ in range(n_muts):
        cname = rng.choices(names, weights=wts, k=1)[0]
        cdrs[cname] = _mutate_cdr(cdrs[cname], cname, rng)

    # Occasionally mutate payload (less frequent than CDR)
    if rng.random() < 0.15 and preferred_warhead is None:
        wh_idx = rng.randrange(len(WARHEAD_LIBRARY))
    if rng.random() < 0.20:
        lnk_idx = rng.randrange(len(LINKER_LIBRARY))

    return (cdrs["CDR_H1"], cdrs["CDR_H2"], cdrs["CDR_H3"],
            cdrs["CDR_L1"], cdrs["CDR_L2"], cdrs["CDR_L3"], wh_idx, lnk_idx)


def _crossover(a: tuple, b: tuple, rng: random.Random) -> tuple:
    """CDR-level crossover, keep best payload."""
    keys = ["CDR_H1", "CDR_H2", "CDR_H3", "CDR_L1", "CDR_L2", "CDR_L3"]
    a_cdrs = dict(zip(keys, a[:6]))
    b_cdrs = dict(zip(keys, b[:6]))
    child_cdrs = {k: (a_cdrs[k] if rng.random() < 0.5 else b_cdrs[k]) for k in keys}
    wh_idx  = a[6] if rng.random() < 0.5 else b[6]
    lnk_idx = a[7] if rng.random() < 0.5 else b[7]
    return tuple(child_cdrs[k] for k in keys) + (wh_idx, lnk_idx)


def _evaluate(cand: tuple, epitope: ADCEpitope,
              gen: int, rng: random.Random, w_aff, w_dev, w_war, w_lnk) -> Optional[ADCCandidate]:
    h1, h2, h3, l1, l2, l3, wh_idx, lnk_idx = cand
    if not _is_valid_cdrs(h1, h2, h3, l1, l2, l3):
        return None

    warhead_data = WARHEAD_LIBRARY[wh_idx]
    linker_data  = LINKER_LIBRARY[lnk_idx]

    aff, dev, vh_charge, pi_vh, charge_comp, hbond_cap = _score_antibody(
        h1, h2, h3, l1, l2, l3, epitope)
    warpot, lnk_score, off_tgt = _score_payload(
        warhead_data[0], warhead_data, linker_data, rng)

    fitness = (w_aff * aff + w_dev * dev + w_war * warpot +
               w_lnk * lnk_score - 0.10 * off_tgt)
    fitness = max(0.0, fitness)

    return ADCCandidate(
        cdr_h1=h1, cdr_h2=h2, cdr_h3=h3, cdr_l1=l1, cdr_l2=l2, cdr_l3=l3,
        generation=gen,
        warhead_class=warhead_data[0],
        warhead_smiles=warhead_data[1],
        linker_name=linker_data[0],
        linker_smiles=linker_data[1],
        linker_cleavable=linker_data[2],
        dar_min=warhead_data[3][0],
        dar_max=warhead_data[3][1],
        affinity_score=round(aff, 4),
        developability=round(dev, 4),
        warhead_potency=round(warpot, 4),
        linker_score=round(lnk_score, 4),
        off_target_penalty=round(off_tgt, 4),
        fitness=round(fitness, 4),
        net_charge_vh=round(vh_charge, 2),
        pi_vh=round(pi_vh, 2),
        charge_comp=round(charge_comp, 4),
        hbond_capacity=hbond_cap,
        origin="seed" if gen == 1 else "evolved",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_adc_design(
    uniprot_id:         str,
    active_data:        Optional[dict] = None,
    physico_data:       Optional[dict] = None,
    ppi_data:           Optional[dict] = None,
    allosteric_data:    Optional[dict] = None,
    n_generations:      int            = MAX_GENERATIONS,
    epitope_mode:       str            = "auto",
    preferred_warhead:  Optional[str]  = None,
    rng_seed:           Optional[int]  = None,
    force:              bool           = False,
) -> ADCResult:
    """
    Run evolutionary ADC design.

    Skips (returns cached) if output file already exists and force=False.
    """
    t0   = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    rng  = random.Random(seed)
    np.random.seed(seed % (2**32))

    uid      = uniprot_id.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uid}_adc.json"

    # ── Skip if already done ──────────────────────────────────────────────────
    if out_path.exists() and not force:
        log.info(f"  [ADC] Cached result found: {out_path} — skipping. Use force=True to re-run.")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        result = ADCResult(uniprot_id=uid)
        result.__dict__.update({k: v for k, v in data.items() if k in result.__dict__})
        return result

    log.info(f"══ Module 18: ADC Design: {uid} [seed={seed}] ══")

    # Gene name
    target_gene = uid
    struct_json = inter_dir / f"{uid}_structure.json"
    if struct_json.exists():
        try:
            target_gene = json.loads(struct_json.read_text()).get("gene_name", uid)
        except Exception:
            pass

    log.info(f"  Target: {uid} ({target_gene})")

    # Epitope
    epitope = _select_adc_epitope(uid, active_data, physico_data,
                                  ppi_data, allosteric_data, epitope_mode)
    if epitope is None:
        epitope = ADCEpitope(source="generic", residue_numbers=[], residue_letters=[],
                             mean_sasa=100.0, net_charge=0.0, mean_hydrophobicity=0.0,
                             surface_area=800.0, n_hbond_donors=5, n_hbond_acceptors=5)

    log.info(f"  Epitope: {len(epitope.residue_numbers)} residues  "
             f"SASA={epitope.surface_area:.0f}Å²  ({epitope.source})")
    if preferred_warhead:
        log.info(f"  Preferred warhead: {preferred_warhead}")

    # Population
    population = _initial_population(POP_SIZE, rng, preferred_warhead)
    hall_of_fame: List[ADCCandidate] = []
    gen_stats: List[dict] = []
    n_evaluated = 0
    best_fitness = 0.0
    stagnation = 0
    seen_keys: set = set()

    log.info(f"  Starting evolution: {n_generations} generations × {POP_SIZE} candidates")

    for gen in range(1, n_generations + 1):
        t = min(1.0, (gen - 1) / max(1, n_generations - 1))
        w_aff = W_AFF_START + t * (W_AFF_END - W_AFF_START)
        w_dev = W_DEV_START + t * (W_DEV_END - W_DEV_START)
        w_war = W_WAR_START + t * (W_WAR_END - W_WAR_START)
        w_lnk = W_LNK_START + t * (W_LNK_END - W_LNK_START)
        total = w_aff + w_dev + w_war + w_lnk
        w_aff /= total; w_dev /= total; w_war /= total; w_lnk /= total
        temperature = max(0.1, 1.0 - 0.8 * t)

        # Evaluate
        evaluated = []
        for cand in population:
            c = _evaluate(cand, epitope, gen, rng, w_aff, w_dev, w_war, w_lnk)
            if c is not None:
                evaluated.append((cand, c))
        n_evaluated += len(evaluated)

        if not evaluated:
            continue

        evaluated.sort(key=lambda x: x[1].fitness, reverse=True)

        # Hall of fame
        for _, cand in evaluated:
            key = _cdr_string(cand.cdr_h1, cand.cdr_h2, cand.cdr_h3,
                              cand.cdr_l1, cand.cdr_l2, cand.cdr_l3) + f"|{cand.warhead_class}|{cand.linker_name}"
            if key not in seen_keys:
                hall_of_fame.append(cand)
                seen_keys.add(key)

        gen_best = evaluated[0][1].fitness
        improved = gen_best > best_fitness
        if improved:
            best_fitness = gen_best
            stagnation = 0
        else:
            stagnation += 1

        fitnesses = [c.fitness for _, c in evaluated]
        gen_stats.append({
            "gen": gen, "best_fitness": round(gen_best, 4),
            "mean_fitness": round(statistics.mean(fitnesses), 4),
            "stagnation": stagnation,
        })

        if gen % 10 == 0:
            log.info(f"  Gen {gen}/{n_generations}: best={gen_best:.4f}  "
                     f"stagnant={stagnation}")

        # Next generation
        next_pop = []
        seen_next: set = set()

        def _add(c):
            k = str(c)
            if k not in seen_next:
                next_pop.append(c); seen_next.add(k); return True
            return False

        # Elitism
        for raw, _ in evaluated[:ELITISM]:
            _add(raw)

        # Crossover
        top_raw = [r for r, _ in evaluated[:10]]
        for _ in range(POP_SIZE // 3):
            if len(top_raw) >= 2:
                p1, p2 = rng.sample(top_raw, 2)
                _add(_crossover(p1, p2, rng))

        # Mutation
        for raw, _ in evaluated[:POP_SIZE // 2]:
            _add(_mutate_candidate(raw, rng, temperature, preferred_warhead))

        # Hard stagnation reset
        if stagnation >= STAGNATION_HARD:
            log.info(f"  [ADC] Hard stagnation reset at gen {gen}")
            stagnation = 0
            next_pop = [evaluated[0][0]]  # keep only the best
            next_pop += _initial_population(POP_SIZE - 1, rng, preferred_warhead)
        else:
            # Fill remainder
            while len(next_pop) < POP_SIZE:
                _add(_initial_population(1, rng, preferred_warhead)[0])

        population = next_pop[:POP_SIZE]

    # Final selection
    hall_of_fame.sort(key=lambda c: c.fitness, reverse=True)
    seen_h3: set = set()
    top_candidates: List[ADCCandidate] = []
    for c in hall_of_fame:
        key = f"{c.cdr_h3}|{c.warhead_class}"
        if key not in seen_h3:
            top_candidates.append(c)
            seen_h3.add(key)
        if len(top_candidates) >= TOP_FOR_FINAL:
            break

    result = ADCResult(
        uniprot_id=uid,
        target_gene=target_gene,
        epitope=epitope,
        epitope_source=epitope.source,
        n_generations=n_generations,
        n_evaluated=n_evaluated,
        top_candidates=top_candidates,
        best_fitness=best_fitness,
        best_cdr_h3=top_candidates[0].cdr_h3 if top_candidates else "",
        best_warhead=top_candidates[0].warhead_class if top_candidates else "",
        best_linker=top_candidates[0].linker_name if top_candidates else "",
        generation_stats=gen_stats,
        notes=f"seed={seed}  epitope={epitope.source}  runtime={time.time()-t0:.1f}s",
    )

    result.to_json(out_path)
    log.info(result.summary())
    log.info(f"\n  Results saved: {out_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

import click

@click.command()
@click.option("--uniprot",    "-u", required=True, help="UniProt ID (e.g. P04637)")
@click.option("--generations","-g", default=MAX_GENERATIONS, type=int,
              help=f"Evolution generations (default: {MAX_GENERATIONS})")
@click.option("--epitope-mode","-e", default="auto",
              type=click.Choice(["auto","active","ppi","surface","allosteric"]),
              help="Epitope selection strategy")
@click.option("--warhead",    "-w", default=None,
              type=click.Choice([w[0] for w in WARHEAD_LIBRARY], case_sensitive=False),
              help="Fix warhead class (default: co-evolve)")
@click.option("--seed",       "-s", default=None, type=int, help="Random seed")
@click.option("--force",      "-f", is_flag=True, default=False,
              help="Re-run even if output already exists")
def main(uniprot, generations, epitope_mode, warhead, seed, force):
    """
    Module 18 — De Novo Antibody-Drug Conjugate (ADC) Design.

    Co-evolves antibody CDR sequences, linker chemistry, and cytotoxic warhead.
    Skips automatically if output already exists (use --force to re-run).

    Examples:
        python pipeline\\adc_design.py --uniprot P04637
        python pipeline\\adc_design.py --uniprot P04637 --warhead MMAE --generations 80
        python pipeline\\adc_design.py --uniprot P04637 --epitope-mode ppi
    """
    uid = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    def _load(f):
        p = inter_dir / f
        return json.loads(p.read_text()) if p.exists() else None

    result = run_adc_design(
        uniprot_id=uid,
        active_data=_load(f"{uid}_active_sites.json"),
        physico_data=_load(f"{uid}_physicochemical.json"),
        ppi_data=_load(f"{uid}_ppi.json"),
        allosteric_data=_load(f"{uid}_allosteric.json"),
        n_generations=generations,
        epitope_mode=epitope_mode,
        preferred_warhead=warhead,
        rng_seed=seed,
        force=force,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()