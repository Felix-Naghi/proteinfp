"""
pipeline/protac_design.py
──────────────────────────
Module 20 — De Novo PROTAC / Protein Degrader Design

Evolves bifunctional PROTAC molecules: POI warhead + linker + E3 ligase ligand.
Skips if data/intermediate/{uid}_protac.json already exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A PROTAC?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTAC = Proteolysis TArgeting Chimera

  [POI Warhead]──[Linker]──[E3 Ligase Ligand]
       ↓              ↓              ↓
  binds target  flexible join   recruits E3 ligase
  protein (POI)               → ubiquitinates POI
                               → proteasomal degradation

Why better than inhibitors for some targets:
  - Catalytic mechanism: one PROTAC molecule can degrade many POI copies
  - Removes ALL functions of the protein (not just catalytic)
  - Overcomes resistance mutations in the binding pocket
  - Especially effective for epigenetic regulators (BRD4, EZH2, KRAS)

Key design parameters:
  POI warhead     — binds the target protein pocket (evolved from denovo_design fragments)
  Linker          — length and rigidity affect ternary complex geometry
  E3 ligase ligand — recruits the E3 to the ternary complex (CRBN, VHL, IAP, MDM2)
  Hook effect     — PROTAC at high concentration can form unproductive binary complexes

This module co-evolves all three components:
  - POI warhead SMILES (fragment-based, drug-like, targets the binding pocket)
  - Linker (length 2-8 PEG units or alkyl, rigid vs flexible)
  - E3 ligase ligand (thalidomide/pomalidomide for CRBN, VH032 for VHL, etc.)

FITNESS FUNCTION:
  poi_affinity     — warhead binding to POI pocket (ADMET-filtered)
  e3_affinity      — E3 ligand binding to E3 enzyme
  linker_score     — length/rigidity score for ternary complex geometry
  dc50_proxy       — predicted DC50 from cooperativity model
  dmax_proxy       — predicted Dmax (max degradation) from Kd ratio
  hook_penalty     — penalises warheads with very high affinities (hook effect)
  admet_pass       — Lipinski-adjacent rules for PROTACs (MW up to 1000 Da)
  composite        — adaptive weighted fitness

SKIP LOGIC:
  Skips if {uid}_protac.json exists and force=False.

OUTPUT:
  data/intermediate/{uid}_protac.json

Usage (standalone):
    python pipeline/protac_design.py --uniprot P04637
    python pipeline/protac_design.py --uniprot P04637 --e3 CRBN --generations 60
    python pipeline/protac_design.py --uniprot P04637 --e3 VHL --linker-type PEG

Usage (from orchestrator):
    from pipeline.protac_design import run_protac_design
    result = run_protac_design("P04637", pocket_data, active_data)
"""

from __future__ import annotations

import json
import math
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

MAX_GENERATIONS  = 50
POP_SIZE         = 30
ELITISM          = 5
TOP_FOR_FINAL    = 8
STAGNATION_HARD  = 10

# PROTAC MW can be up to ~1000 Da — relaxed Lipinski
PROTAC_MW_MAX    = 1000.0
PROTAC_LOGP_MAX  = 7.0
PROTAC_HBD_MAX   = 10
PROTAC_HBA_MAX   = 20

# Fitness weights
W_POI_START = 0.30;  W_POI_END = 0.40
W_E3_START  = 0.20;  W_E3_END  = 0.20
W_LNK_START = 0.20;  W_LNK_END = 0.15
W_DEG_START = 0.20;  W_DEG_END = 0.20
W_ADM_START = 0.10;  W_ADM_END = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# E3 ligase ligands: (name, smiles, e3_kd_nM, selectivity_score, notes)
E3_LIGANDS = {
    "CRBN": [
        ("Thalidomide",  "O=C1CCC(=O)N1C1CC(=O)Nc2ccccc21",
                         320.0, 0.85,
                         "Original CRBN binder — thalidomide (teratogenic, use analogue)"),
        ("Pomalidomide", "O=C1CCC(=O)N1C1CC(=O)Nc2ccc(N)cc21",
                         80.0, 0.90,
                         "Pomalidomide — stronger CRBN binder, clinical PROTAC use"),
        ("Lenalidomide", "O=C1CCC(=O)N1C1CC(=O)Nc2ccc(CN)cc21",
                         150.0, 0.88,
                         "Lenalidomide analogue — FDA-approved IMiD"),
        ("dBET_CRBN",    "O=C1CCC(=O)N1C1CC(=O)Nc2ccccc21",
                         100.0, 0.92,
                         "Optimised CRBN ligand for dBET series"),
    ],
    "VHL": [
        ("VH032",        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]"
                         "(C(C)(C)O)C(=O)O",
                         185.0, 0.88,
                         "VH032 — first VHL ligand for PROTACs"),
        ("VH298",        "CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]"
                         "(C(O)(C)C)C(=O)O",
                         90.0, 0.92,
                         "VH298 — improved VHL affinity"),
        ("VHL_ligand_1", "CC1(C)CC(=O)N[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
                         250.0, 0.85,
                         "Alternative VHL recruiter"),
    ],
    "IAP": [
        ("LCL161",       "CC(C)CC(NC(=O)c1cc2c(CN(C)C(=O)c3noc(C(F)(F)F)n3)cccc2[nH]1)"
                         "C(=O)NC1CCCCC1",
                         20.0, 0.80,
                         "LCL161 — cIAP1/2 binder for PROTAC IAP recruitment"),
        ("GDC-0152",     "CC1(C)c2cccc(NC(=O)c3cc4ccccc4[nH]3)c2N(CC(N)=O)C1=O",
                         14.0, 0.78,
                         "GDC-0152 — potent IAP antagonist"),
    ],
    "MDM2": [
        ("Nutlin-3",     "COc1ccc(-c2nc3n(C(C)(C)C)c(=O)n(CC(=O)O)c3c(=O)n2-c2ccc(Cl)cc2Cl)cc1",
                         90.0, 0.90,
                         "Nutlin-3a — MDM2 inhibitor, frees p53"),
        ("MI-773",       "CC(C)(C)c1cc2c(cc1=O)CC1(c3ccc(Cl)cc3Cl)C(=O)N(CC(=O)O)c1n2Cc1ccc(OC)cc1",
                         5.0, 0.92,
                         "MI-773 — high-affinity MDM2 ligand"),
    ],
}

# POI warhead seed fragments (drug-like, suitable for pocket binding)
# Same fragments as denovo_design but biased toward known kinase/BET inhibitor cores
POI_WARHEAD_SEEDS = [
    # Kinase hinge-binding scaffolds
    "c1ccc2[nH]ccc2c1",          # indole
    "c1cnc2ccccc2n1",             # benzimidazole
    "c1ccc2ncncc2c1",             # purine-like
    "c1cnc2[nH]ccc2n1",          # 7-azaindole
    "Cc1cc2ccncc2[nH]1",         # 4-methylpyrrolo[3,2-b]pyridine
    # BET/bromodomain scaffolds
    "Cc1sc2cc(Cl)ccc2c1C",       # thienopyridine
    "c1ccc2c(c1)noc2C",          # benzisoxazole
    "Cc1cn2ccnc2n1",             # imidazotriazine
    # General privileged fragments
    "O=C1CCc2ccccc21",           # tetralone
    "c1ccc2c(c1)CCCO2",          # chromane
    "CC(=O)Nc1ccc(O)cc1",        # APAP-like
    "c1ccc(-c2ccccn2)cc1",       # biphenyl-pyridine
    "NC(=O)c1cccnc1",            # nicotinamide
]

# Linker building blocks
LINKER_TYPES = [
    # (name, smiles, length_atoms, flexibility, notes)
    ("PEG2",    "OCCOCCO",     6, 0.95, "PEG dimer — most flexible, good cell permeability"),
    ("PEG3",    "OCCOCCOCCO", 9, 0.95, "PEG trimer — optimal for many PROTACs"),
    ("PEG4",    "OCCOCCOCCOCCO", 12, 0.93, "PEG tetramer — longer reach"),
    ("Alkyl3",  "OCCCCO",     5, 0.70, "3-carbon alkyl — semi-rigid"),
    ("Alkyl4",  "OCCCCCO",    7, 0.68, "4-carbon alkyl"),
    ("Alkyl6",  "OCCCCCCCCO", 10, 0.65, "6-carbon alkyl — rigid-ish"),
    ("Piperaz", "OCC1CCNCC1CCO", 9, 0.60, "Piperazine linker — good solubility"),
    ("Mixed1",  "OCCOCCCCO",  9, 0.80, "PEG-alkyl hybrid"),
    ("Mixed2",  "OCCNCCNCCCO", 10, 0.75, "Amine-containing — improves solubility"),
    ("Rigid1",  "OCc1cccc(CO)c1", 8, 0.30, "Benzene — rigid, for pre-organised ternary complex"),
]

# Warhead fragment mutation operators (simplified SMILES fragment pool)
WARHEAD_FRAGMENTS = [
    "C", "CC", "CCC", "N", "O", "F", "Cl",
    "C(=O)N", "C(=O)O", "NC(=O)", "C#N",
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1cc[nH]n1",
    "C1CCNCC1", "C1CCOCC1", "C1CCCC1",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PROTACCandidate:
    # Components
    warhead_smiles:     str
    linker_name:        str
    linker_smiles:      str
    e3_name:            str
    e3_ligand_name:     str
    e3_ligand_smiles:   str
    generation:         int
    # Scores
    poi_affinity:       float   # warhead binding to POI
    e3_affinity:        float   # E3 ligand binding
    linker_score:       float   # geometry/length score
    dc50_proxy:         float   # 0-1 (1=best predicted DC50)
    dmax_proxy:         float   # 0-1 (1=complete degradation)
    hook_penalty:       float   # 0-1 (1=no hook effect)
    admet_pass:         bool
    fitness:            float
    estimated_mw:       float
    origin:             str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, rank: int) -> str:
        # Convert DC50 score back to nM for display: score = 1 - log10(nM)/3
        # → log10(nM) = (1 - score) * 3 → nM = 10^((1-score)*3)
        try:
            dc50_nM = 10 ** ((1.0 - self.dc50_proxy) * 3.0)
            if dc50_nM < 0.001:
                dc50_str = f"{dc50_nM*1e6:.0f}fM"
            elif dc50_nM < 1.0:
                dc50_str = f"{dc50_nM*1000:.0f}pM"
            elif dc50_nM < 1000:
                dc50_str = f"{dc50_nM:.1f}nM"
            else:
                dc50_str = f"{dc50_nM/1000:.1f}µM"
        except Exception:
            dc50_str = f"{self.dc50_proxy:.3f}"
        dmax_pct = f"{self.dmax_proxy * 100:.0f}%"
        return (
            f"  #{rank:<2}  poi={self.poi_affinity:.3f}  "
            f"e3={self.e3_affinity:.3f}  "
            f"DC50~{dc50_str}  "
            f"Dmax~{dmax_pct}  "
            f"e3={self.e3_name}/{self.e3_ligand_name}  "
            f"linker={self.linker_name}  "
            f"MW~{self.estimated_mw:.0f}"
        )


@dataclass
class PROTACResult:
    uniprot_id:         str
    target_gene:        str          = ""
    pocket_used:        str          = ""
    n_generations:      int          = 0
    n_evaluated:        int          = 0
    top_candidates:     List[PROTACCandidate] = field(default_factory=list)
    best_fitness:       float        = 0.0
    best_e3:            str          = ""
    best_linker:        str          = ""
    generation_stats:   List[dict]   = field(default_factory=list)
    notes:              str          = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self

    def summary(self) -> str:
        lines = [
            f"\n{'═'*72}",
            f"  MODULE 20 — PROTAC Design: {self.uniprot_id} ({self.target_gene})",
            f"{'═'*72}",
            f"  Pocket used      : {self.pocket_used}",
            f"  Generations run  : {self.n_generations}",
            f"  Candidates eval. : {self.n_evaluated}",
            f"  Best fitness     : {self.best_fitness:.4f}",
            f"  Best E3 ligase   : {self.best_e3}",
            f"  Best linker      : {self.best_linker}",
            f"\n  Top PROTAC Candidates:",
        ]
        for i, c in enumerate(self.top_candidates[:5], 1):
            lines.append(c.summary_line(i))
        lines.append(f"{'═'*72}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_mw(smiles: str) -> float:
    """
    Improved MW estimate from SMILES — counts both upper and lowercase atoms.
    Lowercase letters in SMILES are aromatic atoms (same mass as uppercase).
    """
    # Count by walking the string, handling two-char symbols first
    atom_mass = {
        "C": 12.0, "N": 14.0, "O": 16.0, "F": 19.0,
        "S": 32.0, "P": 31.0, "Cl": 35.5, "Br": 80.0, "I": 127.0,
    }
    mw = 0.0
    i  = 0
    s  = smiles.upper()   # normalise — aromatic c/n/o same mass as C/N/O
    while i < len(s):
        two = s[i:i+2]
        if two in ("CL", "BR"):
            mw += atom_mass[two.title()]; i += 2; continue
        one = s[i]
        if one in atom_mass:
            mw += atom_mass[one]
        i += 1
    # H contribution: PROTAC-sized molecules have roughly 1 H per heavy atom
    # Use 1.08 multiplier (slightly above the 1.0 heavy-atom baseline)
    return mw * 1.08


def _smiles_heavy_atoms(smiles: str) -> int:
    """Count heavy atoms — both aromatic (lowercase) and aliphatic (uppercase)."""
    return sum(1 for c in smiles if c.isalpha() and c.upper() in
               "CNOSPFIBR")


def _admet_ok(warhead: str, linker: str, e3_lig: str) -> Tuple[bool, float]:
    """
    Check PROTAC ADMET.
    PROTACs are 'beyond rule of 5' (bRo5): MW 700-1100 Da is normal and good.
    We reward the 600-1100 Da range, penalise below 400 (too small — not a real
    PROTAC, just a fragment) or above 1200 (too large for cell penetration).
    """
    combined = warhead + linker + e3_lig
    mw = _estimate_mw(combined)
    hbd = combined.upper().count("N") + combined.upper().count("O")

    # MW range: real PROTACs are 700–1100 Da
    mw_ok    = 400 < mw < 1200
    hbd_ok   = hbd < PROTAC_HBD_MAX
    passes   = mw_ok and hbd_ok

    return passes, mw


def _mutate_warhead(smiles: str, rng: random.Random) -> str:
    """Simple SMILES mutation by appending or substituting fragments."""
    if rng.random() < 0.50 or len(smiles) < 5:
        # Grow
        frag = rng.choice(WARHEAD_FRAGMENTS)
        return smiles + frag
    elif rng.random() < 0.70 and len(smiles) > 10:
        # Truncate
        cut = rng.randint(len(smiles)//2, len(smiles)-2)
        return smiles[:cut]
    else:
        # Swap seed
        return rng.choice(POI_WARHEAD_SEEDS)


def _score_warhead(smiles: str, pocket_data: Optional[dict], rng: random.Random) -> float:
    """
    Score warhead binding to POI pocket.
    For PROTACs the warhead should be 200-500 Da (leaves room for linker + E3 ligand
    to reach the 700-1100 Da total target). Rewards aromatic scaffolds for pocket burial.
    """
    mw       = _estimate_mw(smiles)
    n_rings  = smiles.lower().count("c1") + smiles.count("C1")   # rough ring count
    n_aromat = sum(1 for c in smiles if c.islower() and c.isalpha())  # aromatic atoms
    n_polar  = smiles.upper().count("N") + smiles.upper().count("O")

    # PROTAC warhead MW target: 200–500 Da (the full PROTAC will be 700-1100)
    if 200 <= mw <= 500:
        mw_score = 1.0
    elif 150 <= mw < 200 or 500 < mw <= 600:
        mw_score = 0.75
    elif mw < 150:
        mw_score = 0.4    # too small — fragment only
    else:
        mw_score = 0.5    # too large — linker/E3 will push total over 1200 Da

    # Aromatic ring score — warheads need to bury into hydrophobic pockets
    ring_score  = min(1.0, n_aromat / 10.0)
    polar_score = min(1.0, n_polar / 4.0) if n_polar > 0 else 0.2

    # Pocket polarity bias
    if pocket_data:
        pockets = pocket_data.get("pockets", [])
        if pockets:
            hydrophobic = float(pockets[0].get("mean_hydrophobicity", 0.5))
            if hydrophobic > 0.5:
                ring_score  = min(1.0, ring_score  * 1.3)
            else:
                polar_score = min(1.0, polar_score * 1.3)

    score = 0.40*mw_score + 0.35*ring_score + 0.25*polar_score
    return max(0.0, min(1.0, score + rng.gauss(0, 0.04)))


def _score_e3_ligand(e3_data: tuple) -> float:
    """Normalise E3 Kd to 0-1 affinity score."""
    _, _, kd_nM, selectivity, _ = e3_data
    # Lower Kd → higher score; normalise at 500 nM as poor, 10 nM as excellent
    kd_score = max(0.0, min(1.0, 1.0 - (kd_nM - 10) / 490))
    return 0.6 * kd_score + 0.4 * selectivity


def _score_linker(lnk_data: tuple, warhead_ha: int, e3_ha: int) -> float:
    """Score linker geometry for ternary complex formation."""
    _, _, length_atoms, flexibility, _ = lnk_data
    # Optimal linker length is ~5-12 atoms for most CRBN/VHL PROTACs
    len_score = 1.0 if 5 <= length_atoms <= 12 else max(0.3, 1.0 - 0.08*abs(length_atoms-8))
    flex_score = flexibility
    return 0.5 * len_score + 0.5 * flex_score


def _dc50_dmax_model(poi_aff: float, e3_aff: float, linker: float) -> Tuple[float, float]:
    """
    Cooperative degradation model.

    DC50 (concentration for 50% degradation) — lower is better.
    We report a 0-1 score where 1 = excellent DC50 (< 10 nM range proxy).

    Real DC50 = Kd_POI × Kd_E3 / (α × [PROTAC])
    Proxy: convert 0-1 affinity scores to approximate Kd in µM, compute DC50_nM,
    normalise on a log scale (1 nM = 1.0, 1 µM = 0.0).

    Dmax (maximum degradation %) — higher is better.
    Driven by ternary complex stability = geometric mean of both affinities × linker.
    Typical values: 60-95%. We normalise to 0-1 where 0.85+ = excellent.
    """
    # Convert 0-1 affinity to approximate Kd in µM (0→100 µM, 1→0.001 µM)
    # log10(Kd) = 2 - 5×affinity  →  affinity 0.8 → Kd ~0.01 µM = 10 nM
    kd_poi_uM = 10 ** (2.0 - 5.0 * max(0.01, poi_aff))
    kd_e3_uM  = 10 ** (2.0 - 5.0 * max(0.01, e3_aff))

    # Cooperativity α: linker score 0-1 → α 0.5-10 (PEG linkers ≈ 2-5)
    alpha = 0.5 + 9.5 * linker

    # DC50 in nM = 1000 × Kd_POI_uM × Kd_E3_uM / alpha
    dc50_nM = max(0.01, 1000.0 * kd_poi_uM * kd_e3_uM / alpha)

    # Normalise DC50: 1 nM → 1.0, 10 nM → 0.85, 100 nM → 0.60, 1 µM → 0.25
    # log10(1) = 0, log10(1000) = 3 → score = 1 - log10(dc50_nM)/3
    dc50_score = max(0.0, min(1.0, 1.0 - math.log10(max(0.001, dc50_nM)) / 3.0))

    # Dmax: driven by ternary complex lifetime
    # Geometric mean of both affinities, boosted by linker quality
    # Typical real range: 60-95%; map to 0-1 as (value - 0.55) / 0.45
    ternary_stability = (poi_aff * e3_aff) ** 0.5 * (0.6 + 0.4 * linker)
    dmax_raw   = 0.55 + 0.45 * ternary_stability   # 0.55 to 1.0
    dmax_proxy = max(0.0, min(1.0, dmax_raw))

    return round(dc50_score, 4), round(dmax_proxy, 4)


def _hook_penalty(poi_aff: float, e3_aff: float) -> float:
    """
    Hook effect penalty: very high-affinity warheads fill POI without recruiting E3.
    At high concentrations, binary complex > ternary complex → poor degradation.
    """
    if poi_aff > 0.90:
        return max(0.0, 1.0 - (poi_aff - 0.90) * 5.0)
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE REPRESENTATION: (warhead_smiles, linker_idx, e3_name, e3_ligand_idx)
# ══════════════════════════════════════════════════════════════════════════════

def _make_candidate(rng: random.Random, preferred_e3: Optional[str]) -> tuple:
    warhead  = rng.choice(POI_WARHEAD_SEEDS)
    lnk_idx  = rng.randrange(len(LINKER_TYPES))
    if preferred_e3 and preferred_e3 in E3_LIGANDS:
        e3_name  = preferred_e3
    else:
        e3_name  = rng.choice(list(E3_LIGANDS.keys()))
    e3l_idx  = rng.randrange(len(E3_LIGANDS[e3_name]))
    return (warhead, lnk_idx, e3_name, e3l_idx)


def _mutate_candidate(cand: tuple, rng: random.Random,
                      temperature: float, preferred_e3: Optional[str]) -> tuple:
    warhead, lnk_idx, e3_name, e3l_idx = cand
    # Mutate warhead (most frequently)
    if rng.random() < max(0.5, temperature):
        warhead = _mutate_warhead(warhead, rng)
    # Mutate linker
    if rng.random() < 0.20:
        lnk_idx = rng.randrange(len(LINKER_TYPES))
    # Mutate E3
    if rng.random() < 0.10 and preferred_e3 is None:
        e3_name  = rng.choice(list(E3_LIGANDS.keys()))
        e3l_idx  = rng.randrange(len(E3_LIGANDS[e3_name]))
    elif rng.random() < 0.15:
        e3l_idx = rng.randrange(len(E3_LIGANDS[e3_name]))
    return (warhead, lnk_idx, e3_name, e3l_idx)


def _crossover(a: tuple, b: tuple, rng: random.Random) -> tuple:
    warhead  = a[0] if rng.random() < 0.5 else b[0]
    lnk_idx  = a[1] if rng.random() < 0.5 else b[1]
    e3_name  = a[2] if rng.random() < 0.5 else b[2]
    # E3 ligand index must be valid for the chosen E3
    e3l_idx  = rng.randrange(len(E3_LIGANDS[e3_name]))
    return (warhead, lnk_idx, e3_name, e3l_idx)


def _evaluate(cand: tuple, pocket_data: Optional[dict], gen: int, rng: random.Random,
              w_poi, w_e3, w_lnk, w_deg, w_adm) -> Optional[PROTACCandidate]:
    warhead, lnk_idx, e3_name, e3l_idx = cand
    lnk_data = LINKER_TYPES[lnk_idx]
    e3_data  = E3_LIGANDS[e3_name][e3l_idx]

    poi_aff  = _score_warhead(warhead, pocket_data, rng)
    e3_aff   = _score_e3_ligand(e3_data)
    lnk_scr  = _score_linker(lnk_data, _smiles_heavy_atoms(warhead),
                              _smiles_heavy_atoms(e3_data[1]))
    dc50, dmax = _dc50_dmax_model(poi_aff, e3_aff, lnk_scr)
    hook_pen  = _hook_penalty(poi_aff, e3_aff)

    admet_ok, mw = _admet_ok(warhead, lnk_data[1], e3_data[1])
    admet_scr = 1.0 if admet_ok else 0.4

    fitness = (w_poi*poi_aff + w_e3*e3_aff + w_lnk*lnk_scr +
               w_deg*(dc50*0.5 + dmax*0.5) + w_adm*admet_scr) * hook_pen
    fitness = max(0.0, fitness)

    return PROTACCandidate(
        warhead_smiles=warhead,
        linker_name=lnk_data[0], linker_smiles=lnk_data[1],
        e3_name=e3_name, e3_ligand_name=e3_data[0], e3_ligand_smiles=e3_data[1],
        generation=gen,
        poi_affinity=round(poi_aff,4), e3_affinity=round(e3_aff,4),
        linker_score=round(lnk_scr,4), dc50_proxy=round(dc50,4),
        dmax_proxy=round(dmax,4), hook_penalty=round(hook_pen,4),
        admet_pass=admet_ok, fitness=round(fitness,4),
        estimated_mw=round(mw,1), origin="seed" if gen==1 else "evolved",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_protac_design(
    uniprot_id:     str,
    pocket_data:    Optional[dict] = None,
    active_data:    Optional[dict] = None,
    allosteric_data: Optional[dict] = None,
    n_generations:  int            = MAX_GENERATIONS,
    preferred_e3:   Optional[str]  = None,
    preferred_linker: Optional[str] = None,
    rng_seed:       Optional[int]  = None,
    force:          bool           = False,
) -> PROTACResult:
    """Run PROTAC evolutionary design. Skips if output exists and force=False."""
    t0   = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    rng  = random.Random(seed)
    np.random.seed(seed % (2**32))

    uid = uniprot_id.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uid}_protac.json"

    if out_path.exists() and not force:
        log.info(f"  [PROTAC] Cached: {out_path} — skipping.")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        result = PROTACResult(uniprot_id=uid)
        result.__dict__.update({k: v for k, v in data.items() if k in result.__dict__})
        return result

    log.info(f"══ Module 20: PROTAC Design: {uid} [seed={seed}] ══")

    target_gene = uid
    sj = inter_dir / f"{uid}_structure.json"
    if sj.exists():
        try: target_gene = json.loads(sj.read_text()).get("gene_name", uid)
        except: pass

    # Pocket
    pocket_id = "P1"
    if pocket_data:
        pockets = pocket_data.get("pockets", [])
        if pockets:
            pocket_id = pockets[0].get("pocket_id", "P1")

    log.info(f"  Target: {uid} ({target_gene})  Pocket: {pocket_id}")
    if preferred_e3:
        log.info(f"  Preferred E3: {preferred_e3}")

    # Override linker if specified
    preferred_lnk_idx = None
    if preferred_linker:
        preferred_lnk_idx = next(
            (i for i, l in enumerate(LINKER_TYPES) if l[0].lower() == preferred_linker.lower()),
            None
        )

    population = [_make_candidate(rng, preferred_e3) for _ in range(POP_SIZE)]
    # Force preferred linker into initial pop
    if preferred_lnk_idx is not None:
        population = [(w, preferred_lnk_idx, e, el) for w, _, e, el in population]

    hall_of_fame: List[PROTACCandidate] = []
    seen_keys: set = set()
    gen_stats: List[dict] = []
    n_evaluated = 0
    best_fitness = 0.0
    stagnation = 0

    for gen in range(1, n_generations + 1):
        t = min(1.0, (gen-1)/max(1,n_generations-1))
        w_poi = W_POI_START + t*(W_POI_END - W_POI_START)
        w_e3  = W_E3_START  + t*(W_E3_END  - W_E3_START)
        w_lnk = W_LNK_START + t*(W_LNK_END - W_LNK_START)
        w_deg = W_DEG_START + t*(W_DEG_END - W_DEG_START)
        w_adm = W_ADM_START + t*(W_ADM_END - W_ADM_START)
        total = w_poi+w_e3+w_lnk+w_deg+w_adm
        w_poi/=total; w_e3/=total; w_lnk/=total; w_deg/=total; w_adm/=total
        temperature = max(0.1, 1.0 - 0.8*t)

        evaluated = [(c, _evaluate(c, pocket_data, gen, rng, w_poi, w_e3, w_lnk, w_deg, w_adm))
                     for c in population]
        evaluated = [(c, ev) for c, ev in evaluated if ev is not None]
        n_evaluated += len(evaluated)
        if not evaluated:
            continue
        evaluated.sort(key=lambda x: x[1].fitness, reverse=True)

        for _, cand in evaluated:
            key = f"{cand.warhead_smiles[:20]}|{cand.e3_name}|{cand.linker_name}"
            if key not in seen_keys:
                hall_of_fame.append(cand)
                seen_keys.add(key)

        gen_best = evaluated[0][1].fitness
        improved = gen_best > best_fitness
        best_fitness = max(best_fitness, gen_best)
        stagnation = 0 if improved else stagnation + 1

        gen_stats.append({"gen": gen, "best_fitness": round(gen_best,4),
                          "stagnation": stagnation})
        if gen % 10 == 0:
            log.info(f"  Gen {gen}/{n_generations}: best={gen_best:.4f}")

        next_pop = []
        seen_n: set = set()
        def _add(c):
            k = str(c)
            if k not in seen_n:
                next_pop.append(c); seen_n.add(k)

        for raw, _ in evaluated[:ELITISM]:
            _add(raw)
        top_raw = [r for r, _ in evaluated[:10]]
        for _ in range(POP_SIZE // 3):
            if len(top_raw) >= 2:
                _add(_crossover(*rng.sample(top_raw, 2), rng))
        for raw, _ in evaluated[:POP_SIZE//2]:
            mut = _mutate_candidate(raw, rng, temperature, preferred_e3)
            if preferred_lnk_idx is not None:
                mut = (mut[0], preferred_lnk_idx, mut[2], mut[3])
            _add(mut)

        if stagnation >= STAGNATION_HARD:
            log.info(f"  [PROTAC] Hard reset at gen {gen}")
            stagnation = 0
            next_pop = [evaluated[0][0]]
            new = [_make_candidate(rng, preferred_e3) for _ in range(POP_SIZE-1)]
            if preferred_lnk_idx is not None:
                new = [(w, preferred_lnk_idx, e, el) for w, _, e, el in new]
            next_pop += new

        while len(next_pop) < POP_SIZE:
            _add(_make_candidate(rng, preferred_e3))
        population = next_pop[:POP_SIZE]

    hall_of_fame.sort(key=lambda c: c.fitness, reverse=True)
    seen_k: set = set()
    top_candidates = []
    for c in hall_of_fame:
        key = f"{c.e3_name}|{c.linker_name}|{c.warhead_smiles[:15]}"
        if key not in seen_k:
            top_candidates.append(c)
            seen_k.add(key)
        if len(top_candidates) >= TOP_FOR_FINAL:
            break

    result = PROTACResult(
        uniprot_id=uid, target_gene=target_gene, pocket_used=pocket_id,
        n_generations=n_generations, n_evaluated=n_evaluated,
        top_candidates=top_candidates, best_fitness=best_fitness,
        best_e3=top_candidates[0].e3_name if top_candidates else "",
        best_linker=top_candidates[0].linker_name if top_candidates else "",
        generation_stats=gen_stats,
        notes=f"seed={seed}  runtime={time.time()-t0:.1f}s",
    )
    result.to_json(out_path)
    log.info(result.summary())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

import click

@click.command()
@click.option("--uniprot",     "-u", required=True)
@click.option("--generations", "-g", default=MAX_GENERATIONS, type=int)
@click.option("--e3",          default=None,
              type=click.Choice(list(E3_LIGANDS.keys()), case_sensitive=False),
              help="Fix E3 ligase (default: co-evolve)")
@click.option("--linker-type", default=None,
              type=click.Choice([l[0] for l in LINKER_TYPES], case_sensitive=False),
              help="Fix linker type (default: co-evolve)")
@click.option("--seed",        "-s", default=None, type=int)
@click.option("--force",       "-f", is_flag=True, default=False)
def main(uniprot, generations, e3, linker_type, seed, force):
    """
    Module 20 — De Novo PROTAC / Protein Degrader Design.

    Co-evolves POI warhead, linker, and E3 ligase ligand.
    Skips if output already exists.

    Examples:
        python pipeline\\protac_design.py --uniprot P04637
        python pipeline\\protac_design.py --uniprot P04637 --e3 CRBN --linker-type PEG3
        python pipeline\\protac_design.py --uniprot P04637 --e3 VHL --generations 80
    """
    uid = uniprot.strip().upper()
    inter = Path(cfg.paths["intermediate"])

    def _load(f):
        p = inter / f
        return json.loads(p.read_text()) if p.exists() else None

    result = run_protac_design(
        uniprot_id=uid,
        pocket_data=_load(f"{uid}_binding_pockets.json"),
        active_data=_load(f"{uid}_active_sites.json"),
        allosteric_data=_load(f"{uid}_allosteric.json"),
        n_generations=generations, preferred_e3=e3,
        preferred_linker=linker_type, rng_seed=seed, force=force,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()