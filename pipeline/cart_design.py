"""
pipeline/cart_design.py
────────────────────────
Module 19 — De Novo CAR-T Cell Therapy Design

Evolves chimeric antigen receptor (CAR) construct components targeting a
surface-exposed antigen. Skips if data/intermediate/{uid}_cart.json already exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A CAR-T?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A CAR-T construct = antigen-binding domain + hinge + transmembrane + intracellular
signalling domains, expressed on patient T-cells.

  [scFv (CDR sequences)] ─ [Hinge] ─ [TM] ─ [Co-stimulatory] ─ [CD3ζ signal]
        ↓                      ↓         ↓            ↓
  antigen binding        flexibility  membrane    T-cell activation

GENERATIONS of CAR design:
  1st gen: CD3ζ only (poor persistence)
  2nd gen: CD28 or 4-1BB + CD3ζ (current standard — tisagenlecleucel, axicabtagene)
  3rd gen: CD28 + 4-1BB + CD3ζ (more potent, potential toxicity)
  4th gen: + cytokine payload (TRUCK / armoured CAR)

This module co-evolves:
  - scFv CDR sequences (H1/H2/H3/L1/L2/L3) — same evolution as antibody_design
  - CAR generation (1–4) with associated signalling domain choice
  - Co-stimulatory domain (CD28 vs 4-1BB vs OX40)
  - Hinge region (CD8α vs IgG1 vs CD28) affecting flexibility and CAR spacing

FITNESS FUNCTION:
  affinity_score      — scFv epitope binding
  developability      — antibody engineering quality
  activation_score    — predicted T-cell activation from signalling domain
  persistence_score   — predicted CAR-T persistence (4-1BB > CD28 for memory)
  cytokine_score      — IL-2/IFNγ release profile (2nd gen optimal)
  safety_score        — penalty for likely cytokine release syndrome risk
  composite_fitness   — adaptive weighted sum

SKIP LOGIC:
  If data/intermediate/{uid}_cart.json already exists and force=False, skips.

OUTPUT:
  data/intermediate/{uid}_cart.json

Usage (standalone):
    python pipeline/cart_design.py --uniprot P04637
    python pipeline/cart_design.py --uniprot P04637 --car-gen 2 --costim 4-1BB
    python pipeline/cart_design.py --uniprot P04637 --epitope-mode ppi --generations 80
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

MAX_GENERATIONS  = 50
POP_SIZE         = 30
ELITISM          = 5
TOP_FOR_FINAL    = 8
STAGNATION_HARD  = 10

# Fitness weights schedule
W_AFF_START = 0.35;  W_AFF_END = 0.45
W_DEV_START = 0.15;  W_DEV_END = 0.15
W_ACT_START = 0.25;  W_ACT_END = 0.20
W_PER_START = 0.15;  W_PER_END = 0.15
W_SAF_START = 0.10;  W_SAF_END = 0.05

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

CDR_SEEDS: Dict[str, List[str]] = {
    "CDR_H3": ["ARDYYGSGSYYFDY", "AKDSSSWYFDY", "ARGLGLVRGAMDY",
               "ARDQRSGYYFDY", "ARYGDYYGFAY", "AKDRWGGDAFDM"],
    "CDR_H1": ["GYTFTDYY", "GFSLTNYG", "GYTFTSYW", "GFTFSSYW"],
    "CDR_H2": ["INTYTGEPTYADSVKG", "IYPGDGDTRYSPSFQG"],
    "CDR_L1": ["SSSVSSYLY", "RASESVDNYGISFMN", "QASQDISNYLN"],
    "CDR_L2": ["DTSNLAS", "GASNRAT", "AASTLQS"],
    "CDR_L3": ["QQSYSTPLT", "QQRSNWPYT", "QQYYSYPLT"],
}

CHARGE_AT_PH7 = {"R": +1.0, "K": +1.0, "H": +0.1, "D": -1.0, "E": -1.0}

# CAR architecture options
# (name, activation, persistence, cytokine_risk, notes)
CAR_ARCHITECTURES = {
    1: ("1st_gen_CD3z",      0.55, 0.35, 0.20,
        "CD3ζ only — low persistence, rarely used now"),
    2: ("2nd_gen_CD28",      0.85, 0.65, 0.55,
        "CD28+CD3ζ — fast activation, moderate persistence (axicabtagene)"),
    3: ("2nd_gen_41BB",      0.80, 0.90, 0.35,
        "4-1BB+CD3ζ — slow but durable, better for lymphoma (tisagenlecleucel)"),
    4: ("3rd_gen_CD28_41BB", 0.92, 0.80, 0.70,
        "CD28+4-1BB+CD3ζ — high potency, elevated CRS risk"),
    5: ("4th_gen_TRUCK",     0.90, 0.85, 0.65,
        "4-1BB+CD3ζ+IL-12 payload — armoured CAR for solid tumours"),
}

# Hinge options: (name, flexibility, antigen_distance_ok, expression)
HINGE_OPTIONS = [
    ("CD8a",  0.80, 0.85, 0.90, "Most widely used, good for membrane-proximal epitopes"),
    ("IgG1",  0.70, 0.75, 0.85, "Rigid — better for distal epitopes"),
    ("CD28",  0.90, 0.90, 0.88, "Flexible — pairs naturally with CD28 TM"),
    ("IgG4",  0.75, 0.80, 0.87, "Reduced Fc receptor binding vs IgG1"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CARTEpitope:
    source:              str
    residue_numbers:     List[int]
    residue_letters:     List[str]
    mean_sasa:           float
    net_charge:          float
    mean_hydrophobicity: float
    surface_area:        float
    n_hbond_donors:      int
    n_hbond_acceptors:   int
    description:         str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CARTCandidate:
    """One evaluated CAR-T construct."""
    # scFv component
    cdr_h1:             str
    cdr_h2:             str
    cdr_h3:             str
    cdr_l1:             str
    cdr_l2:             str
    cdr_l3:             str
    generation:         int
    # CAR architecture
    car_gen:            int
    car_arch_name:      str
    hinge_name:         str
    # Scores
    affinity_score:     float
    developability:     float
    activation_score:   float
    persistence_score:  float
    safety_score:       float   # 1 = safe, 0 = dangerous
    fitness:            float
    # Sub-scores
    net_charge_vh:      float
    pi_vh:              float
    charge_comp:        float
    origin:             str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, rank: int) -> str:
        return (
            f"  #{rank:<2}  aff={self.affinity_score:.3f}  "
            f"act={self.activation_score:.2f}  "
            f"persist={self.persistence_score:.2f}  "
            f"safe={self.safety_score:.2f}  "
            f"arch={self.car_arch_name}  "
            f"hinge={self.hinge_name}  "
            f"H3={self.cdr_h3}"
        )


@dataclass
class CARTResult:
    uniprot_id:         str
    target_gene:        str          = ""
    epitope:            Optional[CARTEpitope] = None
    epitope_source:     str          = ""
    n_generations:      int          = 0
    n_evaluated:        int          = 0
    top_candidates:     List[CARTCandidate] = field(default_factory=list)
    best_fitness:       float        = 0.0
    best_cdr_h3:        str          = ""
    best_architecture:  str          = ""
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
            f"  MODULE 19 — CAR-T Design: {self.uniprot_id} ({self.target_gene})",
            f"{'═'*72}",
            f"  Epitope source   : {self.epitope_source}",
            f"  Generations run  : {self.n_generations}",
            f"  Candidates eval. : {self.n_evaluated}",
            f"  Best fitness     : {self.best_fitness:.4f}",
            f"  Best architecture: {self.best_architecture}",
            f"  Best CDR-H3      : {self.best_cdr_h3}",
            f"\n  Top CAR-T Constructs:",
        ]
        for i, c in enumerate(self.top_candidates[:5], 1):
            lines.append(c.summary_line(i))
        lines.append(f"{'═'*72}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _select_epitope(uid, active_data, physico_data, ppi_data,
                    allosteric_data, epitope_mode) -> CARTEpitope:
    try:
        from pipeline.antibody_design import select_epitope
        epi = select_epitope(uid, active_data, physico_data, ppi_data,
                             allosteric_data, epitope_mode)
        if epi:
            return CARTEpitope(
                source=epi.source, residue_numbers=epi.residue_numbers,
                residue_letters=epi.residue_letters, mean_sasa=epi.mean_sasa,
                net_charge=epi.net_charge, mean_hydrophobicity=epi.mean_hydrophobicity,
                surface_area=epi.surface_area, n_hbond_donors=epi.n_hbond_donors,
                n_hbond_acceptors=epi.n_hbond_acceptors, description=epi.description,
            )
    except Exception as e:
        log.warning(f"  Epitope import failed: {e}")
    return CARTEpitope(source="generic", residue_numbers=[], residue_letters=[],
                       mean_sasa=100.0, net_charge=0.0, mean_hydrophobicity=0.0,
                       surface_area=800.0, n_hbond_donors=5, n_hbond_acceptors=5)


def _random_cdr(name, rng):
    lo, hi = CDR_LENGTHS[name]
    return "".join(rng.choices(AAs, k=rng.randint(lo, hi)))


def _seed_cdr(name, rng):
    seeds = CDR_SEEDS.get(name, [])
    if seeds:
        base = list(rng.choice(seeds))
        for _ in range(rng.randint(0, min(2, len(base)))):
            base[rng.randrange(len(base))] = rng.choice(AAs)
        return "".join(base)
    return _random_cdr(name, rng)


def _is_valid(h1, h2, h3, l1, l2, l3):
    for nm, sq in zip(["CDR_H1","CDR_H2","CDR_H3","CDR_L1","CDR_L2","CDR_L3"],
                       [h1,h2,h3,l1,l2,l3]):
        lo, hi = CDR_LENGTHS[nm]
        if not (lo <= len(sq) <= hi) or not all(a in AAs for a in sq):
            return False
    return True


def _mutate_cdr(seq, name, rng):
    lo, hi = CDR_LENGTHS[name]
    seq = list(seq)
    r = rng.random()
    if r < 0.60 and seq:
        seq[rng.randrange(len(seq))] = rng.choice(AAs)
    elif r < 0.80 and len(seq) < hi:
        seq.insert(rng.randrange(len(seq)+1), rng.choice(AAs))
    elif r < 0.95 and len(seq) > lo:
        seq.pop(rng.randrange(len(seq)))
    else:
        cons = {"R":"K","K":"R","D":"E","E":"D","I":"V","V":"I","F":"Y","Y":"F"}
        pos = rng.randrange(len(seq))
        seq[pos] = cons.get(seq[pos], rng.choice(AAs))
    return "".join(seq)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_scfv(h1, h2, h3, l1, l2, l3, epi: CARTEpitope, rng):
    all_cdr = h1 + h2 + h3 + l1 + l2 + l3
    # Charge complementarity
    cdr_q = sum(CHARGE_AT_PH7.get(a, 0.0) for a in all_cdr)
    charge_comp = max(0.0, min(1.0, 0.5 - 0.1*(cdr_q + epi.net_charge)**2))
    # Hydrophobic burial
    h3_hydro = sum({"F":1,"W":1,"Y":0.8,"I":0.7,"L":0.7,"V":0.5}.get(a,0) for a in h3)
    hydro_m  = max(0.0, min(1.0, (h3_hydro/max(1,len(h3))) * epi.mean_hydrophobicity))
    # H-bond capacity
    hbond = sum(1 for a in all_cdr if a in "NQSTHY")
    hbond_n = min(hbond, epi.n_hbond_donors + epi.n_hbond_acceptors)
    # CDR-H3 length: for CAR-T, moderate length (10-18) is preferred to avoid
    # steric clashes at close cell-cell interface
    len_ok = 1.0 if 8 <= len(h3) <= 18 else 0.70
    aff = (0.35*charge_comp + 0.25*hydro_m +
           0.20*(hbond_n/max(1,epi.n_hbond_donors+epi.n_hbond_acceptors)) +
           0.20*len_ok)
    aff = max(0.0, min(1.0, aff + rng.gauss(0, 0.03)))

    vh_q = sum(CHARGE_AT_PH7.get(a, 0.0) for a in h1+h2+h3)
    pi_vh = 7.4 + vh_q * 0.5
    charg_ok = 1.0 if -2 <= vh_q <= 3 else max(0.0, 1.0 - 0.15*abs(abs(vh_q)-2.5))
    agg_p = sum(1 for a in h3 if a in "FWIV") / max(1, len(h3))
    agg_ok = max(0.0, 1.0 - 1.5*max(0, agg_p-0.4))
    dev = max(0.0, min(1.0, 0.50*charg_ok + 0.50*agg_ok))
    return aff, dev, vh_q, pi_vh, charge_comp


def _score_car_arch(car_gen: int, hinge_idx: int) -> Tuple[float, float, float]:
    """Returns (activation, persistence, safety)."""
    arch = CAR_ARCHITECTURES[car_gen]
    _, act, per, crs_risk, _ = arch
    hinge = HINGE_OPTIONS[hinge_idx]
    _, flex, dist_ok, expr, _ = hinge

    # Activation boosted by good hinge expression
    activation = min(1.0, act * (0.8 + 0.2*expr))
    # Persistence is an intrinsic property of the signalling domain
    persistence = per
    # Safety: penalise high CRS risk CAR gens, especially with poor distance-OK
    safety = max(0.0, (1.0 - crs_risk) * dist_ok)

    return activation, persistence, safety


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _make_candidate(rng, preferred_gen=None):
    h1 = _seed_cdr("CDR_H1", rng)
    h2 = _seed_cdr("CDR_H2", rng)
    h3 = _seed_cdr("CDR_H3", rng)
    l1 = _seed_cdr("CDR_L1", rng)
    l2 = _seed_cdr("CDR_L2", rng)
    l3 = _seed_cdr("CDR_L3", rng)
    car_gen = preferred_gen if preferred_gen else rng.choice(list(CAR_ARCHITECTURES.keys()))
    hinge   = rng.randrange(len(HINGE_OPTIONS))
    return (h1, h2, h3, l1, l2, l3, car_gen, hinge)


def _mutate(cand, rng, temp, preferred_gen=None):
    h1,h2,h3,l1,l2,l3,car_gen,hinge = cand
    weights = {"CDR_H3":3.0,"CDR_H2":1.5,"CDR_H1":1.0,"CDR_L3":1.5,"CDR_L1":1.0,"CDR_L2":0.5}
    names = list(weights.keys())
    wts   = [weights[n] for n in names]
    cdrs  = {"CDR_H1":h1,"CDR_H2":h2,"CDR_H3":h3,"CDR_L1":l1,"CDR_L2":l2,"CDR_L3":l3}
    for _ in range(max(1, int(temp*3))):
        nm = rng.choices(names, weights=wts, k=1)[0]
        cdrs[nm] = _mutate_cdr(cdrs[nm], nm, rng)
    if rng.random() < 0.12 and preferred_gen is None:
        car_gen = rng.choice(list(CAR_ARCHITECTURES.keys()))
    if rng.random() < 0.18:
        hinge = rng.randrange(len(HINGE_OPTIONS))
    return (cdrs["CDR_H1"],cdrs["CDR_H2"],cdrs["CDR_H3"],
            cdrs["CDR_L1"],cdrs["CDR_L2"],cdrs["CDR_L3"], car_gen, hinge)


def _crossover(a, b, rng):
    keys = ["CDR_H1","CDR_H2","CDR_H3","CDR_L1","CDR_L2","CDR_L3"]
    a_c = dict(zip(keys, a[:6])); b_c = dict(zip(keys, b[:6]))
    child = {k: a_c[k] if rng.random()<0.5 else b_c[k] for k in keys}
    return tuple(child[k] for k in keys) + (
        a[6] if rng.random()<0.5 else b[6],
        a[7] if rng.random()<0.5 else b[7],
    )


def _evaluate(cand, epi: CARTEpitope, gen, rng, w_aff, w_dev, w_act, w_per, w_saf):
    h1,h2,h3,l1,l2,l3,car_gen,hinge_idx = cand
    if not _is_valid(h1,h2,h3,l1,l2,l3):
        return None
    aff, dev, vh_q, pi_vh, charge_comp = _score_scfv(h1,h2,h3,l1,l2,l3,epi,rng)
    act, per, saf = _score_car_arch(car_gen, hinge_idx)
    fitness = (w_aff*aff + w_dev*dev + w_act*act + w_per*per + w_saf*saf)
    arch_name = CAR_ARCHITECTURES[car_gen][0]
    hinge_name = HINGE_OPTIONS[hinge_idx][0]
    return CARTCandidate(
        cdr_h1=h1, cdr_h2=h2, cdr_h3=h3, cdr_l1=l1, cdr_l2=l2, cdr_l3=l3,
        generation=gen, car_gen=car_gen, car_arch_name=arch_name,
        hinge_name=hinge_name,
        affinity_score=round(aff,4), developability=round(dev,4),
        activation_score=round(act,4), persistence_score=round(per,4),
        safety_score=round(saf,4), fitness=round(fitness,4),
        net_charge_vh=round(vh_q,2), pi_vh=round(pi_vh,2),
        charge_comp=round(charge_comp,4),
        origin="seed" if gen==1 else "evolved",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_cart_design(
    uniprot_id:      str,
    active_data:     Optional[dict] = None,
    physico_data:    Optional[dict] = None,
    ppi_data:        Optional[dict] = None,
    allosteric_data: Optional[dict] = None,
    n_generations:   int            = MAX_GENERATIONS,
    epitope_mode:    str            = "auto",
    preferred_gen:   Optional[int]  = None,
    rng_seed:        Optional[int]  = None,
    force:           bool           = False,
) -> CARTResult:
    """Run CAR-T evolutionary design. Skips if output exists and force=False."""
    t0   = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    rng  = random.Random(seed)
    np.random.seed(seed % (2**32))

    uid = uniprot_id.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uid}_cart.json"

    if out_path.exists() and not force:
        log.info(f"  [CAR-T] Cached result: {out_path} — skipping.")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        result = CARTResult(uniprot_id=uid)
        result.__dict__.update({k: v for k, v in data.items() if k in result.__dict__})
        return result

    log.info(f"══ Module 19: CAR-T Design: {uid} [seed={seed}] ══")

    target_gene = uid
    sj = inter_dir / f"{uid}_structure.json"
    if sj.exists():
        try: target_gene = json.loads(sj.read_text()).get("gene_name", uid)
        except: pass

    epitope = _select_epitope(uid, active_data, physico_data,
                              ppi_data, allosteric_data, epitope_mode)
    log.info(f"  Epitope: {len(epitope.residue_numbers)} residues  "
             f"source={epitope.source}  SASA={epitope.surface_area:.0f}Å²")

    population = [_make_candidate(rng, preferred_gen) for _ in range(POP_SIZE)]
    hall_of_fame: List[CARTCandidate] = []
    seen_keys: set = set()
    gen_stats: List[dict] = []
    n_evaluated = 0
    best_fitness = 0.0
    stagnation = 0

    for gen in range(1, n_generations + 1):
        t = min(1.0, (gen-1)/max(1,n_generations-1))
        w_aff = W_AFF_START + t*(W_AFF_END - W_AFF_START)
        w_dev = W_DEV_START + t*(W_DEV_END - W_DEV_START)
        w_act = W_ACT_START + t*(W_ACT_END - W_ACT_START)
        w_per = W_PER_START + t*(W_PER_END - W_PER_START)
        w_saf = W_SAF_START + t*(W_SAF_END - W_SAF_START)
        total = w_aff+w_dev+w_act+w_per+w_saf
        w_aff/=total; w_dev/=total; w_act/=total; w_per/=total; w_saf/=total
        temperature = max(0.1, 1.0 - 0.8*t)

        evaluated = [(c, _evaluate(c, epitope, gen, rng, w_aff, w_dev, w_act, w_per, w_saf))
                     for c in population]
        evaluated = [(c, ev) for c, ev in evaluated if ev is not None]
        n_evaluated += len(evaluated)
        if not evaluated:
            continue
        evaluated.sort(key=lambda x: x[1].fitness, reverse=True)

        for _, cand in evaluated:
            key = f"{cand.cdr_h3}|{cand.car_arch_name}"
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

        # Next generation
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
            _add(_mutate(raw, rng, temperature, preferred_gen))

        if stagnation >= STAGNATION_HARD:
            log.info(f"  [CAR-T] Hard reset at gen {gen}")
            stagnation = 0
            next_pop = [evaluated[0][0]]
            next_pop += [_make_candidate(rng, preferred_gen) for _ in range(POP_SIZE-1)]

        while len(next_pop) < POP_SIZE:
            _add(_make_candidate(rng, preferred_gen))
        population = next_pop[:POP_SIZE]

    hall_of_fame.sort(key=lambda c: c.fitness, reverse=True)
    seen_h3: set = set()
    top_candidates = []
    for c in hall_of_fame:
        key = f"{c.cdr_h3}|{c.car_arch_name}"
        if key not in seen_h3:
            top_candidates.append(c)
            seen_h3.add(key)
        if len(top_candidates) >= TOP_FOR_FINAL:
            break

    result = CARTResult(
        uniprot_id=uid, target_gene=target_gene, epitope=epitope,
        epitope_source=epitope.source, n_generations=n_generations,
        n_evaluated=n_evaluated, top_candidates=top_candidates,
        best_fitness=best_fitness,
        best_cdr_h3=top_candidates[0].cdr_h3 if top_candidates else "",
        best_architecture=top_candidates[0].car_arch_name if top_candidates else "",
        generation_stats=gen_stats,
        notes=f"seed={seed}  epitope={epitope.source}  runtime={time.time()-t0:.1f}s",
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
@click.option("--epitope-mode","-e", default="auto",
              type=click.Choice(["auto","active","ppi","surface","allosteric"]))
@click.option("--car-gen",     "-c", default=None,
              type=click.IntRange(1, 5),
              help="Fix CAR generation 1-5 (default: co-evolve)")
@click.option("--seed",        "-s", default=None, type=int)
@click.option("--force",       "-f", is_flag=True, default=False)
def main(uniprot, generations, epitope_mode, car_gen, seed, force):
    """
    Module 19 — De Novo CAR-T Cell Therapy Design.

    Co-evolves scFv CDR sequences, CAR generation, hinge region, and
    co-stimulatory domain. Skips if output already exists.

    Examples:
        python pipeline\\cart_design.py --uniprot P04637
        python pipeline\\cart_design.py --uniprot P04637 --car-gen 3
        python pipeline\\cart_design.py --uniprot P04637 --epitope-mode ppi --generations 80
    """
    uid = uniprot.strip().upper()
    inter = Path(cfg.paths["intermediate"])

    def _load(f):
        p = inter / f
        return json.loads(p.read_text()) if p.exists() else None

    result = run_cart_design(
        uniprot_id=uid,
        active_data=_load(f"{uid}_active_sites.json"),
        physico_data=_load(f"{uid}_physicochemical.json"),
        ppi_data=_load(f"{uid}_ppi.json"),
        allosteric_data=_load(f"{uid}_allosteric.json"),
        n_generations=generations, epitope_mode=epitope_mode,
        preferred_gen=car_gen, rng_seed=seed, force=force,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()