"""
pipeline/allosteric_drug_design.py
────────────────────────────────────
Module 21 — De Novo Allosteric Small Molecule Design

Evolves small molecules targeting allosteric sites predicted by Module 05.
Skips if data/intermediate/{uid}_allosteric_drug.json already exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AN ALLOSTERIC DRUG?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Allosteric drugs bind a site AWAY from the active site and modulate protein
function by inducing conformational changes that propagate through the protein.

Advantages over orthosteric inhibitors:
  - Can target proteins with flat/undruggable active sites (KRAS, p53, PPI)
  - Higher selectivity (allosteric sites are less conserved than active sites)
  - Can be activators OR inhibitors (bimodal pharmacology)
  - Less likely to be outcompeted by high substrate concentrations

Allosteric mechanisms:
  Type I  — locks protein in inactive/active conformation (conformational selection)
  Type II — displaces regulatory domain, removes autoinhibition
  Type III — binds interface of oligomer (e.g. PKM2 activators)
  Type IV  — binds intrinsically disordered regions

Key structural targets for allosteric drugs:
  - High ENM correlation sites (Module 05 output) — mechanically coupled to active site
  - Cryptic pockets (Module 18/CrypticPocket) that open upon allosteric binding
  - Oligomer interface pockets (for oligomeric enzymes)
  - Back pocket / DFG-out (kinases) — type II inhibitors

This module evolves small molecules against the TOP allosteric site from Module 05.
It uses the same fragment-based evolution as denovo_design.py but:
  - Seeds fragments with allosteric pharmacophores (wedge, hook, staple)
  - Fitness rewards conformational selectivity (not just potency)
  - Scores communication pathway disruption (key for allosteric efficacy)
  - No Vina required — scores pharmacophore match to allosteric site properties

FITNESS FUNCTION:
  site_complementarity  — shape/charge/hydro match to allosteric site
  communication_score   — predicted disruption of ENM communication pathway
  selectivity_score     — allosteric site uniqueness vs orthosteric site
  admet_score           — standard drug-like properties (Ro5)
  conformational_score  — predicted effect on protein conformational ensemble
  composite_fitness     — adaptive weighted sum

SKIP LOGIC:
  Skips if {uid}_allosteric_drug.json exists and force=False.

OUTPUT:
  data/intermediate/{uid}_allosteric_drug.json

Usage (standalone):
    python pipeline/allosteric_drug_design.py --uniprot P04637
    python pipeline/allosteric_drug_design.py --uniprot P04637 --site A2
    python pipeline/allosteric_drug_design.py --uniprot P04637 --mechanism inhibitor

Usage (from orchestrator):
    from pipeline.allosteric_drug_design import run_allosteric_drug_design
    result = run_allosteric_drug_design("P04637", allosteric_data, physico_data)
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
ELITISM          = 6
TOP_FOR_FINAL    = 10
STAGNATION_HARD  = 10
DIVERSITY_MIN    = 0.20

ADMET_MW_MAX     = 500.0
ADMET_LOGP_MAX   = 5.0
ADMET_HBD_MAX    = 5
ADMET_HBA_MAX    = 10

# Fitness weights (start → end)
W_SITE_START  = 0.30;  W_SITE_END  = 0.40
W_COMM_START  = 0.25;  W_COMM_END  = 0.20
W_SEL_START   = 0.20;  W_SEL_END   = 0.20
W_ADM_START   = 0.15;  W_ADM_END   = 0.10
W_CONF_START  = 0.10;  W_CONF_END  = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# FRAGMENT LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# Allosteric pharmacophore seeds — biased toward known allosteric scaffolds
ALLOSTERIC_SEEDS = [
    # Type II kinase inhibitors (bind DFG-out allosteric back pocket)
    "c1ccc(NC(=O)Nc2ccc(Cl)cn2)cc1",           # urea scaffold
    "c1ccc(-c2cc3ccccc3[nH]2)cn1",             # indole-pyridine
    "COc1ccc(-c2ccc(NC(=O)c3ccccc3)cc2)cc1",   # biphenyl-amide
    # Hydrophobic wedge (inserts into hinge/allosteric junction)
    "c1ccc2c(c1)cc1ccccc1n2",                   # acridine
    "c1ccc2c(c1)[nH]c1ccccc12",                 # carbazole
    "c1ccc2c(c1)cccc2-c1ccccn1",               # quinoline-phenyl
    # Ionic/polar hooks (anchor in charged allosteric pocket)
    "OC(=O)c1ccccc1NC(=O)c1ccccc1",            # anthranilic acid
    "NC(=O)c1ccc(S(N)(=O)=O)cc1",              # sulfonamide
    "OC(=O)CC1=CC=CC=C1",                       # phenylacetic acid
    # Fragment-like seeds (MW 150-250)
    "c1ccc2[nH]ccc2c1",                         # indole
    "Cc1ccc2[nH]ncc2c1",                        # methylindazole
    "c1cnc2ccccc2c1",                           # quinoline
    "O=C1CCc2ccccc21",                          # tetralone
    "CC(=O)Nc1ccc(O)cc1",                       # paracetamol-like
    "c1cc2ccccc2[nH]c1",                        # isoindole
    # Known allosteric drug scaffolds
    "CC1=C(C(=O)Nc2ccc(F)cc2)SC=N1",           # thiazole (similar to dasatinib allosteric)
    "Cc1cnc(NC(=O)c2ccc(Cl)cc2)s1",            # 4-methyl-thiazole
    "O=C(Nc1ccc(-c2ccccc2)cc1)c1ccncc1",       # isonicotinamide-biphenyl
]

GROW_FRAGMENTS = [
    "C", "CC", "CCC", "N", "O", "F", "Cl",
    "C(=O)N", "C(=O)O", "NC(=O)", "C#N",
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1cc[nH]n1",
    "C1CCCCC1", "C1CCNCC1", "C1CCOCC1",
    "OC", "NC", "SC", "CC(=O)", "CCO", "CCN",
]

# Allosteric mechanism types
MECHANISMS = ["inhibitor", "activator", "modulator"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlloSiteTarget:
    """Parsed allosteric site for scoring."""
    site_id:           str
    mean_correlation:  float
    mean_sasa:         float
    mean_hydrophobicity: float
    net_charge:        float
    n_residues:        int
    coupled_residues:  List[int]
    confidence:        str

    @classmethod
    def from_dict(cls, d: dict) -> "AlloSiteTarget":
        return cls(
            site_id=d.get("site_id", "A1"),
            mean_correlation=float(d.get("mean_correlation", 0.5)),
            mean_sasa=float(d.get("mean_sasa", 50.0)),
            mean_hydrophobicity=float(d.get("mean_hydrophobicity", 0.0)),
            net_charge=float(d.get("net_charge", 0.0)),
            n_residues=int(d.get("size", len(d.get("residue_numbers", [])))),
            coupled_residues=d.get("coupled_active_residues", []),
            confidence=d.get("confidence", "MEDIUM"),
        )


@dataclass
class AlloCandidate:
    smiles:               str
    generation:           int
    mechanism:            str     # inhibitor / activator / modulator
    site_id:              str
    # Scores
    site_complementarity: float   # shape/charge/hydro match to allosteric site
    communication_score:  float   # disruption of ENM communication pathway
    selectivity_score:    float   # allosteric vs orthosteric selectivity
    admet_score:          float
    conformational_score: float   # predicted conformational effect
    fitness:              float
    # Mol properties
    mw:                   float
    logp_proxy:           float
    n_rings:              int
    origin:               str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, rank: int) -> str:
        return (
            f"  #{rank:<2}  site={self.site_id}  "
            f"compl={self.site_complementarity:.3f}  "
            f"comm={self.communication_score:.3f}  "
            f"sel={self.selectivity_score:.3f}  "
            f"mech={self.mechanism}  "
            f"MW={self.mw:.0f}  "
            f"SMILES={self.smiles[:40]}"
        )


@dataclass
class AllodrugResult:
    uniprot_id:         str
    target_gene:        str          = ""
    target_site:        str          = ""
    site_correlation:   float        = 0.0
    n_generations:      int          = 0
    n_evaluated:        int          = 0
    top_candidates:     List[AlloCandidate] = field(default_factory=list)
    best_fitness:       float        = 0.0
    best_smiles:        str          = ""
    best_mechanism:     str          = ""
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
            f"  MODULE 21 — Allosteric Drug Design: {self.uniprot_id} ({self.target_gene})",
            f"{'═'*72}",
            f"  Target site      : {self.target_site}  "
            f"(ENM corr={self.site_correlation:.3f})",
            f"  Generations run  : {self.n_generations}",
            f"  Candidates eval. : {self.n_evaluated}",
            f"  Best fitness     : {self.best_fitness:.4f}",
            f"  Best mechanism   : {self.best_mechanism}",
            f"  Best SMILES      : {self.best_smiles[:60]}",
            f"\n  Top Allosteric Drug Candidates:",
        ]
        for i, c in enumerate(self.top_candidates[:5], 1):
            lines.append(c.summary_line(i))
        lines.append(f"{'═'*72}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MOLECULAR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_mw(smiles: str) -> float:
    atoms = {"C": 12, "N": 14, "O": 16, "F": 19, "Cl": 35.5, "S": 32}
    mw = 0.0; i = 0
    while i < len(smiles):
        if i < len(smiles)-1 and smiles[i:i+2] == "Cl":
            mw += atoms["Cl"]; i += 2; continue
        if smiles[i] in atoms:
            mw += atoms[smiles[i]]
        i += 1
    return mw * 1.15  # ~15% H contribution


def _logp_proxy(smiles: str) -> float:
    """Rough LogP from aromatic ring count and polar groups."""
    n_arom = smiles.lower().count("c") // 4
    n_polar = smiles.count("N") + smiles.count("O")
    n_halog = smiles.count("F") + smiles.count("Cl")
    return float(max(-2, n_arom - n_polar * 0.7 + n_halog * 0.5))


def _n_rings(smiles: str) -> int:
    return smiles.count("1") // 2 + smiles.count("2") // 2


def _hbd(smiles: str) -> int:
    return smiles.count("N") + smiles.count("O") - smiles.lower().count("n") // 2


def _admet_props(smiles: str) -> Tuple[bool, float, float]:
    """Returns (admet_pass, mw, logp_proxy)."""
    mw   = _estimate_mw(smiles)
    logp = _logp_proxy(smiles)
    hbd  = _hbd(smiles)
    hba  = smiles.count("N") + smiles.count("O") * 2
    passes = (mw < ADMET_MW_MAX and logp < ADMET_LOGP_MAX and
              hbd < ADMET_HBD_MAX and hba < ADMET_HBA_MAX)
    return passes, mw, logp


def _tanimoto_diversity(smiles_list: List[str]) -> float:
    """Bit-based diversity proxy from character n-gram overlap."""
    if len(smiles_list) < 2:
        return 1.0
    sample = smiles_list[:min(20, len(smiles_list))]
    sims = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            a = set(sample[i][k:k+3] for k in range(len(sample[i])-2))
            b = set(sample[j][k:k+3] for k in range(len(sample[j])-2))
            union = len(a | b)
            sims.append(len(a & b) / union if union else 0.0)
    return 1.0 - (sum(sims) / len(sims)) if sims else 1.0


def _mutate_smiles(smiles: str, rng: random.Random) -> str:
    """Fragment-based SMILES mutation — raw, may produce unbalanced strings."""
    r = rng.random()
    if r < 0.40 and len(smiles) < 80:           # grow
        frag = rng.choice(GROW_FRAGMENTS)
        return smiles + frag
    elif r < 0.65 and len(smiles) > 20:          # truncate at safe point
        # Truncate only up to a balanced-parenthesis boundary
        cut = rng.randint(len(smiles)//2, len(smiles)-5)
        # Walk back to nearest '(' boundary to avoid orphan closing parens
        while cut > len(smiles)//2 and smiles[cut] in ")":
            cut -= 1
        return smiles[:cut]
    elif r < 0.85:                               # swap to known-good seed
        return rng.choice(ALLOSTERIC_SEEDS)
    else:                                        # insert fragment at safe position
        if len(smiles) > 10:
            # Find a position not inside parentheses
            depth = 0
            safe_positions = []
            for i, c in enumerate(smiles):
                if c == "(": depth += 1
                elif c == ")": depth -= 1
                if depth == 0 and 5 <= i <= len(smiles) - 5:
                    safe_positions.append(i)
            if safe_positions:
                pos = rng.choice(safe_positions)
                frag = rng.choice(GROW_FRAGMENTS)
                return smiles[:pos] + frag + smiles[pos:]
        return smiles + rng.choice(GROW_FRAGMENTS)


def _smiles_balanced(smiles: str) -> bool:
    """
    Lightweight SMILES validity check — no RDKit needed.
    Checks: parenthesis balance, ring-closure digit pairing, no empty string.
    Not a full SMILES parser — catches the most common mutation artifacts.
    """
    if not smiles or len(smiles) < 3:
        return False
    # Parenthesis balance
    depth = 0
    for c in smiles:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    # Ring-closure digits should appear in pairs (rough check)
    import re
    digits = re.findall(r"(?<![%])\d", smiles)
    if len(digits) % 2 != 0:
        return False
    # Must contain at least one letter (atom)
    if not any(c.isalpha() for c in smiles):
        return False
    return True


def _safe_mutate_smiles(smiles: str, rng: random.Random, max_attempts: int = 5) -> str:
    """Mutate SMILES, retrying until the result passes the bracket-balance check."""
    for _ in range(max_attempts):
        candidate = _mutate_smiles(smiles, rng)
        if _smiles_balanced(candidate):
            return candidate
    # All attempts produced malformed SMILES — return a known-good seed
    return rng.choice(ALLOSTERIC_SEEDS)


def _safe_crossover(a: Tuple[str, str], b: Tuple[str, str],
                    rng: random.Random, max_attempts: int = 5) -> Tuple[str, str]:
    """Crossover with validity check — retry if splice produces unbalanced SMILES."""
    for _ in range(max_attempts):
        result = _crossover(a, b, rng)
        if _smiles_balanced(result[0]):
            return result
    # Fall back to whichever parent has the better (longer) SMILES
    return a if len(a[0]) >= len(b[0]) else b


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_site_complementarity(smiles: str, site: AlloSiteTarget, rng: random.Random) -> float:
    """
    Score how well the molecule matches allosteric site properties.
    Key properties: hydrophobicity, charge, H-bond capacity, size.
    """
    mw   = _estimate_mw(smiles)
    logp = _logp_proxy(smiles)
    n_pol = smiles.count("N") + smiles.count("O")
    n_aro = smiles.lower().count("c") // 4
    n_rin = _n_rings(smiles)

    # Allosteric sites tend to be partially buried (moderate SASA)
    # Good molecules have intermediate polarity matching site
    sasa_ok = 1.0 if 20 <= site.mean_sasa <= 80 else 0.75

    # Charge complementarity
    mol_charge_proxy = (smiles.count("N") - smiles.count("C(=O)O")) * 0.5
    charge_comp = max(0.0, min(1.0, 0.5 - 0.1*(mol_charge_proxy + site.net_charge)**2))

    # Hydrophobicity match
    site_hydro = site.mean_hydrophobicity
    if site_hydro > 0.3:  # hydrophobic allosteric site
        hydro_score = min(1.0, logp / 3.0)
    else:                 # polar allosteric site
        hydro_score = max(0.0, 1.0 - logp / 5.0) * (n_pol / max(1, n_pol))

    # Size score: molecule should fit the allosteric pocket (not too big/small)
    # Pocket size roughly proportional to n_residues
    ideal_mw = min(500, site.n_residues * 20 + 150)
    mw_score = max(0.0, 1.0 - abs(mw - ideal_mw) / 300)

    # Ring score: allosteric sites prefer rigid aromatic scaffolds
    ring_score = min(1.0, n_rin / max(1, 2)) * min(1.0, n_aro / max(1, 2))

    score = (0.20*sasa_ok + 0.25*charge_comp + 0.20*hydro_score +
             0.20*mw_score + 0.15*ring_score)
    return max(0.0, min(1.0, score + rng.gauss(0, 0.03)))


def _score_communication(smiles: str, site: AlloSiteTarget) -> float:
    """
    Score predicted disruption of the ENM communication pathway.
    High correlation site → stronger disruption if molecule is a good fit.
    Proxy: correlation × site_complementarity × size_factor.
    """
    # Larger molecules disrupt more allosteric communication
    mw = _estimate_mw(smiles)
    size_factor = min(1.0, mw / 350)
    # ENM correlation: higher correlation = more important site
    corr_bonus = site.mean_correlation
    # Number of coupled active residues: more coupling = bigger effect
    coupling_bonus = min(1.0, len(site.coupled_residues) / 5.0)
    return min(1.0, (0.5*corr_bonus + 0.3*coupling_bonus + 0.2*size_factor))


def _score_selectivity(smiles: str, site: AlloSiteTarget,
                       physico_data: Optional[dict]) -> float:
    """
    Reward molecules that preferentially fit the allosteric site over orthosteric.
    Proxy: allosteric sites are typically LESS hydrophobic than active sites.
    A molecule matching allosteric polarity but not orthosteric polarity = selective.
    """
    logp = _logp_proxy(smiles)
    site_hydro = site.mean_hydrophobicity

    # Allosteric selectivity: moderate LogP (2-4) for moderately hydrophobic sites
    if 0.2 <= site_hydro <= 0.5:
        # Moderately hydrophobic allosteric site
        sel_score = 1.0 if 1.5 <= logp <= 4.0 else max(0.3, 1.0 - abs(logp-2.5)/3)
    elif site_hydro < 0.2:
        # Polar allosteric site — prefer polar molecules
        sel_score = max(0.0, 1.0 - logp/5) if logp > 0 else 0.9
    else:
        # Hydrophobic allosteric site — prefer lipophilic
        sel_score = min(1.0, logp / 3.5)

    # Penalise molecules likely to also hit orthosteric pocket
    # (rough proxy: large, flat aromatic = often dual-site)
    n_aro = smiles.lower().count("c") // 4
    if n_aro > 4 and logp > 4.5:
        sel_score *= 0.7  # likely non-selective

    return max(0.0, min(1.0, sel_score))


def _score_conformational(smiles: str, site: AlloSiteTarget,
                           mechanism: str) -> float:
    """
    Predicted conformational effect.
    Inhibitors need to lock the protein in an inactive state.
    Activators need to release autoinhibitory contacts.
    Modulators can do either.
    """
    n_rings = _n_rings(smiles)
    mw = _estimate_mw(smiles)

    # Rigid molecules (high ring count) better lock conformations
    rigidity = min(1.0, n_rings / 4)
    # Larger molecules have more contact surface → stronger effect
    size_eff = min(1.0, mw / 400)

    if mechanism == "inhibitor":
        # Need rigidity to lock inactive conformation
        score = 0.6 * rigidity + 0.4 * size_eff
    elif mechanism == "activator":
        # Activators: flexible molecules can wedge open regulatory interface
        flexibility = 1.0 - rigidity * 0.5
        score = 0.6 * flexibility + 0.4 * size_eff
    else:  # modulator
        score = 0.5 * rigidity + 0.3 * size_eff + 0.2

    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(smiles: str, mechanism: str, site: AlloSiteTarget,
              physico_data: Optional[dict], gen: int, rng: random.Random,
              w_site, w_comm, w_sel, w_adm, w_conf) -> Optional[AlloCandidate]:
    # Reject malformed SMILES before scoring
    if not _smiles_balanced(smiles):
        return None
    admet_ok, mw, logp = _admet_props(smiles)
    if not (50 < mw < ADMET_MW_MAX):
        return None

    site_comp  = _score_site_complementarity(smiles, site, rng)
    comm_score = _score_communication(smiles, site)
    sel_score  = _score_selectivity(smiles, site, physico_data)
    admet_scr  = 1.0 if admet_ok else 0.4
    conf_scr   = _score_conformational(smiles, site, mechanism)

    fitness = (w_site*site_comp + w_comm*comm_score + w_sel*sel_score +
               w_adm*admet_scr + w_conf*conf_scr)
    fitness = max(0.0, fitness)

    return AlloCandidate(
        smiles=smiles, generation=gen, mechanism=mechanism,
        site_id=site.site_id,
        site_complementarity=round(site_comp,4),
        communication_score=round(comm_score,4),
        selectivity_score=round(sel_score,4),
        admet_score=round(admet_scr,4),
        conformational_score=round(conf_scr,4),
        fitness=round(fitness,4),
        mw=round(mw,1), logp_proxy=round(logp,2),
        n_rings=_n_rings(smiles),
        origin="seed" if gen==1 else "evolved",
    )


def _initial_population(size: int, rng: random.Random,
                         preferred_mechanism: Optional[str]) -> List[Tuple[str,str]]:
    pop = []
    for i in range(size):
        smiles = rng.choice(ALLOSTERIC_SEEDS) if i < size//2 else rng.choice(GROW_FRAGMENTS*3 + ALLOSTERIC_SEEDS)
        mech = (preferred_mechanism or rng.choice(MECHANISMS))
        pop.append((smiles, mech))
    return pop


def _crossover(a: Tuple[str,str], b: Tuple[str,str], rng: random.Random) -> Tuple[str,str]:
    """Splice two SMILES strings at random point."""
    sa, ma = a; sb, mb = b
    if len(sa) > 5 and len(sb) > 5:
        pos_a = rng.randint(len(sa)//3, 2*len(sa)//3)
        pos_b = rng.randint(len(sb)//3, 2*len(sb)//3)
        child_smiles = sa[:pos_a] + sb[pos_b:]
    else:
        child_smiles = sa if rng.random()<0.5 else sb
    mech = ma if rng.random()<0.5 else mb
    return (child_smiles, mech)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_allosteric_drug_design(
    uniprot_id:          str,
    allosteric_data:     Optional[dict] = None,
    physico_data:        Optional[dict] = None,
    preferred_site:      Optional[str]  = None,
    preferred_mechanism: Optional[str]  = None,
    n_generations:       int            = MAX_GENERATIONS,
    rng_seed:            Optional[int]  = None,
    force:               bool           = False,
) -> AllodrugResult:
    """Run allosteric drug evolutionary design. Skips if output exists and force=False."""
    t0   = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    rng  = random.Random(seed)
    np.random.seed(seed % (2**32))

    uid = uniprot_id.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    out_path  = inter_dir / f"{uid}_allosteric_drug.json"

    if out_path.exists() and not force:
        log.info(f"  [AlloDrug] Cached: {out_path} — skipping.")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        result = AllodrugResult(uniprot_id=uid)
        result.__dict__.update({k: v for k, v in data.items() if k in result.__dict__})
        return result

    log.info(f"══ Module 21: Allosteric Drug Design: {uid} [seed={seed}] ══")

    target_gene = uid
    sj = inter_dir / f"{uid}_structure.json"
    if sj.exists():
        try: target_gene = json.loads(sj.read_text()).get("gene_name", uid)
        except: pass

    # Select target allosteric site
    site: Optional[AlloSiteTarget] = None
    if allosteric_data:
        sites_raw = allosteric_data.get("allosteric_sites", [])
        if preferred_site:
            site_raw = next((s for s in sites_raw if s.get("site_id") == preferred_site), None)
            if site_raw:
                site = AlloSiteTarget.from_dict(site_raw)
        if site is None and sites_raw:
            # Pick best by correlation
            best = max(sites_raw, key=lambda s: s.get("mean_correlation", 0))
            site = AlloSiteTarget.from_dict(best)

    if site is None:
        log.warning("  No allosteric site data — using generic fallback site")
        site = AlloSiteTarget(
            site_id="A1", mean_correlation=0.5, mean_sasa=50.0,
            mean_hydrophobicity=0.3, net_charge=0.0, n_residues=6,
            coupled_residues=[], confidence="LOW",
        )

    log.info(f"  Target: {uid} ({target_gene})  Site: {site.site_id}  "
             f"corr={site.mean_correlation:.3f}  confidence={site.confidence}")
    if preferred_mechanism:
        log.info(f"  Mechanism: {preferred_mechanism}")

    population = _initial_population(POP_SIZE, rng, preferred_mechanism)
    hall_of_fame: List[AlloCandidate] = []
    seen_smiles: set = set()
    gen_stats: List[dict] = []
    n_evaluated = 0
    best_fitness = 0.0
    stagnation = 0

    for gen in range(1, n_generations + 1):
        t = min(1.0, (gen-1)/max(1,n_generations-1))
        w_site = W_SITE_START + t*(W_SITE_END - W_SITE_START)
        w_comm = W_COMM_START + t*(W_COMM_END - W_COMM_START)
        w_sel  = W_SEL_START  + t*(W_SEL_END  - W_SEL_START)
        w_adm  = W_ADM_START  + t*(W_ADM_END  - W_ADM_START)
        w_conf = W_CONF_START + t*(W_CONF_END  - W_CONF_START)
        total  = w_site+w_comm+w_sel+w_adm+w_conf
        w_site/=total; w_comm/=total; w_sel/=total; w_adm/=total; w_conf/=total
        temperature = max(0.1, 1.0 - 0.8*t)

        evaluated = []
        for smiles, mech in population:
            c = _evaluate(smiles, mech, site, physico_data, gen, rng,
                          w_site, w_comm, w_sel, w_adm, w_conf)
            if c is not None:
                evaluated.append(((smiles, mech), c))
        n_evaluated += len(evaluated)
        if not evaluated:
            continue
        evaluated.sort(key=lambda x: x[1].fitness, reverse=True)

        for _, cand in evaluated:
            key = cand.smiles[:30] + cand.mechanism
            if key not in seen_smiles:
                hall_of_fame.append(cand)
                seen_smiles.add(key)

        gen_best = evaluated[0][1].fitness
        improved = gen_best > best_fitness
        best_fitness = max(best_fitness, gen_best)
        stagnation = 0 if improved else stagnation + 1

        div = _tanimoto_diversity([s for s, _ in population])
        gen_stats.append({"gen": gen, "best_fitness": round(gen_best,4),
                          "diversity": round(div,3), "stagnation": stagnation})

        if gen % 10 == 0:
            log.info(f"  Gen {gen}/{n_generations}: best={gen_best:.4f}  "
                     f"div={div:.2f}  stag={stagnation}")

        # Next generation
        next_pop = []
        seen_n: set = set()
        def _add(c):
            k = str(c)
            if k not in seen_n: next_pop.append(c); seen_n.add(k)

        for raw, _ in evaluated[:ELITISM]:
            _add(raw)
        top_raw = [r for r, _ in evaluated[:10]]
        for _ in range(POP_SIZE // 3):
            if len(top_raw) >= 2:
                _add(_safe_crossover(*rng.sample(top_raw, 2), rng))
        for (smiles, mech), _ in evaluated[:POP_SIZE//2]:
            new_sm = _safe_mutate_smiles(smiles, rng)
            new_mech = mech if preferred_mechanism else (mech if rng.random()<0.8 else rng.choice(MECHANISMS))
            _add((new_sm, new_mech))

        # Diversity injection
        if div < DIVERSITY_MIN or stagnation >= STAGNATION_HARD:
            if stagnation >= STAGNATION_HARD:
                log.info(f"  [AlloDrug] Hard reset at gen {gen}")
                stagnation = 0
                next_pop = [evaluated[0][0]]
            for _ in range(POP_SIZE // 3):
                _add((rng.choice(ALLOSTERIC_SEEDS),
                      preferred_mechanism or rng.choice(MECHANISMS)))

        while len(next_pop) < POP_SIZE:
            _add((rng.choice(ALLOSTERIC_SEEDS + GROW_FRAGMENTS),
                  preferred_mechanism or rng.choice(MECHANISMS)))
        population = next_pop[:POP_SIZE]

    hall_of_fame.sort(key=lambda c: c.fitness, reverse=True)
    seen_k: set = set()
    top_candidates = []
    for c in hall_of_fame:
        key = c.smiles[:20] + c.mechanism
        if key not in seen_k:
            top_candidates.append(c)
            seen_k.add(key)
        if len(top_candidates) >= TOP_FOR_FINAL:
            break

    result = AllodrugResult(
        uniprot_id=uid, target_gene=target_gene,
        target_site=site.site_id, site_correlation=site.mean_correlation,
        n_generations=n_generations, n_evaluated=n_evaluated,
        top_candidates=top_candidates, best_fitness=best_fitness,
        best_smiles=top_candidates[0].smiles if top_candidates else "",
        best_mechanism=top_candidates[0].mechanism if top_candidates else "",
        generation_stats=gen_stats,
        notes=f"seed={seed}  site={site.site_id}  runtime={time.time()-t0:.1f}s",
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
@click.option("--site",        "-s", default=None,
              help="Allosteric site ID to target (e.g. A1, A2). Default: best by correlation.")
@click.option("--mechanism",   "-m", default=None,
              type=click.Choice(MECHANISMS),
              help="Drug mechanism (default: co-evolve)")
@click.option("--seed",        default=None, type=int)
@click.option("--force",       "-f", is_flag=True, default=False)
def main(uniprot, generations, site, mechanism, seed, force):
    """
    Module 21 — De Novo Allosteric Small Molecule Design.

    Evolves small molecules against the best allosteric site from Module 05.
    No Vina required — scores pharmacophore match to allosteric site properties.
    Skips if output already exists.

    Examples:
        python pipeline\\allosteric_drug_design.py --uniprot P04637
        python pipeline\\allosteric_drug_design.py --uniprot P04637 --site A2 --mechanism inhibitor
        python pipeline\\allosteric_drug_design.py --uniprot P04637 --mechanism activator --generations 80
    """
    uid = uniprot.strip().upper()
    inter = Path(cfg.paths["intermediate"])

    def _load(f):
        p = inter / f
        return json.loads(p.read_text()) if p.exists() else None

    result = run_allosteric_drug_design(
        uniprot_id=uid,
        allosteric_data=_load(f"{uid}_allosteric.json"),
        physico_data=_load(f"{uid}_physicochemical.json"),
        preferred_site=site, preferred_mechanism=mechanism,
        n_generations=generations, rng_seed=seed, force=force,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()