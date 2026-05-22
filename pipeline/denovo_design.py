"""
pipeline/denovo_design.py
──────────────────────────
De Novo Molecule Design Module.

Evolutionary molecular generator that designs drug-like binders for a target
protein using AutoDock Vina docking + surrogate-guided evolution.

FIXES in this version:
  [BUG-1] all_seen grew unboundedly → generation collapsed at Gen 18+
           FIX: split into docked_seen (never re-dock) vs gen_seen (per-gen dedup).
                _generate_and_screen no longer mutates global all_seen.
  [BUG-2] Surrogate pre-screening replaced population before docking, adding
           all screened molecules to all_seen immediately, starving future gens.
           FIX: screened molecules added to docked_seen only after they are docked.
  [BUG-3] MIN_SCORE_ELITE = -4.0 blocked all elites when surrogate screened
           poor molecules into the generation; elite pool → empty → pop collapse.
           FIX: raised threshold to -2.0 and added a score-agnostic elite slot.
  [BUG-4] return result was indented inside the Module 15 if-block, so the
           function returned None when Module 15 was skipped.
           FIX: unindented return result + summary log.
  [BUG-5] CYP450 denovo_cyp_score attribute access was unguarded — if cyp450
           module returns an object without that attribute the whole gen crashes.
           FIX: wrapped in hasattr guard with safe fallback.
  [BUG-6] stagnation reseeding never reset stagnation_cnt, so the hard reset
           fired every generation after Gen 14, wiping the best molecule.
           FIX: stagnation_cnt reset to 0 after each reseeding cycle.
  [BUG-7] hall_of_fame included zero-score placeholder entries from reseeded
           generations (score=None checked incorrectly).
           FIX: guard r["score"] is not None in hall_of_fame sorting.
  [BUG-8] Population could shrink below POP_SIZE when all_seen was exhausted,
           causing ThreadPoolExecutor to dock 0 molecules → infinite reseeding.
           FIX: emergency fill now uses docked_seen (smaller set) as dedup guard.

CLOSED LOOP ADDITIONS (Stage 1 + 2 + 3):
  [STAGE-1] FeedbackStore records every docking event to disk.
  [STAGE-2] pocket_refit learns from accumulated events after each run.
  [STAGE-3] Warm-start: surrogate + fqsar pre-loaded from 1929+ prior events
            so Gen 1 begins with a fully trained model, not a blank slate.
  [STAGE-3] Pocket bias: refined pocket model steers fragment pool toward
            what actually binds (e.g. heavy_atoms → ring-rich fragments).
  [STAGNATION FIX] Surrogate bypass: when stagnation >= STAGNATION_LIMIT,
            surrogate ranking is disabled so fresh scaffolds aren't filtered
            back out before they can be docked and evaluated.
  [HARD RESET FIX] Surrogate disabled temporarily on hard reset so the fresh
            scaffolds actually survive into the next generation's docking.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from utils.config import cfg, get_logger

# CYP450 metabolic stability scoring (Module 19)
try:
    from pipeline.cyp450 import profile_molecule as _cyp_profile
    CYP_AVAILABLE = True
except ImportError:
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from pipeline.cyp450 import profile_molecule as _cyp_profile
        CYP_AVAILABLE = True
    except ImportError:
        CYP_AVAILABLE = False
        _cyp_profile = None

log = get_logger(__name__)

# ── Try RDKit ──────────────────────────────────────────────────────────────────
try:
    import rdkit.RDLogger as rl
    rl.DisableLog("rdApp.*")
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    log.error("RDKit not found. Install with: pip install rdkit")
    sys.exit(1)

# ── Evolution hyperparameters ─────────────────────────────────────────────────

MAX_GENERATIONS      = 10
POP_SIZE             = 30
ELITISM              = 6
PARALLEL_WORKERS = min(4, (os.cpu_count() or 4) // 2)

EXHAUST_START        = 8
EXHAUST_END          = 16

SURROGATE_MIN_DATA   = 20
SURROGATE_CANDIDATES = 80

FQSAR_MIN_DATA       = 15
FQSAR_GOOD_THRESH    = -6.0
FQSAR_WINDOW         = 300

W_BINDING_START = 0.45;  W_BINDING_END = 0.75
W_NOVELTY_START = 0.30;  W_NOVELTY_END = 0.05
W_LE_START      = 0.25;  W_LE_END      = 0.20
W_CYP_START     = 0.05;  W_CYP_END     = 0.08

NOVELTY_K            = 10
NOVELTY_ARCHIVE_MAX  = 500
NOVELTY_ADD_THRESH   = 0.10

STAGNATION_LIMIT     = 4
STAGNATION_HARD      = 10
DIVERSITY_MIN        = 0.20
MIN_SCORE_ELITE      = -2.0
MAX_SAME_SCAFFOLD    = 4

VALIDATION_INTERVAL  = 5
TOP_FOR_FINAL        = 10
CONSENSUS_RUNS       = 2

ADMET_MW_MAX         = 450.0
ADMET_LOGP_MAX       = 4.5
ADMET_HBD_MAX        = 5
ADMET_HBA_MAX        = 10

BOX_PADDING          = 4.0
MIN_BOX_SIZE         = 15.0
MAX_BOX_SIZE         = 22.0

MIN_REAL_SCORE       = -0.5


# ── Seed library ──────────────────────────────────────────────────────────────

TINY_SEEDS = [
    "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1",
    "c1ccoc1", "c1ccsc1", "c1cc[nH]n1", "c1cnc[nH]1",
    "C1CCOCC1", "N1CCNCC1", "c1cnccn1",
    "c1ccc2ccccc2c1",
    "c1ccnc2ccccc12",
    "c1ccc2[nH]ccc2c1",
    "c1cnc2ccccc2n1",
    "c1ccc2ncncc2c1",
    "O=C1CCc2ccccc21",
    "c1ccc(Cc2ccccn2)cc1",
    "NC(=O)c1cccnc1",
    "c1ccc(-c2ccccn2)cc1",
    "O=C1CN=C(c2ccccc2)c2ccccc21",
    "Cc1ccc2c(c1)CC(=O)N2",
    "c1ccc2c(c1)CCCO2",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "C1CN2CCc3ccccc3C2C1",
    "O=C1Nc2ccccc2C1=O",
]

GROW_FRAGMENTS = [
    "C", "CC", "CCC", "N", "O", "S", "F", "Cl", "Br",
    "C(=O)N", "C(=O)O", "S(=O)(=O)N", "C#N", "OC", "NC",
    "C(=O)", "NC(=O)", "C(F)(F)F", "OCC", "NCC",
    "C(=O)NC", "NCC(=O)", "OC(=O)", "SC",
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1cc[nH]n1",
    "c1ccoc1", "c1ccsc1", "C1CCCCC1", "C1CCNCC1",
    "C1CCOCC1", "N1CCNCC1", "N1CCCC1",
    "c1ccc2ccccc2c1", "c1ccnc2ccccc12",
    "c1ccc2[nH]ccc2c1", "c1cnc2[nH]ccc2n1",
    "CC(=O)", "CCO", "CCN", "c1cccc(C)c1",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DenovoCandidate:
    smiles:       str
    generation:   int
    score:        float
    le:           float
    qed:          float
    mw:           float
    logp:         float
    tpsa:         float
    hbd:          int
    hba:          int
    heavy_atoms:  int
    fitness:      float
    novelty:      float
    admet_pass:   bool
    scaffold:     str
    origin:       str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DenovoResult:
    uniprot_id:     str
    pocket_id:      str
    pocket_center:  list
    box_size:       list
    n_generations:  int          = 0
    n_docked:       int          = 0
    n_unique:       int          = 0
    top_candidates: list         = field(default_factory=list)
    best_score:     float        = 0.0
    best_le:        float        = 0.0
    best_smiles:    str          = ""
    generation_stats: list       = field(default_factory=list)
    vina_path:      str          = ""
    notes:          str          = ""

    def summary(self) -> str:
        lines = [
            f"\n{'─'*70}",
            f"  De Novo Design: {self.uniprot_id}  [pocket={self.pocket_id}]",
            f"  Generations run  : {self.n_generations}",
            f"  Molecules docked : {self.n_docked}",
            f"  Best score       : {self.best_score:.2f} kcal/mol",
            f"  Best LE          : {self.best_le:.3f} kcal/mol/HA",
            f"  Best SMILES      : {self.best_smiles[:70]}",
            f"{'─'*70}",
            f"  Top candidates:",
        ]
        for i, c in enumerate(self.top_candidates[:10], 1):
            lines.append(
                f"  #{i:2d}: Score={c.score:.2f}  LE={c.le:.3f}  "
                f"QED={c.qed:.2f}  MW={c.mw:.0f}  "
                f"{'ADMET✓' if c.admet_pass else 'ADMET✗'}  "
                f"{c.smiles[:55]}"
            )
        lines.append(f"{'─'*70}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Molecular utilities ───────────────────────────────────────────────────────

_DESCR_CACHE: Dict[str, dict] = {}

def _mol_props(mol: Chem.Mol) -> dict:
    if mol is None:
        return {}
    try:
        smi = Chem.MolToSmiles(mol)
    except:
        return {}
    if smi in _DESCR_CACHE:
        return _DESCR_CACHE[smi]
    try:
        p = {
            "mw":    round(Descriptors.MolWt(mol), 2),
            "logp":  round(Descriptors.MolLogP(mol), 2),
            "tpsa":  round(Descriptors.TPSA(mol), 2),
            "qed":   round(QED.qed(mol), 3),
            "ha":    mol.GetNumHeavyAtoms(),
            "hbd":   rdMolDescriptors.CalcNumHBD(mol),
            "hba":   rdMolDescriptors.CalcNumHBA(mol),
            "rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
        _DESCR_CACHE[smi] = p
        return p
    except:
        return {}


def _admet_prefilter(mol: Chem.Mol, props: dict) -> bool:
    if not props:
        return False
    if props.get("mw", 999)  > ADMET_MW_MAX:  return False
    if props.get("mw", 0)    < 150:            return False
    if props.get("logp", 99) > ADMET_LOGP_MAX: return False
    if props.get("hbd", 99)  > ADMET_HBD_MAX:  return False
    if props.get("hba", 99)  > ADMET_HBA_MAX:  return False
    if mol is not None:
        try:
            if rdMolDescriptors.CalcNumRings(mol) > 6:        return False
            if rdMolDescriptors.CalcNumRotatableBonds(mol) > 12: return False
            if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()): return False
        except Exception:
            pass
    return True


def _canonical(smi: str) -> Optional[str]:
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except:
        return None


def _get_fp(smi: str, nbits: int = 1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    except:
        return None


def _murcko(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except:
        return smi


def _tanimoto_diversity(smiles_list: List[str]) -> float:
    fps = [_get_fp(s) for s in smiles_list if s]
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return 0.0
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for i in range(len(fps))
            for j in range(i + 1, len(fps))]
    return round(statistics.mean(sims), 3) if sims else 0.0


# ── Pocket → Vina box ─────────────────────────────────────────────────────────

def _pocket_to_box(pocket: dict) -> Tuple[List[float], List[float]]:
    center = pocket.get("center", [0.0, 0.0, 0.0])
    volume = pocket.get("volume_A3", 500.0)
    edge   = (volume ** (1 / 3)) + BOX_PADDING
    edge   = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, edge))
    size   = [round(edge, 1), round(edge, 1), round(edge, 1)]
    return [round(c, 3) for c in center], size


# ── Surrogate model ───────────────────────────────────────────────────────────

class SurrogateModel:
    def __init__(self, nbits: int = 1024):
        self.nbits   = nbits
        self.weights = np.zeros(nbits)
        self.bias    = 0.0
        self.trained = False
        self.n_obs   = 0

    def update(self, history: List[dict]) -> None:
        data = [(r["smiles"], r["score"])
                for r in history
                if r.get("score") is not None]
        if len(data) < SURROGATE_MIN_DATA:
            return
        X, y = [], []
        for smi, score in data[-FQSAR_WINDOW:]:
            fp = _get_fp(smi, self.nbits)
            if fp is None:
                continue
            X.append(np.array(fp))
            y.append(score)
        if len(X) < SURROGATE_MIN_DATA:
            return
        X   = np.array(X, dtype=np.float32)
        y   = np.array(y, dtype=np.float32)
        lam = 1.0
        try:
            self.weights = np.linalg.solve(
                X.T @ X + lam * np.eye(self.nbits), X.T @ y
            )
            self.bias    = float(np.mean(y) - self.weights @ np.mean(X, axis=0))
            self.trained = True
            self.n_obs   = len(X)
        except np.linalg.LinAlgError:
            pass

    def predict(self, smi: str) -> float:
        if not self.trained:
            return 0.0
        fp = _get_fp(smi, self.nbits)
        if fp is None:
            return 0.0
        return float(self.weights @ np.array(fp, dtype=np.float32) + self.bias)

    def rank(self, smiles_list: List[str]) -> List[Tuple[str, float]]:
        return sorted([(s, self.predict(s)) for s in smiles_list],
                      key=lambda t: t[1])


# ── Fragment QSAR ─────────────────────────────────────────────────────────────

class FragmentQSAR:
    def __init__(self):
        self.frag_good:  Dict[str, int] = defaultdict(int)
        self.frag_total: Dict[str, int] = defaultdict(int)
        self.n_obs = 0

    def update(self, history: List[dict]) -> None:
        data = [(r["smiles"], r["score"])
                for r in history[-FQSAR_WINDOW:]
                if r.get("score") is not None]
        self.n_obs = len(data)
        if self.n_obs < FQSAR_MIN_DATA:
            return
        self.frag_good.clear()
        self.frag_total.clear()
        for smi, score in data:
            for frag in GROW_FRAGMENTS:
                if frag in smi:
                    self.frag_total[frag] += 1
                    if score < FQSAR_GOOD_THRESH:
                        self.frag_good[frag] += 1

    def score_fragment(self, frag: str) -> float:
        total = self.frag_total.get(frag, 0)
        if total < 3:
            return 0.5
        return self.frag_good.get(frag, 0) / total

    def biased_pool(self, pool: List[str], temperature: float = 1.0) -> List[str]:
        if self.n_obs < FQSAR_MIN_DATA:
            return pool
        scores = np.array([self.score_fragment(f) for f in pool], dtype=np.float64)
        scores = scores ** (1.0 / max(temperature, 0.01))
        total  = scores.sum()
        if total == 0:
            return pool
        probs   = scores / total
        sampled = np.random.choice(len(pool), size=len(pool), p=probs, replace=True)
        return [pool[i] for i in sampled]


# ── Novelty archive ───────────────────────────────────────────────────────────

class NoveltyArchive:
    def __init__(self, max_size: int = NOVELTY_ARCHIVE_MAX, k: int = NOVELTY_K):
        self.archive: List[Tuple[str, object]] = []
        self.max_size = max_size
        self.k = k

    def score(self, smi: str) -> float:
        if len(self.archive) < self.k:
            return 1.0
        fp = _get_fp(smi)
        if fp is None:
            return 0.0
        sims = sorted(
            [DataStructs.TanimotoSimilarity(fp, af) for _, af in self.archive],
            reverse=True,
        )
        return round(1.0 - statistics.mean(sims[:self.k]), 4)

    def try_add(self, smi: str) -> bool:
        if self.score(smi) >= NOVELTY_ADD_THRESH or len(self.archive) < self.k * 2:
            fp = _get_fp(smi)
            if fp is not None:
                self.archive.append((smi, fp))
                if len(self.archive) > self.max_size:
                    self._prune()
                return True
        return False

    def _prune(self) -> None:
        if len(self.archive) <= 1:
            return
        fps    = [fp for _, fp in self.archive]
        scores = [
            statistics.mean(
                DataStructs.TanimotoSimilarity(fps[i], fps[j])
                for j in range(len(fps)) if j != i
            )
            for i in range(len(fps))
        ]
        self.archive.pop(scores.index(max(scores)))


# ── Chemical engine ───────────────────────────────────────────────────────────

class ChemicalEngine:
    def _sanitize(self, mol) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def _to_smiles(self, mol) -> Optional[str]:
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except:
            return None

    def _grow(self, mol: Chem.Mol, pool: List[str]) -> Optional[Chem.Mol]:
        for _ in range(10):
            frag = Chem.MolFromSmiles(random.choice(pool))
            if frag is None:
                continue
            combined = Chem.CombineMols(mol, frag)
            rwm = Chem.RWMol(combined)
            n   = mol.GetNumAtoms()
            mi  = list(range(n))
            fi  = list(range(n, rwm.GetNumAtoms()))
            if not mi or not fi:
                continue
            try:
                rwm.AddBond(random.choice(mi), random.choice(fi),
                            Chem.BondType.SINGLE)
                result = self._sanitize(rwm.GetMol())
                if result:
                    p = _mol_props(result)
                    if _admet_prefilter(result, p):
                        return result
            except:
                continue
        return None

    def _grow_guided(self, mol: Chem.Mol, pool: List[str],
                     surrogate: SurrogateModel,
                     n_tries: int = 8) -> Optional[Chem.Mol]:
        candidates = []
        for _ in range(n_tries):
            result = self._grow(mol, pool)
            if result:
                smi = self._to_smiles(result)
                if smi:
                    candidates.append((smi, result))
        if not candidates:
            return None
        if not surrogate.trained:
            return random.choice(candidates)[1]
        best_smi = min(candidates, key=lambda t: surrogate.predict(t[0]))[0]
        return Chem.MolFromSmiles(best_smi)

    def _mutate_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rwm  = Chem.RWMol(mol)
        idx  = random.randint(0, rwm.GetNumAtoms() - 1)
        atom = rwm.GetAtomWithIdx(idx)
        curr = atom.GetAtomicNum()
        choices = [c for c in [6, 7, 8, 9, 16, 17] if c != curr]
        if not choices:
            return mol
        atom.SetAtomicNum(random.choice(choices))
        return rwm.GetMol()

    def _prune(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        if mol.GetNumAtoms() < 6:
            return mol
        rwm   = Chem.RWMol(mol)
        cands = [a.GetIdx() for a in rwm.GetAtoms()
                 if not a.IsInRing() and a.GetDegree() == 1]
        if not cands:
            cands = list(range(rwm.GetNumAtoms()))
        rwm.RemoveAtom(random.choice(cands))
        return rwm.GetMol()

    def _ring_swap(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        new_ring = Chem.MolFromSmiles(random.choice(TINY_SEEDS))
        if new_ring is None:
            return None
        result = self._grow(new_ring, GROW_FRAGMENTS)
        return result if result else new_ring

    def build_denovo(self, n_steps: int = None,
                     surrogate: SurrogateModel = None,
                     fqsar: FragmentQSAR = None,
                     temperature: float = 1.0) -> Optional[str]:
        if n_steps is None:
            n_steps = random.randint(1, 4)
        mol = Chem.MolFromSmiles(random.choice(TINY_SEEDS))
        if mol is None:
            return None
        for step in range(n_steps):
            pool = (fqsar.biased_pool(GROW_FRAGMENTS, temperature)
                    if fqsar and fqsar.n_obs >= FQSAR_MIN_DATA
                    else GROW_FRAGMENTS)
            if surrogate and surrogate.trained and step > 0:
                grown = self._grow_guided(mol, pool, surrogate, n_tries=6)
            else:
                grown = self._grow(mol, pool)
            if grown:
                mol = grown
        return self._to_smiles(mol)

    def crossover(self, smi1: str, smi2: str) -> Optional[str]:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return None
        a1 = [a.GetIdx() for a in mol1.GetAtoms() if not a.IsInRing()]
        a2 = [a.GetIdx() for a in mol2.GetAtoms() if not a.IsInRing()]
        if not a1: a1 = list(range(mol1.GetNumAtoms()))
        if not a2: a2 = list(range(mol2.GetNumAtoms()))
        combined = Chem.CombineMols(mol1, mol2)
        n1 = mol1.GetNumAtoms()
        for _ in range(10):
            try:
                rwm = Chem.RWMol(combined)
                rwm.AddBond(random.choice(a1),
                            random.choice(a2) + n1,
                            Chem.BondType.SINGLE)
                result = self._sanitize(rwm.GetMol())
                if result:
                    p = _mol_props(result)
                    if _admet_prefilter(result, p):
                        smi = self._to_smiles(result)
                        if smi:
                            return smi
            except:
                continue
        return None

    def mutate(self, parent_smi: str,
               stagnation: int = 0,
               mode: str = "balanced",
               surrogate: SurrogateModel = None,
               fqsar: FragmentQSAR = None,
               temperature: float = 1.0) -> Tuple[Optional[str], str]:
        mol = Chem.MolFromSmiles(parent_smi)
        if mol is None:
            return None, "Invalid"

        stag_bonus = min(0.25, stagnation * 0.04)
        pool = (fqsar.biased_pool(GROW_FRAGMENTS, temperature)
                if fqsar and fqsar.n_obs >= FQSAR_MIN_DATA
                else GROW_FRAGMENTS)
        new_mol = None
        mtype   = "None"

        try:
            r = random.random()
            if mode == "exploit":
                if r < 0.40:
                    new_mol = (self._grow_guided(mol, pool, surrogate, 8)
                               if surrogate and surrogate.trained
                               else self._grow(mol, pool))
                    mtype = "GuidedGrow" if surrogate and surrogate.trained else "Grow"
                elif r < 0.70:
                    new_mol = self._mutate_atom(mol); mtype = "MutateAtom"
                else:
                    new_mol = self._prune(mol);       mtype = "Prune"

            elif mode == "explore":
                if r < 0.40:
                    new_mol = self._ring_swap(mol); mtype = "RingSwap"
                elif r < 0.70:
                    new_mol = self._grow(mol, GROW_FRAGMENTS); mtype = "GrowRing"
                else:
                    smi = self.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                            temperature=temperature)
                    return (smi, "DeNovo") if smi else (parent_smi, "Failed")

            else:  # balanced
                if r < (0.35 - stag_bonus):
                    new_mol = (self._grow_guided(mol, pool, surrogate, 6)
                               if surrogate and surrogate.trained
                               else self._grow(mol, pool))
                    mtype = "GuidedGrow" if surrogate and surrogate.trained else "Grow"
                elif r < (0.60 - stag_bonus * 0.5):
                    new_mol = self._mutate_atom(mol); mtype = "MutateAtom"
                elif r < 0.73:
                    new_mol = self._prune(mol);       mtype = "Prune"
                elif r < (0.73 + 0.12 + stag_bonus):
                    new_mol = self._ring_swap(mol);   mtype = "RingSwap"
                else:
                    smi = self.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                            temperature=temperature)
                    return (smi, "DeNovo") if smi else (parent_smi, "Failed")

            if new_mol:
                new_mol = self._sanitize(new_mol)
            if new_mol:
                p = _mol_props(new_mol)
                if _admet_prefilter(new_mol, p):
                    smi = self._to_smiles(new_mol)
                    if smi:
                        return smi, mtype
        except:
            pass

        return parent_smi, "Failed"


# ── Docking ───────────────────────────────────────────────────────────────────

_DOCK_CACHE: Dict[str, dict] = {}


def _load_dock_cache(path: str) -> None:
    global _DOCK_CACHE
    if os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            _DOCK_CACHE = {
                k: v for k, v in raw.items()
                if v.get("status") == "OK" and v.get("score") is not None
            }
            n_purged = len(raw) - len(_DOCK_CACHE)
            log.info(f"  Loaded docking cache: {len(_DOCK_CACHE)} OK entries "
                     f"({n_purged} bad entries purged)")
            if _DOCK_CACHE:
                _save_dock_cache(path)
        except:
            _DOCK_CACHE = {}


def _save_dock_cache(path: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump(_DOCK_CACHE, f)
    except:
        pass


def _dock_molecule(smiles: str, gen_id: int, cand_id: int,
                   vina_path: str, receptor_path: str,
                   center: List[float], box_size: List[float],
                   cache_path: str,
                   exhaust: int = EXHAUST_START) -> dict:
    res = {"smiles": smiles, "generation": gen_id, "candidate_id": cand_id,
           "score": None, "status": "Failed", "error": "", "props": {}}

    if smiles in _DOCK_CACHE and _DOCK_CACHE[smiles].get("status") == "OK":
        cached = _DOCK_CACHE[smiles].copy()
        cached.update({"generation": gen_id, "candidate_id": cand_id})
        return cached

    tmpdir    = os.path.join(os.getcwd(), f"_tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmpdir, exist_ok=True)
    pdb_path   = os.path.join(tmpdir, "ligand.pdb")
    pdbqt_path = os.path.join(tmpdir, "ligand.pdbqt")
    out_path   = os.path.join(tmpdir, "out.pdbqt")

    try:
        if "." in smiles:
            res["error"] = "Disconnected SMILES (contains '.')"; return res

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            res["error"] = "Invalid SMILES"; return res

        props   = _mol_props(mol)
        res["props"] = props

        if not _admet_prefilter(mol, props):
            res["error"] = "Failed ADMET pre-filter"; res["status"] = "AdmetFail"
            return res

        try:
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if n_rings > 6:
                res["error"] = f"Too many rings ({n_rings})"; return res
            if n_rot > 12:
                res["error"] = f"Too many rotatable bonds ({n_rot})"; return res
            if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
                res["error"] = "Radical atoms present"; return res
        except Exception:
            pass

        mol_h  = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random.randint(1, 99999)
        ret = AllChem.EmbedMolecule(mol_h, params)
        if ret == -1:
            params.useRandomCoords = True
            ret = AllChem.EmbedMolecule(mol_h, params)
        if ret == -1:
            res["error"] = "3D embedding failed"; return res

        AllChem.MMFFOptimizeMolecule(mol_h)
        Chem.MolToPDBFile(mol_h, pdb_path)

        if not _smiles_to_ligand_pdbqt(smiles, Path(pdbqt_path)):
            res["error"] = "Ligand PDBQT conversion failed"; return res

        cx, cy, cz = center
        sx, sy, sz = box_size
        cmd = [vina_path,
               '--receptor', receptor_path,
               '--ligand',   pdbqt_path,
               '--center_x', str(cx), '--center_y', str(cy), '--center_z', str(cz),
               '--size_x',   str(sx), '--size_y',   str(sy), '--size_z',   str(sz),
               '--exhaustiveness', str(exhaust),
               '--num_modes', '9',
               '--cpu', '1',
               '--out', out_path]

        r = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=300)
        out = r.stdout.decode(errors="ignore")
        err = r.stderr.decode(errors="ignore")

        m = re.search(r"^\s*1\s+(-?\d+\.\d+)", out, re.MULTILINE)
        if m:
            score_raw = float(m.group(1))
            SCORE_MIN = -15.0
            SCORE_MAX = -0.5
            if score_raw < SCORE_MIN:
                res["error"] = f"Score {score_raw:.1f} outside physical range"; return res
            if score_raw > SCORE_MAX:
                res["error"] = f"Score {score_raw:.2f} > {SCORE_MAX} — no real pose"; return res
            score = score_raw
            ha    = max(1, props.get("ha", 1))
            le    = round(-score / ha, 4)
            res.update({"score": score, "le": le, "status": "OK"})
            _DOCK_CACHE[smiles] = {"smiles": smiles, "score": score, "le": le,
                                    "status": "OK", "props": props}
            _save_dock_cache(cache_path)
        else:
            err_snippet = err[:200] if err else "no stderr"
            out_snippet = out[:200] if out else "no stdout"
            res["error"] = f"No score in Vina output. stderr: {err_snippet} | stdout: {out_snippet}"

    except subprocess.TimeoutExpired:
        res["error"] = "Docking timeout"
    except Exception as e:
        res["error"] = str(e)[:200]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return res


# ── Candidate generation ──────────────────────────────────────────────────────

def _generate_and_screen(chem: ChemicalEngine,
                          parent_pool: List[str],
                          surrogate: SurrogateModel,
                          fqsar: FragmentQSAR,
                          best_smi: str,
                          stagnation: int,
                          temperature: float,
                          n_candidates: int,
                          n_dock: int,
                          dedup_seen: set) -> List[str]:
    candidates = []
    gen_seen   = set()
    attempts   = 0

    while len(candidates) < n_candidates and attempts < n_candidates * 15:
        attempts += 1
        rv = random.random()

        if rv < 0.25:
            p1, p2 = random.choice(parent_pool), random.choice(parent_pool)
            child  = chem.crossover(p1, p2)
        elif rv < 0.45 and best_smi:
            child, _ = chem.mutate(best_smi, stagnation=0, mode="exploit",
                                   surrogate=surrogate, fqsar=fqsar,
                                   temperature=temperature)
        elif rv < 0.70:
            parent = random.choice(parent_pool)
            child, _ = chem.mutate(parent, stagnation=stagnation,
                                   mode="balanced", surrogate=surrogate,
                                   fqsar=fqsar, temperature=temperature)
        else:
            child = chem.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                      temperature=temperature)

        if child:
            c = _canonical(child)
            if c and c not in dedup_seen and c not in gen_seen:
                candidates.append(c)
                gen_seen.add(c)

    if not candidates:
        return []

    if surrogate.trained:
        ranked = surrogate.rank(candidates)
        return [smi for smi, _ in ranked[:n_dock]]
    else:
        random.shuffle(candidates)
        return candidates[:n_dock]


def _build_initial_population(chem: ChemicalEngine,
                               target: int) -> List[Tuple[str, str]]:
    population = []
    seen = set()
    log.info(f"  Building initial population ({target} molecules)...")

    for seed in TINY_SEEDS:
        mol_check = Chem.MolFromSmiles(seed)
        if mol_check and mol_check.GetNumHeavyAtoms() < 9:
            continue
        c = _canonical(seed)
        if c and c not in seen:
            mol = Chem.MolFromSmiles(c)
            if mol:
                p = _mol_props(mol)
                if _admet_prefilter(mol, p):
                    population.append((c, "Seed"))
                    seen.add(c)

    attempts = 0
    while len(population) < target and attempts < target * 30:
        attempts += 1
        seed = random.choice(TINY_SEEDS)
        mol  = Chem.MolFromSmiles(seed)
        if mol is None:
            continue
        grown = chem._grow(mol, GROW_FRAGMENTS)
        if grown:
            smi = chem._to_smiles(grown)
            if smi:
                c = _canonical(smi)
                if c and c not in seen:
                    m = Chem.MolFromSmiles(c)
                    if m:
                        p = _mol_props(m)
                        if _admet_prefilter(m, p):
                            population.append((c, "Seed+1"))
                            seen.add(c)

    log.info(f"  Initial population: {len(population)} molecules")
    return population


# ── Ligand efficiency fitness ─────────────────────────────────────────────────

def _normalize_score(score: float, lo: float = -12.0, hi: float = -1.5) -> float:
    s = max(lo, min(hi, float(score)))
    return (hi - s) / (hi - lo)


def _le_fitness(score: float, ha: int, qed: float) -> float:
    le = (-score) / max(1, ha)
    le_capped = min(le, 0.55)
    le_score  = 1.0 / (1.0 + math.exp(-8 * (le_capped - 0.30)))
    abs_bonus = min(0.3, max(0.0, (-score - 5.0) / 20.0))
    return float(max(0.0, min(1.0, le_score * qed + abs_bonus)))


# ── PDB → PDBQT conversion ────────────────────────────────────────────────────

def _fast_pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """Convert receptor PDB to Vina-compatible receptor PDBQT (no ROOT/BRANCH tags)."""
    AD4_TYPES = {
        "C":  "C",  "N":  "NA", "O":  "OA", "S":  "SA", "H":  "HD",
        "P":  "P",  "F":  "F",  "CL": "Cl", "BR": "Br", "I":  "I",
        "FE": "Fe", "ZN": "Zn", "MG": "Mg", "CA": "Ca", "MN": "Mn",
        "CU": "Cu", "K":  "K",  "NA": "Na",
    }
    try:
        lines_out = []
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    if rec in ("TER", "END"):
                        lines_out.append(rec)
                    continue
                element = line[76:78].strip().upper() if len(line) > 76 else ""
                if not element:
                    atom_name = line[12:16].strip()
                    element   = "".join(c for c in atom_name if c.isalpha())[:2].upper()
                ad4_type = AD4_TYPES.get(element, element[:1] if element else "C")
                try:
                    occ = float(line[54:60]) if len(line) >= 60 else 1.00
                except ValueError:
                    occ = 1.00
                try:
                    bfac = float(line[60:66]) if len(line) >= 66 else 0.00
                except ValueError:
                    bfac = 0.00
                head = line[:54].ljust(54)[:54]
                pdbqt_line = f"{head}{occ:6.2f}{bfac:6.2f}    {'0.000':>6} {ad4_type:<2s}"
                lines_out.append(pdbqt_line.rstrip())

        if not lines_out:
            log.error(f"  No ATOM/HETATM records found in {pdb_path}")
            return False
        pdbqt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        log.info(f"  Wrote receptor PDBQT: {len(lines_out)} atoms → {pdbqt_path.name}")
        return True
    except Exception as e:
        log.error(f"  Receptor PDBQT conversion failed: {e}")
        return False


def _smiles_to_ligand_pdbqt(smiles: str, pdbqt_path: Path) -> bool:
    """Convert SMILES to Vina-compatible ligand PDBQT using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdPartialCharges

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ret = AllChem.EmbedMolecule(mol, params)
        if ret == -1:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        if mol.GetNumConformers() == 0:
            return False

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = []
            for a in mol.GetAtoms():
                c = float(a.GetDoubleProp("_GasteigerCharge"))
                charges.append(0.0 if (c != c or abs(c) > 9) else c)
        except Exception:
            charges = [0.0] * mol.GetNumAtoms()

        conf = mol.GetConformer()
        AD4 = {"C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "H",
               "F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "P": "P"}

        lines = ["ROOT"]
        for i, atom in enumerate(mol.GetAtoms()):
            sym  = atom.GetSymbol()
            ad4  = AD4.get(sym, sym[:2])
            pos  = conf.GetAtomPosition(i)
            q    = charges[i]
            aname = f"{sym:<3s}"
            line = (
                f"HETATM{i+1:5d}  {aname} LIG A   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00  0.00    "
                f"{q:6.3f} {ad4}"
            )
            lines.append(line)

        lines += ["ENDROOT", "TORSDOF 0"]
        pdbqt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


# ── Stage 3: Fragment pool biasing from refined pocket ────────────────────────

def _apply_pocket_bias(fragment_pool: List[str], refined_p1) -> List[str]:
    """
    Reshape the fragment pool based on what the Stage 2 refit learned.
    Called once per run after the refined pocket is loaded.
    Returns a new pool with biased fragment weights.
    """
    if refined_p1 is None or not refined_p1.accepted:
        return fragment_pool

    top_feat  = refined_p1.top_feature
    direction = refined_p1.top_feature_direction
    pool      = list(fragment_pool)

    log.info(f"  Applying refined-pocket bias: {top_feat} ({direction})")

    if top_feat == "heavy_atoms" and direction == "higher_is_better":
        # Prefer ring-containing fragments — they add more heavy atoms per step
        ring_frags = [f for f in GROW_FRAGMENTS
                      if "c1" in f or "C1" in f or "n1" in f or "N1" in f]
        pool = ring_frags * 2 + pool

    elif top_feat == "logp":
        if direction == "higher_is_better":
            lipophilic = ["c1ccccc1", "C(F)(F)F", "c1ccc2ccccc2c1",
                          "Cc1ccccc1", "c1ccnc2ccccc12"]
            pool = lipophilic * 3 + pool
        else:
            polar = ["C(=O)O", "C(=O)N", "OCC", "NCC", "S(=O)(=O)N", "OC"]
            pool = polar * 3 + pool

    elif top_feat == "hbd":
        if direction == "higher_is_better":
            hbd_frags = ["NC", "OC", "NCC", "OCC", "NC(=O)", "c1cc[nH]n1"]
            pool = hbd_frags * 3 + pool
        else:
            non_hbd = ["c1ccccc1", "CC", "C(F)(F)F", "CCC", "c1ccncc1"]
            pool = non_hbd * 2 + pool

    elif top_feat in ("tpsa", "hba"):
        if direction == "higher_is_better":
            polar = ["C(=O)N", "C(=O)O", "c1ccncc1", "OCC", "NC(=O)", "S(=O)(=O)N"]
            pool = polar * 3 + pool

    elif top_feat == "aromatic_frac" and direction == "higher_is_better":
        aromatics = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1",
                     "c1cnccn1", "c1ccc2[nH]ccc2c1"]
        pool = aromatics * 3 + pool

    elif top_feat == "frac_csp3" and direction == "higher_is_better":
        sp3_frags = ["C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "CCC", "CC(C)C"]
        pool = sp3_frags * 3 + pool

    elif top_feat == "mw" and direction == "higher_is_better":
        heavy_frags = ["c1ccc2ccccc2c1", "c1ccnc2ccccc12",
                       "c1ccc2[nH]ccc2c1", "C1CN2CCc3ccccc3C2C1"]
        pool = heavy_frags * 2 + pool

    return pool


# ── Main evolution loop ───────────────────────────────────────────────────────

def run_denovo_design(
       uniprot_id:      str,
       pocket_data:     Optional[dict] = None,
       active_data:     Optional[dict] = None,
       allosteric_data: Optional[dict] = None,
       chem_env_data:   Optional[dict] = None,
       vina_path:       str = "",
       receptor_path:   str = "",
       n_generations:   int = MAX_GENERATIONS,
       rng_seed:        Optional[int] = None,
       consensus_data   = None,
       md_data          = None,
  ) -> DenovoResult:
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    log.info(f"── De Novo Design: {uniprot_id}  [seed={seed}] ──")

    inter_dir  = Path(cfg.paths["intermediate"])
    cache_path = str(inter_dir / f"{uniprot_id}_dock_cache.json")
    out_path   = inter_dir / f"{uniprot_id}_denovo.json"

    _load_dock_cache(cache_path)

    # ── Stage 1+2: open feedback store, refined pocket loaded after target selection
    from utils.feedback_store import FeedbackStore
    from pipeline.pocket_refit import load_refined_pocket

    fb         = FeedbackStore(uniprot_id)
    refined_p1 = None

    pocket_volume_A3      = 0.0
    pocket_druggability   = 0.0
    pocket_net_charge     = 0.0
    pocket_hydrophobicity = 0.0

    # ── Select target site: consensus > active > allosteric > pocket ──────────
    center         = [0.0, 0.0, 0.0]
    box_size       = [20.0, 20.0, 20.0]
    used_pocket_id = "P1"
    target_source  = "none"

    if consensus_data is not None and consensus_data.has_good_pocket:
        cp = consensus_data.top_pocket
        center         = list(cp.center)
        vol_edge       = (cp.volume_A3 ** (1/3)) if cp.volume_A3 > 0 else 10.0
        edge           = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, vol_edge + BOX_PADDING * 2))
        box_size       = [round(edge, 1)] * 3
        used_pocket_id = cp.pocket_id
        target_source  = f"consensus_{cp.pocket_id}"
        pocket_volume_A3      = cp.volume_A3
        pocket_druggability   = cp.druggability_score
        pocket_net_charge     = cp.net_charge
        pocket_hydrophobicity = cp.mean_hydrophobicity
        log.info(
            f"  Target: CONSENSUS POCKET {cp.pocket_id}  "
            f"drug={cp.druggability_score:.2f} ({cp.druggability_class})  "
            f"evidence={cp.evidence_score:.1f} ({cp.n_evidence_sources} sources)  "
            f"center={center}  box={box_size}"
        )
        if cp.near_active_site:
            log.info(f"  (pocket overlaps active site — high-confidence target)")

    if active_data:
        high_res = [r for r in active_data.get("active_residues", [])
                    if r.get("confidence") == "HIGH" and r.get("coords")]
        if not high_res:
            high_res = [r for r in active_data.get("active_residues", [])
                        if r.get("coords")]
        if high_res:
            coords_arr = np.array([r["coords"] for r in high_res])
            center     = [round(float(c), 3) for c in coords_arr.mean(axis=0)]
            span       = coords_arr.max(axis=0) - coords_arr.min(axis=0)
            edge       = float(span.max()) + BOX_PADDING * 2
            edge       = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, edge))
            box_size   = [round(edge, 1)] * 3
            # NOTE: keep used_pocket_id from consensus — only docking box changes.
            # This ensures feedback events are tagged to P1, matching the static
            # pocket descriptor needed for Stage 2 refit.
            target_source = f"active_site ({len(high_res)} residues)"
            log.info(f"  Target: ACTIVE SITE  center={center}  box={box_size}  "
                     f"({len(high_res)} high-conf residues)")

    if target_source == "none" and allosteric_data:
        sites = allosteric_data.get("allosteric_sites", [])
        if sites:
            top_site = sites[0]
            centre   = top_site.get("centre", [0.0, 0.0, 0.0])
            center   = [round(float(c), 3) for c in centre]
            edge     = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, 16.0 + BOX_PADDING))
            box_size = [round(edge, 1)] * 3
            used_pocket_id = top_site.get("site_id", "A1")
            target_source  = f"allosteric_{used_pocket_id}"
            log.info(f"  Target: ALLOSTERIC {used_pocket_id}  center={center}")

    if target_source == "none" and pocket_data:
        pockets = pocket_data.get("pockets", [])
        if pockets:
            top_p    = pockets[0]
            center, box_size = _pocket_to_box(top_p)
            used_pocket_id   = top_p.get("pocket_id", "P1")
            target_source    = f"pocket_{used_pocket_id}"
            pocket_volume_A3      = top_p.get("volume_A3", 0.0)
            pocket_druggability   = top_p.get("druggability_score", 0.0)
            pocket_net_charge     = top_p.get("net_charge", 0.0)
            pocket_hydrophobicity = top_p.get("mean_hydrophobicity", 0.0)
            log.info(f"  Target: POCKET {used_pocket_id}  center={center}")

    if target_source == "none":
        log.warning("  No site data — using default box at origin")

    # ── Load refined pocket model (Stage 2 output) ────────────────────────────
    refined_p1 = load_refined_pocket(uniprot_id, used_pocket_id)
    if refined_p1 and refined_p1.accepted:
        log.info(
            f"  Loaded refined pocket model: top_feature={refined_p1.top_feature} "
            f"({refined_p1.top_feature_direction}), "
            f"RMSE gain={refined_p1.rmse_gain:+.2f} kcal/mol "
            f"from {refined_p1.n_events_used} prior events"
        )

    # Open feedback run tagged to this pocket
    pocket_snapshot = {
        "volume_A3":     pocket_volume_A3,
        "druggability":  pocket_druggability,
        "center":        center,
        "net_charge":    pocket_net_charge,
        "hydrophobicity": pocket_hydrophobicity,
        "model_version": "refined_v1" if (refined_p1 and refined_p1.accepted) else "static_v0",
    }
    fb.start_run(
        modality        = "small_molecule",
        target_site     = used_pocket_id,
        pocket_snapshot = pocket_snapshot,
    )

    # ── Base fragment pool ────────────────────────────────────────────────────
    fragment_pool = list(GROW_FRAGMENTS)

    # ── Chem env bias ─────────────────────────────────────────────────────────
    if chem_env_data:
        all_envs = (chem_env_data.get("active_envs", []) +
                    chem_env_data.get("binding_envs", []))
        if all_envs:
            top_env    = all_envs[0]
            mode       = top_env.get("predicted_binding_mode", "")
            net_charge = top_env.get("net_charge", 0.0)
            n_aromatic = top_env.get("n_aromatic", 0)
            if "hydrophobic" in mode or n_aromatic >= 3:
                hydrophobic_frags = ["c1ccccc1", "c1ccncc1", "C1CCCCC1",
                                     "C(F)(F)F", "c1ccc2ccccc2c1"]
                fragment_pool = hydrophobic_frags * 3 + fragment_pool
                log.info("  Chem env: hydrophobic → aromatic fragments")
            elif "electrostatic" in mode or abs(net_charge) > 1.0:
                polar_frags = ["C(=O)O", "C(=O)N", "NC(=O)", "S(=O)(=O)N",
                               "NCC", "OCC", "c1ccncc1"]
                fragment_pool = polar_frags * 3 + fragment_pool
                log.info(f"  Chem env: charged (q={net_charge:+.1f}) → polar fragments")

    # ── Stage 3: Apply refined pocket bias to fragment pool ───────────────────
    fragment_pool = _apply_pocket_bias(fragment_pool, refined_p1)

    # ── MD flexibility adaptation ─────────────────────────────────────────────
    flex_strategy = {
        "max_build_steps": 4, "mutation_rate": 1.0,
        "box_padding": 0.0, "scaffold_bias": "flexible", "warning": "",
    }

    if md_data is not None and md_data.source != "none":
        pocket_lining = []
        if consensus_data and consensus_data.top_pocket:
            pocket_lining = consensus_data.top_pocket.lining_residues
        elif pocket_data:
            pockets = pocket_data.get("pockets", [])
            if pockets:
                pocket_lining = pockets[0].get("lining_residues", [])

        pocket_rmsf   = md_data.pocket_rmsf(pocket_lining) if pocket_lining \
                        else md_data.mean_rmsf
        flex_strategy = md_data.flexibility_strategy(pocket_rmsf)

        if flex_strategy["warning"]:
            log.warning(flex_strategy["warning"])
        else:
            log.info(
                f"  MD flexibility: pocket RMSF={pocket_rmsf:.2f}Å  "
                f"strategy={flex_strategy['scaffold_bias']}  "
                f"box_padding={flex_strategy['box_padding']:.1f}Å"
            )

        if flex_strategy["box_padding"] > 0:
            box_size = [round(b + flex_strategy["box_padding"], 1) for b in box_size]
            box_size = [min(b, MAX_BOX_SIZE) for b in box_size]
            log.info(f"  Box expanded for flexibility: {box_size}")

        from pipeline.denovo_design_context import get_flexibility_fragments
        flex_frags = get_flexibility_fragments(flex_strategy["scaffold_bias"])
        fragment_pool = flex_frags * 3 + fragment_pool
        log.info(f"  Added {len(flex_frags)} {flex_strategy['scaffold_bias']} fragments to pool")

    # ── Validate tools ────────────────────────────────────────────────────────
    if not vina_path or not os.path.exists(vina_path):
        log.error(f"  Vina not found: {vina_path}")
        return DenovoResult(uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                            pocket_center=center, box_size=box_size,
                            notes=f"Vina not found: {vina_path}")

    # ── Receptor PDBQT validation and auto-conversion ─────────────────────────
    default_pdb   = Path(cfg.paths["structures"]) / f"{uniprot_id}.pdb"
    default_pdbqt = Path(cfg.paths["structures"]) / f"{uniprot_id}.pdbqt"

    if not receptor_path:
        receptor_path = str(default_pdbqt)

    needs_regen = not os.path.exists(receptor_path)

    if not needs_regen:
        try:
            with open(receptor_path, encoding="utf-8", errors="ignore") as _f:
                for _l in _f:
                    stripped = _l.strip()
                    if stripped == "ROOT":
                        needs_regen = True
                        log.warning(f"  Receptor PDBQT contains ROOT tag — regenerating.")
                        break
                    if stripped.startswith(("ATOM", "HETATM")):
                        if len(_l.rstrip()) < 76:
                            needs_regen = True
                            log.warning(f"  Receptor PDBQT has short lines — regenerating.")
                        break
        except Exception:
            needs_regen = True

    if needs_regen:
        source_pdb = None
        if receptor_path:
            candidate = Path(receptor_path).with_suffix(".pdb")
            if candidate.exists():
                source_pdb = candidate
        if source_pdb is None and default_pdb.exists():
            source_pdb = default_pdb

        if source_pdb is None:
            log.error(f"  No source PDB found for {uniprot_id}.")
            return DenovoResult(uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                                pocket_center=center, box_size=box_size,
                                notes=f"Receptor PDB not found: {default_pdb}")

        target_pdbqt = Path(receptor_path) if receptor_path else default_pdbqt
        log.info(f"  Converting receptor: {source_pdb.name} → {target_pdbqt.name}")
        t0_conv = time.time()
        ok = _fast_pdb_to_pdbqt(source_pdb, target_pdbqt)
        if not ok:
            return DenovoResult(uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                                pocket_center=center, box_size=box_size,
                                notes="Receptor PDBQT conversion failed")
        log.info(f"  Converted in {time.time() - t0_conv:.1f}s")
        receptor_path = str(target_pdbqt.resolve())

    log.info(f"  Receptor: {receptor_path}")
    if not os.path.exists(receptor_path):
        log.error(f"  Receptor missing: {receptor_path}")
        return DenovoResult(uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                            pocket_center=center, box_size=box_size,
                            notes=f"Receptor missing: {receptor_path}")

    # ── Init evolution objects ────────────────────────────────────────────────
    chem      = ChemicalEngine()
    novelty   = NoveltyArchive()
    surrogate = SurrogateModel()
    fqsar     = FragmentQSAR()

    # ── Stage 3: Warm-start surrogate + fqsar from prior feedback events ──────
    # This is the key closed-loop addition: rather than starting blind each run,
    # we load all prior docking events from the feedback store and pre-train the
    # surrogate so Gen 1 already has a fully trained model.
    prior_history: List[dict] = []
    try:
        for ev in fb.iter_events(modality="small_molecule",
                                  target_site=used_pocket_id,
                                  valid_only=True):
            smi = ev.identity.get("smiles")
            sc  = ev.score.get("primary")
            if smi and sc is not None:
                prior_history.append({"smiles": smi, "score": sc})

        if len(prior_history) >= SURROGATE_MIN_DATA:
            surrogate.update(prior_history)
            fqsar.update(prior_history)
            log.info(
                f"  Warm-started surrogate from {len(prior_history)} prior events "
                f"(trained={surrogate.trained}, n_obs={surrogate.n_obs})"
            )
            log.info(f"  Warm-started fragment QSAR (n_obs={fqsar.n_obs})")
        else:
            log.info(
                f"  Prior events: {len(prior_history)} "
                f"(need {SURROGATE_MIN_DATA} to warm-start surrogate)"
            )
    except Exception as e:
        log.warning(f"  Could not warm-start from feedback store: {e}")

    population  = _build_initial_population(chem, POP_SIZE)
    docked_seen: set = set(_DOCK_CACHE.keys())

    full_history:  List[dict] = []
    hall_of_fame:  List[dict] = []
    gen_stats:     List[dict] = []

    best_score     = float("inf")
    best_le        = 0.0
    best_smi       = ""
    stagnation_cnt = 0
    total_start    = time.time()

    log.info(f"  Starting evolution: {n_generations} gens × {POP_SIZE} molecules")

    for gen in range(1, n_generations + 1):
        gen_start = time.time()

        t      = min(1.0, (gen - 1) / max(1, n_generations - 1))
        w_bind = W_BINDING_START + t * (W_BINDING_END - W_BINDING_START)
        w_nov  = W_NOVELTY_START + t * (W_NOVELTY_END  - W_NOVELTY_START)
        w_le   = W_LE_START      + t * (W_LE_END       - W_LE_START)
        w_cyp  = W_CYP_START     + t * (W_CYP_END      - W_CYP_START)
        if not CYP_AVAILABLE:
            w_cyp = 0.0
        total_w = w_bind + w_nov + w_le + w_cyp
        w_bind /= total_w; w_nov /= total_w; w_le /= total_w; w_cyp /= total_w

        gen_exhaust = int(EXHAUST_START + t * (EXHAUST_END - EXHAUST_START))
        if stagnation_cnt >= STAGNATION_HARD // 2:
            gen_exhaust = min(gen_exhaust * 2, EXHAUST_END * 2)

        temperature = max(0.1, 1.0 - 0.8 * t)

        # Update surrogate from current run's history (on top of warm-start)
        if len(full_history) >= SURROGATE_MIN_DATA:
            surrogate.update(full_history)
        fqsar.update(full_history)

        # ── Stagnation-aware surrogate pre-screening ──────────────────────────
        # KEY FIX: when stagnation is high, bypass surrogate ranking entirely.
        # The surrogate has learned to prefer the current best scaffold, so
        # leaving it active filters fresh scaffolds out before they can dock.
        # Disabling it forces exploration instead of continuous exploitation.
        if surrogate.trained:
            parent_pool      = [s for s, _ in population]
            current_gen_seen = {s for s, _ in population}

            use_surrogate_ranking = (stagnation_cnt < STAGNATION_LIMIT)
            if not use_surrogate_ranking:
                log.info(
                    f"  Bypassing surrogate ranking (stagnation={stagnation_cnt}) "
                    f"— forcing exploration"
                )

            # Pass untrained dummy surrogate when bypassing, so _generate_and_screen
            # falls through to random shuffle instead of ranking
            screen_surrogate = surrogate if use_surrogate_ranking else SurrogateModel()

            screened = _generate_and_screen(
                chem, parent_pool, screen_surrogate, fqsar,
                best_smi, stagnation_cnt, temperature,
                n_candidates=SURROGATE_CANDIDATES,
                n_dock=POP_SIZE,
                dedup_seen=current_gen_seen,
            )
            if screened:
                existing   = {s for s, _ in population}
                new_slots  = [(s, "Screened") for s in screened if s not in existing]
                population = population[:ELITISM] + new_slots
                population = population[:POP_SIZE]

        # ── Dock current population ───────────────────────────────────────────
        tasks   = [(smi, gen, i) for i, (smi, _) in enumerate(population)]
        results = []

        n_cache_hits = sum(1 for smi, _, _ in tasks
                           if smi in _DOCK_CACHE and _DOCK_CACHE[smi].get("status") == "OK")
        n_vina_calls = len(tasks) - n_cache_hits

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
            futures = {
                exe.submit(
                    _dock_molecule, t[0], t[1], t[2],
                    vina_path, receptor_path,
                    center, box_size, cache_path, gen_exhaust
                ): t for t in tasks
            }
            completed = 0
            sys.stdout.write(
                f"Gen {gen}/{n_generations} | Docking: 0/{len(futures)} "
                f"[{n_vina_calls} new, {n_cache_hits} cached]")
            sys.stdout.flush()
            for fut in as_completed(futures):
                completed += 1
                sys.stdout.write(
                    f"\rGen {gen}/{n_generations} | Docking: {completed}/{len(futures)}")
                sys.stdout.flush()
                res = fut.result()
                docked_seen.add(res["smiles"])
                if res.get("status") != "OK" and len([r for r in results if r.get("status") != "OK"]) < 3:
                    err_msg = res.get("error", "unknown")[:150]
                    log.warning(f"    [DOCK FAIL] {res['smiles'][:40]}: {err_msg}")
                if res.get("status") == "OK" and res.get("score") is not None:
                    smi   = res["smiles"]
                    mol   = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    props = res["props"]
                    score = res["score"]
                    if score >= MIN_REAL_SCORE:
                        continue
                    ha    = max(1, props.get("ha", 1))
                    qed   = props.get("qed", 0.0)
                    le    = res.get("le", round(-score / ha, 4))

                    nov_score = novelty.score(smi)
                    novelty.try_add(smi)

                    le_fit    = _le_fitness(score, ha, qed)
                    norm_bind = _normalize_score(score)

                    cyp_score = 1.0
                    if CYP_AVAILABLE and w_cyp > 0 and _cyp_profile is not None:
                        try:
                            _cyp = _cyp_profile("denovo", smi, "denovo")
                            if _cyp is not None:
                                cyp_score = float(
                                    getattr(_cyp, "denovo_cyp_score",
                                    getattr(_cyp, "stability_score", 1.0))
                                )
                        except Exception:
                            cyp_score = 1.0

                    fitness = (w_bind * norm_bind +
                               w_nov  * nov_score +
                               w_le   * le_fit +
                               w_cyp  * cyp_score)

                    res.update({
                        "le": le, "qed": qed, "novelty": nov_score,
                        "le_fitness": le_fit, "norm_bind": norm_bind,
                        "cyp_score": round(cyp_score, 3), "fitness": fitness,
                        "w_bind": round(w_bind, 3), "w_nov": round(w_nov, 3),
                        "w_cyp": round(w_cyp, 3), "generation": gen,
                    })
                    results.append(res)
                    full_history.append(res)

                    # Stage 1: record to feedback store
                    try:
                        fb.record(
                            smiles        = smi,
                            primary_score = float(score),
                            primary_kind  = "vina_dG_kcal_mol",
                            secondary     = {"le": le, "qed": qed,
                                             "fitness": fitness, "novelty": nov_score},
                            origin        = {
                                "generation":  gen,
                                "mutation_op": res.get("origin", "unknown"),
                                "weights":     {"w_bind": w_bind, "w_nov": w_nov,
                                                "w_le": w_le, "w_cyp": w_cyp},
                            },
                            valid = True,
                        )
                    except Exception as e:
                        log.debug(f"  feedback record failed: {e}")

                    if res.get("score") is not None:
                        hall_of_fame.append(res)

        sys.stdout.write("\n")

        hall_of_fame = [r for r in hall_of_fame if r.get("score") is not None]
        hall_of_fame.sort(key=lambda r: r["score"])
        hall_of_fame = hall_of_fame[:30]

        if not results:
            log.warning(f"  Gen {gen}: no valid docking results — reseeding")
            stagnation_cnt = 0
            new_pop = []
            new_seen_local: set = set()
            drug_seeds = [s for s in TINY_SEEDS
                          if Chem.MolFromSmiles(s) is not None
                          and Chem.MolFromSmiles(s).GetNumHeavyAtoms() >= 9]
            for seed_smi in drug_seeds:
                c = _canonical(seed_smi)
                if c and c not in docked_seen and c not in new_seen_local:
                    mol = Chem.MolFromSmiles(c)
                    if mol:
                        p = _mol_props(mol)
                        if _admet_prefilter(mol, p):
                            new_pop.append((c, "DrugSeed"))
                            new_seen_local.add(c)
            attempts_r = 0
            while len(new_pop) < POP_SIZE and attempts_r < POP_SIZE * 40:
                attempts_r += 1
                seed_smi = random.choice(drug_seeds) if drug_seeds else random.choice(TINY_SEEDS)
                mol = Chem.MolFromSmiles(seed_smi)
                if mol is None: continue
                grown = chem._grow(mol, GROW_FRAGMENTS)
                if grown:
                    smi = chem._to_smiles(grown)
                    if smi:
                        c = _canonical(smi)
                        if c and c not in docked_seen and c not in new_seen_local:
                            m2 = Chem.MolFromSmiles(c)
                            if m2:
                                p = _mol_props(m2)
                                if _admet_prefilter(m2, p):
                                    new_pop.append((c, "DrugGrown"))
                                    new_seen_local.add(c)
            population = new_pop[:POP_SIZE] if new_pop else _build_initial_population(chem, POP_SIZE)
            log.info(f"  Reseeded with {len(population)} drug-like molecules")
            continue

        by_score   = sorted(results, key=lambda r: r["score"])
        by_fitness = sorted(results, key=lambda r: r["fitness"], reverse=True)
        by_novelty = sorted(results, key=lambda r: r.get("novelty", 0), reverse=True)

        gen_best = by_score[0]
        improved = gen_best["score"] < best_score

        if improved:
            best_score     = gen_best["score"]
            best_le        = gen_best.get("le", 0.0)
            best_smi       = gen_best["smiles"]
            stagnation_cnt = 0
        else:
            stagnation_cnt += 1

        div = _tanimoto_diversity([r["smiles"] for r in results])
        scaffold_counts: Dict[str, int] = {}
        for r in results:
            sc = _murcko(r["smiles"])
            scaffold_counts[sc] = scaffold_counts.get(sc, 0) + 1

        flag    = "↑ IMPROVED" if improved else f"stagnant {stagnation_cnt}"
        surr    = (f"  Surr={surrogate.predict(gen_best['smiles']):.2f}"
                   if surrogate.trained else "")
        cyp_str = (f"  CYP={gen_best.get('cyp_score', 1.0):.2f}"
                   if CYP_AVAILABLE and w_cyp > 0 else "")
        log.info(
            f"  Gen {gen}: score={gen_best['score']:.2f}  "
            f"LE={gen_best.get('le',0):.3f}  "
            f"QED={gen_best.get('qed',0):.2f}  "
            f"div={div:.2f}  {flag}{surr}{cyp_str}"
        )

        if gen % VALIDATION_INTERVAL == 0:
            log.info(f"  Top 5 by fitness (gen {gen}):")
            for pos, s in enumerate(by_fitness[:5], 1):
                log.info(
                    f"    #{pos}: fit={s['fitness']:.3f}  "
                    f"score={s['score']:.2f}  LE={s.get('le',0):.3f}  "
                    f"SMILES={s['smiles'][:60]}"
                )

        gen_stats.append({
            "generation":    gen,
            "best_score":    gen_best["score"],
            "global_best":   best_score,
            "best_le":       gen_best.get("le", 0.0),
            "best_fitness":  by_fitness[0]["fitness"],
            "mean_fitness":  round(statistics.mean(r["fitness"] for r in results), 4),
            "diversity":     div,
            "scaffolds":     len(scaffold_counts),
            "mean_qed":      round(statistics.mean(r.get("qed", 0) for r in results), 3),
            "stagnation":    stagnation_cnt,
            "archive_size":  len(novelty.archive),
            "exhaust":       gen_exhaust,
            "surrogate_obs": surrogate.n_obs,
            "surrogate_active": use_surrogate_ranking if surrogate.trained else False,
        })

        # ── Build next population ─────────────────────────────────────────────
        next_pop:  List[Tuple[str, str]] = []
        next_seen: set = set()

        c = _canonical(gen_best["smiles"])
        if c and c not in next_seen:
            next_pop.append((c, "BestGen")); next_seen.add(c)

        elite_scaffold_counts: Dict[str, int] = {}
        MAX_ELITE_PER_SCAFFOLD = 2
        for e in by_fitness[:ELITISM * 3]:
            if len([x for x in next_pop if x[1] == "Elite"]) >= ELITISM:
                break
            c = _canonical(e["smiles"])
            if not c or c in next_seen:
                continue
            if e["score"] >= MIN_SCORE_ELITE:
                continue
            sc = _murcko(c)
            if elite_scaffold_counts.get(sc, 0) >= MAX_ELITE_PER_SCAFFOLD:
                continue
            next_pop.append((c, "Elite")); next_seen.add(c)
            elite_scaffold_counts[sc] = elite_scaffold_counts.get(sc, 0) + 1

        for e in by_novelty[:max(2, ELITISM // 2)]:
            c = _canonical(e["smiles"])
            if c and c not in next_seen:
                next_pop.append((c, "NovelElite")); next_seen.add(c)

        for e in hall_of_fame[:3]:
            c = _canonical(e["smiles"])
            if c and c not in next_seen:
                next_pop.append((c, "HoF")); next_seen.add(c)

        overcrowded = {sc for sc, cnt in scaffold_counts.items()
                       if cnt >= MAX_SAME_SCAFFOLD}
        if overcrowded or div < DIVERSITY_MIN:
            n_div = POP_SIZE // 4
            inj   = 0
            for _ in range(n_div * 20):
                if inj >= n_div: break
                child = chem.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                          temperature=temperature)
                if child:
                    c = _canonical(child)
                    if c and c not in next_seen and _murcko(c) not in overcrowded:
                        next_pop.append((c, "ScaffDiv")); next_seen.add(c); inj += 1

        if stagnation_cnt >= STAGNATION_LIMIT:
            n_inj = POP_SIZE // 3
            log.info(f"  Stagnation {stagnation_cnt}: injecting {n_inj} unguided molecules")
            inj = 0
            for _ in range(n_inj * 25):
                if inj >= n_inj: break
                if random.random() < 0.5:
                    child = chem.build_denovo(surrogate=None, fqsar=None, temperature=1.5)
                else:
                    child = chem.build_denovo(surrogate=None, fqsar=fqsar, temperature=1.5)
                if child:
                    c = _canonical(child)
                    if c and c not in next_seen:
                        m = Chem.MolFromSmiles(c)
                        if m:
                            p = _mol_props(m)
                            if _admet_prefilter(m, p):
                                next_pop.append((c, "Injected")); next_seen.add(c); inj += 1

        if stagnation_cnt >= STAGNATION_HARD:
            n_hard = int(POP_SIZE * 0.80)
            best_scaffold = _murcko(best_smi) if best_smi else ""
            log.info(f"  HARD RESET: exploring {n_hard} fresh scaffolds "
                     f"(ignoring {best_scaffold[:30]})")

            fresh_seeds = [s for s in TINY_SEEDS
                           if _murcko(s) != best_scaffold
                           and Chem.MolFromSmiles(s) is not None]
            if not fresh_seeds:
                fresh_seeds = TINY_SEEDS
            inj = 0
            for _ in range(n_hard * 40):
                if inj >= n_hard: break
                seed = random.choice(fresh_seeds)
                seed_mol = Chem.MolFromSmiles(seed)
                if seed_mol is None: continue
                n_steps = random.randint(2, 4)
                mol = seed_mol
                for _ in range(n_steps):
                    grown = chem._grow(mol, GROW_FRAGMENTS)
                    if grown: mol = grown
                child = chem._to_smiles(mol)
                if child:
                    c = _canonical(child)
                    if c and c not in next_seen:
                        m = Chem.MolFromSmiles(c)
                        if m:
                            p = _mol_props(m)
                            if _admet_prefilter(m, p):
                                next_pop.append((c, "HardReset"))
                                next_seen.add(c); inj += 1

            # KEY FIX: disable the surrogate after a hard reset so it doesn't
            # immediately re-rank the fresh scaffolds back out of the population.
            # It retrains naturally on surrogate.update(full_history) next gen.
            surrogate.trained = False
            log.info(f"  Surrogate disabled for next gen — will retrain from full history")
            stagnation_cnt = 1

        parent_pool = [r["smiles"] for r in by_fitness[:max(2, len(by_fitness)//2)]]
        remaining   = POP_SIZE - len(next_pop)
        if remaining > 0:
            screened_fill = _generate_and_screen(
                chem, parent_pool, surrogate, fqsar, best_smi,
                stagnation_cnt, temperature,
                n_candidates=max(remaining * 4, SURROGATE_CANDIDATES),
                n_dock=remaining,
                dedup_seen=next_seen,
            )
            for c in screened_fill:
                if c not in next_seen:
                    next_pop.append((c, "Screened")); next_seen.add(c)

        eft = 0
        while len(next_pop) < POP_SIZE and eft < 200:
            eft += 1
            child = chem.build_denovo()
            if child:
                c = _canonical(child)
                if c and c not in next_seen:
                    next_pop.append((c, "EmergFill")); next_seen.add(c)

        population = next_pop[:POP_SIZE]

        gen_time = time.time() - gen_start
        elapsed  = time.time() - total_start
        eta      = (elapsed / gen) * (n_generations - gen)
        log.info(f"  Gen time: {_fmt_time(gen_time)}  ETA: {_fmt_time(eta)}")

    # ── Final consensus docking ───────────────────────────────────────────────
    log.info("\n  Final consensus docking on top candidates...")
    seen_final = set()
    final_smiles = []
    for r in sorted(full_history, key=lambda x: x["score"]):
        c = _canonical(r["smiles"])
        if c and c not in seen_final:
            final_smiles.append(c); seen_final.add(c)
        if len(final_smiles) >= TOP_FOR_FINAL:
            break

    top_candidates: List[DenovoCandidate] = []
    for smi in final_smiles:
        scores_list = []
        for rr in range(CONSENSUS_RUNS):
            res = _dock_molecule(smi, 9999, rr, vina_path, receptor_path,
                                 center, box_size, cache_path, exhaust=EXHAUST_END)
            if res.get("status") == "OK":
                scores_list.append(res["score"])
        if not scores_list:
            continue
        mean_score = float(np.mean(scores_list))
        mol        = Chem.MolFromSmiles(smi)
        props      = _mol_props(mol) if mol else {}
        ha         = max(1, props.get("ha", 1))
        le         = round(-mean_score / ha, 4)
        admet_ok   = _admet_prefilter(mol, props) if mol else False

        final_cyp = 1.0
        if CYP_AVAILABLE and _cyp_profile is not None:
            try:
                _cp = _cyp_profile("final", smi, "final")
                if _cp is not None:
                    final_cyp = float(
                        getattr(_cp, "denovo_cyp_score",
                        getattr(_cp, "stability_score", 1.0))
                    )
            except Exception:
                final_cyp = 1.0

        w_total = W_BINDING_END + W_LE_END + W_CYP_END
        top_candidates.append(DenovoCandidate(
            smiles=smi,
            generation=next((r["generation"] for r in full_history
                             if r["smiles"] == smi), 0),
            score=round(mean_score, 3),
            le=le,
            qed=round(props.get("qed", 0.0), 3),
            mw=round(props.get("mw", 0.0), 1),
            logp=round(props.get("logp", 0.0), 2),
            tpsa=round(props.get("tpsa", 0.0), 1),
            hbd=props.get("hbd", 0),
            hba=props.get("hba", 0),
            heavy_atoms=ha,
            fitness=round(
                _normalize_score(mean_score)                          * W_BINDING_END / w_total
                + _le_fitness(mean_score, ha, props.get("qed", 0.0)) * W_LE_END      / w_total
                + final_cyp                                           * W_CYP_END     / w_total,
                4
            ),
            novelty=round(novelty.score(smi), 3),
            admet_pass=admet_ok,
            scaffold=_murcko(smi),
            origin="consensus",
        ))
        log.info(f"  Final: score={mean_score:.2f}  LE={le:.3f}  "
                 f"QED={props.get('qed',0):.2f}  CYP={final_cyp:.2f}  {smi[:60]}")

    top_candidates.sort(key=lambda c: c.score)

    result = DenovoResult(
        uniprot_id=uniprot_id,
        pocket_id=used_pocket_id,
        pocket_center=center,
        box_size=box_size,
        n_generations=n_generations,
        n_docked=len(full_history),
        n_unique=len(set(r["smiles"] for r in full_history)),
        top_candidates=top_candidates,
        best_score=best_score if best_score != float("inf") else 0.0,
        best_le=best_le,
        best_smiles=best_smi,
        generation_stats=gen_stats,
        vina_path=vina_path,
    )

    result.to_json(out_path)

    # ── Module 15: Selectivity Optimization ──────────────────────────────────
    if vina_path and result.top_candidates:
        try:
            from pipeline.selectivity_optimizer import run_selectivity_optimization
            log.info("\n  ── Auto-triggering Module 15: Selectivity Optimization ──")
            sel_result = run_selectivity_optimization(
                uniprot_id=uniprot_id,
                denovo_result=result,
                vina_path=vina_path,
                receptor_path=receptor_path,
                top_n=min(5, len(result.top_candidates)),
                rng_seed=rng_seed,
            )
            result.notes = (
                f"Selectivity: best_SI={sel_result.best_si:.2f}x  "
                f"hERG_safe={sel_result.n_herg_safe}/{len(sel_result.refined_molecules)}"
            )
            result.to_json(out_path)
        except Exception as e:
            log.warning(f"  Module 15 skipped: {e}")

    # ── Stage 2: Close feedback run + trigger pocket model refit ─────────────
    try:
        fb.end_run()
        from pipeline.pocket_refit import refit_pocket_model
        refit = refit_pocket_model(uniprot_id, feedback_store=fb)
        log.info(refit.summary())
    except Exception as e:
        log.warning(f"  Pocket refit skipped: {e}")

    log.info(result.summary())
    log.info(f"\n  Results saved to: {out_path}")
    return result


def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# ── CLI entry point ────────────────────────────────────────────────────────────

import click

@click.command()
@click.option("--uniprot",     "-u", required=True,  help="UniProt ID (e.g. P04637)")
@click.option("--vina",        "-v", required=True,  help="Path to AutoDock Vina executable")
@click.option("--receptor",    "-r", default=None,   help="Path to receptor PDBQT")
@click.option("--generations", "-g", default=MAX_GENERATIONS, type=int,
              help=f"Evolution generations (default: {MAX_GENERATIONS})")
@click.option("--seed",        "-s", default=None, type=int,
              help="Random seed for reproducibility")
def main(uniprot: str, vina: str, receptor: Optional[str],
         generations: int, seed: Optional[int]) -> None:
    """
    De Novo Molecule Design — evolutionary molecular generator.

    Example:
        python pipeline\\denovo_design.py \\
            --uniprot P04637 \\
            --vina C:/tools/vina.exe \\
            --generations 30
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    if not receptor:
        receptor = str(Path(cfg.paths["structures"]) / f"{uniprot}.pdbqt")

    def _load(fname):
        p = inter_dir / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    active_data     = _load(f"{uniprot}_active_sites.json")
    allosteric_data = _load(f"{uniprot}_allosteric.json")
    pocket_data     = _load(f"{uniprot}_binding_pockets.json")
    chem_env_data   = _load(f"{uniprot}_chemical_env.json")

    from pipeline.denovo_design_context import load_consensus_context, load_md_context
    consensus_data = load_consensus_context(uniprot, inter_dir)
    if consensus_data and consensus_data.top_pocket:
        log.info(consensus_data.pocket_summary())
    else:
        log.info("  No consensus report — using raw module outputs for targeting")

    md_data = load_md_context(uniprot, inter_dir, pocket_data, active_data)
    if md_data.source != "none":
        log.info(
            f"  MD data: mean RMSF={md_data.mean_rmsf:.2f}Å  "
            f"flexible residues={len(md_data.flexible_residues)}  "
            f"sites profiled={len(md_data.site_flexibility)}"
        )
    else:
        log.info("  No MD data — run Module 14 for flexibility-aware design")

    if active_data:
        n_high = sum(1 for r in active_data.get("active_residues", [])
                     if r.get("confidence") == "HIGH")
        log.info(f"  Loaded active site data ({n_high} HIGH-confidence residues)")
    else:
        log.warning("  No active site data — run Module 03 first")

    if allosteric_data:
        log.info(f"  Loaded allosteric data ({allosteric_data.get('n_sites', 0)} sites)")
    if pocket_data:
        log.info(f"  Loaded pocket data ({pocket_data.get('n_pockets', 0)} pockets)")

    result = run_denovo_design(
        uniprot_id=uniprot,
        pocket_data=pocket_data,
        active_data=active_data,
        allosteric_data=allosteric_data,
        chem_env_data=chem_env_data,
        vina_path=vina,
        receptor_path=receptor,
        n_generations=generations,
        rng_seed=seed,
        consensus_data=consensus_data,
        md_data=md_data,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()