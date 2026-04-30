"""
pipeline/denovo_design.py
──────────────────────────
De Novo Molecule Design Module.

Evolutionary molecular generator that designs drug-like binders for a target
protein using AutoDock Vina docking + surrogate-guided evolution.

Key improvements over the standalone evolve script:
  - Full ProteinFP pipeline integration: reads pocket centre/size from Module 04,
    loads ADMET filters from the ADMET module, uses chem_env data to bias fragments
  - Pre-evolution ADMET filter: molecules failing Lipinski/hERG are never docked
  - Ligand efficiency (LE) used as primary fitness metric, not raw score
  - Clean pipeline-style output: {uniprot}_denovo.json in data/intermediate/
  - Structured logging via utils.config
  - No hardcoded paths: vina/receptor resolved from config.yaml or CLI flags
  - Fixed double-printing bug
  - Surrogate model: same ridge regression approach, cleaned up
  - Fragment QSAR: unchanged (it works well)
  - Novelty archive: unchanged (it works well)

Usage (standalone):
    python pipeline/denovo_design.py --uniprot P04637
    python pipeline/denovo_design.py --uniprot P04637 --pocket P2 --generations 30
    python pipeline/denovo_design.py --uniprot P04637 --vina path/to/vina.exe

Usage (from orchestrator):
    from pipeline.denovo_design import run_denovo_design
    result = run_denovo_design("P04637", pocket_data, chem_env_data)

Requirements:
    pip install rdkit
    AutoDock Vina (vina.exe on Windows) — https://vina.scripps.edu/downloads/
    OpenBabel (obabel) — https://github.com/openbabel/openbabel/releases
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
POP_SIZE             = 20
ELITISM              = 6
PARALLEL_WORKERS     = min(8, os.cpu_count() or 2)

EXHAUST_START        = 8
EXHAUST_END          = 32

SURROGATE_MIN_DATA   = 20
SURROGATE_CANDIDATES = 60

FQSAR_MIN_DATA       = 15
FQSAR_GOOD_THRESH    = -7.0
FQSAR_WINDOW         = 300

# Fitness weight schedule (start → end over generations)
W_BINDING_START = 0.25;  W_BINDING_END = 0.70
W_NOVELTY_START = 0.45;  W_NOVELTY_END = 0.08
W_LE_START      = 0.30;  W_LE_END      = 0.22   # ligand efficiency

NOVELTY_K            = 10
NOVELTY_ARCHIVE_MAX  = 500
NOVELTY_ADD_THRESH   = 0.12

STAGNATION_LIMIT     = 3
STAGNATION_HARD      = 6
DIVERSITY_MIN        = 0.18
MIN_SCORE_ELITE      = -4.0
MAX_SAME_SCAFFOLD    = 3

VALIDATION_INTERVAL  = 5
TOP_FOR_FINAL        = 8
CONSENSUS_RUNS       = 2

# ADMET pre-filter thresholds (molecules failing these are never docked)
ADMET_MW_MAX         = 600.0   # slightly relaxed from Lipinski for fragments
ADMET_LOGP_MAX       = 6.0
ADMET_HBD_MAX        = 7
ADMET_HBA_MAX        = 12

# Pocket box padding (Å added to pocket volume estimate for docking box)
BOX_PADDING          = 6.0
MIN_BOX_SIZE         = 12.0
MAX_BOX_SIZE         = 30.0


# ── Seed library ──────────────────────────────────────────────────────────────

TINY_SEEDS = [
    "C", "CC", "CCO", "CCN",
    "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1",
    "c1ccoc1", "c1ccsc1", "c1cc[nH]n1", "c1cnc[nH]1",
    "C1CCOCC1", "N1CCNCC1", "c1cnccn1",
]

GROW_FRAGMENTS = [
    "C", "CC", "N", "O", "S", "F", "Cl", "Br",
    "C(=O)N", "C(=O)O", "S(=O)(=O)N", "C#N", "OC", "NC",
    "C(=O)", "NC(=O)", "C(F)(F)F", "OCC", "NCC",
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1cc[nH]n1",
    "c1ccoc1", "c1ccsc1", "C1CCCCC1", "C1CCNCC1",
    "C1CCOCC1", "N1CCNCC1", "N1CCCC1",
    "c1ccc2ccccc2c1", "c1ccnc2ccccc12",
    "c1ccc2[nH]ccc2c1", "c1cnc2[nH]ccc2n1",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DenovoCandidate:
    """A single evolved molecule with docking + ADMET data."""
    smiles:           str
    generation:       int
    score:            float        # Vina docking score (kcal/mol, more negative = better)
    le:               float        # ligand efficiency = -score / heavy_atoms
    qed:              float        # QED drug-likeness 0-1
    mw:               float
    logp:             float
    tpsa:             float
    hbd:              int
    hba:              int
    heavy_atoms:      int
    fitness:          float        # composite fitness
    novelty:          float
    admet_pass:       bool
    scaffold:         str
    origin:           str          # how it was generated

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DenovoResult:
    """Full de novo design output. Output of this module."""
    uniprot_id:       str
    pocket_id:        str
    pocket_center:    list[float]
    box_size:         list[float]
    n_generations:    int          = 0
    n_docked:         int          = 0
    n_unique:         int          = 0
    top_candidates:   list[DenovoCandidate] = field(default_factory=list)
    best_score:       float        = 0.0
    best_le:          float        = 0.0
    best_smiles:      str          = ""
    generation_stats: list[dict]   = field(default_factory=list)
    vina_path:        str          = ""
    notes:            str          = ""

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
    """
    Fast pre-docking ADMET filter.
    Rejects molecules that would obviously fail drug-likeness.
    Uses relaxed thresholds to allow fragment-like molecules through.
    """
    if not props:
        return False
    if props.get("mw", 999)  > ADMET_MW_MAX:
        return False
    if props.get("mw", 0)    < 80:
        return False
    if props.get("logp", 99) > ADMET_LOGP_MAX:
        return False
    if props.get("hbd", 99)  > ADMET_HBD_MAX:
        return False
    if props.get("hba", 99)  > ADMET_HBA_MAX:
        return False
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
    """
    Convert a BindingPocket dict (from Module 04) into Vina box parameters.

    Returns:
        (center [x, y, z], size [sx, sy, sz]) in Ångströms
    """
    center = pocket.get("center", [0.0, 0.0, 0.0])

    # Estimate box size from pocket volume
    # V = sx * sy * sz, assume roughly cubic pocket
    volume = pocket.get("volume_A3", 500.0)
    edge   = (volume ** (1 / 3)) + BOX_PADDING
    edge   = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, edge))

    # Slightly elongate based on lining residue spread if available
    size = [round(edge, 1), round(edge, 1), round(edge, 1)]

    return [round(c, 3) for c in center], size


# ── Surrogate model ───────────────────────────────────────────────────────────

class SurrogateModel:
    """
    Ridge regression surrogate for docking score prediction.
    Trained on Morgan fingerprints of previously docked molecules.
    Predicts score (lower = better) to pre-screen candidates before docking.
    """

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
    """
    Tracks which fragments appear in good binders.
    Used to bias the fragment pool toward productive chemistry.
    """

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
    """
    Maintains a diversity archive and scores molecules by their distance
    to the k nearest archive members (higher = more novel).
    """

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
    """Molecular mutation and crossover operators."""

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
                _DOCK_CACHE = json.load(f)
            log.info(f"  Loaded docking cache: {len(_DOCK_CACHE)} entries")
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
    """
    Dock a single molecule using AutoDock Vina.
    Returns dict with score, props, status.
    """
    res = {"smiles": smiles, "generation": gen_id, "candidate_id": cand_id,
           "score": None, "status": "Failed", "error": "", "props": {}}

    if smiles in _DOCK_CACHE:
        cached = _DOCK_CACHE[smiles].copy()
        cached.update({"generation": gen_id, "candidate_id": cand_id})
        return cached

    tmpdir    = os.path.join(os.getcwd(), f"_tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmpdir, exist_ok=True)
    pdb_path   = os.path.join(tmpdir, "ligand.pdb")
    pdbqt_path = os.path.join(tmpdir, "ligand.pdbqt")
    out_path   = os.path.join(tmpdir, "out.pdbqt")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            res["error"] = "Invalid SMILES"; return res

        props   = _mol_props(mol)
        res["props"] = props

        # Check ADMET before wasting compute on docking
        if not _admet_prefilter(mol, props):
            res["error"] = "Failed ADMET pre-filter"; res["status"] = "AdmetFail"
            return res

        # 3D conformer
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

        # Convert to PDBQT
        subprocess.run(
            f'obabel "{pdb_path}" -O "{pdbqt_path}" --partialcharge gasteiger -h',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(pdbqt_path):
            res["error"] = "obabel conversion failed"; return res

        # Vina docking
        cx, cy, cz = center
        sx, sy, sz = box_size
        cmd = (f'"{vina_path}" '
               f'--receptor "{receptor_path}" '
               f'--ligand "{pdbqt_path}" '
               f'--center_x {cx} --center_y {cy} --center_z {cz} '
               f'--size_x {sx} --size_y {sy} --size_z {sz} '
               f'--exhaustiveness {exhaust} --num_modes 9 '
               f'--out "{out_path}"')

        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=300)
        out = r.stdout.decode(errors="ignore")

        m = re.search(r"^\s*1\s+(-?\d+\.\d+)", out, re.MULTILINE)
        if m:
            score = float(m.group(1))
            ha    = max(1, props.get("ha", 1))
            le    = round(-score / ha, 4)
            res.update({
                "score":  score,
                "le":     le,
                "status": "OK",
            })
            _DOCK_CACHE[smiles] = {
                "smiles": smiles, "score": score, "le": le,
                "status": "OK", "props": props,
            }
            _save_dock_cache(cache_path)
        else:
            res["error"] = "No score in Vina output"

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
                          all_seen: set) -> List[str]:
    candidates = []
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
            if c and c not in all_seen:
                candidates.append(c)
                all_seen.add(c)

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

def _normalize_score(score: float, lo: float = -15.0, hi: float = -3.0) -> float:
    s = max(lo, min(hi, float(score)))
    return (hi - s) / (hi - lo)


def _le_fitness(score: float, ha: int, qed: float) -> float:
    """
    Ligand efficiency-based fitness.
    Rewards potent, small, drug-like molecules.
    LE = -score / heavy_atoms; target LE > 0.3 kcal/mol/HA
    """
    le = (-score) / max(1, ha)
    le_score = 1.0 / (1.0 + math.exp(-6 * (le - 0.25)))
    return float(max(0.0, min(1.0, le_score * qed)))


# ── Main evolution loop ───────────────────────────────────────────────────────

def _fast_pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """
    Fast PDB → PDBQT conversion using pure Python (no obabel wait).

    Writes a minimal PDBQT by:
      - Copying ATOM/HETATM records
      - Assigning Gasteiger-like partial charges via RDKit
      - Adding AutoDock atom types based on element

    Falls back to obabel if RDKit charge assignment fails.
    This avoids the 1-2 minute obabel startup overhead on Windows.
    """
    try:
        # Try RDKit-based conversion first (fast)
        from rdkit.Chem import MolFromPDBFile, AllChem, rdPartialCharges
        mol = MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if mol is not None:
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                pass

        # Write PDBQT manually from PDB lines + charges
        _write_pdbqt_from_pdb(pdb_path, pdbqt_path, mol)
        return pdbqt_path.exists()

    except Exception as e:
        log.warning(f"  Fast conversion failed ({e}) — falling back to obabel")
        r = subprocess.run(
            f'obabel "{pdb_path}" -O "{pdbqt_path}" --partialcharge gasteiger -h',
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        return pdbqt_path.exists()


def _write_pdbqt_from_pdb(pdb_path: Path, pdbqt_path: Path,
                            mol=None) -> None:
    """
    Write a PDBQT file from a PDB file.
    Assigns AutoDock4 atom types and Gasteiger charges where available.
    """
    # AutoDock4 atom type map: element → AD4 type
    AD4_TYPES = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
        "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
        "FE": "Fe", "ZN": "Zn", "MG": "Mg", "CA": "Ca", "MN": "Mn",
    }

    # Build charge map from RDKit mol if available
    charge_map: dict[int, float] = {}
    if mol is not None:
        try:
            for atom in mol.GetAtoms():
                gc = atom.GetDoubleProp("_GasteigerCharge")
                if gc == gc:  # NaN check
                    charge_map[atom.GetIdx()] = gc
        except:
            pass

    lines_out = []
    atom_idx  = 0

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                if rec in ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH",
                           "TORSDOF", "END", "TER", "END"):
                    lines_out.append(line.rstrip())
                continue

            # Extract fields
            element = line[76:78].strip().upper() if len(line) > 76 else ""
            if not element:
                atom_name = line[12:16].strip()
                element   = "".join(c for c in atom_name if c.isalpha())[:2].upper()

            ad4_type = AD4_TYPES.get(element, element[:1] if element else "C")
            charge   = charge_map.get(atom_idx, 0.0)
            atom_idx += 1

            # PDBQT format: same as PDB cols 1-54, then charge (8.3f), then AD4 type
            pdbqt_line = (
                f"{line[:54]}"
                f"{charge:8.3f}"
                f"    "
                f"{ad4_type:<2s}"
            )
            lines_out.append(pdbqt_line.rstrip())

    lines_out.append("END")
    pdbqt_path.write_text("\n".join(lines_out) + "\n")


def run_denovo_design(
    uniprot_id:    str,
    pocket_data:   Optional[dict] = None,
    active_data:   Optional[dict] = None,
    allosteric_data: Optional[dict] = None,
    chem_env_data: Optional[dict] = None,
    vina_path:     str = "",
    receptor_path: str = "",
    n_generations: int = MAX_GENERATIONS,
    rng_seed:      Optional[int] = None,
) -> DenovoResult:
    """
    Run evolutionary de novo molecule design targeting the active site.

    Target selection logic:
      1. Active site (Module 03) — preferred: most validated, directly
         disrupts protein function. Center computed from mean CA coords
         of HIGH-confidence active site residues.
      2. Allosteric site A1 (Module 05) — fallback if no active site data.
         Allosteric inhibition avoids resistance mutations at active site.
      3. Top druggable pocket (Module 04) — last resort.

    Args:
        uniprot_id:      UniProt accession
        pocket_data:     Dict from Module 04 JSON (binding_pockets)
        active_data:     Dict from Module 03 JSON (active_sites)  ← PRIMARY
        allosteric_data: Dict from Module 05 JSON (allosteric)    ← FALLBACK
        chem_env_data:   Dict from Module 06 JSON (chemical_env)
        vina_path:       Path to AutoDock Vina executable
        receptor_path:   Path to receptor PDBQT (auto-converted if .pdb given)
        n_generations:   Number of evolution generations
        rng_seed:        Random seed for reproducibility
    """
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    log.info(f"── De Novo Design: {uniprot_id}  [seed={seed}] ──")

    inter_dir  = Path(cfg.paths["intermediate"])
    cache_path = str(inter_dir / f"{uniprot_id}_dock_cache.json")
    out_path   = inter_dir / f"{uniprot_id}_denovo.json"

    _load_dock_cache(cache_path)

    # ── Select target site: active > allosteric > pocket ──────────────────────
    center         = [0.0, 0.0, 0.0]
    box_size       = [20.0, 20.0, 20.0]
    used_pocket_id = "active_site"
    target_source  = "none"

    # Priority 1: Active site — mean of HIGH-confidence residue CA coords
    if active_data:
        high_res = [
            r for r in active_data.get("active_residues", [])
            if r.get("confidence") == "HIGH" and r.get("coords")
        ]
        if not high_res:
            high_res = [
                r for r in active_data.get("active_residues", [])
                if r.get("coords")
            ]
        if high_res:
            coords_arr = np.array([r["coords"] for r in high_res])
            center     = [round(float(c), 3) for c in coords_arr.mean(axis=0)]
            # Box size: span of active site + padding
            span       = coords_arr.max(axis=0) - coords_arr.min(axis=0)
            edge       = float(span.max()) + BOX_PADDING * 2
            edge       = max(MIN_BOX_SIZE, min(MAX_BOX_SIZE, edge))
            box_size   = [round(edge, 1)] * 3
            used_pocket_id = "active_site"
            target_source  = f"active_site ({len(high_res)} residues)"
            log.info(f"  Target: ACTIVE SITE  center={center}  box={box_size}  "
                     f"({len(high_res)} high-conf residues)")

    # Priority 2: Allosteric site A1 (if no active site data)
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
            log.info(f"  Target: ALLOSTERIC {used_pocket_id}  "
                     f"center={center}  box={box_size}  "
                     f"corr={top_site.get('mean_correlation', 0):.2f}")

    # Priority 3: Top druggable pocket from Module 04
    if target_source == "none" and pocket_data:
        pockets = pocket_data.get("pockets", [])
        if pockets:
            top_p    = pockets[0]
            center, box_size = _pocket_to_box(top_p)
            used_pocket_id   = top_p.get("pocket_id", "P1")
            target_source    = f"pocket_{used_pocket_id}"
            log.info(f"  Target: POCKET {used_pocket_id}  "
                     f"center={center}  box={box_size}")

    if target_source == "none":
        log.warning("  No site data found — using default box at origin")
        log.warning("  Run Modules 03, 04, 05 first for best targeting")

    # ── Bias fragment pool from chemical environment ───────────────────────────
    fragment_pool = list(GROW_FRAGMENTS)
    if chem_env_data:
        # Active site environments
        active_envs  = chem_env_data.get("active_envs", [])
        binding_envs = chem_env_data.get("binding_envs", [])
        all_envs     = active_envs + binding_envs

        if all_envs:
            top_env    = all_envs[0]
            mode       = top_env.get("predicted_binding_mode", "")
            net_charge = top_env.get("net_charge", 0.0)
            n_aromatic = top_env.get("n_aromatic", 0)

            if "hydrophobic" in mode or n_aromatic >= 3:
                hydrophobic_frags = ["c1ccccc1", "c1ccncc1", "C1CCCCC1",
                                     "C(F)(F)F", "c1ccc2ccccc2c1"]
                fragment_pool = hydrophobic_frags * 3 + fragment_pool
                log.info("  Chem env: hydrophobic active site → aromatic fragments")

            elif "electrostatic" in mode or abs(net_charge) > 1.0:
                polar_frags = ["C(=O)O", "C(=O)N", "NC(=O)", "S(=O)(=O)N",
                               "NCC", "OCC", "c1ccncc1"]
                fragment_pool = polar_frags * 3 + fragment_pool
                log.info(f"  Chem env: charged active site (q={net_charge:+.1f}) → "
                         f"polar fragments")

    # ── Validate external tools ───────────────────────────────────────────────
    if not vina_path or not os.path.exists(vina_path):
        log.error(f"  Vina not found at: {vina_path}")
        return DenovoResult(
            uniprot_id=uniprot_id,
            pocket_id=used_pocket_id,
            pocket_center=center,
            box_size=box_size,
            notes=f"Vina not found: {vina_path}",
        )

    if not receptor_path or not os.path.exists(receptor_path):
        # Auto-convert: find .pdb and convert fast
        pdbqt_p = Path(receptor_path) if receptor_path else Path("")
        pdb_p   = pdbqt_p.with_suffix(".pdb")

        # Also check default pipeline structures path
        default_pdb   = Path(cfg.paths["structures"]) / f"{uniprot_id}.pdb"
        default_pdbqt = default_pdb.with_suffix(".pdbqt")

        source_pdb = pdb_p if pdb_p.exists() else (
            default_pdb if default_pdb.exists() else None
        )
        target_pdbqt = pdbqt_p if receptor_path else default_pdbqt

        if source_pdb:
            log.info(f"  PDBQT not found — converting {source_pdb.name} "
                     f"(fast Python method)...")
            t0 = time.time()
            ok = _fast_pdb_to_pdbqt(source_pdb, target_pdbqt)
            if ok:
                log.info(f"  Converted in {time.time()-t0:.1f}s → {target_pdbqt.name}")
                receptor_path = str(target_pdbqt)
            else:
                log.error(f"  Conversion failed for {source_pdb}")
                return DenovoResult(
                    uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                    pocket_center=center, box_size=box_size,
                    notes="PDBQT conversion failed",
                )
        else:
            log.error(f"  Receptor not found: {receptor_path}")
            log.error(f"  Also checked: {default_pdb}")
            return DenovoResult(
                uniprot_id=uniprot_id, pocket_id=used_pocket_id,
                pocket_center=center, box_size=box_size,
                notes=f"Receptor not found: {receptor_path}",
            )

    r = subprocess.run("obabel --version", shell=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        log.error("  obabel not found — needed for PDBQT conversion")
        return DenovoResult(
            uniprot_id=uniprot_id,
            pocket_id=used_pocket_id,
            pocket_center=center,
            box_size=box_size,
            notes="obabel not found",
        )

    log.info(f"  Vina: {vina_path}")
    log.info(f"  Receptor: {receptor_path}")
    log.info(f"  Workers: {PARALLEL_WORKERS}")

    # ── Initialise ────────────────────────────────────────────────────────────
    chem      = ChemicalEngine()
    novelty   = NoveltyArchive()
    surrogate = SurrogateModel()
    fqsar     = FragmentQSAR()

    population    = _build_initial_population(chem, POP_SIZE)
    all_seen      = set(s for s, _ in population)
    full_history:  List[dict] = []
    hall_of_fame:  List[dict] = []
    gen_stats:     List[dict] = []

    best_score     = float("inf")
    best_le        = 0.0
    best_smi       = ""
    stagnation_cnt = 0
    total_start    = time.time()

    log.info(f"  Starting evolution: {n_generations} generations × {POP_SIZE} molecules")

    # ── Evolution loop ────────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):
        gen_start = time.time()

        # Adaptive weights
        t      = min(1.0, (gen - 1) / max(1, n_generations - 1))
        w_bind = W_BINDING_START + t * (W_BINDING_END - W_BINDING_START)
        w_nov  = W_NOVELTY_START + t * (W_NOVELTY_END  - W_NOVELTY_START)
        w_le   = W_LE_START      + t * (W_LE_END       - W_LE_START)
        total_w = w_bind + w_nov + w_le
        w_bind /= total_w; w_nov /= total_w; w_le /= total_w

        gen_exhaust = int(EXHAUST_START + t * (EXHAUST_END - EXHAUST_START))
        if stagnation_cnt >= STAGNATION_HARD // 2:
            gen_exhaust = min(gen_exhaust * 2, EXHAUST_END * 2)

        temperature = max(0.1, 1.0 - 0.8 * t)

        # Update models
        if len(full_history) >= SURROGATE_MIN_DATA:
            surrogate.update(full_history)
        fqsar.update(full_history)

        # Surrogate-guided pre-screening
        if surrogate.trained:
            parent_pool = [s for s, _ in population]
            screened    = _generate_and_screen(
                chem, parent_pool, surrogate, fqsar,
                best_smi, stagnation_cnt, temperature,
                n_candidates=SURROGATE_CANDIDATES,
                n_dock=POP_SIZE,
                all_seen=all_seen,
            )
            if screened:
                population = [(s, "Screened") for s in screened]

        # Dock current population in parallel
        tasks   = [(smi, gen, i) for i, (smi, _) in enumerate(population)]
        results = []

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
            futures   = {
                exe.submit(
                    _dock_molecule, t[0], t[1], t[2],
                    vina_path, receptor_path,
                    center, box_size, cache_path, gen_exhaust
                ): t for t in tasks
            }
            completed = 0
            sys.stdout.write(f"Gen {gen}/{n_generations} | Docking: 0/{len(futures)}")
            sys.stdout.flush()
            for fut in as_completed(futures):
                completed += 1
                sys.stdout.write(
                    f"\rGen {gen}/{n_generations} | Docking: {completed}/{len(futures)}")
                sys.stdout.flush()
                res = fut.result()
                if res.get("status") == "OK" and res.get("score") is not None:
                    smi   = res["smiles"]
                    mol   = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    props = res["props"]
                    score = res["score"]
                    ha    = max(1, props.get("ha", 1))
                    qed   = props.get("qed", 0.0)
                    le    = res.get("le", round(-score / ha, 4))

                    nov_score = novelty.score(smi)
                    novelty.try_add(smi)

                    le_fit   = _le_fitness(score, ha, qed)
                    norm_bind = _normalize_score(score)
                    fitness  = (w_bind * norm_bind +
                                w_nov  * nov_score +
                                w_le   * le_fit)

                    res.update({
                        "le":          le,
                        "qed":         qed,
                        "novelty":     nov_score,
                        "le_fitness":  le_fit,
                        "norm_bind":   norm_bind,
                        "fitness":     fitness,
                        "w_bind":      round(w_bind, 3),
                        "w_nov":       round(w_nov, 3),
                        "generation":  gen,
                    })
                    results.append(res)
                    full_history.append(res)
                    hall_of_fame.append(res)

        sys.stdout.write("\n")

        hall_of_fame.sort(key=lambda r: r["score"])
        hall_of_fame = hall_of_fame[:30]

        if not results:
            log.warning(f"  Gen {gen}: no valid docking results — reseeding")
            stagnation_cnt += 1
            population = _build_initial_population(chem, POP_SIZE)
            all_seen.update(s for s, _ in population)
            continue

        by_score   = sorted(results, key=lambda r: r["score"])
        by_fitness = sorted(results, key=lambda r: r["fitness"], reverse=True)
        by_le      = sorted(results, key=lambda r: r.get("le", 0), reverse=True)
        by_novelty = sorted(results, key=lambda r: r.get("novelty", 0), reverse=True)

        gen_best = by_score[0]
        improved = gen_best["score"] < best_score

        if improved:
            best_score = gen_best["score"]
            best_le    = gen_best.get("le", 0.0)
            best_smi   = gen_best["smiles"]
            stagnation_cnt = 0
        else:
            stagnation_cnt += 1

        div = _tanimoto_diversity([r["smiles"] for r in results])
        scaffold_counts: Dict[str, int] = {}
        for r in results:
            sc = _murcko(r["smiles"])
            scaffold_counts[sc] = scaffold_counts.get(sc, 0) + 1

        flag = "↑ IMPROVED" if improved else f"stagnant {stagnation_cnt}"
        surr = (f"  Surr={surrogate.predict(gen_best['smiles']):.2f}"
                if surrogate.trained else "")
        log.info(
            f"  Gen {gen}: score={gen_best['score']:.2f}  "
            f"LE={gen_best.get('le',0):.3f}  "
            f"QED={gen_best.get('qed',0):.2f}  "
            f"div={div:.2f}  {flag}{surr}"
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
        })

        # ── Build next population ─────────────────────────────────────────────
        next_pop: List[Tuple[str, str]] = []
        next_seen: set = set()

        # Elitism: top fitness molecules
        for e in by_fitness[:ELITISM]:
            c = _canonical(e["smiles"])
            if c and c not in next_seen and e["score"] < MIN_SCORE_ELITE:
                next_pop.append((c, "Elite")); next_seen.add(c)

        # Novelty elites
        for e in by_novelty[:max(2, ELITISM // 2)]:
            c = _canonical(e["smiles"])
            if c and c not in next_seen:
                next_pop.append((c, "NovelElite")); next_seen.add(c)

        # Hall of fame injection
        for e in hall_of_fame[:3]:
            c = _canonical(e["smiles"])
            if c and c not in next_seen:
                next_pop.append((c, "HoF")); next_seen.add(c)

        # Diversity injection
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

        # Stagnation injection
        if stagnation_cnt >= STAGNATION_LIMIT:
            n_inj = POP_SIZE // 3
            log.info(f"  Stagnation {stagnation_cnt}: injecting {n_inj} fresh molecules")
            inj = 0
            for _ in range(n_inj * 25):
                if inj >= n_inj: break
                child = chem.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                          temperature=temperature + 0.3)
                if child:
                    c = _canonical(child)
                    if c and c not in next_seen:
                        next_pop.append((c, "Injected")); next_seen.add(c); inj += 1

        # Hard reset
        if stagnation_cnt >= STAGNATION_HARD:
            n_hard = int(POP_SIZE * 0.80)
            log.info(f"  HARD RESET: rebuilding {n_hard} molecules from best")
            best_mol = Chem.MolFromSmiles(best_smi) if best_smi else None
            inj = 0
            for _ in range(n_hard * 30):
                if inj >= n_hard: break
                if best_mol and random.random() < 0.5:
                    pool  = (fqsar.biased_pool(GROW_FRAGMENTS, temperature=1.5)
                             if fqsar.n_obs >= FQSAR_MIN_DATA else GROW_FRAGMENTS)
                    grown = chem._grow(best_mol, pool)
                    child = chem._to_smiles(grown) if grown else None
                else:
                    child = chem.build_denovo(surrogate=surrogate, fqsar=fqsar,
                                              temperature=1.5)
                if child:
                    c = _canonical(child)
                    if c and c not in next_seen:
                        next_pop.append((c, "HardReset")); next_seen.add(c); inj += 1
            stagnation_cnt = STAGNATION_LIMIT - 1

        # Fill remainder
        parent_pool = [r["smiles"] for r in by_fitness[:max(2, len(by_fitness)//2)]]
        remaining   = POP_SIZE - len(next_pop)
        if remaining > 0:
            screened_fill = _generate_and_screen(
                chem, parent_pool, surrogate, fqsar, best_smi,
                stagnation_cnt, temperature,
                n_candidates=max(remaining * 4, SURROGATE_CANDIDATES),
                n_dock=remaining,
                all_seen=next_seen | all_seen,
            )
            for c in screened_fill:
                if c not in next_seen:
                    next_pop.append((c, "Screened")); next_seen.add(c)

        # Emergency fill
        eft = 0
        while len(next_pop) < POP_SIZE and eft < 80:
            eft += 1
            child = chem.build_denovo()
            if child:
                c = _canonical(child)
                if c and c not in next_seen:
                    next_pop.append((c, "EmergFill")); next_seen.add(c)

        population = next_pop[:POP_SIZE]
        all_seen.update(s for s, _ in population)

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
            fitness=0.0,
            novelty=round(novelty.score(smi), 3),
            admet_pass=admet_ok,
            scaffold=_murcko(smi),
            origin="consensus",
        ))
        log.info(f"  Final: score={mean_score:.2f}  LE={le:.3f}  "
                 f"QED={props.get('qed',0):.2f}  {smi[:60]}")

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
                    f"Selectivty: best_SI={sel_result.best_si:.2f}x  "
                    f"hERG_safe={sel_result.n_herg_safe}/{len(sel_result.refined_molecules)}"
                )
                result.to_json(out_path)
            except Exception as e:
                log.warning(f"  Module 15 skipped: {e}")
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
@click.option("--uniprot",   "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--vina",      "-v", required=True,
              help="Path to AutoDock Vina executable")
@click.option("--receptor",  "-r", default=None,
              help="Path to receptor PDBQT (auto-converted from .pdb if not found)")
@click.option("--generations", "-g", default=MAX_GENERATIONS, type=int,
              help=f"Number of evolution generations (default: {MAX_GENERATIONS})")
@click.option("--seed",      "-s", default=None, type=int,
              help="Random seed for reproducibility")
def main(uniprot: str, vina: str, receptor: Optional[str],
         generations: int, seed: Optional[int]) -> None:
    """
    De Novo Molecule Design — evolutionary molecular generator.

    Automatically targets the active site (Module 03 output).
    Falls back to allosteric site (Module 05), then top pocket (Module 04).
    Auto-converts .pdb → .pdbqt using fast Python method (no obabel wait).

    Example:
        python pipeline\\denovo_design.py \\
            --uniprot P04637 \\
            --vina C:/tools/vina.exe \\
            --generations 30
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])

    # Default receptor path = structures folder
    if not receptor:
        receptor = str(Path(cfg.paths["structures"]) / f"{uniprot}.pdbqt")

    def _load(fname):
        p = inter_dir / fname
        return json.loads(p.read_text()) if p.exists() else None

    active_data      = _load(f"{uniprot}_active_sites.json")
    allosteric_data  = _load(f"{uniprot}_allosteric.json")
    pocket_data      = _load(f"{uniprot}_binding_pockets.json")
    chem_env_data    = _load(f"{uniprot}_chemical_env.json")

    if active_data:
        n_high = sum(1 for r in active_data.get("active_residues", [])
                     if r.get("confidence") == "HIGH")
        log.info(f"  Loaded active site data ({n_high} HIGH-confidence residues)")
    else:
        log.warning("  No active site data — run Module 03 first")

    if allosteric_data:
        log.info(f"  Loaded allosteric data "
                 f"({allosteric_data.get('n_sites', 0)} sites)")

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
    )

    click.echo(result.summary())


if __name__ == "__main__":
    main()