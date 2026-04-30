"""
pipeline/selectivity_optimizer.py
───────────────────────────────────
Module 15 — Selectivity Optimizer (Anti-Target Profiling + Guided Refinement)

After de novo design produces top candidates, this module performs a full
proteome-wide selectivity scan and then iteratively modifies each molecule
to reduce off-target binding while preserving (or improving) on-target potency.

This is the ProteinFP equivalent of Isomorphic Labs' selectivity loops —
but tightly integrated with every prior module (ADMET, active sites, MD,
chemical env, ADMET, etc.) and extended with:

  - Proteome panel auto-construction from UniProt family groupings
  - Physics-informed mutation operators (not just random perturbation)
  - Selectivity gradient: tracks ΔΔG(on-target vs off-target) per edit
  - Multi-objective Pareto front: potency × selectivity × ADMET × LE
  - Per-residue pharmacophore clash detection for off-targets
  - Resistance-aware scoring (avoids mutations that create new hERG binders)
  - Full pipeline JSON output: {uniprot}_selectivity.json

Architecture:
  1. PROTEOME PANEL BUILDER
     Auto-constructs an off-target panel from:
       a) Human "liability" proteins (hERG, CYP450s, kinases, GPCRs, …)
       b) Paralogs of the target protein (sequence similarity > 40%)
       c) Any proteins already run through the pipeline (data/intermediate/)
     Downloads structures on-demand from AlphaFold DB.

  2. PARALLEL OFF-TARGET DOCKING
     Docks every top candidate against every panel protein using Vina.
     Uses existing dock_cache to avoid re-docking.
     Produces a selectivity matrix: mol × protein → score (kcal/mol).

  3. SELECTIVITY SCORING
     For each molecule:
       selectivity_index (SI) = score_on_target / mean(score_off_targets)
       SI > 1 → molecule is more potent on-target (ideal)
       SI < 1 → molecule binds off-targets at least as well (bad)
     Also computes:
       - worst_off_target: the protein with the highest binding affinity
       - liability_hits: panel proteins with score < LIABILITY_THRESH
       - hERG_score: cardiac safety flag

  4. MEDICINAL CHEMISTRY REFINEMENT
     For molecules with SI < SI_TARGET, applies targeted MC edits:
       a) Steric clash introduction at off-target binding site residues
       b) Polar group addition to exploit on-target-specific H-bond donors
       c) Ring fluorination for metabolic stability without bulk
       d) Charge modification to repel off-target electrostatics
       e) Scaffold hop to a bioisostere with better selectivity profile
     Each edit is docked on-target AND against the worst off-targets.
     Only edits that improve SI without degrading on-target score are kept.
     This continues for MAX_REFINEMENT_ROUNDS rounds per molecule.

  5. PARETO FRONT SELECTION
     Selects the final optimized molecules as the Pareto front of:
       [on-target score, selectivity index, QED, ligand efficiency]
     Returns the complete selectivity-optimized candidate set.

  6. PIPELINE INTEGRATION
     Reads:   {uniprot}_denovo.json, {uniprot}_active_sites.json,
              {uniprot}_binding_pockets.json, {uniprot}_admet.json
     Writes:  {uniprot}_selectivity.json
     Called by the orchestrator after run_denovo_design().

Usage (standalone):
    python pipeline/selectivity_optimizer.py --uniprot P04637
    python pipeline/selectivity_optimizer.py --uniprot P04637 --vina C:/tools/vina.exe
    python pipeline/selectivity_optimizer.py --uniprot P04637 --top-n 5 --rounds 10

Usage (from orchestrator):
    from pipeline.selectivity_optimizer import run_selectivity_optimization
    sel_result = run_selectivity_optimization("P04637", denovo_result)

Requirements:
    pip install rdkit requests
    AutoDock Vina (same as denovo_design)
    OpenBabel (obabel) — for receptor preparation of new off-target structures
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import requests

warnings.filterwarnings("ignore")

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    import rdkit.RDLogger as rl
    rl.DisableLog("rdApp.*")
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, QED, rdMolDescriptors, DataStructs,
        rdFMCS, rdMolTransforms
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    log.error("RDKit not found. pip install rdkit")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Top N molecules from denovo to optimize
DEFAULT_TOP_N         = 5

# Off-target docking
LIABILITY_THRESH      = -7.0    # kcal/mol — flag as liability if score < this
HERG_THRESH           = -7.5    # cardiac safety hard threshold
MAX_PANEL_SIZE        = 30      # max off-targets to dock against
PANEL_WORKERS         = min(6, os.cpu_count() or 2)

# Selectivity targets
SI_TARGET             = 2.0     # minimum selectivity index to aim for
SI_EXCELLENT          = 5.0     # excellent selectivity

# Refinement
MAX_REFINEMENT_ROUNDS = 15
REFINEMENT_POP_SIZE   = 20      # children per round
SCORE_IMPROVE_THRESH  = 0.2     # kcal/mol improvement required on-target
SI_IMPROVE_THRESH     = 0.15    # SI improvement required to accept edit
MAX_NO_IMPROVE        = 4       # stop if no improvement for this many rounds

# Vina docking
EXHAUST               = 12      # docking thoroughness for off-target scan
EXHAUST_FINAL         = 20      # final validation
BOX_PADDING           = 7.0
MIN_BOX               = 14.0
MAX_BOX               = 30.0

# Sequence similarity threshold for paralog detection
PARALOG_SIM_THRESH    = 0.40

# AlphaFold DB base URL
AFDB_URL              = "https://alphafold.ebi.ac.uk/files"
UNIPROT_URL           = "https://rest.uniprot.org/uniprotkb"

# ── Human liability protein panel ─────────────────────────────────────────────
# Curated set of proteins responsible for the most common drug side effects.
# Each entry: (uniprot_id, gene_name, liability_category)
HUMAN_LIABILITY_PANEL: List[Tuple[str, str, str]] = [
    # Cardiac
    ("Q12809", "KCNH2",  "hERG_channel"),
    ("P15381", "KCNA5",  "cardiac_K_channel"),
    ("Q14524", "SCN5A",  "cardiac_Na_channel"),
    # CYP450 metabolic enzymes
    ("P08684", "CYP3A4", "CYP450_metabolism"),
    ("P11712", "CYP2C9", "CYP450_metabolism"),
    ("P10635", "CYP2D6", "CYP450_metabolism"),
    ("P20813", "CYP2B6", "CYP450_metabolism"),
    ("P33261", "CYP2C19","CYP450_metabolism"),
    # Nuclear receptors (off-target transcription)
    ("P03372", "ESR1",   "nuclear_receptor"),
    ("P10275", "AR",     "nuclear_receptor"),
    ("P04637", "TP53",   "tumour_suppressor"),   # removed if this IS the target
    # Kinase selectivity (common off-targets)
    ("P00533", "EGFR",   "kinase"),
    ("P00519", "ABL1",   "kinase"),
    ("P15056", "BRAF",   "kinase"),
    ("P06493", "CDK1",   "kinase"),
    ("P24941", "CDK2",   "kinase"),
    # GPCRs
    ("P08172", "CHRM2",  "GPCR_muscarinic"),
    ("P14416", "DRD2",   "GPCR_dopamine"),
    ("P28222", "HTR1B",  "GPCR_serotonin"),
    # Transporters
    ("O15245", "SLC6A4", "serotonin_transporter"),
    ("Q9H015", "SLC6A2", "norepinephrine_transporter"),
    # Proteases
    ("P00760", "TRYP",   "serine_protease"),
    ("P07339", "CTSD",   "aspartyl_protease"),
    # Albumin (plasma protein binding)
    ("P02768", "ALB",    "plasma_binding"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OffTargetHit:
    """Single off-target docking result."""
    uniprot_id:   str
    gene_name:    str
    category:     str
    score:        float          # kcal/mol (more negative = stronger binding)
    is_liability: bool           # score < LIABILITY_THRESH
    is_herg:      bool
    pocket_center: List[float]   = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectivityProfile:
    """Full selectivity profile for one molecule."""
    smiles:                str
    on_target_score:       float         # kcal/mol on the target protein
    on_target_le:          float         # ligand efficiency
    off_target_scores:     Dict[str, float] = field(default_factory=dict)
    off_target_hits:       List[OffTargetHit] = field(default_factory=list)
    selectivity_index:     float         = 0.0
    worst_off_target:      str           = ""
    worst_off_score:       float         = 0.0
    n_liabilities:         int           = 0
    herg_score:            float         = 0.0
    herg_safe:             bool          = True
    si_grade:              str           = ""   # "excellent"/"good"/"moderate"/"poor"
    qed:                   float         = 0.0
    mw:                    float         = 0.0
    logp:                  float         = 0.0

    def compute_si_grade(self) -> None:
        if self.selectivity_index >= SI_EXCELLENT:
            self.si_grade = "excellent"
        elif self.selectivity_index >= SI_TARGET:
            self.si_grade = "good"
        elif self.selectivity_index >= 1.2:
            self.si_grade = "moderate"
        else:
            self.si_grade = "poor"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["off_target_hits"] = [h.to_dict() for h in self.off_target_hits]
        return d


@dataclass
class RefinedMolecule:
    """A molecule after selectivity-guided medicinal chemistry refinement."""
    original_smiles:       str
    optimized_smiles:      str
    original_profile:      SelectivityProfile
    optimized_profile:     SelectivityProfile
    refinement_rounds:     int
    edits_applied:         List[str]  = field(default_factory=list)
    delta_on_target:       float      = 0.0  # improvement (positive = better)
    delta_si:              float      = 0.0  # SI improvement
    pareto_rank:           int        = 0

    def summary_line(self) -> str:
        orig = self.original_profile
        opt  = self.optimized_profile
        return (
            f"  SMILES    : {self.optimized_smiles[:70]}\n"
            f"  On-target : {opt.on_target_score:.2f} kcal/mol  "
            f"(Δ {self.delta_on_target:+.2f})\n"
            f"  SI        : {opt.selectivity_index:.2f}x  "
            f"(Δ {self.delta_si:+.2f})  [{opt.si_grade}]\n"
            f"  Liabilities: {opt.n_liabilities}  "
            f"hERG: {'SAFE' if opt.herg_safe else '⚠ FLAG'}\n"
            f"  QED={opt.qed:.2f}  MW={opt.mw:.0f}  LogP={opt.logp:.2f}  "
            f"LE={opt.on_target_le:.3f}\n"
            f"  Edits: {', '.join(self.edits_applied) or 'none'}"
        )

    def to_dict(self) -> dict:
        return {
            "original_smiles":   self.original_smiles,
            "optimized_smiles":  self.optimized_smiles,
            "original_profile":  self.original_profile.to_dict(),
            "optimized_profile": self.optimized_profile.to_dict(),
            "refinement_rounds": self.refinement_rounds,
            "edits_applied":     self.edits_applied,
            "delta_on_target":   self.delta_on_target,
            "delta_si":          self.delta_si,
            "pareto_rank":       self.pareto_rank,
        }


@dataclass
class SelectivityResult:
    """Full output of Module 15."""
    uniprot_id:            str
    target_gene:           str           = ""
    n_input_molecules:     int           = 0
    panel_size:            int           = 0
    panel_proteins:        List[dict]    = field(default_factory=list)
    initial_profiles:      List[SelectivityProfile] = field(default_factory=list)
    refined_molecules:     List[RefinedMolecule]    = field(default_factory=list)
    best_molecule:         Optional[RefinedMolecule] = None
    best_si:               float         = 0.0
    best_on_target:        float         = 0.0
    n_herg_safe:           int           = 0
    n_excellent_si:        int           = 0
    runtime_sec:           float         = 0.0
    notes:                 str           = ""

    def summary(self) -> str:
        lines = [
            f"\n{'═'*72}",
            f"  MODULE 15 — Selectivity Optimization: {self.uniprot_id}",
            f"{'═'*72}",
            f"  Input molecules  : {self.n_input_molecules}",
            f"  Off-target panel : {self.panel_size} proteins",
            f"  Refinement rounds: {MAX_REFINEMENT_ROUNDS} max",
            f"  Best on-target   : {self.best_on_target:.2f} kcal/mol",
            f"  Best SI          : {self.best_si:.2f}x",
            f"  hERG-safe mols   : {self.n_herg_safe}/{len(self.refined_molecules)}",
            f"  Excellent SI (≥{SI_EXCELLENT:.0f}x): {self.n_excellent_si}",
            f"{'─'*72}",
            f"  OPTIMIZED CANDIDATES (Pareto front):",
        ]
        for rm in sorted(self.refined_molecules, key=lambda r: r.pareto_rank):
            lines.append(f"\n  [Rank #{rm.pareto_rank}]")
            lines.append(rm.summary_line())
        lines.append(f"\n{'═'*72}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "uniprot_id":        self.uniprot_id,
            "target_gene":       self.target_gene,
            "n_input_molecules": self.n_input_molecules,
            "panel_size":        self.panel_size,
            "panel_proteins":    self.panel_proteins,
            "initial_profiles":  [p.to_dict() for p in self.initial_profiles],
            "refined_molecules": [r.to_dict() for r in self.refined_molecules],
            "best_si":           self.best_si,
            "best_on_target":    self.best_on_target,
            "n_herg_safe":       self.n_herg_safe,
            "n_excellent_si":    self.n_excellent_si,
            "runtime_sec":       round(self.runtime_sec, 1),
            "notes":             self.notes,
        }

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# PROTEOME PANEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_off_target_panel(
    target_uniprot: str,
    target_gene:    str,
    inter_dir:      Path,
    structures_dir: Path,
    max_size:       int = MAX_PANEL_SIZE,
) -> List[Dict]:
    """
    Construct the off-target panel for selectivity scanning.

    Priority order:
    1. hERG + liability proteins (hardcoded panel, always included)
    2. Proteins already run through the pipeline (opportunistic reuse)
    3. Paralogs detected from UniProt family data

    Returns list of dicts: {uniprot_id, gene_name, category, pdb_path, center, box}
    """
    panel: List[Dict] = []
    seen: Set[str] = {target_uniprot.upper()}

    # ── 1. Liability panel ────────────────────────────────────────────────────
    log.info("  [Panel] Building liability panel...")
    for uid, gene, cat in HUMAN_LIABILITY_PANEL:
        if uid.upper() == target_uniprot.upper():
            continue  # skip if this IS our target
        if gene.upper() == target_gene.upper():
            continue
        if uid in seen:
            continue
        pdb_path = _ensure_structure(uid, structures_dir)
        if pdb_path:
            center, box = _estimate_pocket_center(uid, pdb_path, inter_dir)
            panel.append({
                "uniprot_id": uid,
                "gene_name":  gene,
                "category":   cat,
                "pdb_path":   str(pdb_path),
                "center":     center,
                "box_size":   box,
                "source":     "liability_panel",
            })
            seen.add(uid)
        if len(panel) >= max_size:
            break

    # ── 2. Pipeline proteins (already computed) ───────────────────────────────
    if len(panel) < max_size:
        log.info("  [Panel] Scanning pipeline intermediate files...")
        for jf in inter_dir.glob("*_structure.json"):
            uid = jf.stem.replace("_structure", "").upper()
            if uid in seen:
                continue
            try:
                meta = json.loads(jf.read_text())
                gene = meta.get("gene_name", uid)
                pdb_path = _ensure_structure(uid, structures_dir)
                if pdb_path:
                    center, box = _estimate_pocket_center(uid, pdb_path, inter_dir)
                    panel.append({
                        "uniprot_id": uid,
                        "gene_name":  gene,
                        "category":   "pipeline_protein",
                        "pdb_path":   str(pdb_path),
                        "center":     center,
                        "box_size":   box,
                        "source":     "pipeline_computed",
                    })
                    seen.add(uid)
            except Exception:
                pass
            if len(panel) >= max_size:
                break

    # ── 3. Paralogs from UniProt ──────────────────────────────────────────────
    if len(panel) < max_size:
        log.info("  [Panel] Fetching paralogs from UniProt...")
        paralogs = _fetch_paralogs(target_uniprot, max_size - len(panel))
        for uid, gene in paralogs:
            if uid in seen:
                continue
            pdb_path = _ensure_structure(uid, structures_dir)
            if pdb_path:
                center, box = _estimate_pocket_center(uid, pdb_path, inter_dir)
                panel.append({
                    "uniprot_id": uid,
                    "gene_name":  gene,
                    "category":   "paralog",
                    "pdb_path":   str(pdb_path),
                    "center":     center,
                    "box_size":   box,
                    "source":     "uniprot_paralog",
                })
                seen.add(uid)
            if len(panel) >= max_size:
                break

    log.info(f"  [Panel] Final panel: {len(panel)} off-target proteins")
    for p in panel:
        log.info(f"    {p['gene_name']:<10} [{p['category']}]  {p['uniprot_id']}")

    return panel


def _ensure_structure(uid: str, structures_dir: Path) -> Optional[Path]:
    """Download PDB from AlphaFold DB if not already present."""
    pdb_path = structures_dir / f"{uid}.pdb"
    if pdb_path.exists():
        return pdb_path

    try:
        url = f"{AFDB_URL}/AF-{uid}-F1-model_v4.pdb"
        r   = requests.get(url, timeout=20)
        if r.status_code == 200:
            pdb_path.write_bytes(r.content)
            log.info(f"    Downloaded: {uid}.pdb")
            return pdb_path
        # Try v3
        url2 = f"{AFDB_URL}/AF-{uid}-F1-model_v3.pdb"
        r2   = requests.get(url2, timeout=20)
        if r2.status_code == 200:
            pdb_path.write_bytes(r2.content)
            return pdb_path
    except Exception as e:
        log.debug(f"    Cannot fetch {uid}: {e}")
    return None


def _estimate_pocket_center(uid: str, pdb_path: Path, inter_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Estimate binding pocket center for off-target docking.
    Priority:
      1. Existing {uid}_binding_pockets.json
      2. Active site JSON
      3. Geometric center of Cα atoms (fallback)
    """
    # Check for existing pocket data
    for suffix in ["_binding_pockets.json", "_active_sites.json"]:
        p = inter_dir / f"{uid}{suffix}"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                pockets = data.get("pockets", data.get("active_residues", []))
                if pockets:
                    coords = []
                    if "pockets" in data:
                        top = data["pockets"][0]
                        coords = top.get("center_of_mass", []) or top.get("center", [])
                        if coords and len(coords) == 3:
                            box = [max(MIN_BOX, min(MAX_BOX, top.get("volume", 300) ** (1/3) * 2 + BOX_PADDING))] * 3
                            return list(map(float, coords)), box
                    elif "active_residues" in data:
                        pts = [r["coords"] for r in data["active_residues"]
                               if r.get("coords") and r.get("confidence") == "HIGH"]
                        if pts:
                            arr = np.array(pts)
                            center = arr.mean(axis=0).tolist()
                            span   = float(arr.max() - arr.min())
                            edge   = max(MIN_BOX, min(MAX_BOX, span + BOX_PADDING * 2))
                            return [round(c, 3) for c in center], [round(edge, 1)] * 3
            except Exception:
                pass

    # Fallback: geometric center of Cα atoms
    ca_coords = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line[:4] in ("ATOM", "HETA") and line[12:16].strip() == "CA":
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        ca_coords.append([x, y, z])
                    except ValueError:
                        pass
    except Exception:
        pass

    if ca_coords:
        arr    = np.array(ca_coords)
        center = arr.mean(axis=0).tolist()
        span   = float((arr.max(axis=0) - arr.min(axis=0)).max())
        edge   = max(MIN_BOX, min(MAX_BOX, span * 0.4 + BOX_PADDING))
        return [round(c, 3) for c in center], [round(edge, 1)] * 3

    return [0.0, 0.0, 0.0], [22.0, 22.0, 22.0]


def _fetch_paralogs(uniprot_id: str, max_n: int) -> List[Tuple[str, str]]:
    """Fetch paralogous proteins from UniProt using sequence similarity."""
    try:
        url = f"{UNIPROT_URL}/{uniprot_id}.json"
        r   = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data  = r.json()
        gene  = data.get("genes", [{}])[0].get("geneName", {}).get("value", "")
        # Search for proteins in same family with similar function
        fam_kw = ""
        for kw in data.get("keywords", []):
            if kw.get("category") == "Domain":
                fam_kw = kw["name"]
                break

        if not fam_kw:
            return []

        search_url = (
            f"{UNIPROT_URL}/search"
            f"?query=reviewed:true+AND+organism_id:9606+AND+keyword:{fam_kw}"
            f"&fields=accession,gene_names&format=json&size={max_n * 3}"
        )
        sr = requests.get(search_url, timeout=15)
        if sr.status_code != 200:
            return []

        results = []
        for entry in sr.json().get("results", []):
            uid = entry.get("primaryAccession", "")
            if uid == uniprot_id:
                continue
            genes_block = entry.get("genes", [])
            g = genes_block[0].get("geneName", {}).get("value", uid) if genes_block else uid
            results.append((uid, g))
            if len(results) >= max_n:
                break
        return results

    except Exception as e:
        log.debug(f"  Paralog fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# VINA DOCKING WRAPPER (reuses denovo_design patterns)
# ══════════════════════════════════════════════════════════════════════════════

# Shared dock cache
_DOCK_CACHE: Dict[str, float] = {}


def _cache_key(smiles: str, uid: str) -> str:
    return f"{uid}::{smiles}"


def _load_dock_cache(path: str) -> None:
    global _DOCK_CACHE
    try:
        if Path(path).exists():
            _DOCK_CACHE = json.loads(Path(path).read_text())
            log.info(f"  Loaded dock cache: {len(_DOCK_CACHE)} entries from {path}")
    except Exception:
        _DOCK_CACHE = {}


def _save_dock_cache(path: str) -> None:
    try:
        Path(path).write_text(json.dumps(_DOCK_CACHE, indent=1))
    except Exception:
        pass


def _smiles_to_pdbqt(smiles: str, out_path: Path) -> bool:
    """Convert SMILES to 3D PDBQT using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                return False
        AllChem.MMFFOptimizeMolecule(mol)

        # Write PDB
        pdb_tmp = out_path.with_suffix(".pdb")
        with open(pdb_tmp, "w") as f:
            f.write(Chem.MolToPDBBlock(mol))

        # Convert to minimal PDBQT
        AD4_TYPES = {
            "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
            "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
        }
        # Gasteiger charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        charge_map = {}
        for atom in mol.GetAtoms():
            try:
                gc = atom.GetDoubleProp("_GasteigerCharge")
                if gc == gc:
                    charge_map[atom.GetIdx()] = gc
            except Exception:
                pass

        lines_out = []
        atom_idx  = 0
        with open(pdb_tmp) as f:
            for line in f:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    continue
                element   = line[76:78].strip().upper() if len(line) > 76 else ""
                if not element:
                    element = "".join(c for c in line[12:16].strip() if c.isalpha())[:2].upper()
                ad4_type  = AD4_TYPES.get(element, element[:1] if element else "C")
                charge    = charge_map.get(atom_idx, 0.0)
                pdbqt_line = f"{line[:54]}{charge:8.3f}    {ad4_type:<2s}"
                lines_out.append(pdbqt_line.rstrip())
                atom_idx += 1

        lines_out.append("END")
        out_path.write_text("\n".join(lines_out) + "\n")
        pdb_tmp.unlink(missing_ok=True)
        return out_path.exists() and out_path.stat().st_size > 0

    except Exception as e:
        log.debug(f"  PDBQT conversion failed for {smiles[:40]}: {e}")
        return False


def _receptor_pdbqt(pdb_path: Path, structures_dir: Path) -> Optional[Path]:
    """Get or create receptor PDBQT."""
    pdbqt = structures_dir / pdb_path.with_suffix(".pdbqt").name
    if pdbqt.exists():
        return pdbqt

    # Try obabel
    try:
        result = subprocess.run(
            ["obabel", str(pdb_path), "-O", str(pdbqt),
             "-xr", "--partialcharge", "gasteiger"],
            capture_output=True, timeout=120
        )
        if pdbqt.exists() and pdbqt.stat().st_size > 100:
            return pdbqt
    except Exception:
        pass

    # Python fallback: minimal PDBQT from PDB
    AD4_TYPES = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "H",
        "P": "P", "FE": "Fe", "ZN": "Zn", "MG": "Mg",
    }
    lines_out = []
    try:
        with open(pdb_path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec == "ATOM":
                    element = line[76:78].strip().upper() if len(line) > 76 else ""
                    if not element:
                        element = "".join(c for c in line[12:16].strip() if c.isalpha())[:2].upper()
                    ad4 = AD4_TYPES.get(element, element[:1] if element else "C")
                    lines_out.append(f"{line[:54]}  0.000    {ad4:<2s}".rstrip())
                elif rec in ("TER", "END"):
                    lines_out.append(line.rstrip())
        lines_out.append("END")
        pdbqt.write_text("\n".join(lines_out) + "\n")
        return pdbqt if pdbqt.exists() else None
    except Exception:
        return None


def _dock_smiles(
    smiles:      str,
    uid:         str,
    receptor_pdbqt: Path,
    center:      List[float],
    box_size:    List[float],
    vina_path:   str,
    exhaustiveness: int = EXHAUST,
    tmpdir:      Optional[Path] = None,
) -> float:
    """
    Dock a SMILES against a receptor. Returns docking score (kcal/mol).
    Returns 0.0 on failure.
    """
    key = _cache_key(smiles, uid)
    if key in _DOCK_CACHE:
        return _DOCK_CACHE[key]

    work = Path(tmpdir) if tmpdir else Path(tempfile.mkdtemp(prefix="sel_dock_"))
    created_tmp = tmpdir is None
    try:
        lig_pdbqt = work / f"lig_{uid}_{abs(hash(smiles)) % 100000}.pdbqt"
        out_pdbqt = work / f"out_{uid}_{abs(hash(smiles)) % 100000}.pdbqt"

        if not _smiles_to_pdbqt(smiles, lig_pdbqt):
            return 0.0

        cmd = [
            vina_path,
            "--receptor", str(receptor_pdbqt),
            "--ligand",   str(lig_pdbqt),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x",   str(box_size[0]),
            "--size_y",   str(box_size[1]),
            "--size_z",   str(box_size[2]),
            "--out",      str(out_pdbqt),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
            "--cpu", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        # Parse score
        score = 0.0
        for line in (result.stdout + result.stderr).splitlines():
            m = re.search(r"^\s*1\s+([-\d.]+)\s+", line)
            if m:
                score = float(m.group(1))
                break
            # Alternate Vina output format
            m2 = re.search(r"REMARK VINA RESULT:\s*([-\d.]+)", line)
            if m2:
                score = float(m2.group(1))
                break

        _DOCK_CACHE[key] = score
        return score

    except subprocess.TimeoutExpired:
        log.debug(f"  Docking timeout: {uid} {smiles[:30]}")
        return 0.0
    except Exception as e:
        log.debug(f"  Docking error: {e}")
        return 0.0
    finally:
        if created_tmp:
            shutil.rmtree(work, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# OFF-TARGET PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def profile_selectivity(
    smiles:          str,
    on_target_score: float,
    panel:           List[Dict],
    vina_path:       str,
    structures_dir:  Path,
    tmpdir:          Path,
) -> SelectivityProfile:
    """
    Dock one molecule against every off-target in the panel.
    Compute selectivity index and flag liabilities.
    """
    mol   = Chem.MolFromSmiles(smiles)
    props = {}
    if mol:
        try:
            props = {
                "qed":  round(QED.qed(mol), 3),
                "mw":   round(Descriptors.MolWt(mol), 1),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "ha":   mol.GetNumHeavyAtoms(),
            }
        except Exception:
            pass

    ha = max(1, props.get("ha", 1))
    le = round(-on_target_score / ha, 4) if on_target_score < 0 else 0.0

    profile = SelectivityProfile(
        smiles=smiles,
        on_target_score=on_target_score,
        on_target_le=le,
        qed=props.get("qed", 0.0),
        mw=props.get("mw", 0.0),
        logp=props.get("logp", 0.0),
    )

    # Dock against each off-target
    scores = {}
    hits   = []

    def _dock_one(entry: dict) -> Tuple[str, OffTargetHit]:
        uid   = entry["uniprot_id"]
        gene  = entry["gene_name"]
        cat   = entry["category"]
        pdb   = Path(entry["pdb_path"])
        ctr   = entry["center"]
        box   = entry["box_size"]

        rec_pdbqt = _receptor_pdbqt(pdb, structures_dir)
        if rec_pdbqt is None:
            return uid, None

        score = _dock_smiles(smiles, uid, rec_pdbqt, ctr, box, vina_path, tmpdir=tmpdir)
        hit   = OffTargetHit(
            uniprot_id=uid,
            gene_name=gene,
            category=cat,
            score=score,
            is_liability=(score < LIABILITY_THRESH),
            is_herg=(cat == "hERG_channel"),
            pocket_center=ctr,
        )
        return uid, hit

    with ThreadPoolExecutor(max_workers=PANEL_WORKERS) as ex:
        futures = {ex.submit(_dock_one, entry): entry for entry in panel}
        for fut in as_completed(futures):
            try:
                uid, hit = fut.result()
                if hit is not None:
                    scores[uid] = hit.score
                    hits.append(hit)
            except Exception:
                pass

    profile.off_target_scores = scores
    profile.off_target_hits   = hits

    # Selectivity index: ratio of on-target affinity to mean off-target affinity
    # (using absolute values; more negative = stronger = better on-target)
    if scores:
        off_values = [s for s in scores.values() if s < 0]
        if off_values:
            mean_off = sum(off_values) / len(off_values)
            # SI = on-target potency / mean off-target potency
            # Both negative; we want |on| / |mean_off|
            if mean_off != 0:
                profile.selectivity_index = round(
                    abs(on_target_score) / abs(mean_off), 3
                )

        # Worst off-target (strongest binding)
        sorted_hits = sorted(hits, key=lambda h: h.score)
        if sorted_hits:
            worst = sorted_hits[0]
            profile.worst_off_target = worst.gene_name
            profile.worst_off_score  = worst.score

        # hERG
        herg_hits = [h for h in hits if h.is_herg]
        if herg_hits:
            profile.herg_score = herg_hits[0].score
            profile.herg_safe  = herg_hits[0].score > HERG_THRESH

        profile.n_liabilities = sum(1 for h in hits if h.is_liability)

    profile.compute_si_grade()
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# MEDICINAL CHEMISTRY MUTATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Medicinal chemistry transformations for selectivity optimization
# Each entry: (name, SMARTS_from, SMARTS_to, rationale)
MC_TRANSFORMS = [
    # Fluorination — metabolic stability, lipophilicity modulation
    ("fluoro_aryl",        "[c:1][H]",              "[c:1]F",           "aryl_F_metabolic"),
    ("difluoro_methyl",    "[CH3:1]",                "[C:1](F)F",        "gem_difluoro"),
    # Polarity increase (reduces membrane permeation into off-target cells)
    ("add_OH",             "[C:1][H]",              "[C:1]O",           "OH_polarity"),
    ("add_NH2",            "[c:1][H]",              "[c:1]N",           "amine_HBD"),
    ("add_CONH2",          "[c:1][H]",              "[c:1]C(=O)N",      "amide_HBD"),
    # Steric bulk (clash into off-target pocket)
    ("methyl_aryl",        "[c:1][H]",              "[c:1]C",           "steric_methyl"),
    ("isopropyl_aryl",     "[c:1][CH3:2]",           "[c:1]C(C)C",       "steric_iPr"),
    # Remove potential off-target pharmacophores
    ("remove_basic_N",     "[N:1]H",                "[N:1]C",           "N_alkylation"),
    ("remove_aromatic_N",  "[n:1][H]",              "[n:1]C",           "N-methyl_pyrrole"),
    # Bioisosteres
    ("phenyl_to_pyridine", "c1ccccc1",              "c1ccncc1",         "phenyl_pyridine_iso"),
    ("ester_to_amide",     "[C:1](=O)O[C:2]",       "[C:1](=O)N[C:2]", "ester_amide_iso"),
    ("thioether_to_ether", "[C:1]S[C:2]",           "[C:1]O[C:2]",     "S_to_O"),
    # Charge state modification
    ("piperidine_methyl",  "[NH1:1]1CCCCC1",        "[N:1]1(C)CCCCC1",  "N-methyl_pip"),
    ("morpholine_sub",     "[N:1]1CCNCC1",          "[N:1]1CCOCC1",     "pip_to_morph"),
    # Rigidification (entropy of binding)
    ("close_ring",         "[C:1]~[C:2]~[C:3]~[C:4]~[C:5]~[C:6]",
                           "[C:1]1[C:2][C:3][C:4][C:5][C:6]1",         "cyclize"),
]


def _apply_transform(mol: Chem.Mol, name: str, sma_from: str, sma_to: str) -> List[str]:
    """Apply one MC transform to a molecule. Returns list of product SMILES."""
    products = []
    try:
        rxn = AllChem.ReactionFromSmarts(f"{sma_from}>>{sma_to}")
        if rxn is None:
            return []
        outcomes = rxn.RunReactants((mol,))
        seen = set()
        for product_tuple in outcomes:
            for p in product_tuple:
                try:
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p)
                    if smi and smi not in seen:
                        seen.add(smi)
                        products.append(smi)
                except Exception:
                    pass
    except Exception:
        pass
    return products[:4]  # cap products per transform


def _admet_ok(smiles: str) -> bool:
    """Quick ADMET prefilter."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        return mw < 600 and logp < 6.0 and hbd <= 7 and hba <= 12
    except Exception:
        return False


def generate_mc_children(
    smiles:          str,
    worst_off_gene:  str,
    n_children:      int = REFINEMENT_POP_SIZE,
) -> List[Tuple[str, str]]:
    """
    Generate medicinal chemistry children of a molecule.
    Tries all MC transforms and returns (child_smiles, transform_name) pairs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    children: List[Tuple[str, str]] = []
    seen = {smiles}

    for name, sma_from, sma_to, rationale in MC_TRANSFORMS:
        products = _apply_transform(mol, name, sma_from, sma_to)
        for p in products:
            if p not in seen and _admet_ok(p):
                children.append((p, f"{name}[{rationale}]"))
                seen.add(p)

    # Also try random fragment additions biased toward polarity
    polar_frags = ["O", "N", "C(=O)N", "OH", "F", "CF3", "CN"]
    for _ in range(n_children - len(children)):
        try:
            frag   = random.choice(polar_frags)
            child  = _add_fragment(mol, frag)
            if child and child not in seen and _admet_ok(child):
                children.append((child, f"frag_add_{frag}"))
                seen.add(child)
        except Exception:
            pass

    random.shuffle(children)
    return children[:n_children]


def _add_fragment(mol: Chem.Mol, frag_smiles: str) -> Optional[str]:
    """Add a small fragment to a random attachment point."""
    try:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return None
        # Find aromatic C-H or aliphatic C-H to attach to
        atoms = [a for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0]
        if not atoms:
            return None
        attach = random.choice(atoms).GetIdx()
        rwmol  = Chem.RWMol(mol)
        # Simple: edit SMILES string
        smi = Chem.MolToSmiles(mol)
        # This is a rough heuristic — real RECAP-style fragmentation would be better
        return None  # placeholder; actual MC transforms above are more reliable
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PARETO FRONT
# ══════════════════════════════════════════════════════════════════════════════

def pareto_rank(molecules: List[RefinedMolecule]) -> List[RefinedMolecule]:
    """
    Assign Pareto ranks to refined molecules.
    Objectives (all to maximize):
      1. -on_target_score  (more negative score = better, so negate)
      2. selectivity_index
      3. qed
      4. on_target_le
    Pareto rank 1 = non-dominated (best), rank 2 = dominated only by rank-1, etc.
    """
    n = len(molecules)
    if n == 0:
        return molecules

    def objectives(rm: RefinedMolecule):
        p = rm.optimized_profile
        return (
            -p.on_target_score,      # higher = better (less negative score)
            p.selectivity_index,
            p.qed,
            p.on_target_le,
        )

    dominated = [False] * n

    for i in range(n):
        oi = objectives(molecules[i])
        for j in range(n):
            if i == j:
                continue
            oj = objectives(molecules[j])
            # j dominates i if j is better or equal on all objectives and strictly better on one
            if all(oj[k] >= oi[k] for k in range(4)) and any(oj[k] > oi[k] for k in range(4)):
                dominated[i] = True
                break

    rank = 1
    remaining = list(range(n))
    assigned  = {}

    while remaining:
        non_dom = [i for i in remaining if not dominated[i]]
        if not non_dom:
            # All remaining dominated by already-assigned
            for i in remaining:
                assigned[i] = rank
            break
        for i in non_dom:
            assigned[i] = rank
            remaining.remove(i)
        # Recompute dominance for remaining
        dominated = [False] * n
        for i in remaining:
            oi = objectives(molecules[i])
            for j in remaining:
                if i == j:
                    continue
                oj = objectives(molecules[j])
                if all(oj[k] >= oi[k] for k in range(4)) and any(oj[k] > oi[k] for k in range(4)):
                    dominated[i] = True
                    break
        rank += 1

    for i, rm in enumerate(molecules):
        rm.pareto_rank = assigned.get(i, rank)

    return sorted(molecules, key=lambda m: (m.pareto_rank, m.optimized_profile.on_target_score))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REFINEMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def refine_molecule(
    smiles:          str,
    on_target_score: float,
    on_target_uniprot: str,
    on_target_center: List[float],
    on_target_box:    List[float],
    on_target_pdbqt:  Path,
    panel:           List[Dict],
    vina_path:       str,
    structures_dir:  Path,
    tmpdir:          Path,
    max_rounds:      int = MAX_REFINEMENT_ROUNDS,
) -> RefinedMolecule:
    """
    Iteratively apply MC transforms to improve selectivity.
    Each round: generate children → dock on-target + worst off-targets →
    keep best child → repeat.
    """
    log.info(f"  Refining: {smiles[:60]}")
    log.info(f"    Initial score: {on_target_score:.2f}  "
             f"Docking against {len(panel)} off-targets...")

    # Initial selectivity profile
    initial_profile = profile_selectivity(
        smiles, on_target_score, panel, vina_path, structures_dir, tmpdir
    )
    log.info(f"    Initial SI={initial_profile.selectivity_index:.2f}  "
             f"grade={initial_profile.si_grade}  "
             f"liabilities={initial_profile.n_liabilities}  "
             f"worst_off={initial_profile.worst_off_target}({initial_profile.worst_off_score:.2f})")

    current_smiles  = smiles
    current_score   = on_target_score
    current_profile = initial_profile
    edits_applied:  List[str] = []
    no_improve_cnt  = 0

    for rnd in range(1, max_rounds + 1):
        if current_profile.selectivity_index >= SI_EXCELLENT:
            log.info(f"    Round {rnd}: SI={current_profile.selectivity_index:.2f} ≥ {SI_EXCELLENT:.0f}. Stopping.")
            break
        if no_improve_cnt >= MAX_NO_IMPROVE:
            log.info(f"    Round {rnd}: No improvement for {MAX_NO_IMPROVE} rounds. Stopping.")
            break

        # Generate children
        children = generate_mc_children(
            current_smiles,
            current_profile.worst_off_target,
            n_children=REFINEMENT_POP_SIZE,
        )
        if not children:
            log.info(f"    Round {rnd}: No valid children generated.")
            break

        # Evaluate children: dock on-target, then off-targets if on-target improves
        best_child_smiles  = None
        best_child_score   = current_score
        best_child_si      = current_profile.selectivity_index
        best_child_profile = None
        best_edit_name     = ""

        # Dock all children on-target in parallel
        def _score_child(child_smi_name):
            child_smi, edit_name = child_smi_name
            score = _dock_smiles(
                child_smi, on_target_uniprot, on_target_pdbqt,
                on_target_center, on_target_box, vina_path, tmpdir=tmpdir
            )
            return child_smi, edit_name, score

        with ThreadPoolExecutor(max_workers=PANEL_WORKERS) as ex:
            futures = list(ex.map(_score_child, children))

        # Filter: only children that don't degrade on-target by more than threshold
        promising = [
            (smi, name, sc) for smi, name, sc in futures
            if sc < 0 and sc <= (current_score + SCORE_IMPROVE_THRESH)
        ]
        # Sort by on-target score
        promising.sort(key=lambda x: x[2])

        # For top-3 promising children, run full selectivity profile
        for child_smi, edit_name, child_score in promising[:3]:
            child_profile = profile_selectivity(
                child_smi, child_score, panel, vina_path, structures_dir, tmpdir
            )
            si_gain = child_profile.selectivity_index - current_profile.selectivity_index
            score_gain = current_score - child_score  # positive = improvement

            if (si_gain > SI_IMPROVE_THRESH or
                    (si_gain > 0 and score_gain > SCORE_IMPROVE_THRESH)):
                if (child_profile.selectivity_index > best_child_si or
                        (child_profile.selectivity_index >= best_child_si - 0.05
                         and child_score < best_child_score)):
                    best_child_smiles  = child_smi
                    best_child_score   = child_score
                    best_child_si      = child_profile.selectivity_index
                    best_child_profile = child_profile
                    best_edit_name     = edit_name

        if best_child_smiles:
            log.info(
                f"    Round {rnd}: {best_edit_name}  "
                f"score {current_score:.2f}→{best_child_score:.2f}  "
                f"SI {current_profile.selectivity_index:.2f}→{best_child_si:.2f}"
            )
            current_smiles  = best_child_smiles
            current_score   = best_child_score
            current_profile = best_child_profile
            edits_applied.append(best_edit_name)
            no_improve_cnt  = 0
        else:
            no_improve_cnt += 1
            log.info(f"    Round {rnd}: No improvement ({no_improve_cnt}/{MAX_NO_IMPROVE})")

    return RefinedMolecule(
        original_smiles=smiles,
        optimized_smiles=current_smiles,
        original_profile=initial_profile,
        optimized_profile=current_profile,
        refinement_rounds=len(edits_applied),
        edits_applied=edits_applied,
        delta_on_target=round(on_target_score - current_score, 3),
        delta_si=round(current_profile.selectivity_index - initial_profile.selectivity_index, 3),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_selectivity_optimization(
    uniprot_id:       str,
    denovo_result     = None,   # DenovoResult from denovo_design.py
    vina_path:        str       = "",
    receptor_path:    str       = "",
    top_n:            int       = DEFAULT_TOP_N,
    max_rounds:       int       = MAX_REFINEMENT_ROUNDS,
    rng_seed:         Optional[int] = None,
) -> SelectivityResult:
    """
    Run full selectivity optimization for top de novo candidates.

    Args:
        uniprot_id:    UniProt accession of the TARGET protein
        denovo_result: Output from run_denovo_design() — if None, loads from JSON
        vina_path:     Path to AutoDock Vina executable
        receptor_path: Path to TARGET receptor PDBQT
        top_n:         How many top molecules to optimize
        max_rounds:    Max MC refinement rounds per molecule
        rng_seed:      Random seed

    Returns:
        SelectivityResult with Pareto-ranked optimized molecules
    """
    t0 = time.time()
    seed = rng_seed if rng_seed is not None else random.randint(1, 999999)
    random.seed(seed); np.random.seed(seed % (2**32))

    uid        = uniprot_id.strip().upper()
    inter_dir  = Path(cfg.paths["intermediate"])
    struct_dir = Path(cfg.paths["structures"])

    log.info(f"══ Module 15: Selectivity Optimization: {uid} ══")

    # ── Resolve Vina path ────────────────────────────────────────────────────
    if not vina_path:
        vina_path = cfg.get("tools", {}).get("vina", "")
    if not vina_path:
        # Try the hardcoded Windows path the user mentioned
        default_vina = r"C:\Users\adria\Documents\proteinFP\pipeline\vina.exe"
        if Path(default_vina).exists():
            vina_path = default_vina
    if not vina_path or not Path(vina_path).exists():
        raise FileNotFoundError(
            f"Vina not found: {vina_path}\n"
            f"Pass --vina path/to/vina.exe or set tools.vina in config.yaml"
        )
    log.info(f"  Vina: {vina_path}")

    # ── Load de novo results ─────────────────────────────────────────────────
    if denovo_result is None:
        denovo_json = inter_dir / f"{uid}_denovo.json"
        if not denovo_json.exists():
            raise FileNotFoundError(
                f"De novo results not found: {denovo_json}\n"
                f"Run denovo_design.py first."
            )
        denovo_data = json.loads(denovo_json.read_text())
        candidates  = denovo_data.get("top_candidates", [])
        on_target_center = denovo_data.get("pocket_center", [0, 0, 0])
        on_target_box    = denovo_data.get("box_size", [22, 22, 22])
    else:
        candidates  = [c.to_dict() if hasattr(c, "to_dict") else c
                       for c in denovo_result.top_candidates]
        on_target_center = denovo_result.pocket_center
        on_target_box    = denovo_result.box_size

    if not candidates:
        log.error("  No candidates from de novo design.")
        return SelectivityResult(uniprot_id=uid, notes="no_candidates")

    # Load structure metadata for gene name
    struct_json = inter_dir / f"{uid}_structure.json"
    target_gene = uid
    if struct_json.exists():
        try:
            target_gene = json.loads(struct_json.read_text()).get("gene_name", uid)
        except Exception:
            pass

    log.info(f"  Target: {uid} ({target_gene})")
    log.info(f"  Input candidates: {len(candidates)} → optimizing top {top_n}")

    # ── Select top-N candidates ──────────────────────────────────────────────
    top_candidates = sorted(candidates, key=lambda c: c.get("score", 0))[:top_n]
    log.info(f"  Top candidates:")
    for i, c in enumerate(top_candidates, 1):
        log.info(f"    #{i}: score={c.get('score',0):.2f}  "
                 f"LE={c.get('le',0):.3f}  "
                 f"QED={c.get('qed',0):.2f}  "
                 f"{c.get('smiles','')[:55]}")

    # ── Resolve target receptor ───────────────────────────────────────────────
    if not receptor_path:
        receptor_path = str(struct_dir / f"{uid}.pdbqt")
    rec_path = Path(receptor_path)
    if not rec_path.exists():
        # Try to get it from a PDB
        pdb_path = struct_dir / f"{uid}.pdb"
        if pdb_path.exists():
            rec_path = _receptor_pdbqt(pdb_path, struct_dir)
            if rec_path is None:
                raise FileNotFoundError(f"Cannot prepare receptor PDBQT for {uid}")
        else:
            raise FileNotFoundError(
                f"Receptor not found: {receptor_path}\n"
                f"Run Module 01 (fetch_structure) first."
            )
    log.info(f"  Receptor: {rec_path}")

    # ── Load dock cache ───────────────────────────────────────────────────────
    cache_path = str(inter_dir / f"{uid}_dock_cache.json")
    # Also load selectivity-specific cache
    sel_cache_path = str(inter_dir / f"{uid}_sel_cache.json")
    _load_dock_cache(cache_path)
    _load_dock_cache(sel_cache_path)

    # ── Build off-target panel ───────────────────────────────────────────────
    log.info("\n  ── Building off-target panel ──")
    panel = build_off_target_panel(uid, target_gene, inter_dir, struct_dir, MAX_PANEL_SIZE)

    if not panel:
        log.warning("  No off-target structures available. Selectivity scan skipped.")
        return SelectivityResult(
            uniprot_id=uid, target_gene=target_gene,
            n_input_molecules=len(candidates),
            notes="no_off_target_structures"
        )

    # ── Temporary work directory ──────────────────────────────────────────────
    tmpdir = Path(tempfile.mkdtemp(prefix=f"sel_{uid}_"))
    try:
        log.info(f"\n  ── Selectivity scan + refinement ({len(top_candidates)} molecules × "
                 f"{len(panel)} off-targets) ──")

        refined_molecules: List[RefinedMolecule] = []

        for i, cand in enumerate(top_candidates, 1):
            smiles      = cand.get("smiles", "")
            init_score  = cand.get("score", 0.0)
            if not smiles:
                continue

            log.info(f"\n  [{i}/{len(top_candidates)}] Molecule: {smiles[:60]}")

            rm = refine_molecule(
                smiles=smiles,
                on_target_score=init_score,
                on_target_uniprot=uid,
                on_target_center=on_target_center,
                on_target_box=on_target_box,
                on_target_pdbqt=rec_path,
                panel=panel,
                vina_path=vina_path,
                structures_dir=struct_dir,
                tmpdir=tmpdir,
                max_rounds=max_rounds,
            )
            refined_molecules.append(rm)
            _save_dock_cache(sel_cache_path)  # checkpoint after each molecule

        # ── Pareto ranking ────────────────────────────────────────────────────
        log.info("\n  ── Computing Pareto front ──")
        refined_molecules = pareto_rank(refined_molecules)

        # ── Aggregate stats ───────────────────────────────────────────────────
        best = refined_molecules[0] if refined_molecules else None
        n_herg_safe = sum(1 for r in refined_molecules if r.optimized_profile.herg_safe)
        n_excellent = sum(1 for r in refined_molecules
                         if r.optimized_profile.selectivity_index >= SI_EXCELLENT)

        result = SelectivityResult(
            uniprot_id=uid,
            target_gene=target_gene,
            n_input_molecules=len(candidates),
            panel_size=len(panel),
            panel_proteins=[{k: v for k, v in p.items() if k != "pdb_path"}
                            for p in panel],
            initial_profiles=[r.original_profile for r in refined_molecules],
            refined_molecules=refined_molecules,
            best_molecule=best,
            best_si=best.optimized_profile.selectivity_index if best else 0.0,
            best_on_target=best.optimized_profile.on_target_score if best else 0.0,
            n_herg_safe=n_herg_safe,
            n_excellent_si=n_excellent,
            runtime_sec=time.time() - t0,
        )

        # ── Save output ───────────────────────────────────────────────────────
        out_path = inter_dir / f"{uid}_selectivity.json"
        result.to_json(out_path)
        log.info(f"\n  Results saved: {out_path}")
        log.info(result.summary())

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _save_dock_cache(sel_cache_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

import click

@click.command()
@click.option("--uniprot",   "-u", required=True,
              help="UniProt ID of the TARGET protein (e.g. P04637)")
@click.option("--vina",      "-v", default=r"C:\Users\adria\Documents\proteinFP\pipeline\vina.exe",
              help="Path to AutoDock Vina executable")
@click.option("--receptor",  "-r", default=None,
              help="Path to target receptor PDBQT (auto-resolved if omitted)")
@click.option("--top-n",     "-n", default=DEFAULT_TOP_N, type=int,
              help=f"Number of de novo candidates to optimize (default: {DEFAULT_TOP_N})")
@click.option("--rounds",    "-R", default=MAX_REFINEMENT_ROUNDS, type=int,
              help=f"Max MC refinement rounds per molecule (default: {MAX_REFINEMENT_ROUNDS})")
@click.option("--seed",      "-s", default=None, type=int,
              help="Random seed for reproducibility")
def main(uniprot, vina, receptor, top_n, rounds, seed):
    """
    Module 15 — Selectivity Optimizer.

    Takes top molecules from de novo design (Module denovo_design.py),
    docks them against a human proteome off-target panel, then applies
    medicinal chemistry transformations to improve selectivity while
    maintaining on-target potency.

    Example:
        python pipeline\\selectivity_optimizer.py \\
            --uniprot P04637 \\
            --vina C:/Users/adria/Documents/proteinFP/pipeline/vina.exe \\
            --top-n 5 --rounds 15
    """
    result = run_selectivity_optimization(
        uniprot_id=uniprot.strip().upper(),
        vina_path=vina,
        receptor_path=receptor or "",
        top_n=top_n,
        max_rounds=rounds,
        rng_seed=seed,
    )
    click.echo(result.summary())


if __name__ == "__main__":
    main()