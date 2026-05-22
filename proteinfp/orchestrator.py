"""
proteinfp/orchestrator.py
──────────────────────────
Core pipeline orchestrator — runs all available modules for a protein,
gracefully skipping any whose dependencies are not installed.

This is what `proteinfp --uniprot P04637` calls.

MODULE TIERS
────────────
Tier 1 — Core (always run, no optional deps beyond numpy/requests/biopython):
    01  fetch_structure       AlphaFold structure + UniProt metadata
    02  physicochemical       SASA, charge, hydrophobicity  [needs freesasa]
    03  active_sites          Catalytic residue prediction
    04  binding_pockets       Druggable pocket detection
    05  allosteric            Elastic network allosteric sites
    06  chemical_env          Active site chemical environment
    07  homology              Sequence/structure homologs
    13  consensus             Final report aggregation

Tier 2 — ML (needs torch + xgboost + lightgbm):
    08  esm2_embeddings       Protein language model
    10  EC classification     ML enzyme class prediction

Tier 3 — External databases (needs internet, always attempted):
    11  foldseek              Structural analog search
    12  ppi_network           Protein-protein interactions

Tier 4 — Optional heavy modules (separate installs):
    14  molecular_dynamics    OpenMM MD simulation      [needs openmm]
    15  de_novo_design        Evolutionary mol design    [needs rdkit + vina]
    17  ptm_analysis          Post-translational mods    [always available]
    18  cryptic_pockets       RMSF-based cryptic sites   [needs md output]
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class RunResult:
    uniprot_id:   str
    modules_run:  list[str]    = field(default_factory=list)
    modules_skip: list[str]    = field(default_factory=list)
    modules_fail: list[str]    = field(default_factory=list)
    report_path:  Optional[str] = None
    elapsed_sec:  float        = 0.0
    success:      bool         = False

    def summary(self) -> str:
        status = "✓ complete" if self.success else "✗ failed"
        lines = [
            f"\n  {'─'*60}",
            f"  ProteinFP run: {self.uniprot_id}  [{status}]",
            f"  {'─'*60}",
            f"  Modules run  : {len(self.modules_run)}  "
            f"({', '.join(self.modules_run)})",
        ]
        if self.modules_skip:
            lines.append(
                f"  Modules skip : {len(self.modules_skip)}  "
                f"({', '.join(self.modules_skip)})"
            )
        if self.modules_fail:
            lines.append(
                f"  Modules fail : {len(self.modules_fail)}  "
                f"({', '.join(self.modules_fail)})"
            )
        if self.report_path:
            lines.append(f"  Report       : {self.report_path}")
        lines.append(f"  Wall time    : {self.elapsed_sec:.1f}s")
        lines.append(f"  {'─'*60}")
        return "\n".join(lines)


# ── Module runner helper ───────────────────────────────────────────────────────

def _run_module(
    name:        str,
    fn,
    args:        tuple = (),
    kwargs:      dict  = None,
    result:      RunResult = None,
    skip_reason: str   = "",
) -> Optional[object]:
    """
    Run one pipeline module with uniform error handling.

    Returns the module's return value, or None on failure/skip.
    Updates result.modules_run / skip / fail in place.
    """
    if kwargs is None:
        kwargs = {}

    if skip_reason:
        print(f"  [SKIP] {name}: {skip_reason}")
        if result:
            result.modules_skip.append(name)
        return None

    try:
        t0  = time.time()
        ret = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  [ OK ] {name}  ({elapsed:.1f}s)")
        if result:
            result.modules_run.append(name)
        return ret
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        if result:
            result.modules_fail.append(name)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-module wrapper functions — each computes AND saves its JSON
# ══════════════════════════════════════════════════════════════════════════════

def _run_physicochemical(uid: str):
    """Module 02 — compute SASA/DSSP and save intermediate JSON."""
    from pipeline.physicochemical import compute_physicochemical
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    result = compute_physicochemical(parsed)
    result.to_json(inter / f"{uid}_physicochemical.json")
    return result


def _run_esm2(uid: str):
    """Module 08 — compute ESM-2 embeddings and save intermediate JSON."""
    from pipeline.esm2_embeddings import compute_esm2_embeddings
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    result = compute_esm2_embeddings(uid, parsed.sequence)
    result.to_json(inter / f"{uid}_esm2.json")
    return result


def _run_deepfri_go(uid: str):
    """Module 09 — predict GO terms from ESM-2 embeddings + homology and save JSON."""
    from pipeline.deepfri_go import predict_go_terms
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)

    def _load(fname):
        p = inter / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    result = predict_go_terms(
        uniprot_id      = uid,
        sequence        = parsed.sequence,
        esm2_result     = _load(f"{uid}_esm2.json"),
        homology_result = _load(f"{uid}_homology.json"),
        active_result   = _load(f"{uid}_active_sites.json"),
    )
    result.to_json(inter / f"{uid}_go_predictions.json")
    return result


def _run_active_sites(uid: str):
    from pipeline.active_sites import predict_active_sites
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    physico_path = inter / f"{uid}_physicochemical.json"
    physico = (json.loads(physico_path.read_text(encoding="utf-8"))
               if physico_path.exists() else None)
    result = predict_active_sites(parsed, physico)
    result.to_json(inter / f"{uid}_active_sites.json")
    return result


def _run_binding_pockets(uid: str):
    from pipeline.binding_pockets import detect_binding_pockets
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    active_path = inter / f"{uid}_active_sites.json"
    active_set  = set()
    if active_path.exists():
        ad = json.loads(active_path.read_text(encoding="utf-8"))
        active_set = {r["residue_number"]
                      for r in ad.get("active_residues", [])}
    result = detect_binding_pockets(parsed, active_set)
    result.to_json(inter / f"{uid}_binding_pockets.json")
    return result


def _run_allosteric(uid: str):
    from pipeline.allosteric import predict_allosteric_sites
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    result = predict_allosteric_sites(parsed)
    result.to_json(inter / f"{uid}_allosteric.json")
    return result


def _run_chemical_env(uid: str):
    from pipeline.chemical_env import map_chemical_environment
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)

    def _load(fname):
        p = inter / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    result = map_chemical_environment(
        parsed,
        _load(f"{uid}_active_sites.json"),
        _load(f"{uid}_binding_pockets.json"),
        _load(f"{uid}_allosteric.json"),
    )
    result.to_json(inter / f"{uid}_chemical_env.json")
    return result


def _run_homology(uid: str):
    from pipeline.homology import run_homology
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)
    result = run_homology(uid, parsed.sequence)
    result.to_json(inter / f"{uid}_homology.json")
    return result


def _run_ec_prediction(uid: str):
    from pipeline.ec_model_check import predict_ec_ml_checked
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)

    def _load(fname):
        p = inter / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    result = predict_ec_ml_checked(
        uniprot_id      = uid,
        sequence        = parsed.sequence,
        active_result   = _load(f"{uid}_active_sites.json"),
        go_result       = _load(f"{uid}_go_predictions.json"),
        homology_result = _load(f"{uid}_homology.json"),
        esm2_result     = _load(f"{uid}_esm2.json"),
        warn_on_fallback = True,
    )
    result.to_json(inter / f"{uid}_ec_prediction.json")
    return result


def _run_foldseek(uid: str):
    from pipeline.foldseek import run_foldseek
    from utils.config import cfg
    inter  = Path(cfg.paths["intermediate"])
    pdb    = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    result = run_foldseek(uid, pdb)
    result.to_json(inter / f"{uid}_foldseek.json")
    return result


def _run_ppi(uid: str):
    from pipeline.ppi_network import predict_ppi
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)

    sasa_map: dict = {}
    phys_path = inter / f"{uid}_physicochemical.json"
    if phys_path.exists():
        phys = json.loads(phys_path.read_text(encoding="utf-8"))
        for rec in phys.get("residues", []):
            sasa_map[(rec["chain_id"], rec["residue_number"])] = rec.get("sasa", 50.0)

    result = predict_ppi(uid, parsed.sequence, parsed, sasa_map)
    result.to_json(inter / f"{uid}_ppi.json")
    return result


def _run_ptm(uid: str):
    from pipeline.ptm_analysis import analyze_ptms
    from utils.config import cfg
    from utils.pdb_parser import parse_pdb
    inter  = Path(cfg.paths["intermediate"])
    struct = Path(cfg.paths["structures"]) / f"{uid}.pdb"
    parsed = parse_pdb(struct, uid)

    def _load(fname):
        p = inter / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    result = analyze_ptms(
        uniprot_id   = uid,
        sequence     = parsed.sequence,
        active_data  = _load(f"{uid}_active_sites.json"),
        pocket_data  = _load(f"{uid}_binding_pockets.json"),
        physico_data = _load(f"{uid}_physicochemical.json"),
    )
    result.to_json(inter / f"{uid}_ptm.json")
    return result


def _run_denovo(uid: str, vina_path: str):
    from pipeline.denovo_design import run_denovo_design
    from pipeline.denovo_design_context import load_consensus_context, load_md_context
    from utils.config import cfg
    inter  = Path(cfg.paths["intermediate"])

    def _load(fname):
        p = inter / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    result = run_denovo_design(
        uniprot_id      = uid,
        pocket_data     = _load(f"{uid}_binding_pockets.json"),
        active_data     = _load(f"{uid}_active_sites.json"),
        allosteric_data = _load(f"{uid}_allosteric.json"),
        chem_env_data   = _load(f"{uid}_chemical_env.json"),
        vina_path       = vina_path,
        consensus_data  = load_consensus_context(uid, inter),
        md_data         = load_md_context(uid, inter),
    )
    return result


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_pipeline(
    uniprot_id:     str,
    vina_path:      Optional[str]  = None,
    run_md:         bool           = False,
    run_denovo:     bool           = False,
    run_grn:        bool           = False,
    run_antibody:   bool           = False,
    epitope_mode:   str            = "auto",
    ab_generations: int            = 50,
    force:          bool           = False,
    verbose:        bool           = True,
) -> RunResult:
    """
    Run the full ProteinFP pipeline for a single protein.

    Skips modules whose dependencies are not installed rather than crashing.
    All skips are logged with the install command needed to enable them.

    Args:
        uniprot_id:  UniProt accession (e.g. "P04637")
        vina_path:   Path to AutoDock Vina executable (enables de novo)
        run_md:      Run molecular dynamics if OpenMM is available
        run_denovo:  Run de novo design if RDKit + Vina are available
        run_grn:     Run GRN modules if scRNA-seq data is configured
        force:       Re-run even if outputs already exist
        verbose:     Print progress to stdout

    Returns:
        RunResult with lists of run/skipped/failed modules and report path.
    """
    from proteinfp.deps import (
        has_freesasa, has_ml_stack, has_esm2,
        has_openmm, has_rdkit, has_vina, has_grn_stack,
        status_report,
    )

    t_start = time.time()
    result  = RunResult(uniprot_id=uniprot_id.strip().upper())

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ProteinFP  —  {result.uniprot_id}")
        print(f"{'='*60}")

    uid       = result.uniprot_id
    sys.path.insert(0, str(ROOT))

    # Resolve config paths
    try:
        from utils.config import cfg
        inter_dir  = Path(cfg.paths["intermediate"])
        report_dir = Path(cfg.paths["reports"])
        struct_dir = Path(cfg.paths["structures"])
    except Exception:
        inter_dir  = ROOT / "data" / "intermediate"
        report_dir = ROOT / "data" / "reports"
        struct_dir = ROOT / "data" / "structures"
        for d in (inter_dir, report_dir, struct_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Check if report already exists ────────────────────────────────────────
    report_path = report_dir / f"{uid}_report.json"
    if report_path.exists() and not force:
        if verbose:
            print(f"  Report already exists: {report_path}")
            print(f"  Use --force to re-run.")
        result.report_path = str(report_path)
        result.success     = True
        result.elapsed_sec = time.time() - t_start
        return result

    pdb_path = struct_dir / f"{uid}.pdb"

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1 — Core modules (no optional deps)
    # ══════════════════════════════════════════════════════════════════════════

    # Module 01: fetch structure
    from pipeline.fetch_structure import fetch_structure
    struct_result = _run_module(
        "01_fetch_structure",
        fetch_structure,
        args=(uid,),
        kwargs={"force": force},
        result=result,
    )
    if struct_result is None:
        print(f"\n  Cannot continue — Module 01 failed for {uid}.")
        result.elapsed_sec = time.time() - t_start
        return result

    # Module 02: physicochemical (needs freesasa)
    # FIX: use _run_physicochemical wrapper so the JSON is saved to disk
    if has_freesasa():
        _run_module(
            "02_physicochemical",
            _run_physicochemical, args=(uid,),
            result=result,
        )
    else:
        _run_module("02_physicochemical", None,
                    skip_reason="freesasa not installed "
                                "(pip install proteinfp[structure])",
                    result=result)

    # Module 03: active sites
    _run_module(
        "03_active_sites",
        _run_active_sites, args=(uid,), result=result,
    )

    # Module 04: binding pockets
    _run_module(
        "04_binding_pockets",
        _run_binding_pockets, args=(uid,), result=result,
    )

    # Module 05: allosteric sites
    _run_module(
        "05_allosteric",
        _run_allosteric, args=(uid,), result=result,
    )

    # Module 06: chemical environment
    _run_module(
        "06_chemical_env",
        _run_chemical_env, args=(uid,), result=result,
    )

    # Module 07: homology
    _run_module(
        "07_homology",
        _run_homology, args=(uid,), result=result,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — ML modules (needs torch + xgboost + lightgbm)
    # ══════════════════════════════════════════════════════════════════════════

    # Module 08: ESM-2 embeddings
    # FIX: use _run_esm2 wrapper so the JSON is saved to disk
    if has_esm2():
        _run_module(
            "08_esm2",
            _run_esm2, args=(uid,),
            result=result,
        )
    else:
        _run_module("08_esm2", None,
                    skip_reason="fair-esm not installed "
                                "(pip install proteinfp[ml])",
                    result=result)

    # Module 09: DeepFRI GO term prediction (uses ESM-2 + homology + active sites)
    # Runs always — falls back gracefully if ESM-2 JSON is missing
    _run_module(
        "09_deepfri_go",
        _run_deepfri_go, args=(uid,), result=result,
    )

    # Module 10: EC classification
    _run_module(
        "10_ec",
        _run_ec_prediction, args=(uid,), result=result,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3 — External database modules (internet required, always attempted)
    # ══════════════════════════════════════════════════════════════════════════

    _run_module("11_foldseek",   _run_foldseek,   args=(uid,), result=result)
    _run_module("12_ppi",        _run_ppi,        args=(uid,), result=result)

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 17: PTM analysis (always available)
    # ══════════════════════════════════════════════════════════════════════════

    _run_module("17_ptm", _run_ptm, args=(uid,), result=result)

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 4 — Heavy optional modules
    # ══════════════════════════════════════════════════════════════════════════

    # Module 14: molecular dynamics (needs OpenMM)
    if run_md:
        if has_openmm():
            from pipeline.molecular_dynamics import run_md as _md
            _run_module("14_md", _md, args=(uid,), result=result)
        else:
            _run_module("14_md", None,
                        skip_reason="OpenMM not installed "
                                    "(pip install proteinfp[sim])",
                        result=result)

    # Module 15: de novo design (needs RDKit + Vina)
    if run_denovo:
        if has_rdkit() and (vina_path or has_vina()):
            _run_module(
                "15_denovo",
                _run_denovo,
                args=(uid, vina_path or "vina"),
                result=result,
            )
        else:
            missing = []
            if not has_rdkit():
                missing.append("RDKit (pip install proteinfp[chem])")
            if not (vina_path or has_vina()):
                missing.append("AutoDock Vina (https://vina.scripps.edu)")
            _run_module("15_denovo", None,
                        skip_reason=f"missing: {', '.join(missing)}",
                        result=result)

    # Module 16: antibody design (always available)
    if run_antibody:
        try:
            from pipeline.antibody_design import run_antibody_design
            from utils.config import cfg as _cfg

            def _run_ab(uid_: str):
                _inter = Path(_cfg.paths["intermediate"])

                def _load(fname):
                    p = _inter / fname
                    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

                res = run_antibody_design(
                    uniprot_id      = uid_,
                    active_data     = _load(f"{uid_}_active_sites.json"),
                    physico_data    = _load(f"{uid_}_physicochemical.json"),
                    ppi_data        = _load(f"{uid_}_ppi.json"),
                    allosteric_data = _load(f"{uid_}_allosteric.json"),
                    epitope_mode    = epitope_mode,
                    n_generations   = ab_generations,
                )
                res.to_json(_inter / f"{uid_}_antibody.json")
                return res

            _run_module("16_antibody", _run_ab, args=(uid,), result=result)
        except ImportError as e:
            _run_module("16_antibody", None,
                        skip_reason=f"antibody_design not available: {e}",
                        result=result)

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 13: Consensus (always last)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_consensus(uid_: str):
        from pipeline.consensus import build_consensus_report
        from utils.config import cfg as _cfg
        _report_dir = Path(_cfg.paths["reports"])

        # build_consensus_report loads all intermediate JSONs itself
        report = build_consensus_report(uid_)

        out_json = _report_dir / f"{uid_}_report.json"
        out_txt  = _report_dir / f"{uid_}_report.txt"
        report.to_json(out_json)
        out_txt.write_text(report.to_text_report(), encoding="utf-8")
        return report

    consensus_result = _run_module(
        "13_consensus",
        _run_consensus, args=(uid,), result=result,
    )

    # ── Finalise ───────────────────────────────────────────────────────────────
    result.report_path = str(report_dir / f"{uid}_report.json")
    result.success     = consensus_result is not None
    result.elapsed_sec = time.time() - t_start

    if verbose:
        print(result.summary())

    return result