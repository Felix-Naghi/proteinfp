"""
test_evolutionary.py
─────────────────────
Standalone test runner for all five evolutionary design modules:

  Module 16 — antibody_design    (CDR loop evolution)
  Module 18 — adc_design         (CDR + warhead + linker co-evolution)
  Module 19 — cart_design        (CAR-T scFv + architecture co-evolution)
  Module 20 — protac_design      (warhead + linker + E3 ligand co-evolution)
  Module 21 — allosteric_drug    (allosteric small molecule evolution)

REQUIREMENTS:
  Run `proteinfp --uniprot <ID>` first so intermediate JSON files exist.
  No Vina, RDKit, or OpenMM needed — all five modules are pure Python.

USAGE:
  # Test all 5 modules on TP53 (10 fast generations each):
  python test_evolutionary.py --uniprot P04637

  # Test on multiple proteins:
  python test_evolutionary.py --uniprot P04637 P00533 O60885

  # Run more generations for better results:
  python test_evolutionary.py --uniprot P04637 --generations 50

  # Test only specific modules:
  python test_evolutionary.py --uniprot P04637 --modules antibody protac

  # Force re-run even if cached outputs exist:
  python test_evolutionary.py --uniprot P04637 --force

  # Skip modules that already have output:
  python test_evolutionary.py --uniprot P04637 --skip-done
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import click

# ── Add project root to sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_MODULES = ["antibody", "adc", "cart", "protac", "allosteric"]

MODULE_LABELS = {
    "antibody":  "Module 16 — Antibody CDR Design",
    "adc":       "Module 18 — ADC Design",
    "cart":      "Module 19 — CAR-T Design",
    "protac":    "Module 20 — PROTAC Design",
    "allosteric":"Module 21 — Allosteric Drug Design",
}

OUTPUT_KEYS = {
    "antibody":  "_antibody.json",
    "adc":       "_adc.json",
    "cart":      "_cart.json",
    "protac":    "_protac.json",
    "allosteric":"_allosteric_drug.json",
}


# ══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModuleTestResult:
    module:      str
    uniprot_id:  str
    status:      str        # "ok" | "skip" | "fail" | "no_data"
    elapsed_sec: float      = 0.0
    output_path: str        = ""
    top_score:   float      = 0.0
    top_line:    str        = ""
    error:       str        = ""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_intermediates(uid: str, inter_dir: Path) -> Dict[str, Optional[dict]]:
    """Load all intermediate JSON files for a protein."""
    files = {
        "active_sites":     f"{uid}_active_sites.json",
        "physicochemical":  f"{uid}_physicochemical.json",
        "ppi":              f"{uid}_ppi.json",
        "allosteric":       f"{uid}_allosteric.json",
        "binding_pockets":  f"{uid}_binding_pockets.json",
    }
    data = {}
    for key, fname in files.items():
        p = inter_dir / fname
        data[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return data


def _check_prereqs(uid: str, inter_dir: Path, report_dir: Path) -> List[str]:
    """Return list of missing prerequisite files."""
    missing = []
    report = report_dir / f"{uid}_report.json"
    if not report.exists():
        missing.append(f"Consensus report missing — run: proteinfp --uniprot {uid}")
    return missing


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL MODULE RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_antibody(uid: str, data: dict, inter_dir: Path,
                  generations: int, force: bool) -> ModuleTestResult:
    t0 = time.time()
    out_path = inter_dir / f"{uid}_antibody.json"
    if out_path.exists() and not force:
        return ModuleTestResult("antibody", uid, "skip",
                                output_path=str(out_path))
    try:
        from pipeline.antibody_design import run_antibody_design
        result = run_antibody_design(
            uniprot_id      = uid,
            active_data     = data["active_sites"],
            physico_data    = data["physicochemical"],
            ppi_data        = data["ppi"],
            allosteric_data = data["allosteric"],
            n_generations   = generations,
        )
        result.to_json(out_path)

        top = result.top_candidates[0] if result.top_candidates else None
        top_score = top.fitness if top else 0.0
        top_line  = (f"aff={top.affinity_score:.3f}  "
                     f"dev={top.developability:.2f}  "
                     f"H3={top.cdr_h3}") if top else "no candidates"

        return ModuleTestResult("antibody", uid, "ok",
                                elapsed_sec=time.time()-t0,
                                output_path=str(out_path),
                                top_score=top_score,
                                top_line=top_line)
    except Exception as e:
        return ModuleTestResult("antibody", uid, "fail",
                                elapsed_sec=time.time()-t0,
                                error=f"{type(e).__name__}: {e}")


def _run_adc(uid: str, data: dict, inter_dir: Path,
             generations: int, force: bool) -> ModuleTestResult:
    t0 = time.time()
    out_path = inter_dir / f"{uid}_adc.json"
    if out_path.exists() and not force:
        return ModuleTestResult("adc", uid, "skip",
                                output_path=str(out_path))
    try:
        from pipeline.adc_design import run_adc_design
        result = run_adc_design(
            uniprot_id      = uid,
            active_data     = data["active_sites"],
            physico_data    = data["physicochemical"],
            ppi_data        = data["ppi"],
            allosteric_data = data["allosteric"],
            n_generations   = generations,
            force           = force,
        )

        top = result.top_candidates[0] if result.top_candidates else None
        top_score = top.fitness if top else 0.0
        top_line  = (f"aff={top.affinity_score:.3f}  "
                     f"warhead={top.warhead_class}  "
                     f"linker={top.linker_name}  "
                     f"DAR={top.dar_min}-{top.dar_max}") if top else "no candidates"

        return ModuleTestResult("adc", uid, "ok",
                                elapsed_sec=time.time()-t0,
                                output_path=str(out_path),
                                top_score=top_score,
                                top_line=top_line)
    except Exception as e:
        return ModuleTestResult("adc", uid, "fail",
                                elapsed_sec=time.time()-t0,
                                error=f"{type(e).__name__}: {e}")


def _run_cart(uid: str, data: dict, inter_dir: Path,
              generations: int, force: bool) -> ModuleTestResult:
    t0 = time.time()
    out_path = inter_dir / f"{uid}_cart.json"
    if out_path.exists() and not force:
        return ModuleTestResult("cart", uid, "skip",
                                output_path=str(out_path))
    try:
        from pipeline.cart_design import run_cart_design
        result = run_cart_design(
            uniprot_id      = uid,
            active_data     = data["active_sites"],
            physico_data    = data["physicochemical"],
            ppi_data        = data["ppi"],
            allosteric_data = data["allosteric"],
            n_generations   = generations,
            force           = force,
        )

        top = result.top_candidates[0] if result.top_candidates else None
        top_score = top.fitness if top else 0.0
        top_line  = (f"aff={top.affinity_score:.3f}  "
                     f"act={top.activation_score:.2f}  "
                     f"persist={top.persistence_score:.2f}  "
                     f"arch={top.car_arch_name}") if top else "no candidates"

        return ModuleTestResult("cart", uid, "ok",
                                elapsed_sec=time.time()-t0,
                                output_path=str(out_path),
                                top_score=top_score,
                                top_line=top_line)
    except Exception as e:
        return ModuleTestResult("cart", uid, "fail",
                                elapsed_sec=time.time()-t0,
                                error=f"{type(e).__name__}: {e}")


def _run_protac(uid: str, data: dict, inter_dir: Path,
                generations: int, force: bool) -> ModuleTestResult:
    t0 = time.time()
    out_path = inter_dir / f"{uid}_protac.json"
    if out_path.exists() and not force:
        return ModuleTestResult("protac", uid, "skip",
                                output_path=str(out_path))
    try:
        from pipeline.protac_design import run_protac_design
        result = run_protac_design(
            uniprot_id      = uid,
            pocket_data     = data["binding_pockets"],
            active_data     = data["active_sites"],
            allosteric_data = data["allosteric"],
            n_generations   = generations,
            force           = force,
        )

        top = result.top_candidates[0] if result.top_candidates else None
        top_score = top.fitness if top else 0.0
        if top:
            try:
                import math
                dc50_nM = 10 ** ((1.0 - top.dc50_proxy) * 3.0)
                dc50_str = f"{dc50_nM:.1f}nM" if dc50_nM < 1000 else f"{dc50_nM/1000:.1f}µM"
            except Exception:
                dc50_str = f"{top.dc50_proxy:.3f}"
            top_line = (f"poi={top.poi_affinity:.3f}  "
                        f"e3={top.e3_affinity:.3f}  "
                        f"DC50~{dc50_str}  "
                        f"Dmax~{top.dmax_proxy*100:.0f}%  "
                        f"MW~{top.estimated_mw:.0f}  "
                        f"{top.e3_name}/{top.e3_ligand_name}")
        else:
            top_line = "no candidates"

        return ModuleTestResult("protac", uid, "ok",
                                elapsed_sec=time.time()-t0,
                                output_path=str(out_path),
                                top_score=top_score,
                                top_line=top_line)
    except Exception as e:
        return ModuleTestResult("protac", uid, "fail",
                                elapsed_sec=time.time()-t0,
                                error=f"{type(e).__name__}: {e}")


def _run_allosteric(uid: str, data: dict, inter_dir: Path,
                    generations: int, force: bool) -> ModuleTestResult:
    t0 = time.time()
    out_path = inter_dir / f"{uid}_allosteric_drug.json"
    if out_path.exists() and not force:
        return ModuleTestResult("allosteric", uid, "skip",
                                output_path=str(out_path))
    try:
        from pipeline.allosteric_drug_design import run_allosteric_drug_design
        result = run_allosteric_drug_design(
            uniprot_id      = uid,
            allosteric_data = data["allosteric"],
            physico_data    = data["physicochemical"],
            n_generations   = generations,
            force           = force,
        )

        top = result.top_candidates[0] if result.top_candidates else None
        top_score = top.fitness if top else 0.0
        top_line  = (f"site={top.site_id}  "
                     f"compl={top.site_complementarity:.3f}  "
                     f"comm={top.communication_score:.3f}  "
                     f"mech={top.mechanism}  "
                     f"MW={top.mw:.0f}") if top else "no candidates"

        return ModuleTestResult("allosteric", uid, "ok",
                                elapsed_sec=time.time()-t0,
                                output_path=str(out_path),
                                top_score=top_score,
                                top_line=top_line)
    except Exception as e:
        return ModuleTestResult("allosteric", uid, "fail",
                                elapsed_sec=time.time()-t0,
                                error=f"{type(e).__name__}: {e}")


MODULE_RUNNERS = {
    "antibody":  _run_antibody,
    "adc":       _run_adc,
    "cart":      _run_cart,
    "protac":    _run_protac,
    "allosteric":_run_allosteric,
}


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PRINTER
# ══════════════════════════════════════════════════════════════════════════════

STATUS_ICON = {"ok": "✓", "skip": "–", "fail": "✗", "no_data": "?"}
STATUS_LABEL = {"ok": "PASS", "skip": "SKIP", "fail": "FAIL", "no_data": "N/A "}


def _print_results(all_results: List[ModuleTestResult], generations: int) -> None:
    print(f"\n{'═'*72}")
    print(f"  EVOLUTIONARY MODULE TEST RESULTS  ({generations} generations)")
    print(f"{'═'*72}\n")

    # Group by protein
    proteins = list(dict.fromkeys(r.uniprot_id for r in all_results))

    for uid in proteins:
        uid_results = [r for r in all_results if r.uniprot_id == uid]
        n_ok   = sum(1 for r in uid_results if r.status == "ok")
        n_skip = sum(1 for r in uid_results if r.status == "skip")
        n_fail = sum(1 for r in uid_results if r.status == "fail")
        total_time = sum(r.elapsed_sec for r in uid_results)

        print(f"  {'─'*68}")
        print(f"  {uid}  —  {n_ok} passed  {n_skip} skipped  {n_fail} failed  "
              f"({total_time:.1f}s total)")
        print(f"  {'─'*68}")

        for r in uid_results:
            icon  = STATUS_ICON.get(r.status, "?")
            label = STATUS_LABEL.get(r.status, r.status)
            name  = MODULE_LABELS.get(r.module, r.module)

            if r.status == "ok":
                print(f"  {icon} [{label}]  {name}")
                print(f"           Score : {r.top_score:.4f}")
                print(f"           Best  : {r.top_line}")
                print(f"           Time  : {r.elapsed_sec:.1f}s")
                print(f"           Out   : {Path(r.output_path).name}")

            elif r.status == "skip":
                print(f"  {icon} [{label}]  {name}")
                print(f"           Cached output exists — use --force to re-run")

            elif r.status == "fail":
                print(f"  {icon} [{label}]  {name}")
                print(f"           Error : {r.error}")
                print(f"           Time  : {r.elapsed_sec:.1f}s")

            elif r.status == "no_data":
                print(f"  {icon} [{label}]  {name}")
                print(f"           Missing prerequisite data")
            print()

    # Summary table
    print(f"  {'═'*68}")
    print(f"  {'Module':<24} {'Protein':<10} {'Status':<6}  {'Score':>7}  {'Time':>6}")
    print(f"  {'─'*24} {'─'*10} {'─'*6}  {'─'*7}  {'─'*6}")
    for r in all_results:
        score_str = f"{r.top_score:.4f}" if r.top_score > 0 else "  —   "
        time_str  = f"{r.elapsed_sec:.1f}s" if r.elapsed_sec > 0 else "  —  "
        label     = STATUS_LABEL.get(r.status, r.status)
        name      = r.module.ljust(24)[:24]
        print(f"  {name} {r.uniprot_id:<10} {label:<6}  {score_str:>7}  {time_str:>6}")

    print(f"  {'═'*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(
    uniprot_ids: List[str],
    modules:     List[str],
    generations: int,
    force:       bool,
    skip_done:   bool,
    verbose:     bool,
) -> List[ModuleTestResult]:

    try:
        from utils.config import cfg
        inter_dir  = Path(cfg.paths["intermediate"])
        report_dir = Path(cfg.paths["reports"])
    except Exception:
        inter_dir  = ROOT / "data" / "intermediate"
        report_dir = ROOT / "data" / "reports"

    all_results: List[ModuleTestResult] = []

    for uid in uniprot_ids:
        uid = uid.strip().upper()
        print(f"\n{'═'*72}")
        print(f"  Testing evolutionary modules — {uid}")
        print(f"  Generations: {generations}  |  Modules: {', '.join(modules)}")
        print(f"{'═'*72}\n")

        # Check prerequisites
        missing = _check_prereqs(uid, inter_dir, report_dir)
        if missing:
            print(f"  ✗ Cannot test {uid}:")
            for m in missing:
                print(f"    {m}")
            for mod in modules:
                all_results.append(ModuleTestResult(mod, uid, "no_data"))
            continue

        # Load intermediate data
        data = _load_intermediates(uid, inter_dir)
        loaded = [k for k, v in data.items() if v is not None]
        missing_data = [k for k, v in data.items() if v is None]

        print(f"  Intermediate data loaded: {', '.join(loaded)}")
        if missing_data:
            print(f"  Missing (modules will use fallbacks): {', '.join(missing_data)}")
        print()

        # Run each module
        for mod in modules:
            runner = MODULE_RUNNERS[mod]
            name   = MODULE_LABELS[mod]

            # Skip if output already exists and --skip-done
            out_suffix = OUTPUT_KEYS[mod]
            out_path   = inter_dir / f"{uid}{out_suffix}"
            if skip_done and out_path.exists() and not force:
                print(f"  – {name} — skipping (output exists)")
                all_results.append(ModuleTestResult(mod, uid, "skip",
                                                    output_path=str(out_path)))
                continue

            print(f"  ▶ {name}...")
            result = runner(uid, data, inter_dir, generations, force)
            all_results.append(result)

            icon = STATUS_ICON.get(result.status, "?")
            if result.status == "ok":
                print(f"    {icon} fitness={result.top_score:.4f}  "
                      f"({result.elapsed_sec:.1f}s)")
                print(f"      {result.top_line}")
            elif result.status == "skip":
                print(f"    {icon} cached — use --force to re-run")
            elif result.status == "fail":
                print(f"    ✗ FAILED: {result.error}")
                if verbose:
                    print()
            print()

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

@click.command()
@click.argument("uniprot_ids", nargs=-1, required=True,
                metavar="UNIPROT_ID [UNIPROT_ID ...]")
@click.option("--uniprot", "-u", multiple=True,
              help="UniProt ID(s) — alternative to positional args.")
@click.option("--generations", "-g", default=15, type=int, show_default=True,
              help="Evolution generations per module. "
                   "15 = fast test (~5s/module), 50 = better results (~30s/module).")
@click.option("--modules", "-m", multiple=True,
              type=click.Choice(ALL_MODULES, case_sensitive=False),
              help=f"Which modules to test. Default: all. "
                   f"Options: {', '.join(ALL_MODULES)}")
@click.option("--force", "-f", is_flag=True, default=False,
              help="Re-run even if output files already exist.")
@click.option("--skip-done", is_flag=True, default=False,
              help="Skip modules whose output already exists (same as not passing --force).")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print full tracebacks on failure.")
def main(uniprot_ids, uniprot, generations, modules, force, skip_done, verbose):
    """
    Test all evolutionary design modules on one or more proteins.

    UNIPROT_ID: one or more UniProt accessions (e.g. P04637 P00533 O60885)

    Requires the main pipeline to have run first:
        proteinfp --uniprot P04637

    \b
    Examples:
        python test_evolutionary.py P04637
        python test_evolutionary.py P04637 P00533 O60885
        python test_evolutionary.py P04637 --generations 50
        python test_evolutionary.py P04637 --modules antibody protac
        python test_evolutionary.py P04637 --force
        python test_evolutionary.py P04637 --skip-done
    """
    # Merge positional + --uniprot flag inputs
    ids = list(uniprot_ids) + list(uniprot)
    if not ids:
        click.echo("  Error: provide at least one UniProt ID.")
        raise SystemExit(1)

    mods = list(modules) if modules else ALL_MODULES

    all_results = run_tests(
        uniprot_ids = ids,
        modules     = mods,
        generations = generations,
        force       = force,
        skip_done   = skip_done,
        verbose     = verbose,
    )

    _print_results(all_results, generations)

    # Exit code: 1 if any module failed
    if any(r.status == "fail" for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()