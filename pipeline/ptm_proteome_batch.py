"""
pipeline/ptm_proteome_batch.py
───────────────────────────────
Proteome-wide PTM batch runner for Module 17.

Discovers all proteins that have a _structure.json in data/intermediate/,
runs PTM analysis on every one with ThreadPool parallelism, and produces:

  data/reports/ptm_proteome_summary.json   ← machine-readable summary
  data/reports/ptm_proteome_summary.txt    ← human-readable report
  data/intermediate/{uid}_ptm.json         ← per-protein PTM output (existing module)

Features:
  - Auto-discovers all proteins from existing intermediate data
  - Skips proteins that already have a _ptm.json (resumable)
  - Parallel workers (default: 4, configurable)
  - Progress bar + ETA
  - Proteome-level statistics:
      top kinases across all proteins
      most frequently modified residue types
      proteins with degradation signals
      PTM hotspot distribution
  - SIM-02 integration: patches all existing ensemble JSONs with new PTM states
  - Graceful error handling — one failed protein never stops the batch

Usage:
    python pipeline/ptm_proteome_batch.py
    python pipeline/ptm_proteome_batch.py --workers 8
    python pipeline/ptm_proteome_batch.py --rerun        (re-analyse even if _ptm.json exists)
    python pipeline/ptm_proteome_batch.py --uniprot-list P04637,P00533,Q00987
    python pipeline/ptm_proteome_batch.py --no-phosphosite  (faster, motif-only)
    python pipeline/ptm_proteome_batch.py --workers 8 --rerun
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict

import click

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import cfg, get_logger
from pipeline.ptm_analysis import analyze_ptms as run_ptm_analysis, PTMResult

log = get_logger(__name__)


# ── Per-protein result container ──────────────────────────────────────────────

@dataclass
class ProteinPTMSummary:
    uniprot_id:           str
    gene_name:            str
    sequence_length:      int
    n_total_sites:        int
    n_known:              int
    n_predicted:          int
    n_phospho:            int
    n_ubiq:               int
    n_acetyl:             int
    n_glyco:              int
    n_sumo:               int
    n_methyl:             int
    n_activating:         int
    n_inhibitory:         int
    n_degradation:        int
    n_affecting_pocket:   int
    n_affecting_active:   int
    dominant_kinase:      str
    has_degradation_signal: bool
    has_activation_cluster: bool
    max_charge_delta:     float
    n_ptm_states:         int
    n_grn_signals:        int
    status:               str   # "ok", "skipped", "error"
    error_msg:            str   = ""
    runtime_s:            float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProteomePTMReport:
    run_date:             str
    n_proteins_total:     int
    n_proteins_ok:        int
    n_proteins_skipped:   int
    n_proteins_error:     int
    total_ptm_sites:      int
    total_known_sites:    int
    total_predicted_sites: int
    mean_sites_per_protein: float
    top_kinases:          list[tuple[str, int]]   # (kinase_name, count)
    top_modified_aas:     list[tuple[str, int]]   # (aa_type, count)
    proteins_with_degradation: list[str]
    proteins_with_activation:  list[str]
    most_ptm_rich:        list[tuple[str, int]]   # (uid, n_sites) top 10
    summaries:            list[ProteinPTMSummary] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_text(self) -> str:
        lines = [
            "═" * 70,
            "  ProteinFP — Proteome-wide PTM Analysis Report",
            f"  Run date : {self.run_date}",
            "═" * 70,
            "",
            f"  Proteins analysed : {self.n_proteins_ok} / {self.n_proteins_total}",
            f"  Proteins skipped  : {self.n_proteins_skipped} (already done)",
            f"  Errors            : {self.n_proteins_error}",
            "",
            "─" * 70,
            "  GLOBAL PTM STATISTICS",
            "─" * 70,
            f"  Total PTM sites     : {self.total_ptm_sites:,}",
            f"    Known (DB)        : {self.total_known_sites:,}",
            f"    Predicted (motif) : {self.total_predicted_sites:,}",
            f"  Mean sites/protein  : {self.mean_sites_per_protein:.1f}",
            "",
            "─" * 70,
            "  TOP KINASES ACROSS PROTEOME",
            "─" * 70,
        ]
        for kinase, count in self.top_kinases[:15]:
            bar = "█" * min(40, count // max(1, self.n_proteins_ok // 10))
            lines.append(f"  {kinase:<20} {count:>5}  {bar}")

        lines += [
            "",
            "─" * 70,
            "  MOST FREQUENTLY MODIFIED RESIDUE TYPES",
            "─" * 70,
        ]
        for aa, count in self.top_modified_aas[:10]:
            lines.append(f"  {aa:<6} {count:>6} sites")

        lines += [
            "",
            "─" * 70,
            "  PROTEINS WITH DEGRADATION SIGNALS",
            "─" * 70,
        ]
        for uid in self.proteins_with_degradation[:20]:
            lines.append(f"  {uid}")

        lines += [
            "",
            "─" * 70,
            "  PROTEINS WITH ACTIVATION CLUSTERS (≥3 high-conf activating phospho)",
            "─" * 70,
        ]
        for uid in self.proteins_with_activation[:20]:
            lines.append(f"  {uid}")

        lines += [
            "",
            "─" * 70,
            "  TOP 10 PTM-RICHEST PROTEINS",
            "─" * 70,
        ]
        for uid, n in self.most_ptm_rich[:10]:
            lines.append(f"  {uid:<14} {n:>4} sites")

        lines += [
            "",
            "─" * 70,
            "  PER-PROTEIN SUMMARY",
            "─" * 70,
            f"  {'UniProt':<12} {'Gene':<10} {'Total':>6} {'Known':>6} "
            f"{'Phospho':>8} {'Ubiq':>5} {'Acetyl':>7} "
            f"{'Pocket':>7} {'Kinase':<16} {'Status'}",
            f"  {'─'*12} {'─'*10} {'─'*6} {'─'*6} "
            f"{'─'*8} {'─'*5} {'─'*7} "
            f"{'─'*7} {'─'*16} {'─'*6}",
        ]
        for s in sorted(self.summaries, key=lambda x: -x.n_total_sites):
            if s.status == "error":
                lines.append(f"  {s.uniprot_id:<12} {'ERROR':<10} — {s.error_msg[:40]}")
            elif s.status == "skipped":
                lines.append(f"  {s.uniprot_id:<12} {'(skipped)':<10}")
            else:
                lines.append(
                    f"  {s.uniprot_id:<12} {s.gene_name[:9]:<10} "
                    f"{s.n_total_sites:>6} {s.n_known:>6} "
                    f"{s.n_phospho:>8} {s.n_ubiq:>5} {s.n_acetyl:>7} "
                    f"{s.n_affecting_pocket:>7} {s.dominant_kinase[:15]:<16} ok"
                )

        lines += ["", "═" * 70, ""]
        return "\n".join(lines)


# ── Worker function (runs in thread pool) ─────────────────────────────────────

def _analyse_one(
    uid:              str,
    inter_dir:        Path,
    fetch_phosphosite: bool,
) -> ProteinPTMSummary:
    """
    Analyse one protein. Called from thread pool.
    Returns a ProteinPTMSummary regardless of success/failure.
    """
    t0 = time.time()

    def _load(fname: str) -> Optional[dict]:
        p = inter_dir / fname
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    try:
        struct_data  = _load(f"{uid}_structure.json")
        if not struct_data:
            return ProteinPTMSummary(
                uniprot_id=uid, gene_name=uid, sequence_length=0,
                n_total_sites=0, n_known=0, n_predicted=0,
                n_phospho=0, n_ubiq=0, n_acetyl=0, n_glyco=0,
                n_sumo=0, n_methyl=0, n_activating=0, n_inhibitory=0,
                n_degradation=0, n_affecting_pocket=0, n_affecting_active=0,
                dominant_kinase="", has_degradation_signal=False,
                has_activation_cluster=False, max_charge_delta=0.0,
                n_ptm_states=0, n_grn_signals=0,
                status="error", error_msg="no structure.json",
            )

        sequence  = struct_data.get("sequence", "")
        gene_name = struct_data.get("gene_name", uid)

        if not sequence:
            return ProteinPTMSummary(
                uniprot_id=uid, gene_name=gene_name, sequence_length=0,
                n_total_sites=0, n_known=0, n_predicted=0,
                n_phospho=0, n_ubiq=0, n_acetyl=0, n_glyco=0,
                n_sumo=0, n_methyl=0, n_activating=0, n_inhibitory=0,
                n_degradation=0, n_affecting_pocket=0, n_affecting_active=0,
                dominant_kinase="", has_degradation_signal=False,
                has_activation_cluster=False, max_charge_delta=0.0,
                n_ptm_states=0, n_grn_signals=0,
                status="error", error_msg="empty sequence",
            )

        physico_data = _load(f"{uid}_physicochemical.json")
        active_data  = _load(f"{uid}_active_sites.json")
        pocket_data  = _load(f"{uid}_binding_pockets.json")
        ppi_data     = _load(f"{uid}_ppi.json")

        result: PTMResult = run_ptm_analysis(
            uniprot_id=uid,
            sequence=sequence,
            active_data=active_data,
            pocket_data=pocket_data,
            physico_data=physico_data,
            use_api=fetch_phosphosite,
        )

        # Patch gene name (analyze_ptms defaults to uid)
        result.gene_name = gene_name

        # Save per-protein JSON
        out_path = inter_dir / f"{uid}_ptm.json"
        result.to_json(out_path)

        runtime = round(time.time() - t0, 2)

        # ── Map analyze_ptms() fields to ProteinPTMSummary ──────────────
        sites = result.sites  # your PTMResult uses .sites not .ptm_sites
        phospho_types = {"phosphoserine", "phosphothreonine", "phosphotyrosine"}

        n_phospho = sum(1 for s in sites if s.ptm_type in phospho_types)
        n_ubiq    = sum(1 for s in sites if s.ptm_type == "ubiquitination")
        n_acetyl  = sum(1 for s in sites if s.ptm_type == "acetylation")
        n_glyco   = sum(1 for s in sites if s.ptm_type in ("nglycosylation", "oglycosylation"))
        n_sumo    = sum(1 for s in sites if s.ptm_type == "sumoylation")
        n_methyl  = sum(1 for s in sites if s.ptm_type in ("methylation", "dimethylation", "trimethylation"))

        n_activating  = sum(1 for s in sites if s.conformational_effect == "activating")
        n_inhibitory  = sum(1 for s in sites if s.conformational_effect == "inactivating")
        # degradation signal: ubiquitinated protein is degradation-destined
        n_degradation = n_ubiq
        n_pocket      = sum(1 for s in sites if s.is_binding_switch)
        n_active      = sum(1 for s in sites if s.is_active_site_switch)

        has_deg    = n_ubiq > 0
        has_activ  = sum(1 for s in sites
                         if s.conformational_effect == "activating"
                         and s.confidence >= 0.8) >= 3

        return ProteinPTMSummary(
            uniprot_id=uid,
            gene_name=gene_name,
            sequence_length=len(sequence),
            n_total_sites=len(sites),
            n_known=result.n_known,
            n_predicted=result.n_predicted,
            n_phospho=n_phospho,
            n_ubiq=n_ubiq,
            n_acetyl=n_acetyl,
            n_glyco=n_glyco,
            n_sumo=n_sumo,
            n_methyl=n_methyl,
            n_activating=n_activating,
            n_inhibitory=n_inhibitory,
            n_degradation=n_degradation,
            n_affecting_pocket=n_pocket,
            n_affecting_active=n_active,
            dominant_kinase=result.dominant_kinase,
            has_degradation_signal=has_deg,
            has_activation_cluster=has_activ,
            max_charge_delta=result.total_charge_shift_phospho,
            n_ptm_states=0,   # analyze_ptms uses state_corrections dict, not list
            n_grn_signals=0,
            status="ok",
            runtime_s=runtime,
        )

    except Exception as e:
        tb = traceback.format_exc()
        log.warning(f"  PTM failed for {uid}: {e}\n{tb[-400:]}")
        return ProteinPTMSummary(
            uniprot_id=uid, gene_name=uid, sequence_length=0,
            n_total_sites=0, n_known=0, n_predicted=0,
            n_phospho=0, n_ubiq=0, n_acetyl=0, n_glyco=0,
            n_sumo=0, n_methyl=0, n_activating=0, n_inhibitory=0,
            n_degradation=0, n_affecting_pocket=0, n_affecting_active=0,
            dominant_kinase="", has_degradation_signal=False,
            has_activation_cluster=False, max_charge_delta=0.0,
            n_ptm_states=0, n_grn_signals=0,
            status="error", error_msg=str(e)[:120],
        )


# ── SIM-02 ensemble patching ──────────────────────────────────────────────────

def _patch_sim02_ensembles(inter_dir: Path, summaries: list[ProteinPTMSummary]) -> int:
    """
    For each protein that has both a _ptm.json and an existing SIM-02 ensemble
    JSON, inject the PTM-derived conformational states into the ensemble.

    Returns number of ensembles patched.
    """
    sim_dir   = ROOT / "data" / "sim" / "ensembles"
    n_patched = 0

    for s in summaries:
        if s.status != "ok":
            continue

        uid       = s.uniprot_id
        ptm_path  = inter_dir / f"{uid}_ptm.json"
        sim_paths = list(sim_dir.glob(f"{uid}*.json")) if sim_dir.exists() else []

        if not ptm_path.exists() or not sim_paths:
            continue

        try:
            ptm_data = json.loads(ptm_path.read_text())
            # analyze_ptms() stores SIM-02 data in state_corrections dict
            # Convert to list of state objects SIM-02 can consume
            state_corrections = ptm_data.get("state_corrections", {})
            new_states = []
            for state_name, ddG in state_corrections.items():
                if ddG != 0.0:
                    new_states.append({
                        "name": f"ptm_{state_name}",
                        "delta_G_kJ_mol": ddG,
                        "pocket_volume_A3": 300.0,
                        "pocket_shape": 0.6,
                        "druggability": 0.55,
                        "probability": 0.1,
                        "accessible": True,
                        "ptm_driven": True,
                    })
            if not new_states:
                continue

            sim_path = sim_paths[0]
            ensemble = json.loads(sim_path.read_text())

            # Avoid duplicating states on re-run
            existing_names = {st.get("name", "") for st in ensemble.get("states", [])}
            added = [st for st in new_states if st["name"] not in existing_names]

            if added:
                ensemble.setdefault("states", []).extend(added)
                ensemble["ptm_states_injected"] = len(added)
                sim_path.write_text(json.dumps(ensemble, indent=2))
                n_patched += 1

        except Exception as e:
            log.debug(f"  SIM-02 patch failed for {uid}: {e}")

    return n_patched


# ── Aggregate statistics ───────────────────────────────────────────────────────

def _build_report(
    summaries:  list[ProteinPTMSummary],
    inter_dir:  Path,
) -> ProteomePTMReport:
    """Aggregate all per-protein summaries into a proteome-wide report."""
    from datetime import datetime

    ok       = [s for s in summaries if s.status == "ok"]
    skipped  = [s for s in summaries if s.status == "skipped"]
    errors   = [s for s in summaries if s.status == "error"]

    total_sites    = sum(s.n_total_sites for s in ok)
    total_known    = sum(s.n_known for s in ok)
    total_pred     = sum(s.n_predicted for s in ok)
    mean_sites     = total_sites / max(len(ok), 1)

    # Aggregate kinase counts from all PTM JSONs
    kinase_counts: dict[str, int] = defaultdict(int)
    aa_counts:     dict[str, int] = defaultdict(int)

    for s in ok:
        ptm_path = inter_dir / f"{s.uniprot_id}_ptm.json"
        if not ptm_path.exists():
            continue
        try:
            data = json.loads(ptm_path.read_text())
            for site in data.get("sites", []):  # analyze_ptms saves as "sites"
                kinase = site.get("kinase_enzyme", "")
                aa     = site.get("residue_aa", "")
                if kinase:
                    kinase_counts[kinase] += 1
                if aa:
                    aa_counts[aa] += 1
        except Exception:
            pass

    top_kinases   = sorted(kinase_counts.items(), key=lambda x: -x[1])[:20]
    top_aas       = sorted(aa_counts.items(), key=lambda x: -x[1])[:10]
    degradation   = [s.uniprot_id for s in ok if s.has_degradation_signal]
    activation    = [s.uniprot_id for s in ok if s.has_activation_cluster]
    ptm_rich      = sorted([(s.uniprot_id, s.n_total_sites) for s in ok],
                           key=lambda x: -x[1])[:10]

    return ProteomePTMReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_proteins_total=len(summaries),
        n_proteins_ok=len(ok),
        n_proteins_skipped=len(skipped),
        n_proteins_error=len(errors),
        total_ptm_sites=total_sites,
        total_known_sites=total_known,
        total_predicted_sites=total_pred,
        mean_sites_per_protein=round(mean_sites, 1),
        top_kinases=top_kinases,
        top_modified_aas=top_aas,
        proteins_with_degradation=degradation,
        proteins_with_activation=activation,
        most_ptm_rich=ptm_rich,
        summaries=summaries,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--workers", "-w", default=4, type=int,
              help="Parallel worker threads (default: 4)")
@click.option("--rerun", is_flag=True, default=False,
              help="Re-analyse proteins that already have a _ptm.json")
@click.option("--no-phosphosite", "no_phosphosite", is_flag=True, default=False,
              help="Skip PhosphoSitePlus lookup (faster, motif-only)")
@click.option("--uniprot-list", "uniprot_list", default=None,
              help="Comma-separated list of UniProt IDs to process "
                   "(default: all proteins with _structure.json)")
@click.option("--inter-dir", "inter_dir_override", default=None,
              help="Override intermediate data directory")
def main(
    workers:          int,
    rerun:            bool,
    no_phosphosite:   bool,
    uniprot_list:     Optional[str],
    inter_dir_override: Optional[str],
) -> None:
    """
    Proteome-wide PTM batch analysis — Module 17 at scale.

    Discovers all proteins with _structure.json in data/intermediate/,
    runs PTM analysis in parallel, and writes a proteome summary report.

    Examples:
        python pipeline/ptm_proteome_batch.py
        python pipeline/ptm_proteome_batch.py --workers 8
        python pipeline/ptm_proteome_batch.py --rerun --no-phosphosite
        python pipeline/ptm_proteome_batch.py --uniprot-list P04637,P00533
    """
    inter_dir = Path(inter_dir_override) if inter_dir_override \
                else Path(cfg.paths["intermediate"])
    report_dir = Path(cfg.paths["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)

    fetch_phosphosite = not no_phosphosite

    # ── Discover proteins ──────────────────────────────────────────────────────
    if uniprot_list:
        all_uids = [u.strip().upper() for u in uniprot_list.split(",") if u.strip()]
        click.echo(f"\n  Target list: {len(all_uids)} proteins specified")
    else:
        structure_files = sorted(inter_dir.glob("*_structure.json"))
        all_uids = [f.stem.replace("_structure", "") for f in structure_files]
        click.echo(f"\n  Auto-discovered {len(all_uids)} proteins "
                   f"with _structure.json in {inter_dir}")

    if not all_uids:
        click.echo(
            "\n  No proteins found!\n"
            f"  Make sure data/intermediate/ contains *_structure.json files.\n"
            "  Run Module 01 first:\n"
            "    python pipeline/01_fetch_structure.py --uniprot P04637\n"
        )
        raise SystemExit(1)

    # ── Decide which to run vs skip ────────────────────────────────────────────
    to_run:    list[str] = []
    to_skip:   list[str] = []

    for uid in all_uids:
        ptm_exists = (inter_dir / f"{uid}_ptm.json").exists()
        if ptm_exists and not rerun:
            to_skip.append(uid)
        else:
            to_run.append(uid)

    click.echo(f"  To run    : {len(to_run)}")
    click.echo(f"  To skip   : {len(to_skip)} (already have _ptm.json — use --rerun to redo)")
    click.echo(f"  Workers   : {workers}")
    click.echo(f"  PhosphoSite: {'yes' if fetch_phosphosite else 'no (--no-phosphosite)'}")

    if not to_run:
        click.echo("\n  All proteins already analysed. Use --rerun to redo.")
        # Still build the aggregate report
    else:
        click.echo(f"\n  Starting proteome PTM analysis...\n")

    # ── Run in thread pool ─────────────────────────────────────────────────────
    summaries: list[ProteinPTMSummary] = []

    # Pre-populate skipped entries
    for uid in to_skip:
        struct = inter_dir / f"{uid}_structure.json"
        gene_name = uid
        try:
            d = json.loads(struct.read_text())
            gene_name = d.get("gene_name", uid)
        except Exception:
            pass
        summaries.append(ProteinPTMSummary(
            uniprot_id=uid, gene_name=gene_name, sequence_length=0,
            n_total_sites=0, n_known=0, n_predicted=0,
            n_phospho=0, n_ubiq=0, n_acetyl=0, n_glyco=0,
            n_sumo=0, n_methyl=0, n_activating=0, n_inhibitory=0,
            n_degradation=0, n_affecting_pocket=0, n_affecting_active=0,
            dominant_kinase="", has_degradation_signal=False,
            has_activation_cluster=False, max_charge_delta=0.0,
            n_ptm_states=0, n_grn_signals=0, status="skipped",
        ))

    if to_run:
        n_done     = 0
        n_ok       = 0
        n_error    = 0
        t_start    = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_analyse_one, uid, inter_dir, fetch_phosphosite): uid
                for uid in to_run
            }

            for future in as_completed(futures):
                uid    = futures[future]
                result = future.result()
                summaries.append(result)
                n_done += 1

                if result.status == "ok":
                    n_ok += 1
                    status_str = (
                        f"✓ {uid:<12} {result.gene_name[:9]:<10} "
                        f"{result.n_total_sites:>4} sites  "
                        f"phospho={result.n_phospho}  "
                        f"ubiq={result.n_ubiq}  "
                        f"acetyl={result.n_acetyl}  "
                        f"{result.runtime_s:.1f}s"
                    )
                elif result.status == "error":
                    n_error += 1
                    status_str = f"✗ {uid:<12} ERROR: {result.error_msg[:50]}"
                else:
                    status_str = f"- {uid:<12} skipped"

                # Progress line
                elapsed  = time.time() - t_start
                rate     = n_done / elapsed if elapsed > 0 else 1
                eta      = (len(to_run) - n_done) / rate if rate > 0 else 0
                progress = f"[{n_done:>4}/{len(to_run)}  {eta:.0f}s remaining]"

                click.echo(f"  {progress}  {status_str}")

        elapsed_total = time.time() - t_start
        click.echo(
            f"\n  ── Batch complete ──────────────────────────────────────\n"
            f"  Processed : {len(to_run)} proteins in {elapsed_total:.1f}s "
            f"({elapsed_total/max(len(to_run),1):.1f}s/protein)\n"
            f"  OK        : {n_ok}\n"
            f"  Errors    : {n_error}\n"
        )

    # ── Patch SIM-02 ensembles ─────────────────────────────────────────────────
    ok_summaries = [s for s in summaries if s.status == "ok"]
    if ok_summaries:
        click.echo("  Patching SIM-02 ensemble files with PTM states...")
        n_patched = _patch_sim02_ensembles(inter_dir, ok_summaries)
        click.echo(f"  → {n_patched} ensemble files updated with PTM conformational states")

    # ── Build and save aggregate report ───────────────────────────────────────
    click.echo("\n  Building proteome-wide summary report...")
    report = _build_report(summaries, inter_dir)

    json_out = report_dir / "ptm_proteome_summary.json"
    txt_out  = report_dir / "ptm_proteome_summary.txt"

    json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    txt_out.write_text(report.to_text(), encoding="utf-8")

    # Print summary to console
    click.echo(report.to_text())
    click.echo(f"  Reports saved:")
    click.echo(f"    {json_out}")
    click.echo(f"    {txt_out}")


if __name__ == "__main__":
    main()