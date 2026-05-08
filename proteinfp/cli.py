"""
proteinfp/cli.py
─────────────────
Command-line interface for ProteinFP.

After `pip install proteinfp`, this is available as:

    proteinfp --uniprot P04637
    proteinfp --uniprot P04637 --denovo --vina /path/to/vina
    proteinfp --uniprot P04637 --md
    proteinfp --uniprot P04637 --antibody
    proteinfp --uniprot P04637 --antibody --epitope-mode ppi
    proteinfp --check-deps
    proteinfp --list-modules
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


# ── Version ────────────────────────────────────────────────────────────────────

try:
    from importlib.metadata import version
    __version__ = version("proteinfp")
except Exception:
    __version__ = "0.1.0-dev"


# ── Main CLI ───────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--uniprot", "-u",
    default=None,
    metavar="ID",
    help="UniProt accession to analyse (e.g. P04637 for TP53).",
)
@click.option(
    "--vina", "-v",
    default=None,
    metavar="PATH",
    help="Path to AutoDock Vina executable. Enables de novo molecule design.",
)
@click.option(
    "--denovo", "-d",
    is_flag=True, default=False,
    help="Run de novo molecular design (requires RDKit + Vina).",
)
@click.option(
    "--md", "-m",
    is_flag=True, default=False,
    help="Run molecular dynamics simulation (requires OpenMM).",
)
@click.option(
    "--receptor", "-r",
    default=None,
    metavar="PATH",
    help="Path to receptor PDBQT for docking. Auto-prepared from PDB if not provided.",
)
@click.option(
    "--therapy", "-t",
    is_flag=True, default=False,
    help="Run therapy decision + all viable design modules automatically.",
)
@click.option(
    "--interactive", "-i",
    is_flag=True, default=False,
    help="Interactive therapy mode — score all modalities, pick one, run with guided parameters.",
)
@click.option(
    "--grn", "-g",
    is_flag=True, default=False,
    help="Run GRN modules (requires scRNA-seq data in disease_config.yaml).",
)
@click.option(
    "--antibody", "-a",
    is_flag=True, default=False,
    help="Run de novo antibody CDR design (Module 16).",
)
@click.option(
    "--epitope-mode", "-e",
    default="auto",
    type=click.Choice(["auto", "active", "ppi", "surface", "allosteric"]),
    show_default=True,
    help="Epitope selection strategy for antibody design.",
)
@click.option(
    "--ab-generations",
    default=50,
    type=int,
    show_default=True,
    metavar="N",
    help="Evolution generations for antibody design.",
)
@click.option(
    "--force", "-f",
    is_flag=True, default=False,
    help="Re-run even if output files already exist.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    metavar="DIR",
    help="Override output directory for reports.",
)
@click.option(
    "--check-deps",
    is_flag=True, default=False,
    help="Show which optional dependencies are installed and exit.",
)
@click.option(
    "--list-modules",
    is_flag=True, default=False,
    help="Show all pipeline modules and their dependency requirements.",
)
@click.version_option(__version__, "--version", "-V")
def main(
    uniprot:        str,
    vina:           str,
    receptor:       str,
    denovo:         bool,
    md:             bool,
    grn:            bool,
    antibody:       bool,
    epitope_mode:   str,
    ab_generations: int,
    therapy:        bool,
    interactive:    bool,
    force:          bool,
    output_dir:     str,
    check_deps:     bool,
    list_modules:   bool,
) -> None:
    """
    ProteinFP — protein function prediction and drug candidate design.

    Predicts active sites, binding pockets, allosteric sites, GO terms,
    EC number, PPI partners, and (optionally) de novo drug candidates
    for any protein with an AlphaFold structure.

    \b
    Quick start:
        proteinfp --uniprot P04637

    \b
    With de novo design (requires RDKit + AutoDock Vina):
        proteinfp --uniprot P04637 --denovo --vina /path/to/vina

    \b
    With molecular dynamics (requires OpenMM):
        proteinfp --uniprot P04637 --md

    \b
    Therapy mode — interactive picker (recommended):
        proteinfp --uniprot P04637 --interactive
        proteinfp --uniprot P04637 --interactive --vina /path/to/vina

    \b
    Therapy mode — run all viable modalities automatically:
        proteinfp --uniprot P04637 --therapy

    \b
    With antibody CDR design (Module 16):
        proteinfp --uniprot P04637 --antibody
        proteinfp --uniprot P04637 --antibody --epitope-mode ppi

    \b
    Check what optional features are available:
        proteinfp --check-deps
    """

    # ── --check-deps: show dependency table and exit ───────────────────────────
    if check_deps:
        from proteinfp.deps import status_report
        click.echo(status_report())
        return

    # ── --list-modules: show module table and exit ─────────────────────────────
    if list_modules:
        _print_module_table()
        return

    # ── Require --uniprot for all other operations ────────────────────────────
    if not uniprot:
        click.echo(
            "\n  Error: --uniprot is required.\n"
            "  Example: proteinfp --uniprot P04637\n"
            "\n  Run `proteinfp --help` for all options.\n"
        )
        sys.exit(1)

    # ── Show dependency status on first run ───────────────────────────────────
    from proteinfp.deps import (
        has_freesasa, has_ml_stack, has_esm2,
        has_openmm, has_rdkit, has_vina,
        status_report,
    )

    # Warn about missing deps that would improve results
    _warn_if_missing(has_freesasa,  "freesasa",  "structure",
                     "SASA/DSSP surface analysis (Module 02)")
    _warn_if_missing(has_esm2,      "fair-esm",  "ml",
                     "ESM-2 protein embeddings (Module 08)")
    _warn_if_missing(has_ml_stack,  "ml stack",  "ml",
                     "ML EC classification (Module 10) — using rule-based fallback")

    # Override output dir if specified
    if output_dir:
        _set_output_dir(output_dir)

    # ── Run the pipeline ───────────────────────────────────────────────────────
    click.echo(f"\n  ProteinFP v{__version__}  —  {uniprot.upper()}")

    from proteinfp.orchestrator import run_pipeline

    result = run_pipeline(
        uniprot_id      = uniprot.strip().upper(),
        vina_path       = vina,
        run_md          = md,
        run_denovo      = denovo,
        run_grn         = grn,
        run_antibody    = antibody,
        epitope_mode    = epitope_mode,
        ab_generations  = ab_generations,
        force           = force,
        verbose         = True,
    )

    # ── Final output ──────────────────────────────────────────────────────────
    if result.success and result.report_path:
        click.echo(f"\n  Report saved to: {result.report_path}")
        _print_report_summary(result.report_path)
    else:
        click.echo(f"\n  Pipeline did not complete successfully.")
        click.echo(f"  Failed modules: {', '.join(result.modules_fail)}")
        sys.exit(1)

    # ── --therapy / --interactive: therapy decision + design modules ─────────────
    if (therapy or interactive) and result.success:
        try:
            from proteinfp.therapy import run_therapy, interactive_design
            if interactive:
                click.echo(f"\n  Launching interactive therapy design...")
                interactive_design(
                    uniprot_id    = uniprot.strip().upper(),
                    vina_path     = vina,
                    receptor_path = receptor or "",
                    force         = force,
                )
            else:
                click.echo(f"\n  Running therapy analysis...")
                run_therapy(
                    uniprot_id    = uniprot.strip().upper(),
                    vina_path     = vina,
                    receptor_path = receptor or "",
                    run_denovo    = denovo,
                    verbose       = True,
                )
        except Exception as e:
            click.echo(f"\n  Therapy analysis failed: {e}")
            import traceback; traceback.print_exc()


# ── Helper: warn about missing optional deps ───────────────────────────────────

def _warn_if_missing(check_fn, pkg_name: str, extra: str, feature: str) -> None:
    if not check_fn():
        click.echo(
            f"  [optional] {pkg_name} not installed — {feature} unavailable\n"
            f"             install: pip install proteinfp[{extra}]"
        )


# ── Helper: print brief report summary ────────────────────────────────────────

def _print_report_summary(report_path: str) -> None:
    """Print the top 10 lines of the text report if it exists."""
    import json
    rp = Path(report_path)
    # Try text version first
    txt = rp.with_suffix(".txt")
    if txt.exists():
        lines = txt.read_text(encoding="utf-8").splitlines()
        click.echo("\n" + "\n".join(lines[:35]))
        if len(lines) > 35:
            click.echo(f"  ... ({len(lines) - 35} more lines in {txt})")
        return

    # Fall back to JSON summary
    if rp.exists():
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
            click.echo(f"\n  Protein  : {data.get('protein_name', '?')}")
            click.echo(f"  Gene     : {data.get('gene_name', '?')}")
            click.echo(f"  Function : {data.get('top_function', '?')}")
            click.echo(f"  Enzyme   : {data.get('is_enzyme', '?')}")
            click.echo(f"  Pockets  : {len(data.get('binding_pockets', []))}")
            click.echo(f"  Confidence: {data.get('overall_confidence', '?')}")
        except Exception:
            pass


# ── Helper: set output directory ──────────────────────────────────────────────

def _set_output_dir(output_dir: str) -> None:
    """Override the reports path in config at runtime."""
    try:
        from utils import config as cfg_module
        cfg_module.cfg._data.setdefault("paths", {})
        cfg_module.cfg._data["paths"]["reports"] = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ── Helper: module table ───────────────────────────────────────────────────────

def _print_module_table() -> None:
    from proteinfp.deps import (
        has_freesasa, has_esm2, has_ml_stack,
        has_openmm, has_rdkit, has_vina,
    )

    rows = [
        ("01", "fetch_structure",    True,            "AlphaFold structure + UniProt"),
        ("02", "physicochemical",    has_freesasa(),  "SASA, charge, hydrophobicity"),
        ("03", "active_sites",       True,            "Catalytic residue prediction"),
        ("04", "binding_pockets",    True,            "Druggable pocket detection"),
        ("05", "allosteric",         True,            "Elastic network allosteric"),
        ("06", "chemical_env",       True,            "Active site chemistry"),
        ("07", "homology",           True,            "Sequence/structure homologs"),
        ("08", "esm2",               has_esm2(),      "Protein language model"),
        ("10", "ec_prediction",      True,            "EC number (ML or rules)"),
        ("11", "foldseek",           True,            "Structural analogs"),
        ("12", "ppi_network",        True,            "Protein interactions"),
        ("13", "consensus",          True,            "Final report"),
        ("14", "molecular_dyn",      has_openmm(),    "MD simulation"),
        ("15", "denovo_design",      has_rdkit() and has_vina(), "De novo small molecules"),
        ("16", "antibody_design",    True,            "De novo antibody CDR design"),
        ("17", "ptm_analysis",       True,            "Post-translational mods"),
        ("18", "adc_design",         True,            "ADC (warhead+linker+CDR co-evolution)"),
        ("19", "cart_design",        True,            "CAR-T construct design"),
        ("20", "protac_design",      True,            "PROTAC / protein degrader design"),
        ("21", "allosteric_drug",    True,            "Allosteric small molecule design"),
    ]

    click.echo(f"\n  {'#':<4} {'Module':<20} {'Available':<12} {'Description'}")
    click.echo(f"  {'─'*4} {'─'*20} {'─'*12} {'─'*35}")
    for num, name, avail, desc in rows:
        status = "✓  yes" if avail else "✗  no"
        click.echo(f"  {num:<4} {name:<20} {status:<12} {desc}")
    click.echo()


if __name__ == "__main__":
    main()