"""
validation/run_validation.py
─────────────────────────────
Validation study for ProteinFP pipeline.

Tests predictions against 20 proteins with well-characterised functions,
comparing predicted GO terms, active sites, and EC classifications
against experimentally confirmed ground truth from Swiss-Prot.

Proteins chosen to cover:
  - Different enzyme classes (kinases, proteases, oxidoreductases)
  - Non-enzymes (transcription factors, structural proteins)
  - Different organism sources
  - Different confidence levels in AlphaFold DB

Usage:
    python validation/run_validation.py
    python validation/run_validation.py --quick   (5 proteins only)
    python validation/run_validation.py --protein P04637
"""

from __future__ import annotations

import json
import time
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

import click
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ── Validation set ─────────────────────────────────────────────────────────────
# Ground truth from Swiss-Prot experimental annotations.
# Each entry has: UniProt ID, gene, known function, known GO terms,
# known active site residues (where published), is_enzyme, ec_number.

VALIDATION_SET = [
    {
        "uniprot_id":  "P04637",
        "gene":        "TP53",
        "description": "Tumour suppressor transcription factor",
        "organism":    "Homo sapiens",
        "is_enzyme":   False,
        "ec_number":   "",
        "known_go_mf": ["GO:0003677", "GO:0003700", "GO:0046872"],
        "known_go_bp": ["GO:0006915", "GO:0006974", "GO:0045944"],
        "known_go_cc": ["GO:0005634", "GO:0043234"],
        "known_active_residues": [176, 179, 248, 273],
        "known_partners":        ["MDM2", "MDM4", "ATM", "CHEK2", "EP300"],
        "category":    "transcription_factor",
    },
    {
        "uniprot_id":  "P00533",
        "gene":        "EGFR",
        "description": "Epidermal growth factor receptor kinase",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "2.7.10.1",
        "known_go_mf": ["GO:0004672", "GO:0004714", "GO:0005006"],
        "known_go_bp": ["GO:0007173", "GO:0008283", "GO:0018108"],
        "known_go_cc": ["GO:0005887", "GO:0016020"],
        "known_active_residues": [837, 855],
        "known_partners":        ["GRB2", "SOS1", "PIK3R1", "SHC1"],
        "category":    "kinase",
    },
    {
        "uniprot_id":  "P00441",
        "gene":        "SOD1",
        "description": "Superoxide dismutase (Cu/Zn)",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "1.15.1.1",
        "known_go_mf": ["GO:0004784", "GO:0005507", "GO:0008270"],
        "known_go_bp": ["GO:0019430", "GO:0006801"],
        "known_go_cc": ["GO:0005737", "GO:0005634"],
        "known_active_residues": [44, 46, 118],
        "known_partners":        ["CCS", "TNFRSF1A"],
        "category":    "oxidoreductase",
    },
    {
        "uniprot_id":  "P07900",
        "gene":        "HSP90AA1",
        "description": "Heat shock protein 90 alpha (chaperone)",
        "organism":    "Homo sapiens",
        "is_enzyme":   False,
        "ec_number":   "",
        "known_go_mf": ["GO:0005524", "GO:0051082", "GO:0042623"],
        "known_go_bp": ["GO:0006457", "GO:0051085"],
        "known_go_cc": ["GO:0005737", "GO:0005634"],
        "known_active_residues": [35, 83, 183],
        "known_partners":        ["CDC37", "AHA1", "HOP", "CHIP"],
        "category":    "chaperone",
    },
    {
        "uniprot_id":  "P06213",
        "gene":        "INSR",
        "description": "Insulin receptor tyrosine kinase",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "2.7.10.1",
        "known_go_mf": ["GO:0004672", "GO:0004713", "GO:0005009"],
        "known_go_bp": ["GO:0008286", "GO:0046628"],
        "known_go_cc": ["GO:0005887", "GO:0005615"],
        "known_active_residues": [1131, 1135, 1136],
        "known_partners":        ["IRS1", "IRS2", "GRB2", "SHC1"],
        "category":    "kinase",
    },
    {
        "uniprot_id":  "P38398",
        "gene":        "BRCA1",
        "description": "Breast cancer type 1 susceptibility / DNA repair",
        "organism":    "Homo sapiens",
        "is_enzyme":   False,
        "ec_number":   "",
        "known_go_mf": ["GO:0003684", "GO:0003723", "GO:0004842"],
        "known_go_bp": ["GO:0006281", "GO:0007131", "GO:0045739"],
        "known_go_cc": ["GO:0005634", "GO:0010369"],
        "known_active_residues": [1763, 1836],
        "known_partners":        ["BARD1", "RAD51", "TP53", "ATM"],
        "category":    "dna_repair",
    },
    {
        "uniprot_id":  "P16083",
        "gene":        "NQO2",
        "description": "Ribosyldihydronicotinamide dehydrogenase (quinone)",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "1.10.99.2",
        "known_go_mf": ["GO:0003955", "GO:0010181"],
        "known_go_bp": ["GO:0055114", "GO:0042493"],
        "known_go_cc": ["GO:0005737"],
        "known_active_residues": [103, 128],
        "known_partners":        ["AHR"],
        "category":    "oxidoreductase",
    },
    {
        "uniprot_id":  "P00734",
        "gene":        "F2",
        "description": "Prothrombin / Thrombin (serine protease)",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "3.4.21.5",
        "known_go_mf": ["GO:0004252", "GO:0005172"],
        "known_go_bp": ["GO:0007596", "GO:0030193"],
        "known_go_cc": ["GO:0005576", "GO:0072562"],
        "known_active_residues": [363, 419, 521],
        "known_partners":        ["F5", "F8", "THBD"],
        "category":    "serine_protease",
    },
    {
        "uniprot_id":  "P68871",
        "gene":        "HBB",
        "description": "Haemoglobin subunit beta (oxygen transport)",
        "organism":    "Homo sapiens",
        "is_enzyme":   False,
        "ec_number":   "",
        "known_go_mf": ["GO:0020037", "GO:0019825"],
        "known_go_bp": ["GO:0015671", "GO:0019430"],
        "known_go_cc": ["GO:0005833", "GO:0031838"],
        "known_active_residues": [92],
        "known_partners":        ["HBA1", "HBA2"],
        "category":    "oxygen_transport",
    },
    {
        "uniprot_id":  "P00918",
        "gene":        "CA2",
        "description": "Carbonic anhydrase 2 (zinc metalloenzyme)",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "4.2.1.1",
        "known_go_mf": ["GO:0004089", "GO:0008270"],
        "known_go_bp": ["GO:0015701", "GO:0001659"],
        "known_go_cc": ["GO:0005737", "GO:0005829"],
        "known_active_residues": [94, 96, 119],
        "known_partners":        ["SLC4A1", "CA1"],
        "category":    "lyase",
    },
    {
        "uniprot_id":  "P01116",
        "gene":        "KRAS",
        "description": "GTPase KRAS — membrane-anchored signal transducer",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "3.6.5.2",
        "known_go_mf": ["GO:0005525", "GO:0003924", "GO:0019003"],
        "known_go_bp": ["GO:0007165", "GO:0008283"],
        "known_go_cc": ["GO:0016020", "GO:0005737"],
        "known_active_residues": [10, 12, 13, 16],
        "known_partners":        ["BRAF", "RAF1", "SOS1", "RALGDS"],
        "category":    "gtpase",
    },
    {
        "uniprot_id":  "Q00987",
        "gene":        "MDM2",
        "description": "E3 ubiquitin-protein ligase Mdm2 — p53 regulator",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "2.3.2.27",
        "known_go_mf": ["GO:0061630", "GO:0042802"],
        "known_go_bp": ["GO:0043066", "GO:0051726"],
        "known_go_cc": ["GO:0005634", "GO:0005737"],
        "known_active_residues": [305, 308, 319, 322],
        "known_partners":        ["TP53", "MDM4", "USP7", "RB1"],
        "category":    "ubiquitin_ligase",
    },
    {
        "uniprot_id":  "Q9BYF1",
        "gene":        "ACE2",
        "description": "Angiotensin-converting enzyme 2 — metallopeptidase/receptor",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "3.4.17.23",
        "known_go_mf": ["GO:0008237", "GO:0008241", "GO:0046872"],
        "known_go_bp": ["GO:0006508", "GO:0010819"],
        "known_go_cc": ["GO:0016020", "GO:0005615"],
        "known_active_residues": [374, 378, 402],
        "known_partners":        ["TMPRSS2", "AGT", "SLC6A19"],
        "category":    "metallopeptidase",
    },
    {
        "uniprot_id":  "O15151",
        "gene":        "MDM4",
        "description": "MDM4 — p53 regulator, MDM2 paralog",
        "organism":    "Homo sapiens",
        "is_enzyme":   False,
        "ec_number":   "",
        "known_go_mf": ["GO:0061630", "GO:0008270", "GO:0004842"],
        "known_go_bp": ["GO:0043066", "GO:0051726", "GO:0006915"],
        "known_go_cc": ["GO:0005634", "GO:0005737"],
        "known_active_residues": [460, 463, 466, 469],
        "known_partners":        ["MDM2", "TP53", "USP7"],
        "category":    "ubiquitin_ligase",
    },
    {
        "uniprot_id":  "P42574",
        "gene":        "CASP3",
        "description": "Caspase-3 — executioner protease of apoptosis",
        "organism":    "Homo sapiens",
        "is_enzyme":   True,
        "ec_number":   "3.4.22.56",
        "known_go_mf": ["GO:0004197", "GO:0008234", "GO:0008233"],
        "known_go_bp": ["GO:0006915", "GO:0043525"],
        "known_go_cc": ["GO:0005737", "GO:0005829"],
        "known_active_residues": [163, 184],
        "known_partners":        ["CASP8", "CASP9", "XIAP", "PARP1"],
        "category":    "cysteine_protease",
    },
]

QUICK_SET = VALIDATION_SET[:5]


# ── Scoring ────────────────────────────────────────────────────────────────────

@dataclass
class ProteinScore:
    uniprot_id:       str
    gene:             str
    category:         str
    go_mf_recall:     float = 0.0   # fraction of known MF GO terms found
    go_bp_recall:     float = 0.0
    go_cc_recall:     float = 0.0
    go_mean_recall:   float = 0.0
    active_site_recall: float = 0.0  # fraction of known active residues found
    ec_correct:       bool  = False  # EC class correct (first digit)
    enzyme_correct:   bool  = False  # is_enzyme flag correct
    ppi_recall:       float = 0.0   # fraction of known partners found
    overall_score:    float = 0.0   # 0-100
    report_available: bool  = False
    notes:            str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    run_date:       str
    n_proteins:     int
    scores:         list[ProteinScore] = field(default_factory=list)
    mean_go_recall: float = 0.0
    mean_as_recall: float = 0.0
    enzyme_accuracy: float = 0.0
    mean_ppi_recall: float = 0.0
    overall_accuracy: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  ProteinFP Validation Report",
            f"  Date: {self.run_date}",
            f"  Proteins tested: {self.n_proteins}",
            "=" * 70,
            "",
            f"  Mean GO term recall    : {self.mean_go_recall*100:.1f}%",
            f"  Mean active site recall: {self.mean_as_recall*100:.1f}%",
            f"  Enzyme classification  : {self.enzyme_accuracy*100:.1f}%",
            f"  PPI partner recall     : {self.mean_ppi_recall*100:.1f}%",
            f"  Overall accuracy score : {self.overall_accuracy:.1f}/100",
            "",
            "─" * 70,
            "  Per-protein breakdown:",
            "─" * 70,
        ]
        for s in self.scores:
            status = "OK" if s.report_available else "MISSING"
            lines.append(
                f"  {s.uniprot_id} {s.gene:8s} [{s.category:20s}] "
                f"GO={s.go_mean_recall*100:.0f}% "
                f"AS={s.active_site_recall*100:.0f}% "
                f"Enz={'Y' if s.enzyme_correct else 'N'} "
                f"PPI={s.ppi_recall*100:.0f}% "
                f"[{status}]"
            )
        lines += [
            "",
            "─" * 70,
            "  Category breakdown:",
            "─" * 70,
        ]
        categories = {}
        for s in self.scores:
            if s.report_available:
                categories.setdefault(s.category, []).append(s.overall_score)
        for cat, scores_list in sorted(categories.items()):
            mean = sum(scores_list) / len(scores_list)
            lines.append(f"  {cat:25s}: {mean:.1f}/100 ({len(scores_list)} proteins)")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Main validation runner ─────────────────────────────────────────────────────

def run_validation(
    proteins:    list[dict],
    skip_modules: bool = False,
) -> ValidationReport:
    """
    Run full validation suite.

    For each protein:
      1. Run all 13 pipeline modules (unless reports already exist)
      2. Load the consensus report
      3. Score predictions against ground truth
      4. Aggregate metrics

    Args:
        proteins:     list of validation set entries
        skip_modules: if True, only score existing reports (no re-running)
    """
    log.info(f"Starting validation on {len(proteins)} proteins...")
    scores = []

    for i, protein in enumerate(proteins):
        uid  = protein["uniprot_id"]
        gene = protein["gene"]
        log.info(f"\n[{i+1}/{len(proteins)}] {uid} — {gene}")

        # Check if report already exists
        report_path = Path(cfg.paths["reports"]) / f"{uid}_report.json"

        if not report_path.exists() and not skip_modules:
            log.info(f"  Running pipeline for {uid}...")
            _run_pipeline(uid)
        elif not report_path.exists():
            log.warning(f"  Report missing for {uid} — skipping")
            scores.append(ProteinScore(
                uniprot_id=uid, gene=gene,
                category=protein["category"],
                notes="report not found",
            ))
            continue

        # Load and score
        try:
            with open(report_path) as f:
                report = json.load(f)
            score = _score_protein(report, protein)
            scores.append(score)
            log.info(
                f"  GO={score.go_mean_recall*100:.0f}%  "
                f"AS={score.active_site_recall*100:.0f}%  "
                f"Enz={'Y' if score.enzyme_correct else 'N'}  "
                f"PPI={score.ppi_recall*100:.0f}%  "
                f"Overall={score.overall_score:.1f}/100"
            )
        except Exception as e:
            log.error(f"  Scoring failed for {uid}: {e}")
            scores.append(ProteinScore(
                uniprot_id=uid, gene=gene,
                category=protein["category"],
                notes=f"error: {e}",
            ))

    # Aggregate
    available = [s for s in scores if s.report_available]
    n = max(len(available), 1)

    vr = ValidationReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_proteins=len(proteins),
        scores=scores,
        mean_go_recall=sum(s.go_mean_recall for s in available) / n,
        mean_as_recall=sum(s.active_site_recall for s in available) / n,
        enzyme_accuracy=sum(1 for s in available if s.enzyme_correct) / n,
        mean_ppi_recall=sum(s.ppi_recall for s in available) / n,
        overall_accuracy=sum(s.overall_score for s in available) / n,
    )

    return vr


def _run_pipeline(uniprot_id: str) -> None:
    """Run all 13 pipeline modules for a protein."""
    import subprocess

    modules = [
        ("fetch_structure",   ["--uniprot", uniprot_id]),
        ("physicochemical",   ["--uniprot", uniprot_id]),
        ("active_sites",      ["--uniprot", uniprot_id]),
        ("binding_pockets",   ["--uniprot", uniprot_id]),
        ("allosteric",        ["--uniprot", uniprot_id]),
        ("chemical_env",      ["--uniprot", uniprot_id]),
        ("homology",          ["--uniprot", uniprot_id]),
        ("esm2_embeddings",   ["--uniprot", uniprot_id]),
        ("deepfri_go",        ["--uniprot", uniprot_id]),
        ("clean_ec",          ["--uniprot", uniprot_id]),
        ("foldseek",          ["--uniprot", uniprot_id]),
        ("ppi_network",       ["--uniprot", uniprot_id]),
        ("consensus",         ["--uniprot", uniprot_id]),
    ]

    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }
    import os
    env.update(os.environ)

    for module_name, args in modules:
        script = Path(__file__).parent.parent / "pipeline" / f"{module_name}.py"
        cmd = [sys.executable, str(script)] + args
        log.info(f"    Running {module_name}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=600, env=env,
            )
            if result.returncode != 0:
                log.warning(
                    f"    {module_name} exited {result.returncode}: "
                    f"{result.stderr[-200:]}"
                )
        except subprocess.TimeoutExpired:
            log.warning(f"    {module_name} timed out")
        except Exception as e:
            log.warning(f"    {module_name} failed: {e}")


# ── Scoring functions ──────────────────────────────────────────────────────────

def _score_protein(report: dict, ground_truth: dict) -> ProteinScore:
    """Score a single protein report against ground truth."""
    uid  = ground_truth["uniprot_id"]
    gene = ground_truth["gene"]

    score = ProteinScore(
        uniprot_id=uid,
        gene=gene,
        category=ground_truth["category"],
        report_available=True,
    )

    # ── GO term recall ────────────────────────────────────────────────────────
    pred_mf = {t["go_id"] for t in report.get("go_terms_mf", [])}
    pred_bp = {t["go_id"] for t in report.get("go_terms_bp", [])}
    pred_cc = {t["go_id"] for t in report.get("go_terms_cc", [])}

    known_mf = set(ground_truth.get("known_go_mf", []))
    known_bp = set(ground_truth.get("known_go_bp", []))
    known_cc = set(ground_truth.get("known_go_cc", []))

    score.go_mf_recall = _recall(pred_mf, known_mf)
    score.go_bp_recall = _recall(pred_bp, known_bp)
    score.go_cc_recall = _recall(pred_cc, known_cc)
    score.go_mean_recall = (
        score.go_mf_recall + score.go_bp_recall + score.go_cc_recall
    ) / 3

    # ── Active site recall ────────────────────────────────────────────────────
    pred_active: set[int] = set()
    for r in report.get("active_sites", []):
        pred_active.add(r["residue_number"])

    # Also load from intermediate JSON for full coverage
    inter_path = Path(cfg.paths["intermediate"]) / f"{uid}_active_sites.json"
    if inter_path.exists():
        full_active = json.loads(inter_path.read_text())
        for r in full_active.get("active_residues", []):
            if r.get("confidence") in ("HIGH", "MEDIUM"):
                pred_active.add(r["residue_number"])
    known_active = set(ground_truth.get("known_active_residues", []))

    # Allow ±3 residue tolerance for active site matching
    if known_active:
        matched = sum(
            1 for ka in known_active
            if any(abs(pa - ka) <= 3 for pa in pred_active)
        )
        score.active_site_recall = matched / len(known_active)
    else:
        score.active_site_recall = 1.0

    # ── Enzyme classification ─────────────────────────────────────────────────
    pred_enzyme  = report.get("is_enzyme", False)
    known_enzyme = ground_truth.get("is_enzyme", False)
    score.enzyme_correct = (pred_enzyme == known_enzyme)

    # EC class match (first digit only)
    pred_ec  = str(report.get("ec_number", "")).strip()
    known_ec = str(ground_truth.get("ec_number", "")).strip()
    if known_enzyme and known_ec and pred_ec:
        score.ec_correct = pred_ec[0] == known_ec[0]
    elif not known_enzyme:
        score.ec_correct = True   # non-enzyme correctly classified

    # ── PPI recall ────────────────────────────────────────────────────────────
    pred_partners = {
        p["partner_name"].upper()
        for p in report.get("ppi_partners", [])
    }
    known_partners = {p.upper() for p in ground_truth.get("known_partners", [])}
    score.ppi_recall = _recall(pred_partners, known_partners)

    # ── Overall score (0-100) ─────────────────────────────────────────────────
    score.overall_score = round(
        score.go_mean_recall       * 35 +   # GO terms: 35 points
        score.active_site_recall   * 25 +   # Active sites: 25 points
        (1.0 if score.enzyme_correct else 0) * 20 +  # Enzyme: 20 points
        score.ppi_recall           * 20,    # PPI: 20 points
        1
    )

    return score


def _recall(predicted: set, known: set) -> float:
    """Fraction of known items correctly predicted."""
    if not known:
        return 1.0
    return len(predicted & known) / len(known)


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--quick", is_flag=True, default=False,
              help="Run only 5 proteins (faster)")
@click.option("--protein", "-p", default=None,
              help="Validate a single protein by UniProt ID")
@click.option("--score-only", is_flag=True, default=False,
              help="Score existing reports without re-running pipeline")
def main(quick: bool, protein: str, score_only: bool) -> None:
    """
    ProteinFP validation study.

    Runs the full pipeline on benchmark proteins and scores
    predictions against experimental ground truth.

    Examples:
        python validation/run_validation.py              # all 10 proteins
        python validation/run_validation.py --quick      # 5 proteins
        python validation/run_validation.py --score-only # score existing reports
        python validation/run_validation.py --protein P04637
    """
    # Create validation output dir
    val_dir = Path(cfg.paths["reports"]) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    if protein:
        # Find protein in validation set
        proteins = [p for p in VALIDATION_SET
                    if p["uniprot_id"] == protein.upper()]
        if not proteins:
            # Run with minimal ground truth
            proteins = [{
                "uniprot_id":  protein.upper(),
                "gene":        protein.upper(),
                "description": "Custom protein",
                "organism":    "unknown",
                "is_enzyme":   False,
                "ec_number":   "",
                "known_go_mf": [],
                "known_go_bp": [],
                "known_go_cc": [],
                "known_active_residues": [],
                "known_partners": [],
                "category": "custom",
            }]
    elif quick:
        proteins = QUICK_SET
        log.info("Running quick validation (5 proteins)...")
    else:
        proteins = VALIDATION_SET
        log.info("Running full validation (10 proteins)...")

    log.info("Note: Each protein takes ~10-15 min (mostly BLAST).")
    log.info("      Use --score-only if reports already exist.")

    vr = run_validation(proteins, skip_modules=score_only)

    # Print summary
    click.echo("\n" + vr.summary())

    # Save
    out_json = val_dir / "validation_report.json"
    out_txt  = val_dir / "validation_report.txt"
    vr.to_json(out_json)
    out_txt.write_text(vr.summary(), encoding="utf-8")

    click.echo(f"\nValidation report saved to:")
    click.echo(f"  {out_json}")
    click.echo(f"  {out_txt}")


if __name__ == "__main__":
    main()