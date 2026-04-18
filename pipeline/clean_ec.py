"""
pipeline/10_clean_ec.py
────────────────────────
Module 10 — Enzyme Commission (EC) number prediction.

Predicts whether a protein is an enzyme and, if so, what EC number class
it belongs to. Uses a combination of:

  1. Sequence-based features (amino acid composition, motifs)
  2. Active site chemistry from Module 03
  3. GO term predictions from Module 09
  4. Homology-based inference from Module 07

EC number hierarchy:
  EC 1.x.x.x — Oxidoreductases  (transfer electrons)
  EC 2.x.x.x — Transferases     (transfer functional groups)
  EC 3.x.x.x — Hydrolases       (cleave bonds with water)
  EC 4.x.x.x — Lyases           (cleave bonds without water)
  EC 5.x.x.x — Isomerases       (rearrange atoms)
  EC 6.x.x.x — Ligases          (join molecules using ATP)
  EC 7.x.x.x — Translocases     (move molecules across membranes)

Non-enzymes are labelled "non-enzyme" with a confidence score.

Usage (standalone):
    python pipeline/10_clean_ec.py --uniprot P04637

Usage (from orchestrator):
    from pipeline.clean_ec import predict_ec_number
    result = predict_ec_number(uniprot_id, sequence, active_result, go_result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np

from utils.config import cfg, get_logger

log = get_logger(__name__)

# ── EC class definitions ───────────────────────────────────────────────────────

EC_CLASSES = {
    "1": ("Oxidoreductase",  "Catalyse oxidation/reduction reactions"),
    "2": ("Transferase",     "Transfer functional groups between molecules"),
    "3": ("Hydrolase",       "Catalyse hydrolysis of chemical bonds"),
    "4": ("Lyase",           "Cleave bonds by means other than hydrolysis"),
    "5": ("Isomerase",       "Catalyse isomerisation changes"),
    "6": ("Ligase",          "Join two molecules with covalent bonds"),
    "7": ("Translocase",     "Catalyse movement of ions or molecules"),
}

# GO terms that indicate enzymatic activity and their EC class
GO_TO_EC: dict[str, str] = {
    "GO:0016491": "1",   # oxidoreductase activity
    "GO:0016614": "1",   # oxidoreductase activity, acting on CH-OH
    "GO:0016616": "1",   # oxidoreductase activity, acting on NADH
    "GO:0016746": "2",   # acyltransferase activity
    "GO:0016747": "2",   # transferase activity, acyl groups
    "GO:0016301": "2",   # kinase activity → transferase
    "GO:0004672": "2",   # protein kinase activity
    "GO:0016787": "3",   # hydrolase activity
    "GO:0004252": "3",   # serine-type endopeptidase
    "GO:0008233": "3",   # peptidase activity
    "GO:0016829": "4",   # lyase activity
    "GO:0016853": "5",   # isomerase activity
    "GO:0016874": "6",   # ligase activity
    "GO:0016879": "6",   # ligase activity, forming C-N bonds
    "GO:0016817": "3",   # hydrolase activity, acting on acid anhydrides
    "GO:0003924": "3",   # GTPase activity → EC 3.6.5 (phosphoric monoester hydrolase)
    "GO:0005525": "3",   # GTP binding → GTPase context
    "GO:0016887": "3",   # ATPase activity → EC 3.6.3
    "GO:0005524": "3",   # ATP binding → ATPase context
    "GO:0042623": "3",   # ATPase activity, coupled → EC 3.6.1
    "GO:0061630": "2",   # ubiquitin protein ligase → EC 2.3.2
    "GO:0004842": "2",   # ubiquitin-protein transferase activity
    "GO:0008237": "3",   # metallopeptidase activity
    "GO:0008241": "3",   # peptidyl-dipeptidase activity
    "GO:0003955": "1",   # NAD(P)H dehydrogenase (quinone) activity → oxidoreductase
    "GO:0010181": "1",   # FMN binding → flavoenzyme / oxidoreductase context
}

# GO terms that map to specific EC sub-numbers (beyond just class digit)
GO_TO_SPECIFIC_EC: dict[str, tuple[str, str]] = {
    "GO:0003924": ("3.6.5", "GTPase"),
    "GO:0016887": ("3.6.1", "ATPase"),
    "GO:0042623": ("3.6.1", "ATPase, coupled"),
    "GO:0061630": ("2.3.2", "Ubiquitin-protein ligase"),
    "GO:0004842": ("2.3.2", "Ubiquitin-protein transferase"),
    "GO:0008237": ("3.4.24", "Metallopeptidase"),
    "GO:0004252": ("3.4.21", "Serine-type endopeptidase"),
}

# Active site motif types that indicate enzymatic function
ENZYMATIC_MOTIFS = {
    "serine_protease_triad":  ("3.4.21", "Serine protease"),
    "cysteine_protease_dyad": ("3.4.22", "Cysteine protease"),
    "zinc_binding_cluster":   ("3.4.24", "Metallopeptidase"),
    "p_loop_walker_a":        ("3.6.5",  "GTPase/ATPase"),
    "ghkl_atpase":            ("3.6.1",  "ATPase/Chaperone"),
    "flavin_binding":         ("1.6.5",  "NADH dehydrogenase"),
}

# Amino acid composition features correlated with EC class
# (from statistical analysis of Swiss-Prot enzymes)
EC_COMPOSITION_SIGNALS: dict[str, dict[str, float]] = {
    "1": {"C": 0.02, "H": 0.03, "F": 0.04},   # oxidoreductases: aromatic residues
    "2": {"K": 0.06, "R": 0.05, "D": 0.05},   # transferases: charged residues
    "3": {"S": 0.08, "H": 0.03, "D": 0.06},   # hydrolases: Ser-His-Asp triad
    "4": {"D": 0.07, "E": 0.06, "K": 0.05},   # lyases: charged
    "5": {"R": 0.06, "K": 0.05, "E": 0.05},   # isomerases: charged
    "6": {"K": 0.06, "R": 0.05, "G": 0.09},   # ligases: ATP-binding Gly-rich
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ECPrediction:
    """Prediction for a single EC class."""
    ec_class:    str       # "1", "2", ... "7", or "non-enzyme"
    ec_name:     str
    ec_full:     str       # e.g. "EC 3.4.21" if subclass known
    score:       float     # 0-1 confidence
    evidence:    list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ECResult:
    """Full EC number prediction output. Output of Module 10."""
    uniprot_id:       str
    sequence_length:  int
    is_enzyme:        bool
    enzyme_confidence: float
    top_prediction:   Optional[ECPrediction]         = None
    all_predictions:  list[ECPrediction]             = field(default_factory=list)
    specific_ec:      str                            = ""  # e.g. "3.4.21.4"
    specific_ec_name: str                            = ""
    non_enzyme_score: float                          = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  EC prediction: {self.uniprot_id}",
            f"  Is enzyme    : {'yes' if self.is_enzyme else 'no'} "
            f"(confidence={self.enzyme_confidence:.2f})",
        ]
        if self.top_prediction:
            lines.append(
                f"  Top EC class : EC {self.top_prediction.ec_class}.x.x.x — "
                f"{self.top_prediction.ec_name} "
                f"(score={self.top_prediction.score:.2f})"
            )
        if self.specific_ec:
            lines.append(f"  Specific EC  : {self.specific_ec} "
                         f"— {self.specific_ec_name}")
        for pred in self.all_predictions[:5]:
            lines.append(
                f"  EC {pred.ec_class} {pred.ec_name[:30]} "
                f"score={pred.score:.2f} "
                f"evidence={','.join(pred.evidence[:2])}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Main function ──────────────────────────────────────────────────────────────

def predict_ec_number(
    uniprot_id:      str,
    sequence:        str,
    active_result:   Optional[dict] = None,
    go_result:       Optional[dict] = None,
    homology_result: Optional[dict] = None,
) -> ECResult:
    """
    Predict EC number class for a protein.

    Args:
        uniprot_id:      UniProt accession
        sequence:        Amino acid sequence
        active_result:   Dict from Module 03 JSON
        go_result:       Dict from Module 09 JSON
        homology_result: Dict from Module 07 JSON

    Returns:
        ECResult with enzyme classification and EC predictions.
    """
    log.info(f"── Module 10: EC number prediction for {uniprot_id} ──")

    ec_scores: dict[str, dict] = {
        cls: {"score": 0.0, "evidence": []}
        for cls in EC_CLASSES
    }
    non_enzyme_score = 0.0
    specific_ec      = ""
    specific_ec_name = ""

    # ── Step 1: GO term evidence ──────────────────────────────────────────────
    log.info("  [1/3] Analysing GO term evidence...")
    if go_result:
        n_go, go_spec_ec, go_spec_name = _add_go_evidence(ec_scores, go_result)
        log.info(f"    {n_go} GO terms processed")
        # GO-derived specific EC (e.g. GTPase → 3.6.5) takes precedence when present
        if go_spec_ec:
            specific_ec      = go_spec_ec
            specific_ec_name = go_spec_name

    # ── Step 2: Active site motif evidence ───────────────────────────────────
    log.info("  [2/3] Analysing active site motif evidence...")
    if active_result:
        spec_ec, spec_name = _add_motif_evidence(ec_scores, active_result)
        # Motif evidence only overrides GO-derived specific EC if GO gave nothing
        if spec_ec and not specific_ec:
            specific_ec      = spec_ec
            specific_ec_name = spec_name

    # ── Step 3: Sequence composition evidence ────────────────────────────────
    log.info("  [3/3] Analysing sequence composition...")
    _add_composition_evidence(ec_scores, sequence)

    # ── Homology-based inference ──────────────────────────────────────────────
    if homology_result:
        _add_homology_ec_evidence(ec_scores, homology_result)

    # ── Determine enzyme vs non-enzyme ────────────────────────────────────────
    max_ec_score = max(v["score"] for v in ec_scores.values())
    enzyme_conf  = float(max_ec_score)

    # Non-enzyme indicators: DNA-binding, structural proteins, transporters
    if go_result:
        all_go = go_result.get("mf_predictions", []) + \
                 go_result.get("bp_predictions", [])
        for pred in all_go:
            go_name = pred.get("go_name", "")
            go_id   = pred.get("go_id", "")
            if "DNA binding" in go_name:
                non_enzyme_score += 0.3
            if "transcription factor" in go_name:
                non_enzyme_score += 0.4
            # Chaperones are ATPases but function primarily as non-enzymes;
            # soften the enzyme classification when chaperone GO terms are present
            if go_id == "GO:0051082" or "unfolded protein binding" in go_name:
                non_enzyme_score += 0.5
            if go_id == "GO:0042623":
                non_enzyme_score += 0.2
    # Strong non-enzyme signals override structural motif evidence
    # Transcription factors with DNA-binding clusters are NOT enzymes
    # even if they have zinc-coordinating residues (structural zinc, not catalytic)
    if non_enzyme_score >= 0.6:
        is_enzyme = False
    else:
        is_enzyme = enzyme_conf > 0.4 and enzyme_conf > non_enzyme_score

    # ── Build ranked predictions ───────────────────────────────────────────────
    predictions = []
    for cls, data in ec_scores.items():
        if data["score"] < 0.2:
            continue
        name, desc = EC_CLASSES[cls]
        predictions.append(ECPrediction(
            ec_class=cls,
            ec_name=name,
            ec_full=f"EC {specific_ec}" if specific_ec and specific_ec.startswith(cls) else f"EC {cls}.x.x.x",
            score=round(data["score"], 3),
            evidence=data["evidence"],
        ))

    predictions.sort(key=lambda p: p.score, reverse=True)
    top = predictions[0] if predictions else None

    result = ECResult(
        uniprot_id=uniprot_id,
        sequence_length=len(sequence),
        is_enzyme=is_enzyme,
        enzyme_confidence=round(enzyme_conf, 3),
        top_prediction=top,
        all_predictions=predictions,
        specific_ec=specific_ec,
        specific_ec_name=specific_ec_name,
        non_enzyme_score=round(non_enzyme_score, 3),
    )

    log.info(result.summary())
    return result


# ── Evidence helpers ───────────────────────────────────────────────────────────

def _add_go_evidence(
    ec_scores: dict,
    go_result: dict,
) -> tuple[int, str, str]:
    """Map GO terms to EC classes. Returns (n_mapped, specific_ec, specific_name)."""
    n = 0
    specific_ec   = ""
    specific_name = ""
    best_score    = 0.0

    all_preds = (
        go_result.get("mf_predictions", []) +
        go_result.get("bp_predictions", [])
    )
    for pred in all_preds:
        go_id = pred.get("go_id", "")
        score = pred.get("score", 0.0)
        if go_id in GO_TO_EC:
            ec_class = GO_TO_EC[go_id]
            ec_scores[ec_class]["score"] = max(
                ec_scores[ec_class]["score"], score * 0.9
            )
            src = "go_prediction"
            if src not in ec_scores[ec_class]["evidence"]:
                ec_scores[ec_class]["evidence"].append(src)
            n += 1
        # Derive specific EC from GO when available and high-confidence
        if go_id in GO_TO_SPECIFIC_EC and score > best_score:
            best_score = score
            specific_ec, specific_name = GO_TO_SPECIFIC_EC[go_id]
    return n, specific_ec, specific_name


def _add_motif_evidence(
    ec_scores: dict,
    active_result: dict,
) -> tuple[str, str]:
    """Map catalytic motifs to EC classes."""
    specific_ec   = ""
    specific_name = ""

    for motif in active_result.get("catalytic_motifs", []):
        mtype     = motif.get("motif_type", "")
        conf      = motif.get("confidence", "LOW")
        zinc_type = motif.get("zinc_type", "")
        score     = 0.85 if conf == "HIGH" else 0.65

        # Structural zinc (RING domains, zinc fingers — Cys4/Cys3His1 pattern)
        # must NOT drive EC prediction toward metallopeptidase (EC 3.4.24).
        # Only catalytic zinc (His2Glu pattern) is indicative of enzymatic activity.
        if mtype == "zinc_binding_cluster":
            if not zinc_type:
                # Infer from residue_letters when zinc_type is not stored
                letters = motif.get("residue_letters", [])
                if letters:
                    cys_count = letters.count("C")
                    his_count = letters.count("H")
                    glu_count = letters.count("E")
                    if cys_count >= 3:
                        zinc_type = "structural"
                    elif his_count >= 2 and glu_count >= 1 and cys_count == 0:
                        zinc_type = "catalytic"
                    else:
                        zinc_type = "structural"
                # If residue_letters missing, assume catalytic (backward compat)
                # so old-format motif data still contributes to EC predictions
            if zinc_type == "structural":
                continue  # structural zinc does not indicate metallopeptidase

        if mtype in ENZYMATIC_MOTIFS:
            ec_sub, ec_name = ENZYMATIC_MOTIFS[mtype]
            ec_class = ec_sub.split(".")[0]
            ec_scores[ec_class]["score"] = max(
                ec_scores[ec_class]["score"], score
            )
            src = "structural_motif"
            if src not in ec_scores[ec_class]["evidence"]:
                ec_scores[ec_class]["evidence"].append(src)
            if score >= 0.80:
                specific_ec   = ec_sub
                specific_name = ec_name

    return specific_ec, specific_name


def _add_composition_evidence(ec_scores: dict, sequence: str) -> None:
    """Add weak evidence from amino acid composition."""
    L = len(sequence)
    if L == 0:
        return

    comp = {}
    for aa in sequence:
        comp[aa] = comp.get(aa, 0) + 1
    for aa in comp:
        comp[aa] /= L

    for ec_class, signals in EC_COMPOSITION_SIGNALS.items():
        score = 0.0
        for aa, expected_frac in signals.items():
            actual = comp.get(aa, 0.0)
            if actual >= expected_frac * 0.8:
                score += 0.08
        if score > 0:
            ec_scores[ec_class]["score"] = max(
                ec_scores[ec_class]["score"], score
            )
            src = "sequence_composition"
            if src not in ec_scores[ec_class]["evidence"]:
                ec_scores[ec_class]["evidence"].append(src)


def _add_homology_ec_evidence(ec_scores: dict, homology_result: dict) -> None:
    """Infer EC class from homolog GO terms."""
    for go_id in homology_result.get("all_go_terms", []):
        if go_id in GO_TO_EC:
            ec_class = GO_TO_EC[go_id]
            ec_scores[ec_class]["score"] = max(
                ec_scores[ec_class]["score"], 0.70
            )
            src = "homology_inference"
            if src not in ec_scores[ec_class]["evidence"]:
                ec_scores[ec_class]["evidence"].append(src)


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
def main(uniprot: str) -> None:
    """
    Module 10 — EC number prediction.

    Example:
        python pipeline/clean_ec.py --uniprot P04637
    """
    uniprot   = uniprot.strip().upper()
    inter_dir = Path(cfg.paths["intermediate"])
    pdb_path  = Path(cfg.paths["structures"]) / f"{uniprot}.pdb"
    out_path  = inter_dir / f"{uniprot}_ec_prediction.json"

    if not pdb_path.exists():
        log.error(f".pdb not found — run Module 01 first")
        raise SystemExit(1)

    from utils.pdb_parser import parse_pdb
    parsed   = parse_pdb(pdb_path, uniprot)
    sequence = parsed.sequence

    def _load(fname):
        p = inter_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    active_result   = _load(f"{uniprot}_active_sites.json")
    go_result       = _load(f"{uniprot}_go_predictions.json")
    homology_result = _load(f"{uniprot}_homology.json")

    result = predict_ec_number(
        uniprot, sequence, active_result, go_result, homology_result
    )
    result.to_json(out_path)
    click.echo(f"\nDone. Results written to:\n  {out_path}")


if __name__ == "__main__":
    main()