"""
pipeline/ml_ec_predict.py
──────────────────────────
Module 10-ML: Production inference wrapper.

Drop-in replacement for Module 10 (clean_ec.py).
Accepts the same inputs, returns a compatible ECResult-style dict,
but uses the trained ML ensemble instead of hand-crafted heuristics.

Falls back to the legacy rule-based system if no model is found,
so the pipeline never breaks during the transition period.

Usage (standalone):
    python pipeline/ml_ec_predict.py --uniprot P04637

Usage (from orchestrator / consensus.py):
    from pipeline.ml_ec_predict import predict_ec_ml
    result = predict_ec_ml(uniprot_id, sequence, active_result, go_result, ...)
    # result is a dict compatible with the legacy ECResult.to_dict() format

Usage (as complete module 10 replacement):
    # In your orchestrator, replace:
    #   from pipeline.clean_ec import predict_ec_number
    # with:
    #   from pipeline.ml_ec_predict import predict_ec_ml as predict_ec_number
"""

from __future__ import annotations

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import numpy as np

from pipeline.ml_ec_classifier import ECClassifierEnsemble
from pipeline.ml_ec_features import MLECFeatures

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@click.command()
@click.option("--protein-data", type=str)
def predict(protein_data):
    pass

# ── Default model location ─────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = Path("models/ec_ensemble")


# ── Legacy-compatible result dataclass ────────────────────────────────────────

@dataclass
class ECPrediction:
    ec_class:    str
    ec_name:     str
    ec_full:     str
    score:       float
    evidence:    list[str] = field(default_factory=list)
    probability: float = 0.0    # NEW: calibrated ML probability

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ECResult:
    """
    Backward-compatible result — identical fields to legacy Module 10,
    plus new ml_* fields for extended analysis.
    """
    uniprot_id:         str
    sequence_length:    int
    is_enzyme:          bool
    enzyme_confidence:  float
    top_prediction:     Optional[ECPrediction]       = None
    all_predictions:    list[ECPrediction]           = field(default_factory=list)
    specific_ec:        str                          = ""
    specific_ec_name:   str                          = ""
    non_enzyme_score:   float                        = 0.0
    # Extended ML fields
    ml_used:            bool                         = False
    ml_model_version:   str                          = ""
    ml_inference_ms:    float                        = 0.0
    ml_feature_dim:     int                          = 0
    ml_top2_class:      str                          = ""
    ml_top2_prob:       float                        = 0.0
    ml_entropy:         float                        = 0.0   # prediction uncertainty

    def summary(self) -> str:
        backend = "ML-ensemble" if self.ml_used else "rule-based"
        lines = [
            f"\n{'─'*60}",
            f"  EC prediction [{backend}]: {self.uniprot_id}",
            f"  Is enzyme    : {'yes' if self.is_enzyme else 'no'} "
            f"(confidence={self.enzyme_confidence:.3f})",
        ]
        if self.top_prediction:
            lines.append(
                f"  Top EC class : EC {self.top_prediction.ec_class}.x.x.x — "
                f"{self.top_prediction.ec_name} "
                f"(p={self.top_prediction.probability:.3f})"
            )
        if self.specific_ec:
            lines.append(f"  Specific EC  : {self.specific_ec} — {self.specific_ec_name}")
        if self.ml_used:
            lines.append(f"  Uncertainty  : entropy={self.ml_entropy:.3f} | "
                         f"2nd={self.ml_top2_class} (p={self.ml_top2_prob:.3f})")
            lines.append(f"  Inference    : {self.ml_inference_ms:.1f}ms | "
                         f"dim={self.ml_feature_dim}")
        for pred in self.all_predictions[:5]:
            lines.append(
                f"  EC {pred.ec_class:12s} {pred.ec_name[:25]:25s} "
                f"p={pred.probability:.3f}"
            )
        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── EC class names ─────────────────────────────────────────────────────────────

_EC_CLASSES = {
    "1": ("Oxidoreductase",  "Catalyse oxidation/reduction reactions"),
    "2": ("Transferase",     "Transfer functional groups between molecules"),
    "3": ("Hydrolase",       "Catalyse hydrolysis of chemical bonds"),
    "4": ("Lyase",           "Cleave bonds by means other than hydrolysis"),
    "5": ("Isomerase",       "Catalyse isomerisation changes"),
    "6": ("Ligase",          "Join two molecules with covalent bonds"),
    "7": ("Translocase",     "Catalyse movement of ions or molecules"),
}


# ── Model loader (singleton cache) ────────────────────────────────────────────

_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_dir: Path):
    """Load (and cache) the ECClassifierEnsemble."""
    key = str(model_dir.resolve())
    if key not in _MODEL_CACHE:
        from pipeline.ml_ec_classifier import ECClassifierEnsemble
        try:
            clf = ECClassifierEnsemble.load(model_dir)
            _MODEL_CACHE[key] = clf
            log.info(f"  ML ensemble loaded from {model_dir}/")
        except Exception as e:
            log.warning(f"  Could not load ML model from {model_dir}: {e}")
            _MODEL_CACHE[key] = None
    return _MODEL_CACHE[key]


# ── Main predict function ──────────────────────────────────────────────────────

def predict_ec_ml(
    uniprot_id:      str,
    sequence:        str,
    active_result:   Optional[dict] = None,
    go_result:       Optional[dict] = None,
    homology_result: Optional[dict] = None,
    esm2_result:     Optional[dict] = None,
    pdb_result:      Optional[dict] = None,
    pocket_result:   Optional[dict] = None,
    enm_result:      Optional[dict] = None,
    physico_result:  Optional[dict] = None,
    model_dir:       Path           = DEFAULT_MODEL_DIR,
) -> ECResult:
    """
    Predict EC number class using the ML ensemble.

    Falls back to legacy rule-based prediction if model not found.
    Returns an ECResult with both legacy fields and new ML-specific fields.
    """
    log.info(f"── Module 10-ML: EC number prediction for {uniprot_id} ──")

    clf = _load_model(model_dir)

    # ── ML path ───────────────────────────────────────────────────────────────
    if clf is not None:
        try:
            ml_result = clf.predict(
                sequence        = sequence,
                esm2_result     = esm2_result,
                pdb_result      = pdb_result,
                active_result   = active_result,
                pocket_result   = pocket_result,
                enm_result      = enm_result,
                physico_result  = physico_result,
                go_result       = go_result,
                homology_result = homology_result,
                uniprot_id      = uniprot_id,
            )

            # Convert MLECResult → ECResult (legacy-compatible)
            probas = np.array([p.probability for p in ml_result.ec_predictions])
            entropy = float(-np.sum(probas * np.log(probas + 1e-10)))

            all_preds = []
            for pred in ml_result.ec_predictions:
                name, _ = _EC_CLASSES.get(pred.ec_class, ("Unknown", ""))
                all_preds.append(ECPrediction(
                    ec_class    = pred.ec_class,
                    ec_name     = name,
                    ec_full     = (
                        f"EC {ml_result.specific_ec}"
                        if ml_result.specific_ec and ml_result.specific_ec.startswith(pred.ec_class)
                        else f"EC {pred.ec_class}.x.x.x"
                    ),
                    score       = pred.probability,
                    probability = pred.probability,
                    evidence    = ["ml_ensemble"],
                ))

            top = all_preds[0] if all_preds else None
            top2 = all_preds[1] if len(all_preds) > 1 else None

            result = ECResult(
                uniprot_id        = uniprot_id,
                sequence_length   = len(sequence),
                is_enzyme         = ml_result.is_enzyme,
                enzyme_confidence = round(ml_result.enzyme_probability, 4),
                top_prediction    = top,
                all_predictions   = all_preds,
                specific_ec       = ml_result.specific_ec,
                specific_ec_name  = ml_result.specific_ec_name,
                non_enzyme_score  = round(1.0 - ml_result.enzyme_probability, 4),
                ml_used           = True,
                ml_model_version  = ml_result.model_version,
                ml_inference_ms   = ml_result.inference_time_ms,
                ml_feature_dim    = ml_result.feature_dim,
                ml_top2_class     = top2.ec_class if top2 else "",
                ml_top2_prob      = round(top2.probability, 4) if top2 else 0.0,
                ml_entropy        = round(entropy, 4),
            )
            log.info(result.summary())
            return result

        except Exception as e:
            log.error(f"  ML prediction failed: {e}. Falling back to rule-based.")

    # ── Fallback: rule-based (legacy Module 10) ────────────────────────────────
    log.warning("  Using legacy rule-based EC prediction (ML model not available)")
    from pipeline.clean_ec import predict_ec_number
    legacy = predict_ec_number(
        uniprot_id      = uniprot_id,
        sequence        = sequence,
        active_result   = active_result,
        go_result       = go_result,
        homology_result = homology_result,
    )

    # Wrap legacy result in our richer ECResult (with ml_used=False)
    all_preds = []
    for pred in (legacy.all_predictions or []):
        all_preds.append(ECPrediction(
            ec_class    = pred.ec_class,
            ec_name     = pred.ec_name,
            ec_full     = pred.ec_full,
            score       = pred.score,
            probability = pred.score,   # use heuristic score as pseudo-probability
            evidence    = pred.evidence,
        ))

    top_legacy = None
    if legacy.top_prediction:
        top_legacy = ECPrediction(
            ec_class    = legacy.top_prediction.ec_class,
            ec_name     = legacy.top_prediction.ec_name,
            ec_full     = legacy.top_prediction.ec_full,
            score       = legacy.top_prediction.score,
            probability = legacy.top_prediction.score,
            evidence    = legacy.top_prediction.evidence,
        )

    return ECResult(
        uniprot_id        = uniprot_id,
        sequence_length   = len(sequence),
        is_enzyme         = legacy.is_enzyme,
        enzyme_confidence = legacy.enzyme_confidence,
        top_prediction    = top_legacy,
        all_predictions   = all_preds,
        specific_ec       = legacy.specific_ec,
        specific_ec_name  = legacy.specific_ec_name,
        non_enzyme_score  = legacy.non_enzyme_score,
        ml_used           = False,
    )


# ── Convenience: load all intermediate JSONs for a protein ────────────────────

def _load_intermediate(data_dir: str, uniprot_id: str) -> dict:
    """Load all available intermediate JSON files for a protein."""
    results: dict = {}
    inter = Path(data_dir)
    mapping = {
        "esm2":           f"{uniprot_id}_esm2.json",
        "active_sites":   f"{uniprot_id}_active_sites.json",
        "pockets":        f"{uniprot_id}_pockets.json",
        "physicochemical":f"{uniprot_id}_physicochemical.json",
        "enm":            f"{uniprot_id}_enm.json",
        "go_terms":       f"{uniprot_id}_go_terms.json",
        "homology":       f"{uniprot_id}_homology.json",
        "structure":      f"{uniprot_id}_structure.json",
    }
    for key, fname in mapping.items():
        fpath = inter / fname
        if fpath.exists():
            try:
                results[key] = json.loads(fpath.read_text())
            except Exception:
                pass
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--uniprot", "-u", required=True,
              help="UniProt ID (e.g. P04637)")
@click.option("--data-dir", default="data/intermediate",
              help="Pipeline intermediate directory")
@click.option("--model-dir", default=str(DEFAULT_MODEL_DIR),
              help="Trained ML model directory")
@click.option("--output", "-o", default=None,
              help="Save result JSON to this path")
def main(uniprot: str, data_dir: str, model_dir: str, output: Optional[str]) -> None:
    """
    Module 10-ML: ML-based enzyme commission number prediction.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Try to load from pipeline intermediate first
    inter = _load_intermediate(data_dir, uniprot)

    # Try to get sequence
    sequence = ""
    struct = inter.get("structure", {})
    if struct:
        residues = struct.get("residues", [])
        aa_map = "ACDEFGHIKLMNPQRSTVWY"
        aa_3to1 = {
            "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
            "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
            "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
            "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
        }
        sequence = "".join(aa_3to1.get(r.get("residue_name", ""), "X") for r in residues)

    if not sequence:
        # Fetch from UniProt
        log.info(f"  Fetching sequence for {uniprot} from UniProt...")
        import urllib.request
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
            with urllib.request.urlopen(url, timeout=15) as r:
                fasta = r.read().decode()
            sequence = "".join(fasta.strip().split("\n")[1:])
        except Exception as e:
            log.error(f"  Failed to fetch sequence: {e}")
            raise SystemExit(1)

    result = predict_ec_ml(
        uniprot_id      = uniprot,
        sequence        = sequence,
        active_result   = inter.get("active_sites"),
        go_result       = inter.get("go_terms"),
        homology_result = inter.get("homology"),
        esm2_result     = inter.get("esm2"),
        pdb_result      = inter.get("structure"),
        pocket_result   = inter.get("pockets"),
        enm_result      = inter.get("enm"),
        physico_result  = inter.get("physicochemical"),
        model_dir       = Path(model_dir),
    )

    print(result.summary())

    if output:
        result.to_json(output)
        log.info(f"  Result saved to {output}")
    else:
        out_path = Path(data_dir) / f"{uniprot}_ec_ml.json"
        result.to_json(out_path)
        log.info(f"  Result saved to {out_path}")


if __name__ == "__main__":
    main()