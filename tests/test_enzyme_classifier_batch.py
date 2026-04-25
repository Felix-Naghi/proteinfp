"""
tests/test_enzyme_classifier_batch.py
──────────────────────────────────────
Batch-test the EC ENSEMBLE (models/ec_ensemble/) against all proteins
that have ESM-2 embeddings in data/intermediate/.

This uses ECClassifierEnsemble — the same 99.93%-accuracy model trained
by pipeline/ml_ec_train.py — NOT the old enzyme_classifier.pkl.

Usage:
    python tests/test_enzyme_classifier_batch.py
    python tests/test_enzyme_classifier_batch.py --out results/enzyme_predictions.csv
    python tests/test_enzyme_classifier_batch.py --data-dir data/intermediate
"""
import json
import csv
import sys
import logging
from pathlib import Path

import numpy as np
import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,   # suppress pipeline noise; we handle our own output
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

EC_CLASS_NAMES = {
    "non-enzyme": "Non-enzyme",
    "1": "Oxidoreductase",
    "2": "Transferase",
    "3": "Hydrolase",
    "4": "Lyase",
    "5": "Isomerase",
    "6": "Ligase",
    "7": "Translocase",
}


def load_intermediate(inter_dir: Path, uid: str) -> dict:
    """Load all available module JSONs for a protein."""
    results = {}
    files = {
        "esm2":            f"{uid}_esm2.json",
        "active_result":   f"{uid}_active_sites.json",
        "pdb_result":      f"{uid}_structure.json",
        "pocket_result":   f"{uid}_pockets.json",
        "enm_result":      f"{uid}_enm.json",
        "physico_result":  f"{uid}_physicochemical.json",
        "go_result":       f"{uid}_go_predictions.json",
        "homology_result": f"{uid}_homology.json",
    }
    for key, fname in files.items():
        p = inter_dir / fname
        if p.exists():
            try:
                results[key] = json.loads(p.read_text())
            except Exception:
                pass
    return results


def get_sequence(inter_dir: Path, uid: str) -> str:
    """Pull sequence from the structure JSON (written by Module 01)."""
    p = inter_dir / f"{uid}_structure.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("sequence", "")
        except Exception:
            pass
    return ""


@click.command()
@click.option("--model-dir", default=str(ROOT / "models" / "ec_ensemble"),
              help="Path to trained ECClassifierEnsemble directory")
@click.option("--data-dir",  default=str(ROOT / "data" / "intermediate"),
              help="Folder containing *_esm2.json files")
@click.option("--out",       default=None,
              help="Save CSV results to this path")
def main(model_dir, data_dir, out):
    inter_dir  = Path(data_dir)
    model_path = Path(model_dir)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\nProteinFP — EC Ensemble Batch Test")
    print(f"{'─'*60}")
    print(f"  Model dir      : {model_path}")
    print(f"  Intermediate   : {inter_dir}")

    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("  Train it first:  python pipeline\\ml_ec_train.py")
        sys.exit(1)

    esm2_files = sorted(inter_dir.glob("*_esm2.json"))
    print(f"  Proteins found : {len(esm2_files)}")
    if not esm2_files:
        print(f"\nERROR: No *_esm2.json files in {inter_dir}")
        print("  Run Module 08:  python pipeline\\08_esm2_embeddings.py --uniprot P04637")
        sys.exit(1)

    # ── Load the real model ───────────────────────────────────────────────────
    from pipeline.ml_ec_classifier import ECClassifierEnsemble
    print(f"\n  Loading ECClassifierEnsemble...")
    clf = ECClassifierEnsemble.load(model_path)
    print(f"  Loaded OK.\n")

    # ── Batch predict ─────────────────────────────────────────────────────────
    print(f"  {'UniProt':<12}  {'Prediction':<12}  {'Prob':>6}  EC Class Name")
    print(f"  {'─'*12}  {'─'*12}  {'─'*6}  {'─'*20}")

    rows = []
    ec_counts = {}

    for esm2_path in esm2_files:
        uid = esm2_path.stem.replace("_esm2", "")
        mod = load_intermediate(inter_dir, uid)
        seq = get_sequence(inter_dir, uid)

        try:
            result = clf.predict(
                sequence        = seq,
                esm2_result     = mod.get("esm2"),
                pdb_result      = mod.get("pdb_result"),
                active_result   = mod.get("active_result"),
                pocket_result   = mod.get("pocket_result"),
                enm_result      = mod.get("enm_result"),
                physico_result  = mod.get("physico_result"),
                go_result       = mod.get("go_result"),
                homology_result = mod.get("homology_result"),
                uniprot_id      = uid,
            )

            # MLECResult fields (not ECResult — no enzyme_confidence here)
            is_enzyme  = result.is_enzyme
            enz_prob   = result.enzyme_probability          # ← correct field name
            top        = result.top_prediction              # MLECPrediction or None
            ec_class   = top.ec_class if top else "non-enzyme"
            top_prob   = top.probability if top else (1.0 - enz_prob)
            ec_name    = EC_CLASS_NAMES.get(ec_class, ec_class)

            mark  = "✓" if is_enzyme else " "
            label = f"EC{ec_class}" if is_enzyme else "non-enzyme"
            print(f"  [{mark}] {uid:<12}  {label:<12}  {enz_prob:>6.3f}  {ec_name}")

            ec_counts[label] = ec_counts.get(label, 0) + 1
            rows.append({
                "uniprot_id":       uid,
                "prediction":       label,
                "ec_class":         ec_class,
                "enzyme_prob":      round(enz_prob, 4),
                "top_class_prob":   round(top_prob, 4),
                "is_enzyme":        is_enzyme,
                "ec_name":          ec_name,
                "model_version":    result.model_version,
                "inference_ms":     result.inference_time_ms,
            })

        except Exception as e:
            print(f"  [!] {uid:<12}  ERROR: {e}")
            rows.append({"uniprot_id": uid, "prediction": "error", "ec_class": "",
                         "enzyme_prob": 0, "top_class_prob": 0, "is_enzyme": False,
                         "ec_name": "", "model_version": "", "inference_ms": 0})

    # ── Summary ───────────────────────────────────────────────────────────────
    n         = len(rows)
    n_enzyme  = sum(1 for r in rows if r["is_enzyme"])
    n_none    = n - n_enzyme

    print(f"\n{'─'*60}")
    print(f"  Total proteins : {n}")
    print(f"  Enzymes        : {n_enzyme}  ({n_enzyme/max(n,1)*100:.1f}%)")
    print(f"  Non-enzymes    : {n_none}")
    print(f"\n  Breakdown by EC class:")
    for label, count in sorted(ec_counts.items()):
        name = EC_CLASS_NAMES.get(label.replace("EC",""), label)
        print(f"    {label:<12}  {count:>3}  ({count/max(n,1)*100:.1f}%)  {name}")
    print(f"{'─'*60}")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["uniprot_id","prediction","ec_class","enzyme_prob",
          "top_class_prob","is_enzyme","ec_name","model_version","inference_ms"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()