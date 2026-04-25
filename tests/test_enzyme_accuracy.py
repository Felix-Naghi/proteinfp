"""
tests/test_enzyme_accuracy.py
──────────────────────────────
Score the EC ensemble against the built-in validation set ground truth.
No pipeline re-run needed — just uses existing *_esm2.json files.

Usage:
    python tests/test_enzyme_accuracy.py
    python tests/test_enzyme_accuracy.py --score-only   # skip proteins missing ESM2
"""
import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Silence sklearn/lgbm warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)


def load_intermediate(inter_dir: Path, uid: str) -> dict:
    keys = {
        "esm2":            f"{uid}_esm2.json",
        "active_result":   f"{uid}_active_sites.json",
        "pdb_result":      f"{uid}_structure.json",
        "pocket_result":   f"{uid}_pockets.json",
        "enm_result":      f"{uid}_enm.json",
        "physico_result":  f"{uid}_physicochemical.json",
        "go_result":       f"{uid}_go_predictions.json",
        "homology_result": f"{uid}_homology.json",
    }
    out = {}
    for k, fname in keys.items():
        p = inter_dir / fname
        if p.exists():
            try:
                out[k] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def get_sequence(inter_dir: Path, uid: str) -> str:
    p = inter_dir / f"{uid}_structure.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("sequence", "")
        except Exception:
            pass
    return ""


@click.command()
@click.option("--model-dir", default=str(ROOT / "models" / "ec_ensemble"))
@click.option("--data-dir",  default=str(ROOT / "data" / "intermediate"))
@click.option("--score-only", is_flag=True, default=False,
              help="Skip proteins that have no ESM2 file instead of erroring")
def main(model_dir, data_dir, score_only):
    from pipeline.ml_ec_classifier import ECClassifierEnsemble
    from validation.run_validation import VALIDATION_SET

    inter_dir  = Path(data_dir)
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print(f"\nProteinFP — EC Ensemble Accuracy vs Ground Truth")
    print(f"{'─'*65}")
    clf = ECClassifierEnsemble.load(model_path)
    print(f"  Model loaded:    {model_path}")
    print(f"  Validation set:  {len(VALIDATION_SET)} proteins\n")

    results = []
    header = f"  {'UniProt':<10} {'Gene':<8} {'True':^9} {'Pred':^9} {'Enz_Prob':>8}  {'Correct':>7}"
    print(header)
    print(f"  {'─'*10} {'─'*8} {'─'*9} {'─'*9} {'─'*8}  {'─'*7}")

    for gt in VALIDATION_SET:
        uid          = gt["uniprot_id"]
        gene         = gt["gene"]
        true_enzyme  = gt["is_enzyme"]
        true_ec      = gt.get("ec_number", "")   # e.g. "2.7.10.1"
        true_ec_class = true_ec[0] if true_ec else "—"  # first digit only

        esm2_path = inter_dir / f"{uid}_esm2.json"
        if not esm2_path.exists():
            if score_only:
                print(f"  {uid:<10} {gene:<8} {'SKIP — no ESM2 file':}")
                continue
            else:
                print(f"  {uid:<10} {gene:<8}  [no ESM2 — run: python pipeline\\08_esm2_embeddings.py --uniprot {uid}]")
                results.append({"uid": uid, "gene": gene,
                                 "true_enzyme": true_enzyme, "pred_enzyme": None,
                                 "enzyme_prob": None, "correct": None})
                continue

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

            pred_enzyme   = result.is_enzyme
            enz_prob      = result.enzyme_probability
            pred_ec_class = result.top_prediction.ec_class if result.top_prediction else "—"

            # Correctness checks
            enzyme_correct = (pred_enzyme == true_enzyme)
            ec_correct     = (pred_ec_class == true_ec_class) if true_enzyme else None

            true_lbl = f"{'ENZ':>4} EC{true_ec_class}" if true_enzyme else "non-enzyme"
            pred_lbl = f"{'ENZ':>4} EC{pred_ec_class}" if pred_enzyme else "non-enzyme"
            tick     = "✓" if enzyme_correct else "✗"

            print(f"  {uid:<10} {gene:<8} {true_lbl:^9} {pred_lbl:^9} {enz_prob:>8.3f}  [{tick}]")

            results.append({
                "uid": uid, "gene": gene,
                "true_enzyme":   true_enzyme,
                "pred_enzyme":   pred_enzyme,
                "enzyme_prob":   enz_prob,
                "true_ec_class": true_ec_class,
                "pred_ec_class": pred_ec_class,
                "enzyme_correct": enzyme_correct,
                "ec_correct":     ec_correct,
            })

        except Exception as e:
            print(f"  {uid:<10} {gene:<8}  ERROR: {e}")
            results.append({"uid": uid, "gene": gene, "correct": None})

    # ── Metrics ───────────────────────────────────────────────────────────────
    scored    = [r for r in results if r.get("enzyme_correct") is not None]
    correct   = [r for r in scored  if r["enzyme_correct"]]
    tp        = [r for r in scored  if r["true_enzyme"] and r["pred_enzyme"]]
    fp        = [r for r in scored  if not r["true_enzyme"] and r["pred_enzyme"]]
    fn        = [r for r in scored  if r["true_enzyme"] and not r["pred_enzyme"]]
    tn        = [r for r in scored  if not r["true_enzyme"] and not r["pred_enzyme"]]

    n         = len(scored)
    accuracy  = len(correct) / n if n else 0
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0
    recall    = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0

    # EC class accuracy (only for true enzymes where we have ground truth)
    ec_scored  = [r for r in scored if r.get("ec_correct") is not None]
    ec_correct_n = sum(1 for r in ec_scored if r["ec_correct"])
    ec_acc     = ec_correct_n / len(ec_scored) if ec_scored else 0

    print(f"\n{'─'*65}")
    print(f"  Proteins scored   : {n}")
    print(f"  Enzyme accuracy   : {accuracy*100:.1f}%  ({len(correct)}/{n} correct)")
    print(f"  Precision         : {precision*100:.1f}%")
    print(f"  Recall            : {recall*100:.1f}%")
    print(f"  F1 score          : {f1:.3f}")
    print(f"  EC class accuracy : {ec_acc*100:.1f}%  ({ec_correct_n}/{len(ec_scored)} EC1-7 correct)")
    print(f"\n  Confusion matrix:")
    print(f"    TP={len(tp)}  FP={len(fp)}")
    print(f"    FN={len(fn)}  TN={len(tn)}")
    print(f"{'─'*65}")

    if n < len(VALIDATION_SET):
        missing = len(VALIDATION_SET) - n
        print(f"\n  NOTE: {missing} proteins skipped (no ESM2 file).")
        print(f"  Run Module 08 on missing proteins for a complete score:")
        for gt in VALIDATION_SET:
            uid = gt["uniprot_id"]
            if not (inter_dir / f"{uid}_esm2.json").exists():
                print(f"    python pipeline\\08_esm2_embeddings.py --uniprot {uid}")


if __name__ == "__main__":
    main()