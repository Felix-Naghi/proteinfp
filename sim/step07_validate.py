"""
sim/step07_validate.py
───────────────────────
Validates Modules 3-4 against 25 known drug-protein pairs
with experimental pKi values from ChEMBL/BindingDB.

For each pair:
  1. Load protein ensemble from Module 2 output
  2. Get drug concentration in target compartment (Module 3 logic)
  3. Compute binding score (Module 4 physics)
  4. Compare predicted pKi vs experimental pKi

Metrics:
  - Pearson r (correlation)
  - RMSE (root mean square error in pKi units)
  - MAE  (mean absolute error)
  - Enrichment: % of top-10 predicted = top-10 experimental

A good scoring function achieves:
  - Pearson r > 0.6
  - RMSE < 1.5 pKi units
  - MAE  < 1.2 pKi units

Usage:
    python sim/step07_validate.py
    python sim/step07_validate.py --verbose
"""

from __future__ import annotations

import json
import math
import argparse
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "data" / "sim"
INTER   = ROOT / "data" / "intermediate"


# ── JSON encoder that handles numpy scalar types ──────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to native Python types for JSON output."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_python(obj):
    """Recursively convert numpy types in a nested structure to Python natives."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Load validation pairs ─────────────────────────────────────────────────────

def load_pairs() -> list[dict]:
    path = SIM_DIR / "validation_pairs.json"
    if not path.exists():
        print("ERROR: Run sim/fix_all.py first to create validation_pairs.json")
        return []
    return json.loads(path.read_text())


# ── Load or create ensemble for a protein ────────────────────────────────────

def get_ensemble(uid: str) -> dict:
    """Load ensemble if exists, otherwise build minimal one from report."""
    ens_path = SIM_DIR / "ensembles" / f"{uid}_ensemble.json"
    if ens_path.exists():
        return json.loads(ens_path.read_text())

    # Build minimal ensemble from report data
    report_path = ROOT / "data" / "reports" / f"{uid}_report.json"
    struct_path = INTER / f"{uid}_structure.json"

    if not report_path.exists():
        return {}

    report = json.loads(report_path.read_text())
    struct = json.loads(struct_path.read_text()) if struct_path.exists() else {}

    pockets = report.get("binding_pockets", [])
    vol     = pockets[0].get("volume_A3", 500) if pockets else 500
    drug    = pockets[0].get("druggability_score", 0.5) if pockets else 0.5
    length  = struct.get("length", report.get("length", 300))
    plddt   = struct.get("mean_plddt", report.get("mean_plddt", 70.0))

    # Build simple ensemble with 3 states
    total = math.exp(0) + math.exp(-drug * 2) + math.exp(-1)
    states = [
        {"name": "active",   "probability": round(math.exp(0) / total, 4),
         "pocket_volume_A3": vol, "druggability": drug},
        {"name": "apo",      "probability": round(math.exp(-drug*2) / total, 4),
         "pocket_volume_A3": vol * 0.85, "druggability": drug * 0.85},
        {"name": "inactive", "probability": round(math.exp(-1) / total, 4),
         "pocket_volume_A3": vol * 0.3,  "druggability": drug * 0.2},
    ]

    compartment = "nucleus"
    go_cc = [t.get("go_name", "").lower()
             for t in report.get("go_terms_cc", [])]
    if any("plasma membrane" in g or "cell surface" in g for g in go_cc):
        compartment = "plasma_membrane"
    elif any("mitochondri" in g for g in go_cc):
        compartment = "mitochondria"

    return {
        "uniprot_id":        uid,
        "gene_name":         report.get("gene_name", uid),
        "compartment":       compartment,
        "length":            length,
        "mean_plddt":        plddt,
        "states":            states,
        "mean_pocket_volume": vol,
        "mean_druggability":  drug,
        "conformational_entropy": 1.0,
        "embedding_norm":    7.0,
    }


# ── Predict pKi for a drug-protein pair ──────────────────────────────────────

def predict_pKi(pair: dict, verbose: bool = False) -> dict:
    """Run Module 4 scoring for a validation pair."""

    # Import scoring functions from Module 4
    import sys
    sys.path.insert(0, str(ROOT))
    from sim.step04_binding_probability import (
        score_binding, KNOWN_DRUGS
    )

    uid      = pair["uniprot_id"]
    ensemble = get_ensemble(uid)

    if not ensemble:
        return {"predicted_pKi": None, "error": "no ensemble"}

    # Build drug dict from pair
    drug = {
        "name":             pair["drug_name"],
        "smiles":           pair["smiles"],
        "molecular_weight": pair["mw"],
        "logP":             pair["logP"],
        "pKa_basic":        6.0,
        "pKa_acidic":       12.0,
        "hbd":              pair["hbd"],
        "hba":              pair["hba"],
        "psa":              pair["psa"],
        "charge_at_pH74":   pair["charge"],
        "rotatable_bonds":  pair["rotors"],
        "permeability":     pair["permeability"],
    }

    # Load cell environment
    env_path = SIM_DIR / "cell_environment.json"
    if not env_path.exists():
        return {"predicted_pKi": None, "error": "no cell environment"}
    cell_env  = json.loads(env_path.read_text())["cell_environment"]
    sim_concs = json.loads(env_path.read_text())["simulation"]["steady_state"]

    comp      = ensemble.get("compartment", "nucleus")
    drug_conc = sim_concs.get(comp, 1.0)

    try:
        score = score_binding(drug, uid, ensemble, cell_env,
                              drug_conc, verbose=False)
        return {
            "predicted_pKi":   float(score.pKi),
            "predicted_Kd_uM": float(score.Kd_corrected_uM),
            "dG_corrected":    float(score.dG_corrected_kJ),
            "fill_ratio":      float(score.fill_ratio),
            "error":           None,
        }
    except Exception as e:
        return {"predicted_pKi": None, "error": str(e)}


# ── Statistics ────────────────────────────────────────────────────────────────

def pearson_r(x: list, y: list) -> float:
    n    = len(x)
    mx   = sum(x) / n
    my   = sum(y) / n
    num  = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sum((xi-mx)**2 for xi in x) *
                      sum((yi-my)**2 for yi in y))
    return num / denom if denom > 0 else 0.0


def rmse(pred: list, exp: list) -> float:
    return math.sqrt(sum((p-e)**2 for p,e in zip(pred,exp)) / len(pred))


def mae(pred: list, exp: list) -> float:
    return sum(abs(p-e) for p,e in zip(pred,exp)) / len(pred)


# ── Main validation ───────────────────────────────────────────────────────────

def run_validation(verbose: bool = False):
    print("=" * 65)
    print("  SIM-07: Validation — Modules 3-4 vs Experimental pKi")
    print("  25 drug-protein pairs from ChEMBL / BindingDB")
    print("=" * 65)

    pairs = load_pairs()
    if not pairs:
        return

    print(f"\n  Running {len(pairs)} predictions...")
    print(f"  {'Drug':<18} {'Target':<10} {'Exp pKi':>8} "
          f"{'Pred pKi':>9} {'Error':>7} {'Status'}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*9} {'-'*7} {'-'*10}")

    results   = []
    exp_pKis  = []
    pred_pKis = []

    for pair in pairs:
        result = predict_pKi(pair, verbose=verbose)
        exp    = pair["exp_pKi"]
        pred   = result.get("predicted_pKi")
        err    = result.get("error")

        if pred is not None and pred > 0:
            error  = pred - exp
            status = "OK"
            exp_pKis.append(exp)
            pred_pKis.append(pred)
        else:
            error  = None
            status = f"SKIP ({err[:20] if err else 'none'})"

        results.append({
            **pair,
            "predicted_pKi":   pred,
            "predicted_Kd_uM": result.get("predicted_Kd_uM"),
            "dG_corrected":    result.get("dG_corrected"),
            "fill_ratio":      result.get("fill_ratio"),
            "error_pKi":       error,
            "status":          status,
        })

        err_str  = f"{error:+.2f}" if error is not None else "  N/A"
        pred_str = f"{pred:.2f}"   if pred  is not None else "  N/A"
        print(f"  {pair['drug_name']:<18} {pair['gene']:<10} "
              f"{exp:>8.1f} {pred_str:>9} {err_str:>7}  {status}")

    # ── Statistics ────────────────────────────────────────────────────────
    n_valid = len(exp_pKis)
    print(f"\n{'='*65}")
    print(f"  VALIDATION STATISTICS  (n={n_valid}/{len(pairs)} valid predictions)")
    print(f"{'='*65}")

    if n_valid < 3:
        print("  Too few valid predictions — check protein reports exist")
        return results

    pearson  = pearson_r(pred_pKis, exp_pKis)   # renamed from 'r' to avoid
    rms      = rmse(pred_pKis, exp_pKis)         # shadowing by loop variable
    ma       = mae(pred_pKis, exp_pKis)
    bias     = sum(pred_pKis[i] - exp_pKis[i]
                   for i in range(n_valid)) / n_valid

    print(f"\n  Pearson r        : {pearson:.3f}  "
          f"{'GOOD' if pearson > 0.6 else 'NEEDS IMPROVEMENT'}")
    print(f"  RMSE             : {rms:.3f} pKi units  "
          f"{'GOOD' if rms < 1.5 else 'NEEDS IMPROVEMENT'}")
    print(f"  MAE              : {ma:.3f} pKi units  "
          f"{'GOOD' if ma < 1.2 else 'NEEDS IMPROVEMENT'}")
    print(f"  Systematic bias  : {bias:+.3f} pKi units  "
          f"({'overpredicting' if bias > 0 else 'underpredicting'})")

    # Enrichment: top-5 by predicted vs top-5 by experimental
    pairs_scored = [
        (res["drug_name"], res["exp_pKi"], res["predicted_pKi"])
        for res in results if res["predicted_pKi"] is not None
    ]
    top5_exp  = set(d for d, e, p in
                    sorted(pairs_scored, key=lambda x: -x[1])[:5])
    top5_pred = set(d for d, e, p in
                    sorted(pairs_scored, key=lambda x: -(x[2] or 0))[:5])
    enrichment = len(top5_exp & top5_pred) / 5 * 100
    print(f"  Enrichment (top5): {enrichment:.0f}%  "
          f"({'GOOD' if enrichment >= 60 else 'NEEDS IMPROVEMENT'})")

    # Per-class breakdown
    print(f"\n  Per-class performance:")
    from collections import defaultdict
    by_class = defaultdict(list)
    for res in results:                          # 'res' — not 'r', avoids shadowing
        if res["predicted_pKi"] is not None:
            by_class[res["target_class"]].append(
                abs(res["exp_pKi"] - res["predicted_pKi"])
            )
    for cls, errors in sorted(by_class.items()):
        mean_e = sum(errors) / len(errors)
        print(f"    {cls:<20} MAE={mean_e:.2f}  n={len(errors)}")

    # ── Save — convert all numpy types before serialising ─────────────────
    out = to_python({
        "n_pairs":    len(pairs),
        "n_valid":    n_valid,
        "pearson_r":  round(pearson, 4),
        "rmse":       round(rms, 4),
        "mae":        round(ma, 4),
        "bias":       round(bias, 4),
        "enrichment": enrichment,
        "results":    results,
    })
    out_path = SIM_DIR / "validation_results.json"
    out_path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"\n  Full results saved to {out_path}")

    print(f"\n{'='*65}")
    print(f"  INTERPRETATION")
    print(f"{'='*65}")
    if pearson > 0.6 and rms < 1.5:
        print("  Pipeline validated — predictions correlate with experiment.")
        print("  Systematic bias can be corrected by training ML correction layer.")
    elif pearson > 0.4:
        print("  Moderated correlation — pipeline captures binding trends.")
        print("  Improvement needed: train ML correction on ChEMBL data.")
    else:
        print("  Low correlation — structural data may be incomplete.")
        print("  Check: do all proteins have full-length structure files?")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIM-07: Validation against known drug-protein pairs"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_validation(args.verbose)