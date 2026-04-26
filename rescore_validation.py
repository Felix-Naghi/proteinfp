"""
rescore_validation.py
──────────────────────
Re-scores the enzyme classification component of the 76-protein validation
report using the new ML ensemble (models/ec_ensemble_v2), then recomputes
overall_score for every protein and saves an updated report.

The other components (GO recall, active site recall, PPI recall) are kept
exactly as-is from the original report — only enzyme_correct, ec_correct,
and overall_score are updated.

Usage (from project root):
    python rescore_validation.py

    # Use a different model or report:
    python rescore_validation.py \\
        --report  data/reports/validation/validation_report.json \\
        --model-dir models/ec_ensemble_v2 \\
        --cache-dir data/training_cache_v2 \\
        --out     data/reports/validation/validation_report_v2.json
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional

import certifi
import click
import numpy as np

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.ml_ec_classifier import ECClassifierEnsemble
from pipeline.ml_ec_features import build_feature_vector

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Scoring weights — must match validation/run_validation.py exactly
W_GO      = 35
W_AS      = 25
W_ENZYME  = 20
W_PPI     = 20


def _fetch_sequence(uid: str) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "ProteinFP-rescore/1.0",
            "Accept": "text/x-fasta",
        })
        with urllib.request.urlopen(req, timeout=20) as r:
            fasta = r.read().decode()
        lines = fasta.strip().split("\n")
        return "".join(lines[1:]).upper()
    except Exception as e:
        log.warning(f"  Could not fetch {uid}: {e}")
        return None


def _get_sequence_from_report(uid: str, data_dir: str) -> Optional[str]:
    """Try to get sequence from existing pipeline intermediate files."""
    inter = Path(data_dir)
    for suffix in ["_structure.json", "_esm2.json"]:
        fpath = inter / f"{uid}{suffix}"
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                seq = data.get("sequence", "")
                if seq:
                    return seq.upper()
            except Exception:
                pass
    return None


def _load_esm2_cache(cache_dir: str) -> dict:
    cache_path = Path(cache_dir) / "esm2_cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            log.info(f"  ESM-2 cache loaded: {len(cache):,} embeddings")
            return cache
        except Exception as e:
            log.warning(f"  Could not load ESM-2 cache: {e}")
    else:
        log.warning(f"  No ESM-2 cache at {cache_path} — will use sequence-only features")
    return {}


def _compute_fresh_esm2(sequence: str) -> list:
    """Compute a single ESM-2 embedding on the fly."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        mdl = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(device)

        inputs = tok(sequence, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        emb = out.last_hidden_state[0].mean(dim=0).cpu().tolist()
        return emb
    except Exception as e:
        log.warning(f"  ESM-2 compute failed ({e}), using zeros")
        return [0.0] * 1280


def _predict_for_protein(
    uid: str,
    sequence: str,
    clf: ECClassifierEnsemble,
    esm2_cache: dict,
    fresh_esm2_model: list,   # mutable list [tok, mdl, device] or []
    intermediate_dir: Optional[str],
) -> dict:
    """
    Run the ML model on one protein.
    Returns dict with: is_enzyme, enzyme_prob, top_ec_class, top_ec_prob
    """
    # Get ESM-2 embedding
    emb = esm2_cache.get(uid)
    if emb is None:
        # Try pipeline intermediate
        if intermediate_dir:
            esm2_path = Path(intermediate_dir) / f"{uid}_esm2.json"
            if esm2_path.exists():
                try:
                    esm2_data = json.loads(esm2_path.read_text())
                    emb = esm2_data.get("protein_embedding")
                except Exception:
                    pass
    if emb is None:
        log.info(f"    {uid}: computing fresh ESM-2 embedding...")
        emb = _compute_fresh_esm2(sequence)

    esm2_result = {"protein_embedding": emb, "contact_map": []}

    # Also load any available pipeline intermediate data
    pdata: dict = {}
    if intermediate_dir:
        inter = Path(intermediate_dir)
        for suffix, key in [
            ("_active_sites", "active_sites"),
            ("_go_terms", "go_terms"),
            ("_homology", "homology"),
            ("_pockets", "pockets"),
            ("_enm", "enm"),
            ("_physicochemical", "physicochemical"),
        ]:
            fpath = inter / f"{uid}{suffix}.json"
            if fpath.exists():
                try:
                    pdata[key] = json.loads(fpath.read_text())
                except Exception:
                    pass

    feat = build_feature_vector(
        sequence        = sequence,
        esm2_result     = esm2_result,
        active_result   = pdata.get("active_sites"),
        pocket_result   = pdata.get("pockets"),
        enm_result      = pdata.get("enm"),
        physico_result  = pdata.get("physicochemical"),
        go_result       = pdata.get("go_terms"),
        homology_result = pdata.get("homology"),
    )

    X_pp  = clf.preprocessor.transform(feat.reshape(1, -1))
    proba = clf._predict_proba_raw(X_pp)[0]

    non_enz_prob = float(proba[0])
    enzyme_prob  = float(proba[1:].sum())
    is_enzyme    = enzyme_prob > 0.5

    top_ec_idx  = int(proba[1:].argmax()) + 1   # 1-7
    top_ec_prob = float(proba[top_ec_idx])

    return {
        "is_enzyme":    is_enzyme,
        "enzyme_prob":  round(enzyme_prob, 4),
        "top_ec_class": str(top_ec_idx),
        "top_ec_prob":  round(top_ec_prob, 4),
        "non_enz_prob": round(non_enz_prob, 4),
    }


def rescore(
    report_path: str,
    model_dir: str,
    cache_dir: str,
    out_path: str,
    intermediate_dir: Optional[str],
) -> None:
    # ── Load original report ──────────────────────────────────────────────────
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    scores = report["scores"]
    log.info(f"  Loaded {len(scores)} protein scores from {report_path}")

    # Load original summary stats for comparison
    orig_enzyme_acc = sum(1 for s in scores if s.get("enzyme_correct", False)) / len(scores)
    orig_overall    = sum(s["overall_score"] for s in scores) / len(scores)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"  Loading ML model from {model_dir}/")
    clf = ECClassifierEnsemble.load(Path(model_dir))

    # ── Load ESM-2 cache ──────────────────────────────────────────────────────
    esm2_cache = _load_esm2_cache(cache_dir)

    # ── Process each protein ──────────────────────────────────────────────────
    log.info(f"  Re-scoring {len(scores)} proteins...")
    n_changed_enzyme = 0
    n_changed_ec     = 0

    for i, score in enumerate(scores):
        uid = score["uniprot_id"]
        log.info(f"  [{i+1:>3}/{len(scores)}] {uid} ({score.get('gene','?')})")

        # Get sequence
        seq = None
        if intermediate_dir:
            seq = _get_sequence_from_report(uid, intermediate_dir)
        if not seq:
            seq = _fetch_sequence(uid)
            time.sleep(0.15)   # polite rate limit

        if not seq:
            log.warning(f"    Could not get sequence for {uid} — skipping")
            continue

        # ML prediction
        pred = _predict_for_protein(
            uid, seq, clf, esm2_cache, [], intermediate_dir
        )

        # Determine new enzyme_correct
        # Ground truth is implicit in the original score:
        #   - enzyme_correct=True AND ec_correct=True  → known enzyme with EC
        #   - enzyme_correct=True AND ec_correct could be either
        #   - the original ec_correct already captured "was the EC class right"
        # We need to know is_enzyme ground truth. We re-derive it:
        #   if original ec_correct==True and enzyme_correct==True → known enzyme
        #   if original score had ec_correct from non-enzyme logic → non-enzyme
        # Safest: look at what the original report says was expected.
        # The validation report doesn't store ground truth directly, but we can
        # infer: any protein where ec_correct was set to True via the non-enzyme
        # shortcut had known_enzyme=False in the ground truth set.
        # We use a heuristic: if overall_score had enzyme component=20 before, keep it.
        # Actually the cleanest approach: store old enzyme_correct to compare later.

        old_enzyme_correct = score.get("enzyme_correct", False)
        old_ec_correct     = score.get("ec_correct", False)

        # We need ground truth is_enzyme. Infer from original report logic:
        # The scoring formula gives enzyme_correct=True if:
        #   a) protein is known non-enzyme AND was predicted non-enzyme, OR
        #   b) protein is known enzyme AND was predicted enzyme
        # We can't recover ground truth from just the score, but we CAN look at
        # whether ec_correct was set via the "non-enzyme → True" shortcut.
        # Better approach: just use the known_is_enzyme stored implicitly.
        # If ec_correct=True and the protein has no ec_number (non-enzyme case),
        # then known_is_enzyme=False.
        # For now, we use the ML prediction's is_enzyme directly, and compare
        # against what the original scoring produced — the net change in
        # enzyme_correct tells us the improvement.

        new_enzyme_correct = pred["is_enzyme"] == old_enzyme_correct or (
            # If old was correct, preserve only if ML agrees
            # (we trust ML over old rule-based)
            True  # always use ML prediction
        )

        # Simpler and correct: just directly use ML output and recompute
        # enzyme_correct by comparing to the ground truth embedded in the
        # original report's enzyme_correct value ONLY for the cases where
        # the old system was clearly wrong (enzyme_correct=False)
        # For the cases where old was correct, check if ML agrees.

        # The real ground truth: a protein is an enzyme if it has an EC number.
        # We can check this from the original validation set definition.
        # Load the validation set from run_validation.py to get ground truth.
        new_enzyme_correct = _derive_enzyme_correct(
            pred["is_enzyme"],
            pred["top_ec_class"],
            score,
        )
        new_ec_correct = _derive_ec_correct(
            pred["top_ec_class"],
            pred["is_enzyme"],
            score,
        )

        if new_enzyme_correct != old_enzyme_correct:
            n_changed_enzyme += 1
            log.info(f"    enzyme_correct: {old_enzyme_correct} → {new_enzyme_correct}  "
                     f"(enzyme_p={pred['enzyme_prob']:.3f})")
        if new_ec_correct != old_ec_correct:
            n_changed_ec += 1

        # Recompute overall_score with same weights as run_validation.py
        new_overall = round(
            score["go_mean_recall"]     * W_GO     +
            score["active_site_recall"] * W_AS     +
            (1.0 if new_enzyme_correct else 0.0) * W_ENZYME +
            score["ppi_recall"]         * W_PPI,
            1
        )

        # Write back
        score["enzyme_correct"]     = new_enzyme_correct
        score["ec_correct"]         = new_ec_correct
        score["overall_score"]      = new_overall
        score["ml_enzyme_prob"]     = pred["enzyme_prob"]
        score["ml_top_ec_class"]    = pred["top_ec_class"]
        score["ml_top_ec_prob"]     = pred["top_ec_prob"]

    # ── Summary stats ─────────────────────────────────────────────────────────
    new_enzyme_acc = sum(1 for s in scores if s.get("enzyme_correct", False)) / len(scores)
    new_overall    = sum(s["overall_score"] for s in scores) / len(scores)

    print("\n" + "═" * 66)
    print("  VALIDATION RESCORE RESULTS")
    print("═" * 66)
    print(f"  Proteins rescored       : {len(scores)}")
    print(f"  Enzyme class changes    : {n_changed_enzyme}")
    print(f"  EC class changes        : {n_changed_ec}")
    print("─" * 66)
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Delta':>8}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'Enzyme classification':<30} "
          f"{orig_enzyme_acc*100:>9.1f}% "
          f"{new_enzyme_acc*100:>9.1f}% "
          f"{(new_enzyme_acc-orig_enzyme_acc)*100:>+7.1f}%")
    print(f"  {'Overall accuracy score':<30} "
          f"{orig_overall:>9.1f}/100 "
          f"{new_overall:>9.1f}/100 "
          f"{new_overall-orig_overall:>+7.1f}")
    print("═" * 66)

    # ── Save updated report ───────────────────────────────────────────────────
    report["run_date"]        = report.get("run_date", "") + f" (rescored {datetime.now().strftime('%Y-%m-%d %H:%M')})"
    report["ml_model_dir"]    = model_dir
    report["rescore_summary"] = {
        "orig_enzyme_acc":  round(orig_enzyme_acc, 4),
        "new_enzyme_acc":   round(new_enzyme_acc,  4),
        "orig_overall":     round(orig_overall,    2),
        "new_overall":      round(new_overall,     2),
        "n_enzyme_changed": n_changed_enzyme,
        "n_ec_changed":     n_changed_ec,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"  Updated report saved → {out_path}")

    # Also write a readable txt summary
    txt_path = Path(out_path).with_suffix(".txt")
    _write_txt_summary(scores, new_overall, new_enzyme_acc,
                       orig_overall, orig_enzyme_acc, txt_path)
    log.info(f"  Text summary saved   → {txt_path}")


def _derive_enzyme_correct(ml_is_enzyme: bool, ml_ec: str, score: dict) -> bool:
    """
    Derive whether ML prediction is correct for enzyme/non-enzyme classification.
    We infer ground truth from the original report's scoring pattern.
    If the protein has a non-empty ml_top_ec_class we assume it was scored as enzyme.
    We use the following rule from run_validation.py:
      - known_enzyme=True  → enzyme_correct iff predicted enzyme
      - known_enzyme=False → enzyme_correct iff predicted non-enzyme
    Ground truth is inferred: if original report says ec_correct=True and
    enzyme_correct=True, the protein is likely a known enzyme. But this is
    lossy. Instead we load the ground truth from the validation set module.
    """
    # Load ground truth from validation module
    gt = _get_ground_truth(score["uniprot_id"])
    if gt is None:
        # Fall back: infer from original report
        # If protein had ec_correct=False and enzyme_correct=False in original,
        # ground truth is likely enzyme (system missed it).
        # If ec_correct=True and enzyme_correct=False, system correctly IDed
        # non-enzyme (ec_correct=True for non-enzymes per the scoring logic).
        # This is ambiguous, keep original.
        return score.get("enzyme_correct", False)

    known_enzyme = gt.get("is_enzyme", False)
    if known_enzyme:
        return ml_is_enzyme
    else:
        return not ml_is_enzyme


def _derive_ec_correct(ml_ec: str, ml_is_enzyme: bool, score: dict) -> bool:
    """Check if ML EC class matches ground truth first digit."""
    gt = _get_ground_truth(score["uniprot_id"])
    if gt is None:
        return score.get("ec_correct", False)

    known_enzyme = gt.get("is_enzyme", False)
    known_ec     = str(gt.get("ec_number", "")).strip()

    if not known_enzyme:
        # Non-enzyme: ec_correct=True iff we didn't predict an enzyme
        return not ml_is_enzyme
    if known_ec and ml_is_enzyme:
        return ml_ec == known_ec[0]
    return False


# ── Ground truth lookup ───────────────────────────────────────────────────────
# Inlined from validation/run_validation.py VALIDATION_SET to avoid import
# issues. Only stores the fields we need: uniprot_id, is_enzyme, ec_number.

_GT_TABLE: Optional[dict] = None

def _load_gt_table() -> dict:
    """Load ground truth from VALIDATION_SET + new_entries.json."""
    global _GT_TABLE
    if _GT_TABLE is not None:
        return _GT_TABLE

    _GT_TABLE = {}

    # Source 1: VALIDATION_SET from run_validation.py
    try:
        sys.path.insert(0, str(Path("validation")))
        from run_validation import VALIDATION_SET
        for p in VALIDATION_SET:
            _GT_TABLE[p["uniprot_id"]] = p
        log.info(f"  Ground truth: {len(VALIDATION_SET)} proteins from VALIDATION_SET")
    except Exception as e:
        log.warning(f"  Could not import VALIDATION_SET: {e}")

    # Source 2: new_entries.json (the extra 56 proteins)
    new_entries_path = Path("validation/new_entries.json")
    if new_entries_path.exists():
        try:
            entries = json.loads(new_entries_path.read_text(encoding="utf-8"))
            before = len(_GT_TABLE)
            for p in entries:
                uid = p.get("uniprot_id", "")
                if uid and uid not in _GT_TABLE:
                    _GT_TABLE[uid] = p
            added = len(_GT_TABLE) - before
            log.info(f"  Ground truth: +{added} proteins from new_entries.json "
                     f"(total {len(_GT_TABLE)})")
        except Exception as e:
            log.warning(f"  Could not load new_entries.json: {e}")

    if not _GT_TABLE:
        log.error("  No ground truth loaded! Rescore will be inaccurate.")
    return _GT_TABLE


def _get_ground_truth(uid: str) -> Optional[dict]:
    table = _load_gt_table()
    return table.get(uid)


def _write_txt_summary(scores, new_overall, new_enzyme_acc,
                        orig_overall, orig_enzyme_acc, txt_path: Path) -> None:
    lines = [
        "=" * 70,
        "  ProteinFP Validation Report — Updated with ML EC Classifier v2",
        f"  Rescored: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Proteins: {len(scores)}",
        "=" * 70,
        "",
        f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Delta':>8}",
        f"  {'─'*28} {'─'*8} {'─'*8} {'─'*6}",
        f"  {'Enzyme classification':<30} "
        f"{orig_enzyme_acc*100:>9.1f}% "
        f"{new_enzyme_acc*100:>9.1f}% "
        f"{(new_enzyme_acc-orig_enzyme_acc)*100:>+7.1f}%",
        f"  {'Overall accuracy score':<30} "
        f"{orig_overall:>9.1f}/100 "
        f"{new_overall:>9.1f}/100 "
        f"{new_overall-orig_overall:>+7.1f}",
        "",
        "─" * 70,
        "  Per-protein breakdown (updated):",
        "─" * 70,
    ]

    for s in scores:
        uid   = s["uniprot_id"]
        gene  = s.get("gene", "?")
        cat   = s.get("category", "?")
        go    = s.get("go_mean_recall", 0)
        as_   = s.get("active_site_recall", 0)
        enz   = s.get("enzyme_correct", False)
        ppi   = s.get("ppi_recall", 0)
        ov    = s.get("overall_score", 0)
        ml_p  = s.get("ml_enzyme_prob", "?")
        lines.append(
            f"  {uid:<10} {gene:<10} [{cat:<20}] "
            f"GO={go*100:.0f}% AS={as_*100:.0f}% "
            f"Enz={'Y' if enz else 'N'}({ml_p:.2f}) "
            f"PPI={ppi*100:.0f}% "
            f"Overall={ov:.1f}"
        )

    lines += ["=" * 70]
    txt_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--report",   default="data/reports/validation/validation_report.json",
              help="Original validation report JSON")
@click.option("--model-dir", default="models/ec_ensemble_v2",
              help="Trained ML ensemble directory")
@click.option("--cache-dir", default="data/training_cache_v2",
              help="ESM-2 embedding cache from training")
@click.option("--out",      default="data/reports/validation/validation_report_v2.json",
              help="Output path for updated report")
@click.option("--data-dir", default="data/intermediate",
              help="Pipeline intermediate dir (for sequences + extra features)")
def main(report, model_dir, cache_dir, out, data_dir):
    """Re-score enzyme classification on the 76-protein validation set."""
    log.info("═" * 66)
    log.info("  Validation Rescore — ML EC Classifier v2")
    log.info("═" * 66)
    rescore(report, model_dir, cache_dir, out, data_dir)


if __name__ == "__main__":
    main()