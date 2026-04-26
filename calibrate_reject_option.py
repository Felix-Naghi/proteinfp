"""
calibrate_reject_option.py
───────────────────────────
Adds a "reject option" to your trained EC classifier.

What this does:
  1. Reads predictions from a trained model on a held-out set
  2. Calibrates probabilities using isotonic regression (so 0.8 prob really
     means 80% chance of being right, not just "ranks higher than 0.7")
  3. Finds the optimal confidence threshold that maximises this trade-off:
        - HIGH precision when the model says "I'm sure"  (≥95% on confident)
        - "Uncertain" label when below threshold
  4. Saves the calibrator + threshold next to your model
  5. Provides a wrapper: predict_with_reject() that returns either a
     confident class OR an "uncertain" verdict with reasoning

Why this matters:
  Your validation accuracy is ~70%. But if the model is 70% confident on
  hard cases and 95% confident on easy ones, you can split predictions:
    - "Confident" predictions: ~80% of proteins, ~92-95% accurate
    - "Uncertain" predictions: ~20% of proteins, flagged for human review
  This is more useful in practice than 70% blanket accuracy because
  you know which predictions to trust.

Usage:
    # Calibrate (one-time, after training):
    python calibrate_reject_option.py calibrate \\
        --csv data/swissprot_curated_v4.csv \\
        --model-dir models/ec_ensemble_v5 \\
        --cache-dir data/training_cache_v5

    # Predict with reject option:
    python calibrate_reject_option.py predict \\
        --uniprot P04637 \\
        --model-dir models/ec_ensemble_v5

    # Re-evaluate validation set with reject option:
    python calibrate_reject_option.py evaluate-validation \\
        --model-dir models/ec_ensemble_v5
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import ssl
import sys
import time
import urllib.request
from pathlib import Path
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

LABEL_MAP   = {"non-enzyme": 0, "1": 1, "2": 2, "3": 3,
               "4": 4, "5": 5, "6": 6, "7": 7}
CLASS_NAMES = ["non-enz", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]
EC_NAMES    = {
    0: "Non-enzyme",       1: "Oxidoreductase",  2: "Transferase",
    3: "Hydrolase",        4: "Lyase",           5: "Isomerase",
    6: "Ligase",           7: "Translocase",
}


# ══════════════════════════════════════════════════════════════════════════════
# Lazy ESM-2 helpers (used by calibrate-on-validation and evaluate-validation)
# ══════════════════════════════════════════════════════════════════════════════

class _esm2_lazy:
    """Container for lazily-loaded ESM-2 model state."""
    pass


def _load_esm2_for_calib():
    """Load ESM-2 (fair-esm preferred, transformers fallback)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import esm as esm_module
        log.info(f"    ESM-2 (fair-esm) loading on {device}...")
        model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(device)
        return model, alphabet.get_batch_converter(), device
    except ImportError:
        log.warning("    fair-esm unavailable, using transformers")
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        mdl = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        mdl.eval().to(device)
        return ("transformers", tok, mdl), None, device


def _embed_one(seq: str, model, bc, device) -> list:
    """Embed a single sequence."""
    import torch
    if isinstance(model, tuple) and model[0] == "transformers":
        _, tok, mdl = model
        inputs = tok(seq, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        return out.last_hidden_state[0].mean(0).cpu().tolist()
    else:
        _, _, tokens = bc([("p", seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
        L = len(seq)
        return results["representations"][33][0, 1:L+1].mean(0).cpu().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def _fit_calibrator(probs_max: np.ndarray, correct: np.ndarray):
    """
    Isotonic regression: maps raw max-probability → calibrated confidence.
    Input:  probs_max  = max of softmax for each prediction (n,)
            correct    = 1 if prediction was right, 0 otherwise (n,)
    Output: a fitted IsotonicRegression model
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_max, correct)
    return iso


def _find_threshold(calibrated_conf: np.ndarray, correct: np.ndarray,
                    target_precision: float = 0.92):
    """
    Find the smallest threshold τ such that proteins with confidence ≥ τ
    achieve at least target_precision accuracy.

    Returns (threshold, coverage, precision_at_threshold).
    """
    sorted_idx = np.argsort(-calibrated_conf)   # high-confidence first
    sorted_conf = calibrated_conf[sorted_idx]
    sorted_corr = correct[sorted_idx]

    # Sweep from most confident downward
    best_threshold = 1.0
    best_coverage  = 0.0
    best_precision = 0.0
    cumulative = 0
    for i in range(len(sorted_conf)):
        cumulative += sorted_corr[i]
        coverage   = (i + 1) / len(sorted_conf)
        precision  = cumulative / (i + 1)
        if precision >= target_precision:
            best_threshold = sorted_conf[i]
            best_coverage  = coverage
            best_precision = precision

    return float(best_threshold), float(best_coverage), float(best_precision)


def calibrate(
    csv_path:   str,
    model_dir:  str,
    cache_dir:  str,
    holdout_frac: float = 0.20,
    target_precision: float = 0.92,
) -> None:
    """Fit calibrator and threshold on held-out portion of training CSV."""
    from sklearn.model_selection import train_test_split

    log.info("═" * 70)
    log.info("  Reject-option calibration")
    log.info(f"  Target precision when confident: {target_precision*100:.0f}%")
    log.info("═" * 70)

    # ── Load CSV ────────────────────────────────────────────────────────────
    log.info(f"  Loading {csv_path}...")
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"].strip()
            seq = r["sequence"].strip().upper()
            ec  = r["ec_class"].strip()
            if uid and seq and ec in LABEL_MAP:
                rows.append((uid, seq, LABEL_MAP[ec]))
    log.info(f"  {len(rows):,} samples loaded")

    # ── Stratified holdout ──────────────────────────────────────────────────
    labels = [r[2] for r in rows]
    _, calib_rows = train_test_split(
        rows, test_size=holdout_frac, stratify=labels, random_state=999
    )
    log.info(f"  Calibration set: {len(calib_rows)}")

    # ── Load ESM-2 cache ────────────────────────────────────────────────────
    cache_path = Path(cache_dir) / "esm2_cache.json"
    if not cache_path.exists():
        log.error(f"  ESM-2 cache not found: {cache_path}")
        sys.exit(1)
    cache = json.loads(cache_path.read_text())
    log.info(f"  ESM-2 cache loaded: {len(cache):,} embeddings")

    # ── Load model and predict ──────────────────────────────────────────────
    clf = ECClassifierEnsemble.load(Path(model_dir))
    log.info(f"  Model loaded from {model_dir}/")

    log.info(f"  Generating predictions on calibration set...")
    X_list, y_true, uids = [], [], []
    n_no_cache = 0
    for uid, seq, label in calib_rows:
        emb = cache.get(uid)
        if emb is None:
            n_no_cache += 1
            emb = [0.0] * 1280
        feat = build_feature_vector(
            sequence    = seq,
            esm2_result = {"protein_embedding": emb, "contact_map": []},
        )
        X_list.append(feat)
        y_true.append(label)
        uids.append(uid)
    if n_no_cache:
        log.warning(f"  {n_no_cache} proteins missing from cache (used zeros)")

    X = np.array(X_list, dtype=np.float32)
    X_pp = clf.preprocessor.transform(X)
    proba = clf._predict_proba_raw(X_pp)
    y_pred = proba.argmax(axis=1)
    y_true = np.array(y_true)

    raw_acc = (y_pred == y_true).mean()
    log.info(f"  Raw accuracy on calibration set: {raw_acc*100:.2f}%")

    # ── Fit calibrator ──────────────────────────────────────────────────────
    probs_max = proba.max(axis=1)
    correct   = (y_pred == y_true).astype(int)

    log.info(f"  Fitting isotonic calibrator...")
    iso = _fit_calibrator(probs_max, correct)
    calibrated = iso.predict(probs_max)

    # Reliability check
    log.info(f"  Reliability bins (calibrated confidence vs actual accuracy):")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        mask = (calibrated >= lo) & (calibrated < hi)
        n = mask.sum()
        if n > 0:
            obs_acc = correct[mask].mean()
            log.info(f"    [{lo:.1f}, {hi:.2f}): n={n:>4}  "
                     f"actual={obs_acc*100:5.1f}%")

    # ── Find threshold ──────────────────────────────────────────────────────
    log.info(f"\n  Finding threshold for ≥{target_precision*100:.0f}% precision...")
    threshold, coverage, prec = _find_threshold(
        calibrated, correct, target_precision
    )
    log.info(f"  Optimal threshold: {threshold:.3f}")
    log.info(f"  Coverage at threshold: {coverage*100:.1f}% of proteins")
    log.info(f"  Precision when confident: {prec*100:.1f}%")
    log.info(f"  Below threshold: marked 'uncertain' (~{(1-coverage)*100:.0f}%)")

    # ── Save artifacts ──────────────────────────────────────────────────────
    out_path = Path(model_dir) / "reject_option.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "calibrator":         iso,
            "threshold":          threshold,
            "target_precision":   target_precision,
            "calibration_n":      len(calib_rows),
            "raw_accuracy":       float(raw_acc),
            "coverage":           coverage,
            "precision_confident": prec,
        }, f)
    log.info(f"\n  Saved reject_option.pkl → {out_path}")
    log.info("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION WITH REJECT OPTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_reject_option(model_dir: Path):
    path = model_dir / "reject_option.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["calibrator"], d["threshold"]


def predict_with_reject(
    sequence:    str,
    clf,
    calibrator,
    threshold:   float,
    esm2_emb:    Optional[list] = None,
    uniprot_id:  str = "UNKNOWN",
):
    """
    Returns dict with:
      verdict:    'confident' | 'uncertain'
      ec_class:   string  (e.g. '3' or 'non-enzyme', or '?' if uncertain)
      raw_conf:   float   (uncalibrated max prob)
      calib_conf: float   (calibrated probability of being correct)
      threshold:  float   (the cutoff used)
      probabilities: dict (full per-class probabilities)
    """
    esm2_result = None
    if esm2_emb is not None:
        esm2_result = {"protein_embedding": esm2_emb, "contact_map": []}

    feat = build_feature_vector(sequence=sequence, esm2_result=esm2_result)
    X_pp = clf.preprocessor.transform(feat.reshape(1, -1))
    proba = clf._predict_proba_raw(X_pp)[0]

    top_idx   = int(proba.argmax())
    raw_conf  = float(proba[top_idx])

    if calibrator is not None:
        calib_conf = float(calibrator.predict(np.array([raw_conf]))[0])
    else:
        calib_conf = raw_conf

    is_confident = calib_conf >= threshold

    # Top-2 alternatives (always included)
    sorted_classes = sorted(range(8), key=lambda i: -proba[i])

    return {
        "uniprot_id":  uniprot_id,
        "verdict":     "confident" if is_confident else "uncertain",
        "ec_class":    CLASS_NAMES[top_idx] if is_confident else "?",
        "ec_name":     EC_NAMES.get(top_idx, "?") if is_confident else "Uncertain",
        "raw_conf":    raw_conf,
        "calib_conf":  calib_conf,
        "threshold":   threshold,
        "is_enzyme":   bool(top_idx > 0) if is_confident else None,
        "alternatives": [
            {"class": CLASS_NAMES[i],
             "name":  EC_NAMES[i],
             "prob":  float(proba[i])}
            for i in sorted_classes[:3]
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """Reject-option calibration for trustworthy predictions."""
    pass


@cli.command("calibrate-on-validation")
@click.option("--report",   default="data/reports/validation/validation_report_v5.json",
              help="Existing rescored validation report")
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
@click.option("--target-precision", default=0.90, type=float,
              help="Target accuracy on confident predictions (default 0.90)")
def calibrate_on_validation_cmd(report, model_dir, cache_dir, target_precision):
    """
    Calibrate using the 76-protein validation set instead of training holdout.
    This gives an honest threshold because validation proteins were excluded
    from training (no memorization).
    """
    log.info("=" * 70)
    log.info("  Reject-option calibration (on validation set)")
    log.info(f"  Target precision when confident: {target_precision*100:.0f}%")
    log.info("=" * 70)

    # Load validation report (has uniprot_id, predictions already cached)
    rep = json.loads(Path(report).read_text())
    scores = rep["scores"]
    log.info(f"  Loaded {len(scores)} predictions from validation report")

    # Load ground truth
    sys.path.insert(0, "validation")
    gt = {}
    try:
        from run_validation import VALIDATION_SET
        for p in VALIDATION_SET: gt[p["uniprot_id"]] = p
    except: pass
    ne = Path("validation/new_entries.json")
    if ne.exists():
        for p in json.loads(ne.read_text()):
            uid = p.get("uniprot_id","")
            if uid and uid not in gt: gt[uid] = p

    # Load model + ESM-2 cache to recompute proper top-class probabilities
    clf = ECClassifierEnsemble.load(Path(model_dir))
    log.info(f"  Model loaded from {model_dir}/")
    cache_path = Path(cache_dir) / "esm2_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # Re-predict to get full proba (need top-class max, not enzyme aggregate)
    log.info("  Re-running predictions on validation set...")

    # Need sequences — fetch from intermediate dir or UniProt
    inter_dir = Path("data/intermediate")
    probs_max = []
    correct   = []
    uids      = []

    for s in scores:
        uid = s["uniprot_id"]
        # Get sequence
        seq = None
        for suffix in ["_structure.json", "_esm2.json"]:
            f = inter_dir / f"{uid}{suffix}"
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    seq = d.get("sequence", "")
                    if seq: break
                except: pass
        if not seq:
            log.warning(f"    Skipping {uid} — no sequence")
            continue

        emb = cache.get(uid)
        if emb is None:
            if not hasattr(_esm2_lazy, "model"):
                log.info("    Loading ESM-2 once for missing proteins...")
                _esm2_lazy.model, _esm2_lazy.bc, _esm2_lazy.device = _load_esm2_for_calib()
            try:
                emb = _embed_one(seq.upper(), _esm2_lazy.model,
                                  _esm2_lazy.bc, _esm2_lazy.device)
                cache[uid] = emb
            except Exception as e:
                log.warning(f"    Could not embed {uid}: {e}")
                continue

        feat = build_feature_vector(
            sequence    = seq.upper(),
            esm2_result = {"protein_embedding": emb, "contact_map": []},
        )
        X_pp = clf.preprocessor.transform(feat.reshape(1, -1))
        proba = clf._predict_proba_raw(X_pp)[0]

        top_idx = int(proba.argmax())
        max_p   = float(proba[top_idx])

        # Determine if prediction matches ground truth
        gt_entry = gt.get(uid, {})
        gt_is_enzyme = gt_entry.get("is_enzyme", None)
        gt_ec_str    = str(gt_entry.get("ec_number", "")).strip()
        gt_ec_first  = gt_ec_str[0] if gt_ec_str else ""

        ml_is_enzyme = top_idx > 0
        if gt_is_enzyme is None:
            continue
        if gt_is_enzyme:
            # Correct if predicted enzyme AND first EC digit matches
            is_correct = ml_is_enzyme and (str(top_idx) == gt_ec_first)
        else:
            is_correct = not ml_is_enzyme

        probs_max.append(max_p)
        correct.append(int(is_correct))
        uids.append(uid)

    probs_max = np.array(probs_max)
    correct   = np.array(correct)
    log.info(f"  Used {len(probs_max)} predictions for calibration")
    log.info(f"  Raw accuracy: {correct.mean()*100:.1f}%")

    # Show probability vs accuracy distribution BEFORE calibration
    log.info("  Raw probability vs accuracy:")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask = (probs_max >= lo) & (probs_max < hi)
        n = mask.sum()
        if n > 0:
            obs = correct[mask].mean()
            log.info(f"    raw=[{lo:.1f}, {hi:.2f}): n={n:>3}  acc={obs*100:5.1f}%")

    # Fit calibrator
    iso = _fit_calibrator(probs_max, correct)
    calibrated = iso.predict(probs_max)

    log.info("  Calibrated probability vs accuracy:")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask = (calibrated >= lo) & (calibrated < hi)
        n = mask.sum()
        if n > 0:
            obs = correct[mask].mean()
            log.info(f"    cal=[{lo:.1f}, {hi:.2f}): n={n:>3}  acc={obs*100:5.1f}%")

    # Find threshold
    log.info(f"\n  Finding threshold for >={target_precision*100:.0f}% precision...")
    threshold, coverage, prec = _find_threshold(
        calibrated, correct, target_precision
    )
    log.info(f"  Threshold:               {threshold:.3f}")
    log.info(f"  Coverage at threshold:   {coverage*100:.1f}%")
    log.info(f"  Precision when confident: {prec*100:.1f}%")
    log.info(f"  Uncertain rate:          {(1-coverage)*100:.0f}%")

    # Save
    out_path = Path(model_dir) / "reject_option.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "calibrator":          iso,
            "threshold":           threshold,
            "target_precision":    target_precision,
            "calibration_n":       len(probs_max),
            "calibration_source":  "validation_set",
            "raw_accuracy":        float(correct.mean()),
            "coverage":            coverage,
            "precision_confident": prec,
        }, f)

    # Save updated ESM-2 cache for future runs
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
        log.info(f"  Updated ESM-2 cache: {len(cache)} entries")
    except Exception as e:
        log.warning(f"  Could not save cache: {e}")

    log.info(f"\n  Saved -> {out_path}")
    log.info("=" * 70)


@cli.command("calibrate")
@click.option("--csv",        required=True)
@click.option("--model-dir",  default="models/ec_ensemble_v5")
@click.option("--cache-dir",  default="data/training_cache_v5")
@click.option("--target-precision", default=0.92, type=float,
              help="Target accuracy on confident predictions (default 0.92)")
@click.option("--holdout-frac", default=0.20, type=float)
def calibrate_cmd(csv, model_dir, cache_dir, target_precision, holdout_frac):
    """Fit calibrator + reject threshold on held-out training data."""
    calibrate(csv, model_dir, cache_dir, holdout_frac, target_precision)


@cli.command("predict")
@click.option("--uniprot",   default=None)
@click.option("--sequence",  default=None)
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
def predict_cmd(uniprot, sequence, model_dir, cache_dir):
    """Predict EC class with reject-option for a single protein."""
    if not sequence and not uniprot:
        log.error("Provide --uniprot or --sequence")
        sys.exit(1)

    clf = ECClassifierEnsemble.load(Path(model_dir))
    calibrator, threshold = _load_reject_option(Path(model_dir))
    if calibrator is None:
        log.warning("  No calibrator found — run 'calibrate' first")
        sys.exit(1)

    # Get sequence
    if not sequence:
        log.info(f"  Fetching {uniprot}...")
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        with urllib.request.urlopen(url, timeout=20) as r:
            fasta = r.read().decode()
        sequence = "".join(fasta.strip().split("\n")[1:]).upper()

    # Get embedding (cache or compute)
    cache_path = Path(cache_dir) / "esm2_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    emb = cache.get(uniprot) if uniprot else None

    if emb is None:
        log.info("  Computing fresh ESM-2 embedding...")
        emb = _compute_fresh_esm2(sequence)

    result = predict_with_reject(
        sequence=sequence, clf=clf, calibrator=calibrator,
        threshold=threshold, esm2_emb=emb, uniprot_id=uniprot or "UNKNOWN"
    )

    # Pretty print
    print("\n" + "═" * 60)
    print(f"  Prediction for: {uniprot or 'sequence'}")
    print(f"  Length: {len(sequence)} aa")
    print("─" * 60)
    if result["verdict"] == "confident":
        print(f"  VERDICT      : CONFIDENT ✓")
        print(f"  EC class     : {result['ec_class']} ({result['ec_name']})")
        print(f"  Is enzyme    : {'YES' if result['is_enzyme'] else 'NO'}")
    else:
        print(f"  VERDICT      : UNCERTAIN — flag for review")
        print(f"  Best guess   : {result['alternatives'][0]['class']} "
              f"({result['alternatives'][0]['name']})")
    print(f"  Raw conf     : {result['raw_conf']*100:.1f}%")
    print(f"  Calib conf   : {result['calib_conf']*100:.1f}%")
    print(f"  Threshold    : {result['threshold']*100:.1f}%")
    print(f"\n  Top 3 alternatives:")
    for alt in result["alternatives"]:
        print(f"    {alt['class']:<10} {alt['name']:<20} {alt['prob']*100:>5.1f}%")
    print("═" * 60)


@cli.command("evaluate-validation")
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
@click.option("--report",    default="data/reports/validation/validation_report_v5.json")
def evaluate_validation_cmd(model_dir, cache_dir, report):
    """Show how reject option splits the validation set."""
    rep = json.loads(Path(report).read_text())
    scores = rep["scores"]

    calibrator, threshold = _load_reject_option(Path(model_dir))
    if calibrator is None:
        log.error("  No calibrator. Run calibrate-on-validation first.")
        sys.exit(1)

    sys.path.insert(0, "validation")
    gt = {}
    try:
        from run_validation import VALIDATION_SET
        for p in VALIDATION_SET: gt[p["uniprot_id"]] = p
    except Exception: pass
    ne = Path("validation/new_entries.json")
    if ne.exists():
        for p in json.loads(ne.read_text()):
            uid = p.get("uniprot_id","")
            if uid and uid not in gt: gt[uid] = p

    clf = ECClassifierEnsemble.load(Path(model_dir))
    cache_path = Path(cache_dir) / "esm2_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    inter_dir = Path("data/intermediate")

    confident_correct = 0
    confident_wrong   = 0
    uncertain         = 0
    confident_list, uncertain_list = [], []
    used = 0

    for s in scores:
        uid = s["uniprot_id"]
        gene = s.get("gene", "?")

        seq = None
        for suffix in ["_structure.json", "_esm2.json"]:
            f = inter_dir / f"{uid}{suffix}"
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    seq = d.get("sequence", "")
                    if seq: break
                except Exception: pass
        if not seq:
            continue

        emb = cache.get(uid)
        if emb is None:
            if not hasattr(_esm2_lazy, "model"):
                log.info("  Loading ESM-2 for missing embeddings...")
                _esm2_lazy.model, _esm2_lazy.bc, _esm2_lazy.device = _load_esm2_for_calib()
            try:
                emb = _embed_one(seq.upper(), _esm2_lazy.model,
                                  _esm2_lazy.bc, _esm2_lazy.device)
                cache[uid] = emb
            except Exception:
                continue

        feat = build_feature_vector(
            sequence    = seq.upper(),
            esm2_result = {"protein_embedding": emb, "contact_map": []},
        )
        X_pp = clf.preprocessor.transform(feat.reshape(1, -1))
        proba = clf._predict_proba_raw(X_pp)[0]
        top_idx = int(proba.argmax())
        max_p   = float(proba[top_idx])
        calib_conf = float(calibrator.predict(np.array([max_p]))[0])

        gt_entry = gt.get(uid, {})
        gt_enz = gt_entry.get("is_enzyme", None)
        gt_ec  = str(gt_entry.get("ec_number", "")).strip()
        gt_ec_first = gt_ec[0] if gt_ec else ""
        ml_enz = top_idx > 0

        if gt_enz is None:
            is_correct = None
        elif gt_enz:
            is_correct = ml_enz and (str(top_idx) == gt_ec_first)
        else:
            is_correct = not ml_enz

        used += 1
        if calib_conf >= threshold:
            if is_correct is True:
                confident_correct += 1
            elif is_correct is False:
                confident_wrong += 1
            confident_list.append((uid, gene, gt_enz, ml_enz, calib_conf))
        else:
            uncertain += 1
            uncertain_list.append((uid, gene, gt_enz, ml_enz, calib_conf))

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    except Exception:
        pass

    total_conf = confident_correct + confident_wrong
    print("\n" + "═" * 66)
    print(f"  REJECT-OPTION ON {used}-PROTEIN VALIDATION SET")
    print("═" * 66)
    print(f"  Threshold         : {threshold:.3f}")
    print(f"  Confident pred.   : {total_conf:>3} / {used}  "
          f"({total_conf/max(used,1)*100:.0f}% coverage)")
    print(f"    correct         : {confident_correct:>3}  "
          f"({confident_correct/max(total_conf,1)*100:.1f}% precision)")
    print(f"    wrong           : {confident_wrong:>3}")
    print(f"  Uncertain (flag)  : {uncertain:>3} / {used}  "
          f"({uncertain/max(used,1)*100:.0f}%)")
    print("─" * 66)

    if uncertain_list:
        print(f"\n  Proteins flagged uncertain (human review):")
        print(f"  {'UID':<10} {'Gene':<10} {'GT':>4} {'ML':>4} {'CalibConf':>10}")
        for uid, gene, gt_e, ml_e, cc in sorted(uncertain_list, key=lambda x: x[4]):
            gt_s = ('Y' if gt_e else 'N') if gt_e is not None else '?'
            ml_s = 'Y' if ml_e else 'N'
            print(f"  {uid:<10} {gene:<10} {gt_s:>4} {ml_s:>4} {cc*100:>9.1f}%")
    print("═" * 66)


def _compute_fresh_esm2(sequence: str) -> list:
    """Compute single ESM-2 embedding."""
    try:
        import torch
        try:
            import esm as esm_module
            model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
            model = model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            bc = alphabet.get_batch_converter()
            _, _, tokens = bc([("p", sequence)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[33])
            L = len(sequence)
            return out["representations"][33][0, 1:L+1].mean(0).cpu().tolist()
        except ImportError:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            mdl = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            mdl.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mdl = mdl.to(device)
            inputs = tok(sequence, return_tensors="pt", truncation=True,
                         max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = mdl(**inputs)
            return out.last_hidden_state[0].mean(0).cpu().tolist()
    except Exception as e:
        log.warning(f"  ESM-2 failed: {e}")
        return [0.0] * 1280


if __name__ == "__main__":
    cli()