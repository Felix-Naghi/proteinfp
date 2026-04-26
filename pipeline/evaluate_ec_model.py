"""
evaluate_ec_model.py
─────────────────────
Evaluation tool for the trained EC classifier ensemble.

Three modes:

  1. HOLDOUT   — evaluate on a CSV file (your training CSV or a separate test CSV)
  2. SINGLE    — predict a single protein by UniProt ID or raw sequence
  3. CONFUSION — pretty-print the saved confusion matrix from training

Usage:

  # Evaluate on a held-out test CSV (stratified split from your training data)
  python evaluate_ec_model.py holdout \\
      --csv data/swissprot_curated.csv \\
      --model-dir models/ec_ensemble_v2 \\
      --test-fraction 0.15

  # Predict a single protein by UniProt ID (fetches sequence automatically)
  python evaluate_ec_model.py single --uniprot P04637 --model-dir models/ec_ensemble_v2

  # Predict a single protein from a raw sequence string
  python evaluate_ec_model.py single \\
      --sequence MKTAYIAKQRQISFVKTTAMQFYEIG... \\
      --model-dir models/ec_ensemble_v2

  # Show confusion matrix from your last training run
  python evaluate_ec_model.py confusion --model-dir models/ec_ensemble_v2

IMPORTANT — ESM-2 and the holdout test:
  The model was trained WITH ESM-2 embeddings. If you run holdout without
  pointing it at the training ESM-2 cache, the 128 ESM-2 feature dims will
  be zeros and accuracy will collapse (~42% instead of ~69%).

  Always pass --cache-dir so holdout can reuse the pre-computed embeddings:

  python pipeline/evaluate_ec_model.py holdout \\
      --csv data/swissprot_curated.csv \\
      --model-dir models/ec_ensemble_v2 \\
      --cache-dir data/training_cache_v2 \\
      --test-fraction 0.15 --seed 99
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import ssl
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import certifi
import click
import numpy as np

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ml_ec_classifier import ECClassifierEnsemble
from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

LABEL_MAP   = {"non-enzyme": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
CLASS_NAMES = ["non-enz", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]
EC_NAMES    = {
    "0": "Non-enzyme",
    "1": "Oxidoreductase",
    "2": "Transferase",
    "3": "Hydrolase",
    "4": "Lyase",
    "5": "Isomerase",
    "6": "Ligase",
    "7": "Translocase",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_sequence(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={
        "User-Agent": "ProteinFP-eval/1.0",
        "Accept": "text/x-fasta",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        fasta = r.read().decode()
    lines = fasta.strip().split("\n")
    return "".join(lines[1:]).upper()


def _get_esm2_embedding(sequence: str) -> list[float]:
    """Compute a single ESM-2 embedding."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        log.info("  Loading ESM-2 for embedding...")
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
        log.info(f"  ESM-2 embedding computed ({len(emb)}-dim)")
        return emb
    except Exception as e:
        log.warning(f"  ESM-2 unavailable ({e}), using zeros")
        return [0.0] * 1280


def _load_clf(model_dir: str) -> ECClassifierEnsemble:
    clf = ECClassifierEnsemble.load(Path(model_dir))
    log.info(f"  Model loaded from {model_dir}/")
    return clf


def _load_esm2_cache(cache_dir: Optional[str]) -> dict:
    """Load the pre-computed ESM-2 embedding cache from training."""
    if not cache_dir:
        return {}
    cache_path = Path(cache_dir) / "esm2_cache.json"
    if cache_path.exists():
        log.info(f"  Loading ESM-2 cache: {cache_path} ...")
        try:
            cache = json.loads(cache_path.read_text())
            log.info(f"  {len(cache):,} cached embeddings loaded")
            return cache
        except Exception as e:
            log.warning(f"  Could not load ESM-2 cache: {e}")
    else:
        log.warning(f"  ESM-2 cache not found at {cache_path}")
        log.warning("  Features will use zero embeddings — accuracy will be lower!")
    return {}


def _feature_from_sequence(sequence: str, esm2_cache: dict,
                            uid: str, compute_fresh: bool) -> np.ndarray:
    """Build feature vector, using cache first, then computing fresh if needed."""
    emb = esm2_cache.get(uid)
    if emb is None and compute_fresh:
        emb = _get_esm2_embedding(sequence)
    esm2_result = {"protein_embedding": emb, "contact_map": []} if emb else None
    return build_feature_vector(sequence=sequence, esm2_result=esm2_result)


# ── Mode 1: Holdout evaluation ────────────────────────────────────────────────

def _run_holdout(csv_path: str, model_dir: str, test_fraction: float,
                 seed: int, cache_dir: Optional[str], compute_fresh_esm2: bool) -> None:
    """Split CSV into train/test, evaluate model on test set."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )

    random.seed(seed)

    # Load ESM-2 cache from training
    esm2_cache = _load_esm2_cache(cache_dir)
    n_cached = len(esm2_cache)

    # Load CSV
    log.info(f"  Loading {csv_path}...")
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uniprot_id", "").strip()
            seq = row.get("sequence", "").strip().upper()
            ec  = row.get("ec_class", "").strip()
            if uid and seq and ec in LABEL_MAP:
                rows.append((uid, seq, LABEL_MAP[ec]))

    log.info(f"  {len(rows):,} samples loaded")

    # Stratified split — test set only
    labels = [r[2] for r in rows]
    _, test_rows, _, _ = train_test_split(
        rows, labels, test_size=test_fraction, stratify=labels, random_state=seed
    )
    log.info(f"  Test set: {len(test_rows)} samples")

    # Coverage check
    covered = sum(1 for uid, _, _ in test_rows if uid in esm2_cache)
    log.info(f"  ESM-2 cache coverage: {covered}/{len(test_rows)} test proteins "
             f"({'%.0f' % (100*covered/max(len(test_rows),1))}%)")
    if covered == 0 and not compute_fresh_esm2:
        log.warning("  *** No ESM-2 embeddings available for test set! ***")
        log.warning("  *** Pass --cache-dir data/training_cache_v2 to fix this. ***")
        log.warning("  *** Accuracy will be significantly lower than training.   ***")

    # Build features
    clf = _load_clf(model_dir)
    log.info(f"  Building features...")

    X_list, y_true, uids = [], [], []
    for i, (uid, seq, label) in enumerate(test_rows):
        if i % 100 == 0 and i > 0:
            log.info(f"    {i}/{len(test_rows)} features built...")
        feat = _feature_from_sequence(seq, esm2_cache, uid, compute_fresh_esm2)
        X_list.append(feat)
        y_true.append(label)
        uids.append(uid)

    X = np.array(X_list, dtype=np.float32)
    X_pp = clf.preprocessor.transform(X)
    proba = clf._predict_proba_raw(X_pp)
    y_pred = proba.argmax(axis=1)
    y_true = np.array(y_true)

    # ── Metrics ───────────────────────────────────────────────────────────────
    top1  = accuracy_score(y_true, y_pred)
    top2  = np.mean([y_true[i] in proba[i].argsort()[-2:] for i in range(len(y_true))])
    mac_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wei_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    enz_true = (y_true > 0).astype(int)
    enz_pred = (y_pred > 0).astype(int)
    bin_acc  = accuracy_score(enz_true, enz_pred)

    print("\n" + "═" * 66)
    print("  EVALUATION RESULTS")
    print("═" * 66)
    print(f"  EC top-1 accuracy  : {top1*100:.2f}%")
    print(f"  EC top-2 accuracy  : {top2*100:.2f}%")
    print(f"  Macro F1           : {mac_f1:.4f}")
    print(f"  Weighted F1        : {wei_f1:.4f}")
    print(f"  Binary enzyme acc  : {bin_acc*100:.2f}%")
    print(f"  Test samples       : {len(y_true)}")
    print("─" * 66)

    # Per-class report
    present = sorted(set(y_true))
    names_present = [CLASS_NAMES[i] for i in present]
    print(classification_report(
        y_true, y_pred,
        labels=present, target_names=names_present, zero_division=0
    ))

    # Confusion matrix (text)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "         " + "  ".join(f"{n:>7s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:>8s} " + "  ".join(f"{v:>7d}" for v in cm[i])
        print(row)

    # Worst predictions
    errors = [(uids[i], int(y_true[i]), int(y_pred[i]), float(proba[i].max()))
              for i in range(len(y_true)) if y_pred[i] != y_true[i]]
    errors.sort(key=lambda x: -x[3])   # highest-confidence wrong predictions first

    print(f"\n  Top 10 most-confident mistakes:")
    print(f"  {'UniProt':<12} {'True':>8} {'Predicted':>12} {'Confidence':>12}")
    print("  " + "-" * 46)
    for uid, tr, pr, conf in errors[:10]:
        print(f"  {uid:<12} {CLASS_NAMES[tr]:>8} {CLASS_NAMES[pr]:>12} {conf*100:>11.1f}%")

    print("═" * 66)

    # Save results JSON
    out = Path(model_dir) / "evaluation_results.json"
    result_data = {
        "top1_accuracy":    round(top1, 4),
        "top2_accuracy":    round(top2, 4),
        "macro_f1":         round(mac_f1, 4),
        "weighted_f1":      round(wei_f1, 4),
        "binary_enzyme_acc": round(bin_acc, 4),
        "n_test":           len(y_true),
        "confusion_matrix": cm.tolist(),
        "class_names":      CLASS_NAMES,
        "top_errors":       [
            {"uniprot_id": uid, "true": CLASS_NAMES[tr], "predicted": CLASS_NAMES[pr],
             "confidence": round(conf, 4)}
            for uid, tr, pr, conf in errors[:20]
        ],
    }
    out.write_text(json.dumps(result_data, indent=2))
    log.info(f"  Results saved to {out}")


# ── Mode 2: Single protein prediction ────────────────────────────────────────

def _run_single(uniprot_id: Optional[str], sequence: Optional[str],
                model_dir: str, cache_dir: Optional[str]) -> None:
    """Predict EC class for a single protein."""
    if not sequence and not uniprot_id:
        log.error("Provide either --uniprot or --sequence")
        sys.exit(1)

    clf = _load_clf(model_dir)
    esm2_cache = _load_esm2_cache(cache_dir)

    # Get sequence
    if not sequence:
        log.info(f"  Fetching sequence for {uniprot_id}...")
        try:
            sequence = _fetch_sequence(uniprot_id)
            log.info(f"  Sequence length: {len(sequence)} aa")
        except Exception as e:
            log.error(f"  Could not fetch {uniprot_id}: {e}")
            sys.exit(1)

    sequence = sequence.strip().upper()

    # Get embedding — from cache first, then compute fresh
    emb = esm2_cache.get(uniprot_id) if uniprot_id else None
    if emb:
        log.info(f"  ESM-2: loaded from cache")
        esm2_result = {"protein_embedding": emb, "contact_map": []}
    else:
        log.info(f"  ESM-2: not in cache — computing fresh (this takes ~10s)...")
        fresh_emb = _get_esm2_embedding(sequence)
        esm2_result = {"protein_embedding": fresh_emb, "contact_map": []}
        emb_source = "computed fresh"

    # Build features
    t0 = time.time()
    feat = build_feature_vector(sequence=sequence, esm2_result=esm2_result)
    X_pp = clf.preprocessor.transform(feat.reshape(1, -1))
    proba = clf._predict_proba_raw(X_pp)[0]
    elapsed_ms = (time.time() - t0) * 1000

    # Interpret
    non_enz_prob = float(proba[0])
    enzyme_prob  = float(proba[1:].sum())
    is_enzyme    = enzyme_prob > 0.5

    sorted_preds = sorted(
        [(i, float(proba[i])) for i in range(8)],
        key=lambda x: -x[1]
    )

    # Pretty output
    print("\n" + "═" * 60)
    if uniprot_id:
        print(f"  Prediction for: {uniprot_id}")
    print(f"  Sequence length: {len(sequence)} aa")
    esm2_used = bool(esm2_result and esm2_result.get("protein_embedding") and
                     any(v != 0 for v in esm2_result["protein_embedding"][:10]))
    print(f"  ESM-2: {'from cache' if emb else 'computed fresh'}")
    print("─" * 60)
    print(f"  Is enzyme  : {'YES' if is_enzyme else 'NO'}  "
          f"(enzyme p={enzyme_prob:.3f} | non-enzyme p={non_enz_prob:.3f})")
    print("─" * 60)
    print(f"  {'Rank':<6} {'Class':<14} {'EC Name':<20} {'Probability':>12}")
    print(f"  {'─'*4}   {'─'*12}   {'─'*18}   {'─'*10}")

    for rank, (cls_idx, prob) in enumerate(sorted_preds, 1):
        marker = " ◄ top" if rank == 1 else ""
        label = CLASS_NAMES[cls_idx]
        name  = EC_NAMES.get(str(cls_idx), "Unknown")
        print(f"  {rank:<6} {label:<14} {name:<20} {prob*100:>10.2f}%{marker}")
        if rank >= 5 and prob < 0.02:
            break

    print("─" * 60)
    print(f"  Inference time: {elapsed_ms:.1f} ms  |  Feature dim: {len(feat)}")
    print("═" * 60)


# ── Mode 3: Confusion matrix display ─────────────────────────────────────────

def _run_confusion(model_dir: str) -> None:
    """Pretty-print the confusion matrix saved during training."""
    txt_path  = Path(model_dir) / "confusion_matrix.txt"
    json_path = Path(model_dir) / "confusion_matrix.json"

    if txt_path.exists():
        print("\n" + txt_path.read_text())
    elif json_path.exists():
        data = json.loads(json_path.read_text())
        cm   = np.array(data["matrix"])
        labels = data["labels"]
        print("\nConfusion matrix (rows=true, cols=predicted):")
        header = "         " + "  ".join(f"{n:>7s}" for n in labels)
        print(header)
        for i, name in enumerate(labels):
            row = f"{name:>8s} " + "  ".join(f"{v:>7d}" for v in cm[i])
            print(row)
        # Per-row accuracy
        print("\nPer-class recall (diagonal / row sum):")
        for i, name in enumerate(labels):
            total = cm[i].sum()
            if total > 0:
                recall = cm[i, i] / total
                bar = "█" * int(recall * 20)
                print(f"  {name:>8s}  {recall*100:5.1f}%  {bar}")
    else:
        print(f"No confusion matrix found in {model_dir}/")
        print("Run training first, or run: evaluate_ec_model.py holdout ...")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """EC classifier evaluation tool."""
    pass


@cli.command("holdout")
@click.option("--csv",           required=True, help="Training/test CSV")
@click.option("--model-dir",     default="models/ec_ensemble_v2")
@click.option("--cache-dir",     default="data/training_cache_v2",
              help="ESM-2 cache dir used during training (REQUIRED for correct results)")
@click.option("--test-fraction", default=0.15, type=float)
@click.option("--seed",          default=99,   type=int,
              help="Random seed — use a DIFFERENT seed than training (42)")
@click.option("--fresh-esm2",    is_flag=True, default=False,
              help="Compute fresh ESM-2 for proteins not in cache (very slow)")
def holdout_cmd(csv, model_dir, cache_dir, test_fraction, seed, fresh_esm2):
    """Evaluate model on a held-out portion of a CSV."""
    _run_holdout(csv, model_dir, test_fraction, seed, cache_dir, fresh_esm2)


@cli.command("single")
@click.option("--uniprot",   default=None, help="UniProt ID (e.g. P04637)")
@click.option("--sequence",  default=None, help="Raw amino acid sequence")
@click.option("--model-dir", default="models/ec_ensemble_v2")
@click.option("--cache-dir", default="data/training_cache_v2",
              help="ESM-2 cache dir — proteins in cache get instant embeddings")
def single_cmd(uniprot, sequence, model_dir, cache_dir):
    """Predict EC class for a single protein."""
    _run_single(uniprot, sequence, model_dir, cache_dir)


@cli.command("confusion")
@click.option("--model-dir", default="models/ec_ensemble_v2")
def confusion_cmd(model_dir):
    """Display confusion matrix from last training run."""
    _run_confusion(model_dir)


if __name__ == "__main__":
    cli()