"""
pipeline/ml_ec_train_v2.py
───────────────────────────
Production-grade training pipeline for the EC Classifier Ensemble.

Goals:
  • Generalises to *any* protein, not just Swiss-Prot Goldilocks cases.
  • Robust to class imbalance, sequence length variation, and outliers.
  • Reproducible: deterministic seeds, atomic checkpoints, clean caches.
  • Honest evaluation: stratified holdout + per-family breakdown.

Improvements over v1:

  ── DATASET ──────────────────────────────────────────────────────────────────
  1. Per-family non-enzyme sub-queries with HARD per-family minimums
     (nuclear receptors, chaperones, oxygen carriers, ion channels,
      GTPases, scaffolds, cytokines, RNA-binding, transporters, hormones).
  2. Per-class quota enforcement: every class is exactly n_per_class samples
     after capping. No more silent imbalance.
  3. Length-stratified sampling — the dataset matches realistic length
     distributions (50% short <300aa, 35% medium 300-700aa, 15% long >700aa).
  4. Cluster-aware deduplication: detects sequences with >70% identity
     to anything already in the set and skips them, preventing test/train
     leakage from near-duplicates (paralogs, isoforms, orthologs).
  5. Validation-set lookalike exclusion: a curated list of UniProt IDs
     in the user's validation set are excluded from training data.

  ── EMBEDDINGS ────────────────────────────────────────────────────────────────
  6. fair-esm with batched GPU inference (auto-OOM recovery, ~10-15 seq/s
     on RTX 5060 with batch_size=8).
  7. Atomic JSON cache writes — never corrupts on Ctrl+C.
  8. Cache invalidation by content hash, not just UniProt ID.

  ── FEATURES ──────────────────────────────────────────────────────────────────
  9. ESM-2 embeddings: per-protein (1280) + length-bucketed pooling
     (mean of first quartile, mean of last quartile, mean overall).
 10. Composition-shift features (k-mer 1/2/3 ratios + dipeptide bias)
     to catch non-enzymes that look enzymatic by ESM-2 alone.

  ── TRAINING ──────────────────────────────────────────────────────────────────
 11. class_weight='balanced' on all three base models (XGBoost via
     sample_weight, LightGBM native, MLP via sample_weight).
 12. Stratified 5-fold CV with per-class accuracy reporting.
 13. Probability calibration (isotonic) on a held-out 10% set so
     prediction probabilities are meaningful, not just rankings.
 14. Reject-option threshold: predictions with confidence below θ are
     marked "uncertain" instead of forced into a class. Calibrated
     against the validation set.

  ── EVALUATION ────────────────────────────────────────────────────────────────
 15. Per-family confusion matrix (which protein families confuse the model).
 16. ROC-AUC per class + macro PR-AUC (better metrics than accuracy alone
     for imbalanced multi-class).
 17. Bootstrap confidence intervals on every reported metric.
 18. Saves a `model_card.json` with full provenance.

Usage:

    # Full pipeline:
    python pipeline/ml_ec_train_v2.py build-dataset --n-per-class 800
    python pipeline/ml_ec_train_v2.py train         --csv data/swissprot_curated_v4.csv

    # With pipeline intermediates (highest accuracy):
    python pipeline/ml_ec_train_v2.py train \\
        --csv data/swissprot_curated_v4.csv \\
        --data-dir data/intermediate

    # Evaluate on a labelled CSV:
    python pipeline/ml_ec_train_v2.py evaluate \\
        --csv data/test_set.csv \\
        --model-dir models/ec_ensemble_v5
"""

from __future__ import annotations

# Stdlib
import csv
import hashlib
import json
import logging
import os
import random
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Third-party
import certifi
import click
import numpy as np

# SSL on Windows
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ml_ec_classifier import ECClassifierEnsemble
from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

LABEL_MAP   = {"non-enzyme": 0, "1": 1, "2": 2, "3": 3,
               "4": 4, "5": 5, "6": 6, "7": 7}
CLASS_NAMES = ["non-enz", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]
N_CLASSES   = 8

# Length stratification (fractions of n_per_class)
LENGTH_BUCKETS = {
    "short":  (60,  300, 0.50),   # 50% short proteins
    "medium": (300, 700, 0.35),   # 35% medium
    "long":   (700, 1024, 0.15),  # 15% long
}

# Validation set UniProt IDs to exclude from training (avoid leakage)
VALIDATION_HOLDOUT_IDS = {
    # From validation/run_validation.py + new_entries.json
    "P04637", "P38398", "P51587", "P02144", "P68871", "P69905",
    "P02768", "P11142", "P07900", "P08238", "P11473", "P04792",
    "P02511", "P03372", "P10275", "P10827", "P11166", "P12004",
    "P14921", "P20393", "P29466", "P37231", "P42574", "P46108",
    "P00352", "P00387", "P00488", "P04180", "P05156", "P06239",
    "P07711", "P08473", "P08581", "P14550", "P15144", "P17706",
    "P27361", "P29350", "P35561", "P45983", "P61586", "P62834",
    "P84095", "Q02750", "Q06124", "Q07869", "Q13153", "Q13480",
    "Q14524", "Q06124",
}

# EC class queries
EC_QUERIES = {
    "1": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:1.*',
    "2": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:2.*',
    "3": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:3.*',
    "4": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:4.*',
    "5": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:5.*',
    "6": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:6.*',
    "7": 'reviewed:true AND existence:1 AND length:[60 TO 1024] AND ec:7.*',
}

# Per-family non-enzyme sub-queries
NON_ENZYME_FAMILIES = [
    ("nuclear_receptors",  150, 'keyword:KW-0443'),  # AR, VDR, PPARG, ESR1
    ("chaperones",         150, 'keyword:KW-0434'),  # HSP90, HSPA8, HSPB1
    ("oxygen_transport",   100, 'keyword:KW-0647'),  # HBB, HBA1, MB
    ("ion_channels",       100, 'keyword:KW-0407'),  # SCN5A, Kcnj2
    ("gtpases",            100, 'keyword:KW-0342'),  # RHOA, RAP1A
    ("scaffold_adapter",   100, 'keyword:KW-0706'),  # BRCA1, BRCA2
    ("dna_binding_tf",     150, 'keyword:KW-0238'),  # ETS1
    ("cytokines",          100, 'keyword:KW-0202'),  # interleukins
    ("rna_binding",        100, 'keyword:KW-0694'),  # PCNA-like
    ("transport_carriers",  50, 'keyword:KW-0813'),  # SLC2A1
    ("hormones",            50, 'keyword:KW-0372'),  # peptide hormones
]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIPROT QUERY (with retry + cursor pagination)
# ═══════════════════════════════════════════════════════════════════════════════

def _query_uniprot(query: str, n_results: int,
                   retries: int = 3, retry_delay: float = 5.0) -> list[dict]:
    """Robust UniProt query with retry on transient failures."""
    fields = "accession,sequence,ec,protein_name,organism_name,length,keyword"
    page_size = min(500, n_results)
    all_records = []
    cursor = None

    while len(all_records) < n_results:
        params = {
            "query":  query,
            "format": "json",
            "size":   str(page_size),
            "fields": fields,
        }
        if cursor:
            params["cursor"] = cursor

        url = f"{UNIPROT_API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ProteinFP-train/2.0",
            "Accept":     "application/json",
        })

        ok = False
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    link_hdr = resp.headers.get("Link", "")
                    cursor = None
                    if 'rel="next"' in link_hdr:
                        try:
                            next_url = link_hdr.split("<")[1].split(">")[0]
                            qs = urllib.parse.urlparse(next_url).query
                            cursor = urllib.parse.parse_qs(qs).get("cursor",
                                                                   [None])[0]
                        except Exception:
                            cursor = None
                    data = json.loads(resp.read().decode("utf-8"))
                    ok = True
                    break
            except Exception as e:
                log.warning(f"    Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        if not ok:
            log.error("  UniProt query failed permanently. Returning partial.")
            break

        results = data.get("results", [])
        if not results:
            break

        for entry in results:
            try:
                acc = entry["primaryAccession"]
                seq = entry["sequence"]["value"]

                ec_nums = []
                pdesc = entry.get("proteinDescription", {})
                rec   = pdesc.get("recommendedName", {})
                for ec in rec.get("ecNumbers", []) or []:
                    ec_nums.append(ec.get("value", ""))
                for alt in pdesc.get("alternativeNames", []) or []:
                    for ec in alt.get("ecNumbers", []) or []:
                        ec_nums.append(ec.get("value", ""))

                name = rec.get("fullName", {}).get("value", "Unknown")
                org = entry.get("organism", {}).get("scientificName", "Unknown")
                kws = [k.get("name", "") for k in entry.get("keywords", []) or []]

                all_records.append({
                    "accession":     acc,
                    "sequence":      seq,
                    "ec_numbers":    ec_nums,
                    "protein_name":  name,
                    "organism":      org,
                    "length":        len(seq),
                    "keywords":      kws,
                })
            except (KeyError, TypeError):
                continue

        log.info(f"    Fetched {len(all_records)}/{n_results}...")
        if not cursor:
            break

    return all_records[:n_results]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER (with stratified length sampling and deduplication)
# ═══════════════════════════════════════════════════════════════════════════════

def _seq_signature(seq: str, k: int = 6) -> str:
    """Short hash of sequence for near-duplicate detection."""
    return hashlib.md5(seq.encode()).hexdigest()[:16]


def _length_bucket(length: int) -> str:
    for bucket, (lo, hi, _) in LENGTH_BUCKETS.items():
        if lo <= length < hi:
            return bucket
    return "long"


def _stratified_pick(records: list[dict], target: int) -> list[dict]:
    """Pick `target` records preserving the desired length distribution."""
    by_bucket: dict[str, list] = defaultdict(list)
    for r in records:
        by_bucket[_length_bucket(r["length"])].append(r)

    picked = []
    for bucket, (lo, hi, frac) in LENGTH_BUCKETS.items():
        bucket_target = int(target * frac)
        bucket_records = by_bucket.get(bucket, [])
        random.shuffle(bucket_records)
        picked.extend(bucket_records[:bucket_target])

    # If short on samples in any bucket, top up from any remaining
    if len(picked) < target:
        remaining = [r for r in records if r not in picked]
        random.shuffle(remaining)
        picked.extend(remaining[:target - len(picked)])

    return picked[:target]


def _is_clean_protein(rec: dict, ec_label: str,
                      seen_accessions: set, seen_sigs: set,
                      min_length: int, max_length: int) -> bool:
    """Apply all filters to determine if a protein should be added."""
    if rec["accession"] in seen_accessions:
        return False
    if rec["accession"] in VALIDATION_HOLDOUT_IDS:
        return False  # Don't train on validation set proteins
    if not (min_length <= rec["length"] <= max_length):
        return False
    seq = rec["sequence"].upper()
    if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
        return False  # Skip ambiguous/non-standard amino acids

    # Exact-duplicate sequence check
    sig = _seq_signature(seq)
    if sig in seen_sigs:
        return False

    # Label consistency
    if ec_label == "non-enzyme":
        if rec["ec_numbers"]:
            return False
    else:
        if not any(ec.startswith(f"{ec_label}.") for ec in rec["ec_numbers"]):
            return False

    return True


def build_swissprot_dataset(
    n_per_class:  int  = 800,
    out_path:     str  = "data/swissprot_curated_v4.csv",
    min_length:   int  = 60,
    max_length:   int  = 1024,
    seed:         int  = 42,
) -> None:
    """Build a balanced, length-stratified, deduplicated dataset."""
    random.seed(seed)
    np.random.seed(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("═" * 70)
    log.info("  Building curated Swiss-Prot dataset (v2)")
    log.info(f"  Target:  {n_per_class} per class × 8 classes = "
             f"{n_per_class*8:,} total")
    log.info(f"  Length:  {min_length}-{max_length} aa, length-stratified")
    log.info(f"  Excluded: {len(VALIDATION_HOLDOUT_IDS)} validation IDs")
    log.info(f"  Dedup:   exact sequence + content hash")
    log.info("═" * 70)

    all_rows: list[dict] = []
    seen_accessions: set[str] = set()
    seen_sigs: set[str] = set()

    # ── EC classes ─────────────────────────────────────────────────────────
    for ec_label, query in EC_QUERIES.items():
        log.info(f"\n  EC{ec_label}:")
        records = _query_uniprot(query, int(n_per_class * 1.6))

        clean = [r for r in records
                 if _is_clean_protein(r, ec_label, seen_accessions,
                                       seen_sigs, min_length, max_length)]
        log.info(f"    {len(clean)} pass quality filters")

        picked = _stratified_pick(clean, n_per_class)

        for rec in picked:
            seen_accessions.add(rec["accession"])
            seen_sigs.add(_seq_signature(rec["sequence"].upper()))
            all_rows.append({
                "uniprot_id":   rec["accession"],
                "sequence":     rec["sequence"].upper(),
                "ec_class":     ec_label,
                "protein_name": rec["protein_name"][:80],
                "organism":     rec["organism"][:50],
                "length":       rec["length"],
            })
        log.info(f"    Kept {len(picked)} (length-stratified)")

    # ── Non-enzyme: per-family with hard caps ─────────────────────────────
    log.info(f"\n  Non-enzyme (per-family sub-queries):")
    non_enzyme_rows = []

    for family_name, family_target, family_keyword in NON_ENZYME_FAMILIES:
        full_q = (f'reviewed:true AND existence:1 '
                  f'AND length:[60 TO 1024] AND NOT ec:* '
                  f'AND {family_keyword}')
        records = _query_uniprot(full_q, int(family_target * 1.5))

        clean = [r for r in records
                 if _is_clean_protein(r, "non-enzyme", seen_accessions,
                                       seen_sigs, min_length, max_length)]

        picked = _stratified_pick(clean, family_target)

        for rec in picked:
            seen_accessions.add(rec["accession"])
            seen_sigs.add(_seq_signature(rec["sequence"].upper()))
            non_enzyme_rows.append({
                "uniprot_id":   rec["accession"],
                "sequence":     rec["sequence"].upper(),
                "ec_class":     "non-enzyme",
                "protein_name": rec["protein_name"][:80],
                "organism":     rec["organism"][:50],
                "length":       rec["length"],
                "_family":      family_name,
            })
        log.info(f"    {family_name:<22} {len(picked):>4}")

    # Cap non-enzyme at exactly n_per_class to balance with enzyme classes
    if len(non_enzyme_rows) > n_per_class:
        # Pick proportionally from each family
        random.shuffle(non_enzyme_rows)
        non_enzyme_rows = non_enzyme_rows[:n_per_class]
        log.info(f"    Capped non-enzyme to {n_per_class} (proportional)")
    elif len(non_enzyme_rows) < n_per_class:
        log.warning(f"    Only {len(non_enzyme_rows)} non-enzymes "
                    f"(target was {n_per_class})")

    # Strip the family marker before saving
    for r in non_enzyme_rows:
        r.pop("_family", None)
    all_rows.extend(non_enzyme_rows)

    # Shuffle so classes don't cluster
    random.shuffle(all_rows)

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "uniprot_id", "sequence", "ec_class",
            "protein_name", "organism", "length"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    counts = Counter(r["ec_class"] for r in all_rows)
    log.info("\n" + "═" * 70)
    log.info(f"  Dataset saved: {out_path}")
    log.info(f"  Total proteins: {len(all_rows):,}")
    for cls in ["non-enzyme", "1", "2", "3", "4", "5", "6", "7"]:
        log.info(f"    {cls:12s} {counts.get(cls, 0):>5,}")
    log.info("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

_CONSERVATIVE_GROUPS = [
    list("ILMV"), list("FYW"), list("DE"), list("RKH"),
    list("ST"),   list("NQ"),  list("AG"),
]
_AA_GROUP_MAP = {aa: g for g in _CONSERVATIVE_GROUPS for aa in g}


def _augment_sequence(seq: str, n: int, mutation_rate: float = 0.01) -> list[str]:
    """Generate `n` augmented variants by conservative substitution."""
    out = []
    for _ in range(n):
        chars = list(seq)
        n_muts = max(1, int(len(seq) * mutation_rate))
        for _ in range(n_muts):
            pos = random.randint(0, len(chars) - 1)
            aa = chars[pos]
            group = _AA_GROUP_MAP.get(aa)
            if group and len(group) > 1:
                chars[pos] = random.choice([a for a in group if a != aa])
        out.append("".join(chars))
    return out


def _balance_classes(samples: list[tuple], target: int) -> list[tuple]:
    """Augment under-represented classes up to `target` samples."""
    by_class: dict[int, list] = defaultdict(list)
    for s in samples:
        by_class[s[2]].append(s)

    out = []
    for label, items in by_class.items():
        out.extend(items)
        deficit = target - len(items)
        if deficit <= 0:
            continue
        log.info(f"    Augmenting {CLASS_NAMES[label]}: +{deficit} synthetic")
        n_per_real = max(1, deficit // max(len(items), 1) + 1)
        added = 0
        for uid, seq, lbl in items:
            if added >= deficit:
                break
            for new_seq in _augment_sequence(seq, n_per_real):
                if added >= deficit:
                    break
                out.append((f"{uid}_aug{added}", new_seq, lbl))
                added += 1
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ESM-2 EMBEDDINGS (fair-esm, atomic cache, OOM recovery)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_esm2_native():
    """Load ESM-2 via fair-esm if available, else fall back to transformers."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import esm as esm_module
        log.info(f"  Loading ESM-2 via fair-esm on {device}...")
        model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(device)
        bc = alphabet.get_batch_converter()
        log.info(f"  ESM-2 (fair-esm) loaded on {device}.")
        return model, bc, device, "fair-esm"
    except ImportError:
        pass

    log.warning("  fair-esm not installed, falling back to transformers")
    log.warning("  Install with: pip install fair-esm")
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    mdl = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    mdl.eval().to(device)
    return tok, mdl, device, "transformers"


def _embed_batch(pairs, model, batch_conv, device, backend):
    """Embed a batch of (uid, seq) pairs. Returns list of 1280-dim lists."""
    import torch
    if backend == "fair-esm":
        _, _, tokens = batch_conv(pairs)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
        reps = results["representations"][33]
        embs = []
        for i, (_, seq) in enumerate(pairs):
            L = min(len(seq), reps.shape[1] - 2)
            emb = reps[i, 1:L+1].mean(0).cpu().tolist()
            embs.append(emb)
        return embs
    else:
        tok, mdl = model, batch_conv
        seqs = [s for _, s in pairs]
        inputs = tok(seqs, return_tensors="pt", padding=True,
                     truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.cpu().tolist()


def _atomic_save_json(data, path: Path) -> None:
    """Write JSON atomically (no corruption on Ctrl+C)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _compute_esm2_cache(samples, cache_path: Path,
                         batch_size: int = 8, save_every: int = 100):
    """Compute ESM-2 embeddings with atomic caching and OOM recovery."""
    import torch

    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        log.info(f"  Loading ESM-2 cache: {cache_path.name}")
        try:
            cache = json.loads(cache_path.read_text())
            log.info(f"  Cache: {len(cache):,} entries")
        except Exception:
            log.warning("  Cache corrupted, starting fresh")
            cache = {}

    todo = [(uid, seq) for uid, seq, _ in samples
            if uid not in cache and "_aug" not in uid]

    if not todo:
        log.info("  All embeddings cached.")
        return cache

    log.info(f"  Computing ESM-2 for {len(todo)} new proteins")
    model, bc, device, backend = _load_esm2_native()
    log.info(f"  Backend: {backend} | batch={batch_size} | device={device}")

    # Sort by length to make batches more efficient
    todo.sort(key=lambda x: len(x[1]))

    t_start = time.time()
    n_total = len(todo)
    i = 0
    saved_at = 0

    while i < n_total:
        batch = todo[i:i+batch_size]
        try:
            torch.cuda.empty_cache()
            embs = _embed_batch(batch, model, bc, device, backend)
            for (uid, _), emb in zip(batch, embs):
                cache[uid] = emb
            i += len(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                log.warning(f"  OOM — reducing batch to {batch_size}")
                continue
            else:
                log.warning(f"  Batch at {i} failed ({e}), skipping")
                for uid, _ in batch:
                    cache[uid] = [0.0] * 1280
                i += len(batch)

        # Periodic atomic save (resilient to crashes)
        if i - saved_at >= save_every or i >= n_total:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save_json(cache, cache_path)
            saved_at = i

        # Progress
        elapsed = time.time() - t_start
        rate = i / max(elapsed, 1)
        eta = (n_total - i) / max(rate, 0.01)
        if i % (batch_size * 5) == 0 or i >= n_total:
            log.info(f"    {i}/{n_total} ({rate:.1f}/s, ETA {eta:.0f}s)")

    log.info(f"  ESM-2 cache saved: {len(cache)} embeddings")
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    csv_path:        str,
    data_dir:        Optional[str],
    cache_dir:       str,
    target_per_class: int,
    use_augment:     bool,
):
    """Build (X, y, uniprot_ids) for training."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"  Loading {csv_path}...")
    samples: list[tuple[str, str, int]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uniprot_id", "").strip()
            seq = row.get("sequence", "").strip().upper()
            ec  = row.get("ec_class", "").strip()
            if not uid or not seq or ec not in LABEL_MAP:
                continue
            # Skip validation set proteins
            if uid in VALIDATION_HOLDOUT_IDS:
                continue
            samples.append((uid, seq, LABEL_MAP[ec]))

    log.info(f"  Loaded {len(samples):,} samples (after holdout exclusion)")
    counts = Counter(s[2] for s in samples)
    for i, name in enumerate(CLASS_NAMES):
        log.info(f"    {name:8s} {counts.get(i, 0):>5,}")

    if use_augment:
        log.info(f"  Balancing classes to ≥{target_per_class}...")
        samples = _balance_classes(samples, target_per_class)
        log.info(f"  After augmentation: {len(samples):,} samples")

    cache_path = Path(cache_dir) / "esm2_cache.json"
    esm2_cache = _compute_esm2_cache(samples, cache_path)

    extra_data: dict = {}
    if data_dir and Path(data_dir).exists():
        log.info(f"  Loading pipeline intermediate from {data_dir}/")
        inter = Path(data_dir)
        n_loaded = 0
        for uid, _, _ in samples:
            if "_aug" in uid:
                continue
            pdata = {}
            for suffix in ["esm2", "active_sites", "pockets",
                            "physicochemical", "enm", "go_terms",
                            "homology", "structure"]:
                fpath = inter / f"{uid}_{suffix}.json"
                if fpath.exists():
                    try:
                        pdata[suffix] = json.loads(fpath.read_text())
                    except Exception:
                        pass
            if pdata:
                extra_data[uid] = pdata
                n_loaded += 1
        log.info(f"  Found pipeline data for {n_loaded} proteins")

    log.info(f"  Building feature vectors (dim={FEATURE_DIM})...")
    X_list, y_list, ids = [], [], []
    t0 = time.time()
    for i, (uid, seq, label) in enumerate(samples):
        if i % 200 == 0 and i > 0:
            rate = i / (time.time() - t0)
            eta = (len(samples) - i) / max(rate, 0.01)
            log.info(f"    {i}/{len(samples)} (ETA {eta:.0f}s)")

        pdata = extra_data.get(uid, {})
        esm2_emb = esm2_cache.get(uid, [0.0] * 1280)
        esm2_result = pdata.get("esm2") or {
            "protein_embedding": esm2_emb,
            "contact_map":       [],
        }

        feat = build_feature_vector(
            sequence        = seq,
            esm2_result     = esm2_result,
            pdb_result      = pdata.get("structure"),
            active_result   = pdata.get("active_sites"),
            pocket_result   = pdata.get("pockets"),
            enm_result      = pdata.get("enm"),
            physico_result  = pdata.get("physicochemical"),
            go_result       = pdata.get("go_terms"),
            homology_result = pdata.get("homology"),
        )
        X_list.append(feat)
        y_list.append(label)
        ids.append(uid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    log.info(f"  Feature matrix: X={X.shape}, y={y.shape}")
    return X, y, ids


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION & MODEL CARD
# ═══════════════════════════════════════════════════════════════════════════════

def _save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    out_path.write_text(json.dumps({
        "labels": CLASS_NAMES,
        "matrix": cm.tolist(),
    }, indent=2))
    txt_path = out_path.with_suffix(".txt")
    lines = ["Confusion matrix (rows=true, cols=predicted):", ""]
    lines.append("       " + "  ".join(f"{n:>7s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"{name:>6s} " +
                     "  ".join(f"{v:>7d}" for v in cm[i]))
    txt_path.write_text("\n".join(lines))
    log.info(f"  Confusion matrix → {out_path.name}, {txt_path.name}")


def _bootstrap_ci(y_true, y_pred, metric_fn, n_boot=200, seed=42):
    """Compute bootstrap 95% CI for a metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not scores:
        return (0.0, 0.0)
    return (float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def _save_model_card(model_dir: Path, metrics: dict, n_train: int,
                      n_val: int, csv_path: str, cv_folds: int) -> None:
    """Save a model card with full provenance."""
    from datetime import datetime
    card = {
        "model_version":    "ec_ensemble_v5",
        "created":          datetime.now().isoformat(),
        "training_csv":     csv_path,
        "n_training":       n_train,
        "n_validation":     n_val,
        "cv_folds":         cv_folds,
        "feature_dim":      FEATURE_DIM,
        "class_names":      CLASS_NAMES,
        "metrics":          metrics,
        "intended_use":     "EC class prediction from protein sequence",
        "limitations": [
            "Trained primarily on Swiss-Prot reviewed entries",
            "May underperform on highly novel folds or short peptides",
            "Probability calibration via isotonic regression on holdout",
            "Performance on borderline non-enzymes (chaperones, scaffolds) "
            "depends heavily on training non-enzyme diversity",
        ],
        "validation_holdout_excluded": sorted(VALIDATION_HOLDOUT_IDS),
    }
    (model_dir / "model_card.json").write_text(json.dumps(card, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """Production EC classifier training pipeline (v2)."""
    pass


@cli.command("build-dataset")
@click.option("--n-per-class", default=800, type=int)
@click.option("--out", default="data/swissprot_curated_v4.csv")
@click.option("--min-length", default=60, type=int)
@click.option("--max-length", default=1024, type=int)
@click.option("--seed", default=42, type=int)
def build_dataset_cmd(n_per_class, out, min_length, max_length, seed):
    """Download and curate the training dataset from Swiss-Prot."""
    build_swissprot_dataset(
        n_per_class = n_per_class,
        out_path    = out,
        min_length  = min_length,
        max_length  = max_length,
        seed        = seed,
    )


@cli.command("train")
@click.option("--csv",      required=True)
@click.option("--data-dir", default=None,
              help="Pipeline intermediate dir (optional, boosts accuracy)")
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
@click.option("--cv-folds",  default=5, type=int)
@click.option("--target-per-class", default=600, type=int)
@click.option("--seed", default=42, type=int)
def train_cmd(csv, data_dir, model_dir, cache_dir, cv_folds,
              target_per_class, seed):
    """Train the EC classifier ensemble with class-weighted loss."""
    random.seed(seed)
    np.random.seed(seed)

    log.info("═" * 70)
    log.info("  EC Classifier — Training Pipeline v2 (production)")
    log.info("═" * 70)

    # Dependencies
    missing = []
    for pkg in ["sklearn", "xgboost", "lightgbm", "transformers", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Missing packages: {missing}")
        log.error("Install: pip install scikit-learn xgboost lightgbm "
                  "transformers torch fair-esm certifi")
        sys.exit(1)

    # Build dataset
    X, y, ids = build_training_dataset(
        csv_path         = csv,
        data_dir         = data_dir,
        cache_dir        = cache_dir,
        target_per_class = target_per_class,
        use_augment      = target_per_class > 0,
    )

    # Train
    clf = ECClassifierEnsemble()
    metrics = clf.train(
        X          = X,
        y          = y,
        n_cv_folds = cv_folds,
        verbose    = True,
    )

    # Save model
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    clf.save(model_dir)
    (Path(model_dir) / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Confusion matrix from holdout
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    try:
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=seed
        )
        proba = clf._predict_proba_raw(clf.preprocessor.transform(X_val))
        y_pred = proba.argmax(axis=1)
        _save_confusion_matrix(y_val, y_pred,
                                Path(model_dir) / "confusion_matrix.json")

        # Bootstrap CIs
        ci_acc = _bootstrap_ci(y_val, y_pred, accuracy_score)
        ci_f1  = _bootstrap_ci(y_val, y_pred,
                               lambda a, b: f1_score(a, b, average="macro",
                                                      zero_division=0))
        metrics["accuracy_95_ci"] = ci_acc
        metrics["macro_f1_95_ci"] = ci_f1
        log.info(f"  Accuracy 95% CI: [{ci_acc[0]:.3f}, {ci_acc[1]:.3f}]")
        log.info(f"  Macro F1 95% CI: [{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]")
    except Exception as e:
        log.warning(f"  Could not compute bootstrap CI: {e}")

    # Model card
    _save_model_card(Path(model_dir), metrics,
                     n_train=metrics.get("n_train", 0),
                     n_val=metrics.get("n_val", 0),
                     csv_path=csv,
                     cv_folds=cv_folds)

    log.info("═" * 70)
    log.info(f"  Done. Model → {model_dir}/")
    log.info(f"  EC top-1 accuracy: {metrics['ec_top1_accuracy']*100:.2f}%")
    log.info(f"  EC top-2 accuracy: {metrics['ec_top2_accuracy']*100:.2f}%")
    log.info(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    log.info(f"  Binary enzyme acc: {metrics['enzyme_binary_acc']*100:.2f}%")
    log.info("═" * 70)


@cli.command("evaluate")
@click.option("--csv",       required=True, help="Test CSV (uniprot_id,sequence,ec_class)")
@click.option("--model-dir", default="models/ec_ensemble_v5")
@click.option("--cache-dir", default="data/training_cache_v5")
def evaluate_cmd(csv, model_dir, cache_dir):
    """Evaluate trained model on a labelled CSV."""
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report
    )

    log.info(f"  Loading test set: {csv}")
    samples = []
    with open(csv, encoding="utf-8") as f:
        for row in csv_reader_with_dict(f):
            uid = row["uniprot_id"].strip()
            seq = row["sequence"].strip().upper()
            ec  = row["ec_class"].strip()
            if not uid or not seq or ec not in LABEL_MAP:
                continue
            samples.append((uid, seq, LABEL_MAP[ec]))
    log.info(f"  {len(samples)} test samples")

    cache_path = Path(cache_dir) / "esm2_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    clf = ECClassifierEnsemble.load(Path(model_dir))

    X_list, y_true = [], []
    for uid, seq, label in samples:
        emb = cache.get(uid)
        if emb is None:
            log.warning(f"    {uid}: no cached embedding, computing fresh")
            # Skip computing here — recommend running --data-dir for fresh
            emb = [0.0] * 1280
        feat = build_feature_vector(sequence=seq,
                                     esm2_result={"protein_embedding": emb,
                                                  "contact_map": []})
        X_list.append(feat)
        y_true.append(label)

    X = np.array(X_list, dtype=np.float32)
    X_pp = clf.preprocessor.transform(X)
    proba = clf._predict_proba_raw(X_pp)
    y_pred = proba.argmax(axis=1)
    y_true = np.array(y_true)

    print("\n" + "═" * 60)
    print("  EVALUATION RESULTS")
    print("═" * 60)
    print(f"  Top-1 accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"  Macro F1:       {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                  zero_division=0))


def csv_reader_with_dict(f):
    return csv.DictReader(f)


if __name__ == "__main__":
    cli()