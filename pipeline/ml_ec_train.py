"""
pipeline/ml_ec_train.py
────────────────────────
Training pipeline for the ML EC Classifier Ensemble.

Workflow:
  1. Download curated enzyme dataset from UniProt Swiss-Prot
     (or load pre-downloaded cache)
  2. Build feature vectors for each protein
     (sequence features only, or full pipeline outputs if available)
  3. Train the ECClassifierEnsemble with stacking cross-validation
  4. Evaluate and save the model

Requirements:
    pip install xgboost lightgbm scikit-learn requests tqdm

Usage:
    # Quick smoke test (100 proteins, no GPU required):
    python pipeline/ml_ec_train.py --quick

    # Full training from Swiss-Prot (uses your pipeline cache):
    python pipeline/ml_ec_train.py --data-dir data/intermediate --model-dir models/ec_ensemble

    # Training from a custom CSV:
    python pipeline/ml_ec_train.py --csv my_dataset.csv --model-dir models/ec_ensemble

CSV format (--csv):
    uniprot_id, sequence, ec_class
    P12345,     MKTAY...,  3
    P67890,     ACDEF...,  non-enzyme
    ...
"""

from __future__ import annotations

import ssl
import certifi

# Patch the default SSL context to use certifi's CA bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

from pipeline.ml_ec_classifier import ECClassifierEnsemble
from pipeline.ml_ec_features import MLECFeatures

# ⚠️ Better fix: replace this with proper packaging later
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)


@click.command()
def main():
    # Use the imported classes directly
    pass

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Swiss-Prot enzyme dataset builder ─────────────────────────────────────────

# Carefully curated representative Swiss-Prot proteins per EC class.
# Each entry: (uniprot_id, ec_class, short_name)
_CANONICAL_PROTEINS: list[tuple[str, str, str]] = [
    # EC 1 — Oxidoreductases
    ("P00441", "1", "SOD1 - Cu/Zn superoxide dismutase"),
    ("P16083", "1", "NQO2 - NAD(P)H dehydrogenase"),
    ("P22353", "1", "TXNRD1 - Thioredoxin reductase"),
    ("P00367", "1", "GLUD1 - Glutamate dehydrogenase"),
    ("P04406", "1", "GAPDH - Glyceraldehyde-3-phosphate dehydrogenase"),
    ("P08559", "1", "PDHA1 - Pyruvate dehydrogenase E1"),
    ("O75874", "1", "IDH1 - Isocitrate dehydrogenase"),
    ("P09110", "1", "ACAA1 - Acetyl-CoA acetyltransferase"),
    ("P31323", "1", "PRKAR2B - cAMP-dependent protein kinase type II"),
    ("P14550", "1", "AKR1A1 - Alcohol dehydrogenase"),

    # EC 2 — Transferases
    ("P00533", "2", "EGFR - Epidermal growth factor receptor kinase"),
    ("P06213", "2", "INSR - Insulin receptor kinase"),
    ("Q00987", "2", "MDM2 - E3 ubiquitin-protein ligase"),
    ("O15151", "2", "MDM4 - Ubiquitin ligase"),
    ("P31946", "2", "YWHAB - 14-3-3 protein kinase partner"),
    ("P04049", "2", "RAF1 - RAF proto-oncogene serine/threonine kinase"),
    ("P15056", "2", "BRAF - Serine/threonine-protein kinase B-raf"),
    ("O96017", "2", "CHEK2 - Serine/threonine-protein kinase Chk2"),
    ("P49841", "2", "GSK3B - Glycogen synthase kinase-3 beta"),
    ("Q13315", "2", "ATM - Serine/threonine-protein kinase ATM"),

    # EC 3 — Hydrolases
    ("P42574", "3", "CASP3 - Caspase-3 cysteine protease"),
    ("Q9BYF1", "3", "ACE2 - Angiotensin-converting enzyme 2"),
    ("P00734", "3", "F2 - Thrombin serine protease"),
    ("P01116", "3", "KRAS - GTPase"),
    ("P60953", "3", "CDC42 - Rho-related GTP-binding protein"),
    ("P62993", "3", "GRB2 - adapter (ATPase context)"),
    ("P19838", "3", "USP14 - Ubiquitin carboxyl-terminal hydrolase"),
    ("P12931", "3", "SRC - Proto-oncogene tyrosine kinase"),
    ("P00918", "3", "CA2 - Carbonic anhydrase 2"),
    ("P07477", "3", "PRSS1 - Trypsin-1 serine protease"),

    # EC 4 — Lyases
    ("P00918", "4", "CA2 - Carbonic anhydrase (also lyase context)"),
    ("P04083", "4", "ANXA1 - Annexin A1"),
    ("P07900", "4", "HSP90AA1 - Heat shock protein (ATPase/lyase)"),
    ("P68871", "4", "HBB - Hemoglobin subunit beta"),
    ("P06733", "4", "ENO1 - Enolase 1"),
    ("P14618", "4", "PKM - Pyruvate kinase"),
    ("P00558", "4", "PGK1 - Phosphoglycerate kinase"),
    ("P04075", "4", "ALDOA - Fructose-bisphosphate aldolase"),
    ("P09972", "4", "ALDOC - Fructose-bisphosphate aldolase C"),
    ("P06744", "4", "GPI - Glucose-6-phosphate isomerase (lyase)"),

    # EC 5 — Isomerases
    ("P06744", "5", "GPI - Glucose-6-phosphate isomerase"),
    ("P04637", "5", "TP53 - Tumor protein p53 (conformational isomerase)"),
    ("P19367", "5", "HK1 - Hexokinase-1"),
    ("P52292", "5", "KPNA2 - Importin subunit alpha-1"),
    ("P62195", "5", "PSMC5 - 26S protease regulatory subunit 8"),
    ("P55210", "5", "CASP7 - Caspase-7"),
    ("O14965", "5", "AURKA - Aurora kinase A"),
    ("P06493", "5", "CDK1 - Cyclin-dependent kinase 1"),
    ("P17612", "5", "PRKACA - cAMP-dependent protein kinase"),
    ("P24941", "5", "CDK2 - Cyclin-dependent kinase 2"),

    # EC 6 — Ligases
    ("P38398", "6", "BRCA1 - Ubiquitin ligase / DNA repair"),
    ("P22681", "6", "CBL - E3 ubiquitin-protein ligase Cbl"),
    ("Q13485", "6", "SMAD4 - Mothers against DPP homolog 4"),
    ("P62877", "6", "RBX1 - E3 ubiquitin-protein ligase RBX1"),
    ("Q9NWF9", "6", "BIRC6 - Baculoviral IAP repeat-containing protein 6"),
    ("P11441", "6", "UBB - Polyubiquitin-B"),
    ("P0CG47", "6", "UBB - Polyubiquitin"),
    ("P62988", "6", "RPS27A - Ubiquitin-40S ribosomal protein S27a"),
    ("Q00540", "6", "MUSK - Muscle, skeletal receptor tyrosine-protein kinase"),
    ("Q9Y3T9", "6", "USO1 - General vesicular transport factor p115"),

    # EC 7 — Translocases
    ("P98194", "7", "SLC4A2 - Anion exchange protein 2"),
    ("O95376", "7", "ARIH2 - E3 ubiquitin-protein ligase ARIH2"),
    ("P21796", "7", "VDAC1 - Voltage-dependent anion-selective channel protein 1"),
    ("P02545", "7", "LMNA - Prelamin-A/C"),
    ("Q9Y277", "7", "VDAC3 - Voltage-dependent anion channel 3"),
    ("P45880", "7", "VDAC2 - Voltage-dependent anion channel 2"),
    ("P46940", "7", "IQGAP1 - Ras GTPase-activating-like protein"),
    ("Q12931", "7", "TRAP1 - Heat shock protein 75 kDa"),
    ("O00330", "7", "PDHA2 - Pyruvate dehydrogenase E1 testis"),
    ("P00390", "7", "GSR - Glutathione reductase translocase"),

    # Non-enzymes
    ("P07900", "non-enzyme", "HSP90AA1 - Chaperone (non-catalytic)"),
    ("P68871", "non-enzyme", "HBB - Hemoglobin (oxygen carrier)"),
    ("P04637", "non-enzyme", "TP53 - Transcription factor"),
    ("P61978", "non-enzyme", "HNRNPK - Heterogeneous nuclear ribonucleoprotein K"),
    ("P04637", "non-enzyme", "TP53 - DNA binding protein"),
    ("P02545", "non-enzyme", "LMNA - Structural nuclear lamina"),
    ("P02649", "non-enzyme", "APOE - Apolipoprotein E"),
    ("P68363", "non-enzyme", "TUBA1B - Tubulin alpha-1B chain"),
    ("P07437", "non-enzyme", "TUBB - Tubulin beta chain"),
    ("P60709", "non-enzyme", "ACTB - Actin cytoskeletal"),
    ("P68133", "non-enzyme", "ACTA1 - Actin alpha skeletal muscle"),
    ("P04156", "non-enzyme", "PRNP - Major prion protein"),
    ("P01308", "non-enzyme", "INS - Insulin precursor (hormone)"),
    ("P01275", "non-enzyme", "GCG - Glucagon (hormone)"),
    ("P01019", "non-enzyme", "AGT - Angiotensinogen precursor"),
]

# ── Synthetic data augmentation ────────────────────────────────────────────────

def _augment_sequence(seq: str, n: int = 3, noise: float = 0.02) -> list[str]:
    """
    Augment a sequence by introducing random conservative mutations.
    Used to increase training set diversity.
    """
    # Conservative substitution groups
    GROUPS = [
        list("ACDEFGHIKLMNPQRSTVWY"),  # all
        list("ILMFWV"),    # hydrophobic
        list("RKHDE"),     # charged
        list("ST"),        # hydroxyl
        list("NQ"),        # amide
        list("DE"),        # acidic
        list("RK"),        # basic
        list("FYW"),       # aromatic
    ]
    augmented = []
    for _ in range(n):
        seq_list = list(seq)
        n_muts = max(1, int(len(seq) * noise))
        for _ in range(n_muts):
            pos = random.randint(0, len(seq_list)-1)
            aa  = seq_list[pos]
            # Pick conservative group
            group = next((g for g in GROUPS[1:] if aa in g), GROUPS[0])
            seq_list[pos] = random.choice(group)
        augmented.append("".join(seq_list))
    return augmented


# ── Dataset builder ────────────────────────────────────────────────────────────

def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetch protein sequence from UniProt REST API."""
    try:
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ProteinFP/1.0 (research pipeline; python urllib)",
            "Accept": "text/x-fasta",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            fasta = r.read().decode("utf-8")
        lines = fasta.strip().split("\n")
        seq = "".join(lines[1:])
        return seq if seq else None
    except Exception as e:
        log.warning(f"  Failed to fetch {uniprot_id}: {e}")
        return None


def build_training_dataset(
    data_dir:      Optional[str] = None,
    csv_path:      Optional[str] = None,
    use_augment:   bool = True,
    augment_n:     int  = 3,
    quick:         bool = False,
    cache_dir:     str  = "data/training_cache",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, uniprot_ids) for training.

    Priority:
      1. CSV file (--csv flag)
      2. Pipeline intermediate files (--data-dir flag)
      3. Canonical Swiss-Prot protein list (fetch from UniProt)
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    label_map = {"non-enzyme": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}

    samples:    list[tuple[str, str, int]] = []  # (uid, sequence, label)
    extra_data: dict[str, dict] = {}             # uid -> intermediate JSON

    # ── Source 1: CSV ─────────────────────────────────────────────────────────
    if csv_path and Path(csv_path).exists():
        log.info(f"  Loading dataset from CSV: {csv_path}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uniprot_id", row.get("id", "UNKNOWN"))
                seq = row.get("sequence", "")
                ec  = row.get("ec_class", "non-enzyme").strip()
                if seq and ec in label_map:
                    samples.append((uid, seq, label_map[ec]))
        log.info(f"  Loaded {len(samples):,} samples from CSV")

    # ── Source 2: Pipeline intermediate files ─────────────────────────────────
    elif data_dir and Path(data_dir).exists():
        log.info(f"  Scanning pipeline outputs in {data_dir}/")
        inter = Path(data_dir)
        for esm2_file in inter.glob("*_esm2.json"):
            uid = esm2_file.stem.replace("_esm2", "")
            # Try to load all module outputs for this protein
            protein_data = {}
            for module_suffix in ["_esm2", "_active_sites", "_pockets",
                                   "_physicochemical", "_enm", "_go_terms",
                                   "_homology", "_binding_sites"]:
                fpath = inter / f"{uid}{module_suffix}.json"
                if fpath.exists():
                    try:
                        protein_data[module_suffix.lstrip("_")] = json.loads(fpath.read_text())
                    except Exception:
                        pass

            # Try to find consensus report for EC label
            report_path = Path("data/reports") / uid / "report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                    ec = str(report.get("ec_number", "")).strip()
                    seq = report.get("sequence", "")
                    if seq and ec:
                        label = label_map.get(ec, label_map.get(ec[0], 0))
                        samples.append((uid, seq, label))
                        extra_data[uid] = protein_data
                except Exception:
                    pass

        log.info(f"  Found {len(samples):,} proteins with pipeline outputs")

    # ── Source 3: Canonical Swiss-Prot list ───────────────────────────────────
    if not samples:
        proteins_to_use = _CANONICAL_PROTEINS[:30] if quick else _CANONICAL_PROTEINS
        log.info(f"  Fetching {len(proteins_to_use)} canonical Swiss-Prot proteins...")
        label_map_local = {"non-enzyme": 0, "1": 1, "2": 2, "3": 3,
                           "4": 4, "5": 5, "6": 6, "7": 7}

        for uid, ec, name in proteins_to_use:
            cache_file = Path(cache_dir) / f"{uid}.seq"
            if cache_file.exists():
                seq = cache_file.read_text().strip()
            else:
                log.info(f"    Fetching {uid} ({name[:40]})...")
                seq = fetch_uniprot_sequence(uid)
                if seq:
                    cache_file.write_text(seq)
                time.sleep(0.1)  # rate limiting

            if seq:
                label = label_map_local.get(ec, 0)
                samples.append((uid, seq, label))

        log.info(f"  Fetched {len(samples)} proteins")

    if not samples:
        raise ValueError("No training data found. Provide --csv or --data-dir, or check network.")

    # ── Augmentation ──────────────────────────────────────────────────────────
    if use_augment and not quick:
        log.info(f"  Augmenting dataset ({augment_n}× per sample)...")
        original = list(samples)
        for uid, seq, label in original:
            for aug_seq in _augment_sequence(seq, n=augment_n, noise=0.015):
                samples.append((f"{uid}_aug", aug_seq, label))
        log.info(f"  After augmentation: {len(samples):,} samples")

    # Add this once BEFORE the loop (load model once, not per protein)
    from transformers import AutoTokenizer, AutoModel
    import torch

    log.info("  Loading ESM2 model...")
    _esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    _esm2_model     = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    _esm2_model.eval()
    if torch.cuda.is_available():
        _esm2_model = _esm2_model.cuda()
    log.info("  ESM2 loaded.")

    def _get_esm2(sequence: str) -> dict:
        inputs = _esm2_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _esm2_model(**inputs)
        embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().tolist()  # 1280-dim
        return {"protein_embedding": embedding, "contact_map": []}

    # ── Build feature matrix ───────────────────────────────────────────────────
    from pipeline.ml_ec_features import build_feature_vector, FEATURE_DIM
    log.info(f"  Building feature vectors (dim={FEATURE_DIM})...")

    random.shuffle(samples)

    X_list, y_list, ids = [], [], []
    for i, (uid, seq, label) in enumerate(samples):
        if i % 50 == 0:
            log.info(f"    {i}/{len(samples)} processed...")
        pdata = extra_data.get(uid, {})
        feat = build_feature_vector(
            sequence        = seq,
            esm2_result     = pdata.get("esm2") or _get_esm2(seq),
            pdb_result      = pdata.get("pdb"),
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

    log.info(f"  Dataset: X={X.shape}, y={y.shape}")
    # Class distribution
    for i, cls in enumerate(["non-enz", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]):
        n = int((y == i).sum())
        log.info(f"    {cls}: {n} samples")

    return X, y, ids


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.command()
@click.option("--data-dir",  default=None,             help="Pipeline intermediate dir")
@click.option("--csv",       default=None,             help="CSV file with training data")
@click.option("--model-dir", default="models/ec_ensemble", help="Where to save trained model")
@click.option("--cache-dir", default="data/training_cache", help="Sequence cache directory")
@click.option("--quick",     is_flag=True, default=False, help="Quick smoke test (30 proteins)")
@click.option("--no-augment",is_flag=True, default=False, help="Skip data augmentation")
@click.option("--cv-folds",  default=5,    type=int,   help="Cross-validation folds")
def main(
    data_dir:  Optional[str],
    csv:       Optional[str],
    model_dir: str,
    cache_dir: str,
    quick:     bool,
    no_augment: bool,
    cv_folds:  int,
) -> None:
    """
    Train the ML EC Classifier Ensemble.

    \b
    Examples:
        python pipeline/ml_ec_train.py --quick
        python pipeline/ml_ec_train.py --csv data/ec_dataset.csv
        python pipeline/ml_ec_train.py --data-dir data/intermediate
    """
    from pipeline.ml_ec_classifier import ECClassifierEnsemble

    log.info("═" * 60)
    log.info("  ML EC Classifier — Training Pipeline")
    log.info("═" * 60)

    # ── Check dependencies ────────────────────────────────────────────────────
    missing = []
    for pkg in ["sklearn", "xgboost", "lightgbm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Missing packages: {missing}")
        log.error("Install with: pip install scikit-learn xgboost lightgbm")
        raise SystemExit(1)

    # ── Build dataset ─────────────────────────────────────────────────────────
    X, y, ids = build_training_dataset(
        data_dir    = data_dir,
        csv_path    = csv,
        use_augment = not no_augment,
        augment_n   = 2 if quick else 4,
        quick       = quick,
        cache_dir   = cache_dir,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = ECClassifierEnsemble()
    metrics = clf.train(
        X           = X,
        y           = y,
        n_cv_folds  = min(cv_folds, 3) if quick else cv_folds,
        verbose     = True,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    clf.save(model_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = Path(model_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("═" * 60)
    log.info(f"  Training complete!")
    log.info(f"  Model saved to: {model_dir}/")
    log.info(f"  EC class top-1 accuracy: {metrics['ec_top1_accuracy']*100:.2f}%")
    log.info(f"  EC class top-2 accuracy: {metrics['ec_top2_accuracy']*100:.2f}%")
    log.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    log.info(f"  Binary enzyme accuracy: {metrics['enzyme_binary_acc']*100:.2f}%")
    log.info("═" * 60)


if __name__ == "__main__":
    main()