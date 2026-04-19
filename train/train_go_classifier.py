"""
train/train_go_classifier.py
─────────────────────────────
Proper GO term classifier trained on Swiss-Prot + ESM-2 embeddings.

Architecture:
  - Input: 1280-dim ESM-2 protein embedding
  - Hidden: 2-layer MLP with BatchNorm + Dropout
  - Output: Multi-label sigmoid per GO term
  - Loss: Binary cross-entropy with class weighting

Training data: UniProt Swiss-Prot human proteins with GO annotations
Train/test split: 80/20 stratified

Usage:
    python train/train_go_classifier.py --n-proteins 20000 --min-go-count 30
    python train/train_go_classifier.py --n-proteins 50000 --min-go-count 30 --epochs 30
"""

import json
import pickle
import time
import gc
import requests
import numpy as np
from pathlib import Path
from collections import Counter

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")


class FocalBCELoss(nn.Module):
    """
    Focal loss for multi-label classification.
    Down-weights easy examples so the model focuses on hard ones.
    gamma=2 is standard; higher gamma = more focus on hard examples.
    """
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.gamma    = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# ── Neural network architecture ────────────────────────────────────────────────

class GOClassifier(nn.Module):
    """
    Multi-label GO term classifier on top of ESM-2 embeddings.

    Architecture:
        ESM-2 embedding (1280)
        → Linear(1280, 1024) + BatchNorm + ReLU + Dropout(0.3)
        → Linear(1024, 512)  + BatchNorm + ReLU + Dropout(0.3)
        → Linear(512, 256)   + BatchNorm + ReLU + Dropout(0.2)
        → Linear(256, n_go_terms) + Sigmoid

    Multi-label: each output node is independent (protein can have many GO terms).
    """

    def __init__(self, input_dim: int = 1280, n_go_terms: int = 500,
                 hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims=[2048, 1024, 512, 256],

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.7),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_go_terms))
        # No sigmoid here — use BCEWithLogitsLoss for numerical stability

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# ── Dataset ────────────────────────────────────────────────────────────────────

class GODataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data download ──────────────────────────────────────────────────────────────

def download_swissprot_annotations(
    max_proteins: int = 20000,
    cache_dir: Path = Path("models/embedding_cache"),
    organisms: list = None,
) -> list[dict]:
    """
    Download Swiss-Prot protein sequences and GO annotations.
    By default downloads human proteins first, then other organisms
    to maximise diversity.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"swissprot_{max_proteins}.json"

    if cache_file.exists():
        print(f"Loading cached annotations from {cache_file}...")
        with open(cache_file) as f:
            proteins = json.load(f)
        print(f"Loaded {len(proteins)} proteins.")
        if len(proteins) > 0:
            return proteins[:max_proteins]
        else:
            print("Cache empty, re-downloading...")

    if organisms is None:
        # Human first, then mouse, rat, yeast, e.coli for diversity
        organisms = [9606, 10090, 10116, 559292, 83333]

    proteins = []
    seen_ids = set()

    for organism_id in organisms:
        if len(proteins) >= max_proteins:
            break

        print(f"  Downloading organism {organism_id}...")
        remaining = max_proteins - len(proteins)

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query":   f"reviewed:true AND organism_id:{organism_id}",
            "fields":  "accession,sequence,go,protein_name,gene_names,xref_brenda,ec",
            "format":  "json",
            "size":    min(remaining, 500),
        }

        next_url = url
        page_params = params.copy()
        org_count = 0

        while next_url and len(proteins) < max_proteins:
            try:
                resp = requests.get(
                    next_url if next_url != url else url,
                    params=page_params if next_url == url else None,
                    timeout=30,
                    headers={"User-Agent": "ProteinFP/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()

                for entry in data.get("results", []):
                    uid = entry.get("primaryAccession", "")
                    if uid in seen_ids:
                        continue

                    seq = entry.get("sequence", {}).get("value", "")
                    if not seq or len(seq) < 30 or len(seq) > 1022:
                        continue

                    go_terms = []
                    go_aspects = {}  # go_id → aspect (F/P/C)
                    for xref in entry.get("uniProtKBCrossReferences", []):
                        if xref.get("database") == "GO":
                            go_id = xref.get("id", "")
                            if go_id:
                                go_terms.append(go_id)
                                for prop in xref.get("properties", []):
                                    if prop.get("key") == "GoTerm":
                                        val = prop.get("value", "")
                                        if val.startswith("F:"):
                                            go_aspects[go_id] = "MF"
                                        elif val.startswith("P:"):
                                            go_aspects[go_id] = "BP"
                                        elif val.startswith("C:"):
                                            go_aspects[go_id] = "CC"

                    if not go_terms:
                        continue

                    # Check if enzyme — has EC number cross-reference
                    # Check if enzyme via EC number in protein description
                    # Enzyme detection via GO catalytic activity terms
                    ENZYME_GO = {
                        "GO:0003824", "GO:0016787", "GO:0016301",
                        "GO:0016740", "GO:0016829", "GO:0016853",
                        "GO:0016874", "GO:0016491", "GO:0004672",
                        "GO:0008233", "GO:0003824", "GO:0061630",
                    }
                    is_enzyme = bool(set(go_terms) & ENZYME_GO)

                    seen_ids.add(uid)
                    proteins.append({
                        "uniprot_id":  uid,
                        "sequence":    seq,
                        "go_terms":    go_terms,
                        "go_aspects":  go_aspects,
                        "organism_id": organism_id,
                        "is_enzyme":   is_enzyme,
                    })
                    org_count += 1

                print(f"    {uid[:6]}... total={len(proteins)}")

                # Next page
                link = resp.headers.get("Link", "")
                if 'rel="next"' in link:
                    next_url = link.split("<")[1].split(">")[0]
                    page_params = None
                else:
                    break

                time.sleep(0.3)

            except Exception as e:
                print(f"    Download error: {e}")
                time.sleep(2)
                break

        print(f"  Got {org_count} proteins from organism {organism_id}")

    print(f"\nTotal: {len(proteins)} proteins downloaded")

    with open(cache_file, "w") as f:
        json.dump(proteins, f)

    return proteins[:max_proteins]


# ── ESM-2 embedding ────────────────────────────────────────────────────────────

def compute_embeddings(
    proteins: list[dict],
    cache_dir: Path,
    max_seq_len: int = 512,
) -> dict[str, np.ndarray]:
    """
    Compute ESM-2 protein-level embeddings. Processes one protein at a time
    to avoid OOM crashes. Saves progress every 100 proteins.
    """
    cache_file = cache_dir / "embeddings_cache.pkl"

    # Load existing cache
    if cache_file.exists():
        print("Loading embedding cache...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        print(f"  {len(cache)} embeddings cached")
    else:
        cache = {}

    to_compute = [p for p in proteins if p["uniprot_id"] not in cache]
    if not to_compute:
        print("All embeddings cached.")
        return cache

    print(f"Computing {len(to_compute)} embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import esm as esm_module
    model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    failed = 0
    for i, p in enumerate(to_compute):
        uid = p["uniprot_id"]
        seq = p["sequence"][:max_seq_len]

        try:
            data = [(uid, seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[33])
                L = len(seq)
                emb = results["representations"][33][0, 1:L+1].mean(0)
                cache[uid] = emb.cpu().numpy().astype(np.float32)

            del tokens, results, emb
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Retry with shorter sequence
                try:
                    seq = p["sequence"][:256]
                    data = [(uid, seq)]
                    _, _, tokens = batch_converter(data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        results = model(tokens, repr_layers=[33])
                        L = len(seq)
                        emb = results["representations"][33][0, 1:L+1].mean(0)
                        cache[uid] = emb.cpu().numpy().astype(np.float32)
                    del tokens, results, emb
                    torch.cuda.empty_cache()
                except:
                    failed += 1
                    continue
            else:
                failed += 1
                continue
        except Exception as e:
            failed += 1
            continue

        # Save progress
        if (i + 1) % 100 == 0:
            elapsed = i + 1
            print(f"  [{elapsed}/{len(to_compute)}] cached={len(cache)} failed={failed}")
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
            gc.collect()

    # Final save
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    print(f"Done. {len(cache)} total embeddings. {failed} failed.")
    return cache


# ── Training ───────────────────────────────────────────────────────────────────

def train_neural_go_classifier(
    proteins:      list[dict],
    embeddings:    dict[str, np.ndarray],
    min_go_count:  int   = 30,
    test_size:     float = 0.2,
    epochs:        int   = 20,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    output_path:   Path  = Path("models/go_classifier.pkl"),
) -> dict:
    """
    Train a 3-layer MLP on ESM-2 embeddings to predict GO terms.
    Uses proper train/test split and reports metrics on held-out test set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to proteins with embeddings
    valid = [p for p in proteins if p["uniprot_id"] in embeddings]
    print(f"\nTraining on {len(valid)} proteins with embeddings")

    # Count GO frequencies
    go_counts = Counter()
    for p in valid:
        go_counts.update(p["go_terms"])

    # Select target GO terms
    target_go_terms = [go for go, count in go_counts.most_common()
                       if count >= min_go_count]
    print(f"Target GO terms: {len(target_go_terms)} "
          f"(appearing in >= {min_go_count} proteins)")

    # Build GO aspect map from protein data
    go_aspect_map = {}
    for p in valid:
        for go_id, aspect in p.get("go_aspects", {}).items():
            if go_id in go_aspect_map:
                continue
            go_aspect_map[go_id] = aspect

    # Build feature matrix and label matrix
    X = np.array([embeddings[p["uniprot_id"]] for p in valid], dtype=np.float32)
    go_set = set(target_go_terms)

    # Multi-label binarizer
    mlb = MultiLabelBinarizer(classes=target_go_terms)
    y_lists = [list(set(p["go_terms"]) & go_set) for p in valid]
    Y = mlb.fit_transform(y_lists).astype(np.float32)

    print(f"Feature matrix: {X.shape}")
    print(f"Label matrix:   {Y.shape}")
    print(f"Label density:  {Y.mean():.4f} ({Y.sum()/len(Y):.1f} GO terms per protein)")

    # Train/test split — stratify on most common GO term
    most_common_go = go_counts.most_common(1)[0][0]
    stratify_col = Y[:, target_go_terms.index(most_common_go)]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_col,
    )
    print(f"\nTrain: {len(X_train)} proteins")
    print(f"Test:  {len(X_test)} proteins")

    # Compute class weights (inverse frequency) for loss weighting
    pos_counts = Y_train.sum(axis=0) + 1
    neg_counts = len(Y_train) - pos_counts + 1
    raw_weights = neg_counts / pos_counts
    # Cap at 10 to prevent rare terms from overwhelming the loss
    pos_weight = torch.tensor(np.clip(raw_weights, 1.0, 10.0), dtype=torch.float32)

    # Datasets and loaders
    train_dataset = GODataset(X_train, Y_train)
    test_dataset  = GODataset(X_test,  Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}...")

    n_go = len(target_go_terms)
    model = GOClassifier(
        input_dim=X.shape[1],
        n_go_terms=n_go,
        hidden_dims=[1024, 512, 256],
        dropout=0.3,
    ).to(device)

    pos_weight = pos_weight.to(device)
    criterion = FocalBCELoss(pos_weight=pos_weight, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_f1   = 0.0
    best_state = None
    history   = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Test Loss':>10} "
          f"{'Test F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        test_loss  = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                test_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        test_loss /= len(test_loader)
        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Threshold at 0.5 for metrics
        binary_preds = (all_preds >= 0.5).astype(int)
        f1   = f1_score(all_labels,    binary_preds, average="micro", zero_division=0)
        prec = precision_score(all_labels, binary_preds, average="micro", zero_division=0)
        rec  = recall_score(all_labels,    binary_preds, average="micro", zero_division=0)

        print(f"{epoch:>6} {train_loss:>12.4f} {test_loss:>10.4f} "
              f"{f1:>8.4f} {prec:>10.4f} {rec:>8.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "f1": f1,
        })

        # Save best model
        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest test F1: {best_f1:.4f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # ── Per-GO-term metrics on test set ───────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(y_batch.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    binary_preds = (all_preds >= 0.5).astype(int)

    per_go_f1 = f1_score(all_labels, binary_preds, average=None, zero_division=0)
    good_terms = [(target_go_terms[i], float(per_go_f1[i]))
                  for i in range(len(target_go_terms))
                  if per_go_f1[i] >= 0.3]
    good_terms.sort(key=lambda x: x[1], reverse=True)
    print(f"\nGO terms with F1 >= 0.3: {len(good_terms)}/{len(target_go_terms)}")
    print("Top 20:")
    for go_id, f1_val in good_terms[:20]:
        aspect = go_aspect_map.get(go_id, "?")
        print(f"  {go_id} [{aspect}] F1={f1_val:.3f}  "
              f"(n={int(go_counts[go_id])})")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Save PyTorch model weights
    weights_path = output_path.parent / "go_classifier_weights.pt"
    torch.save(model.state_dict(), weights_path)

    # Save full model package
    package = {
        # Model architecture params
        "input_dim":       X.shape[1],
        "n_go_terms":      n_go,
        "hidden_dims":     [1024, 512, 256],
        "dropout":         0.3,
        "target_go_terms": target_go_terms,
        "go_aspect_map":   go_aspect_map,
        "go_counts":       dict(go_counts),
        "weights_path":    str(weights_path),
        # Training metadata
        "n_proteins_trained": len(X_train),
        "n_proteins_test":    len(X_test),
        "best_f1":         best_f1,
        "history":         history,
        "min_go_count":    min_go_count,
    }

    with open(output_path, "wb") as f:
        pickle.dump(package, f)

    print(f"\nSaved model package to {output_path}")
    print(f"Saved weights to {weights_path}")
    return package


# ── Inference ──────────────────────────────────────────────────────────────────

def load_go_classifier(model_path: Path):
    """Load the trained model. Returns (model, package) or (None, None)."""
    if not model_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        package = pickle.load(f)

    weights_path = Path(package["weights_path"])
    if not weights_path.exists():
        # Try relative to model_path
        weights_path = model_path.parent / "go_classifier_weights.pt"
        if not weights_path.exists():
            return None, None

    model = GOClassifier(
        input_dim=package["input_dim"],
        n_go_terms=package["n_go_terms"],
        hidden_dims=package["hidden_dims"],
        dropout=package["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, package


def predict_go_with_nn(
    protein_embedding: np.ndarray,
    model_path: Path = Path("models/go_classifier.pkl"),
    threshold: float = 0.4,
) -> list[dict]:
    """
    Predict GO terms for a protein using the trained neural network.
    Returns list of {go_id, go_name, score, namespace, evidence} dicts.
    """
    model, package = load_go_classifier(model_path)
    if model is None:
        return []

    target_go_terms = package["target_go_terms"]
    go_aspect_map   = package.get("go_aspect_map", {})

    X = torch.tensor(protein_embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).squeeze(0).numpy()

    predictions = []
    for i, go_id in enumerate(target_go_terms):
        score = float(probs[i])
        if score >= threshold:
            namespace = go_aspect_map.get(go_id, "MF")
            predictions.append({
                "go_id":     go_id,
                "go_name":   "",
                "score":     score,
                "namespace": namespace,
                "evidence":  ["neural_classifier"],
            })

    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions

def train_enzyme_classifier(
    proteins:   list[dict],
    embeddings: dict[str, np.ndarray],
    output_path: Path = Path("models/enzyme_classifier.pkl"),
) -> dict:
    """
    Train a binary enzyme/non-enzyme classifier on ESM-2 embeddings.
    Label: protein has an EC number → enzyme=True.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = [p for p in proteins
             if p["uniprot_id"] in embeddings and "is_enzyme" in p]

    if len(valid) < 100:
        print("Not enough labelled proteins for enzyme classifier")
        return {}

    X = np.array([embeddings[p["uniprot_id"]] for p in valid], dtype=np.float32)
    y = np.array([1 if p["is_enzyme"] else 0 for p in valid])

    n_pos = y.sum()
    print(f"Enzyme classifier: {len(valid)} proteins, {n_pos} enzymes ({n_pos/len(valid)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        max_iter=500, class_weight="balanced", C=1.0, solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred,
                                target_names=["non-enzyme", "enzyme"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    package = {
        "classifier": clf,
        "n_trained":  len(X_train),
        "roc_auc":    float(roc_auc_score(y_test, y_prob)),
    }
    with open(output_path, "wb") as f:
        pickle.dump(package, f)
    print(f"Saved to {output_path}")
    return package

# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--n-proteins",    default=20000, help="Proteins to train on")
@click.option("--min-go-count",  default=30,    help="Min proteins per GO term")
@click.option("--epochs",        default=40,    help="Training epochs")
@click.option("--batch-size",    default=256,   help="Training batch size")
@click.option("--lr",            default=1e-3,  help="Learning rate")
@click.option("--test-size",     default=0.2,   help="Test set fraction")
@click.option("--output",        default="models/go_classifier.pkl")
@click.option("--cache-dir",     default="models/embedding_cache")
@click.option("--max-seq-len",   default=512,   help="Max sequence length for ESM-2")
@click.option("--organisms",     default=None,  help="Comma-separated organism IDs e.g. 9606")
def main(n_proteins, min_go_count, epochs, batch_size, lr,
         test_size, output, cache_dir, max_seq_len, organisms):
    """Train neural GO classifier on Swiss-Prot + ESM-2."""

    organism_list = None
    if organisms:
        organism_list = [int(x) for x in organisms.split(",")]

    cache_dir_path = Path(cache_dir)

    # Step 1: Download
    proteins = download_swissprot_annotations(n_proteins, cache_dir_path, organism_list)

    print("=" * 60)
    print("ProteinFP Neural GO Classifier Training")
    print("=" * 60)
    print(f"  Proteins:     {n_proteins}")
    print(f"  Min GO count: {min_go_count}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Test split:   {test_size:.0%}")
    print()

    cache_dir_path = Path(cache_dir)

    # Step 1: Download
    proteins = download_swissprot_annotations(n_proteins, cache_dir_path)

    # Step 2: Embed
    embeddings = compute_embeddings(proteins, cache_dir_path, max_seq_len)

    # Step 3: Train
    package = train_neural_go_classifier(
        proteins=proteins,
        embeddings=embeddings,
        min_go_count=min_go_count,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_path=Path(output),
    )

# ── Train enzyme classifier ────────────────────────────────────────────────
    proteins_with_label = [p for p in proteins if "is_enzyme" in p]
    print(f"\nFound {len(proteins_with_label)} proteins with enzyme labels")

    if len(proteins_with_label) > 500:
        print("Training enzyme classifier...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split as tts

        valid_enz = [p for p in proteins_with_label
                     if p["uniprot_id"] in embeddings]
        X_enz = np.array([embeddings[p["uniprot_id"]] for p in valid_enz],
                         dtype=np.float32)
        y_enz = np.array([1 if p["is_enzyme"] else 0 for p in valid_enz])

        n_pos = int(y_enz.sum())
        n_neg = len(y_enz) - n_pos
        print(f"  {len(valid_enz)} proteins: {n_pos} enzymes ({n_pos/len(valid_enz)*100:.1f}%), {n_neg} non-enzymes")

        X_tr, X_te, y_tr, y_te = tts(
            X_enz, y_enz, test_size=0.2, random_state=42, stratify=y_enz
        )

        clf_enz = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
        )
        clf_enz.fit(X_tr, y_tr)

        y_pred = clf_enz.predict(X_te)
        y_prob = clf_enz.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)

        print(classification_report(y_te, y_pred,
                                    target_names=["non-enzyme", "enzyme"]))
        print(f"  ROC-AUC: {auc:.4f}")

        # Find optimal threshold
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = float(thresholds[np.argmax(f1_scores)])
        print(f"  Optimal threshold: {best_thresh:.3f}")

        enz_package = {
            "classifier":  clf_enz,
            "roc_auc":     float(auc),
            "threshold":   best_thresh,
            "n_trained":   len(X_tr),
            "n_enzymes":   n_pos,
        }
        enz_path = Path(output).parent / "enzyme_classifier.pkl"
        with open(enz_path, "wb") as f:
            pickle.dump(enz_package, f)
        print(f"  Saved to {enz_path}")
    else:
        print("Not enough labelled proteins — skipping enzyme classifier")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Proteins trained: {package['n_proteins_trained']}")
    print(f"  Proteins tested:  {package['n_proteins_test']}")
    print(f"  GO terms covered: {package['n_go_terms']}")
    print(f"  Best test F1:     {package['best_f1']:.4f}")
    print()
    print("To use in pipeline, run:")
    print("  python pipeline\\deepfri_go.py --uniprot P04637")
    print("  (The classifier is loaded automatically if models/go_classifier.pkl exists)")
    print("=" * 60)


if __name__ == "__main__":
    main()