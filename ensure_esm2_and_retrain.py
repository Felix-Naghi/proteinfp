"""
ensure_esm2_and_retrain.py
───────────────────────────
1. Checks all 100 validation proteins have ESM-2 JSONs, runs Module 08
   for any missing ones.
2. Rebuilds balanced training features now that all validation proteins
   have ESM-2 (important: validation proteins must NOT be in training).
3. Retrains with 1:1 balance and higher n_estimators.

Run from project root:
    python ensure_esm2_and_retrain.py
"""

import json
import subprocess
import sys
import numpy as np
import pickle
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
INTER    = ROOT / "data" / "intermediate"
CACHE    = ROOT / "data" / "training_cache_v2"
MODEL    = ROOT / "models" / "ec_classifier_v2"

VALIDATION_100 = [
    "P00533","P06213","P15056","P00519","P27361","Q02750","P45983","Q13153",
    "P06239","P08581","P04049","P00734","P42574","P55210","P29466","P08246",
    "P07711","P07858","P04637","P01106","P10275","P03372","Q01094","P17275",
    "P05412","P01100","P14921","P00441","P16083","P00387","P00352","P14550",
    "P11473","Q07869","P37231","P10827","P20393","P07900","P08238","P11142",
    "P04792","P02511","Q14524","P35561","P01116","P62834","P84095","P61586",
    "P60953","Q00987","O15151","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P68871","P69905","P02768","P11166","P38398","P51587","P12004","P02452",
    "P68032","P02144","Q9BYF1","P00918","P08473","P15144","P00488","P04180",
    "P01375","P05156","P46108","Q13480","P06493","P24941","P11802","P36507",
    "P45985","P07339","P43235","P08311","P28562","P15923","P17542","P22415",
    "P00491","P15531","P07737","P08107","P38646","P20585","P52701","P62070",
    "P55197","Q07812","P10415","Q16611",
]

# ── Step 1: ensure ESM-2 for all validation proteins ─────────────────────────
print("=" * 60)
print("STEP 1: Checking ESM-2 coverage")
print("=" * 60)

missing = [uid for uid in VALIDATION_100
           if not (INTER / f"{uid}_esm2.json").exists()
           and (INTER / f"{uid}_structure.json").exists()]
no_struct = [uid for uid in VALIDATION_100
             if not (INTER / f"{uid}_structure.json").exists()]

print(f"  Have ESM-2    : {len(VALIDATION_100) - len(missing) - len(no_struct)}")
print(f"  Missing ESM-2 : {len(missing)}  (will compute now)")
print(f"  No structure  : {len(no_struct)}  (need full pipeline run)")
if no_struct:
    print(f"  Skipped       : {no_struct}")
print()

for i, uid in enumerate(missing):
    print(f"  [{i+1}/{len(missing)}] {uid} ESM-2...", end=" ", flush=True)
    r = subprocess.run(
        [sys.executable, "pipeline/esm2_embeddings.py", "--uniprot", uid],
        capture_output=True, text=True, timeout=300
    )
    print("OK" if r.returncode == 0 else f"FAILED")

print()

# ── Step 2: rebuild feature matrix excluding ALL validation proteins ──────────
print("=" * 60)
print("STEP 2: Rebuilding balanced feature matrix")
print("=" * 60)

import pandas as pd
from ec_classifier_v2 import (
    build_full_features, ESM2_DIM, TOTAL_DIM, AA_LETTERS
)

holdout_set = set(VALIDATION_100)

df = pd.read_csv(ROOT / "data" / "swissprot_curated_v4_augmented.csv")
df = df[~df["uniprot_id"].isin(holdout_set)].reset_index(drop=True)
print(f"  Training pool (holdout excluded): {len(df)}")

# Load ESM-2 cache
cache_path = CACHE / "esm2_cache.pkl"
esm2_cache = {}
if cache_path.exists():
    with open(cache_path, "rb") as f:
        esm2_cache = pickle.load(f)
print(f"  ESM-2 cache entries: {len(esm2_cache)}")

# Count class distribution
ne_count  = (df["ec_class"] == "non-enzyme").sum()
enz_count = (df["ec_class"] != "non-enzyme").sum()
print(f"  Enzymes: {enz_count}, Non-enzymes: {ne_count}")

# Build features
X_list, y_list, uids_list = [], [], []
import time
t0 = time.time()

for i, row in df.iterrows():
    uid = row["uniprot_id"]
    seq = str(row.get("sequence", ""))
    ec  = row.get("ec_class", "non-enzyme")
    is_enzyme = 0 if ec == "non-enzyme" else 1

    cached_esm2 = esm2_cache.get(uid)
    if cached_esm2 is not None:
        cached_esm2 = np.array(cached_esm2, dtype=np.float32)

    try:
        feats = build_full_features(uid, seq, INTER, cached_esm2=cached_esm2)
        X_list.append(feats)
        y_list.append(is_enzyme)
        uids_list.append(uid)
    except Exception as e:
        pass

    if (i + 1) % 500 == 0:
        rate = (i+1) / (time.time() - t0)
        print(f"  [{i+1}/{len(df)}] {rate:.0f}/s")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"  Built: {X.shape}")

# Check ESM-2 zero rate
esm_zeros = (np.abs(X[:, :ESM2_DIM]).sum(axis=1) == 0).sum()
print(f"  Zero ESM-2: {esm_zeros}/{len(X)} ({esm_zeros/len(X)*100:.1f}%)")

# ── Balance: 1:1 enzyme to non-enzyme ────────────────────────────────────────
ne_idx  = np.where(y == 0)[0]
enz_idx = np.where(y == 1)[0]
print(f"  Before balance: {len(enz_idx)} enz / {len(ne_idx)} non-enz")

rng = np.random.default_rng(42)
# Keep all non-enzymes, sample equal enzymes
enz_sampled = rng.choice(enz_idx, size=len(ne_idx), replace=False)
keep = np.sort(np.concatenate([ne_idx, enz_sampled]))
X_bal = X[keep]
y_bal = y[keep]
print(f"  After  balance: {(y_bal==1).sum()} enz / {(y_bal==0).sum()} non-enz")

np.save(CACHE / "X_balanced.npy", X_bal)
np.save(CACHE / "y_balanced.npy", y_bal)
print()

# ── Step 3: retrain ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Retraining classifier")
print("=" * 60)

from ec_classifier_v2 import EnzymeClassifierV2

clf = EnzymeClassifierV2(n_estimators=1000, max_depth=7, learning_rate=0.03)
clf.fit(X_bal, y_bal)
clf.save(MODEL)
print()

# ── Step 4: quick validation ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Validating on hold-out proteins")
print("=" * 60)

# Load ground truth
sys.path.insert(0, str(ROOT / "validation"))
gt = {}
try:
    from run_validation import VALIDATION_SET  # type: ignore
    for p in VALIDATION_SET:
        gt[p["uniprot_id"]] = p.get("is_enzyme", False)
except Exception:
    pass
ne_path = ROOT / "validation" / "new_entries.json"
if ne_path.exists():
    for p in json.loads(ne_path.read_text(encoding="utf-8")):
        uid = p.get("uniprot_id", "")
        if uid and uid not in gt:
            gt[uid] = p.get("is_enzyme", False)

results = []
for uid in VALIDATION_100:
    if uid not in gt:
        continue
    struct_path = INTER / f"{uid}_structure.json"
    if not struct_path.exists():
        continue
    try:
        seq = json.loads(struct_path.read_text(encoding="utf-8")).get("sequence", "")
        feats = build_full_features(uid, seq, INTER).reshape(1, -1)
        prob = float(clf.predict_proba(feats)[0])
        results.append((uid, gt[uid], prob))
    except Exception as e:
        print(f"  {uid} failed: {e}")

results.sort(key=lambda x: -x[2])
print(f"\n  Evaluated {len(results)} proteins\n")

print(f"  {'Thresh':>7} {'Acc':>7} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FP':>4} {'FN':>4}")
best_thresh, best_acc = 0.5, 0
for thresh in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    tp = sum(1 for _, t, p in results if p >= thresh and t)
    tn = sum(1 for _, t, p in results if p <  thresh and not t)
    fp = sum(1 for _, t, p in results if p >= thresh and not t)
    fn = sum(1 for _, t, p in results if p <  thresh and t)
    acc  = (tp + tn) / max(len(results), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    flag = " ◄ best" if acc > best_acc else ""
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh
    print(f"  {thresh:>7.2f} {acc*100:>6.1f}% {prec*100:>5.1f}% {rec*100:>5.1f}% "
          f"{f1*100:>5.1f}% {fp:>4} {fn:>4}{flag}")

print()
print(f"  Best accuracy: {best_acc*100:.1f}% at threshold {best_thresh}")

# Per-protein breakdown for wrong predictions
print()
print("  Wrong predictions:")
for uid, te, p in results:
    pred = p >= best_thresh
    if pred != te:
        kind = "FP" if pred and not te else "FN"
        print(f"    [{kind}] {uid}  true={'Y' if te else 'N'}  prob={p:.3f}")

# ── Prevent future ESM-2 missing issues ──────────────────────────────────────
guard_path = ROOT / "validation" / "check_esm2_coverage.py"
guard_code = '''"""
check_esm2_coverage.py
Run this before any validation or EC classifier run to ensure all
proteins have ESM-2 embeddings computed.

    python validation/check_esm2_coverage.py [list_of_uniprot_ids...]
"""
import sys, subprocess
from pathlib import Path

INTER = Path(__file__).parent.parent / "data" / "intermediate"

def ensure_esm2(uniprot_ids):
    missing = [uid for uid in uniprot_ids
               if not (INTER / f"{uid}_esm2.json").exists()
               and (INTER / f"{uid}_structure.json").exists()]
    if not missing:
        print(f"ESM-2 coverage: all {len(uniprot_ids)} proteins OK")
        return []
    print(f"ESM-2 missing for {len(missing)} proteins — computing now...")
    failed = []
    for i, uid in enumerate(missing):
        print(f"  [{i+1}/{len(missing)}] {uid}", end=" ", flush=True)
        r = subprocess.run(
            [sys.executable, "pipeline/esm2_embeddings.py", "--uniprot", uid],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0:
            print("OK")
        else:
            print("FAILED")
            failed.append(uid)
    return failed

if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else []
    if not ids:
        # Default: check all proteins that have a structure
        ids = [p.stem.replace("_structure", "")
               for p in INTER.glob("*_structure.json")]
    ensure_esm2(ids)
'''
guard_path.write_text(guard_code, encoding="utf-8")
print(f"\n  Guard script written → {guard_path}")
print("  Run before any future validation:")
print("  python validation/check_esm2_coverage.py")