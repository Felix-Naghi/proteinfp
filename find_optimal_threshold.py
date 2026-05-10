"""
augment_nonenzyme_training.py
──────────────────────────────
Fetches ~500 diverse non-enzyme proteins from UniProt covering the
families that are currently missing from the training set:
  - GTPases / small G-proteins
  - Chaperones / heat shock proteins
  - Nuclear receptors / transcription factors
  - Ion channels / transporters
  - Scaffold / adaptor proteins
  - Structural proteins (actin, tubulin, collagen)
  - Signalling adaptors (SH2/SH3 domain proteins)
  - Apoptosis regulators (Bcl-2 family)

Appends them to data/swissprot_curated_v4.csv, creating
data/swissprot_curated_v4_augmented.csv

Run from project root:
    python augment_nonenzyme_training.py
"""

import csv
import json
import time
import urllib.request
import urllib.parse
import shutil
import sys
from pathlib import Path

INPUT_CSV  = Path("data/swissprot_curated_v4.csv")
OUTPUT_CSV = Path("data/swissprot_curated_v4_augmented.csv")
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# Validation set IDs to exclude (don't leak into training)
HOLDOUT = {
    "P04637","P38398","P51587","P02144","P68871","P69905","P02768","P11142",
    "P07900","P08238","P11473","P04792","P02511","P03372","P10275","P10827",
    "P11166","P12004","P14921","P20393","P29466","P37231","P42574","P46108",
    "P00352","P00387","P00488","P04180","P05156","P06239","P07711","P08473",
    "P08581","P14550","P15144","P17706","P27361","P29350","P35561","P45983",
    "P61586","P62834","P84095","Q02750","Q06124","Q07869","Q13153","Q13480",
    "Q14524","P00533","P06213","P15056","P00519","P00441","P16083","P00734",
    "P68871","P69905","P01116","Q00987","Q9BYF1","O15151","P01106","P10275",
    "P05412","P01100","P00387","P00352","P14550","Q14524","P35561","P62834",
    "P84095","P61586","P60953","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P02768","P11166","P12004","P02452","P68032","P08473","P15144","P00488",
    "P04180","P01375","P05156","P46108","Q13480","P06493","P24941","P11802",
    "P36507","P45985","P07339","P43235","P08311","P28562","P15923","P17542",
    "P22415","P00491","P15531","P07737","P08107","P38646","P20585","P52701",
    "P62070","P55197","Q07812","P10415","Q16611",
}

# Non-enzyme family queries — each pulls proteins with NO ec: annotation
FAMILY_QUERIES = [
    ("gtpase",          'reviewed:true AND existence:1 AND length:[150 TO 800] AND keyword:KW-0342 NOT ec:*', 80),
    ("chaperone",       'reviewed:true AND existence:1 AND length:[200 TO 900] AND keyword:KW-0143 NOT ec:*', 80),
    ("nuclear_receptor",'reviewed:true AND existence:1 AND length:[200 TO 900] AND keyword:KW-0539 NOT ec:*', 60),
    ("ion_channel",     'reviewed:true AND existence:1 AND length:[200 TO 1024] AND keyword:KW-0407 NOT ec:*', 60),
    ("transcr_factor",  'reviewed:true AND existence:1 AND length:[200 TO 900] AND keyword:KW-0804 NOT ec:*', 60),
    ("scaffold_adaptor",'reviewed:true AND existence:1 AND length:[150 TO 700] AND keyword:KW-0597 NOT ec:*', 60),
    ("structural",      'reviewed:true AND existence:1 AND length:[200 TO 800] AND keyword:KW-0261 NOT ec:*', 40),
    ("apoptosis_reg",   'reviewed:true AND existence:1 AND length:[100 TO 600] AND keyword:KW-0053 NOT ec:*', 40),
    ("dna_binding",     'reviewed:true AND existence:1 AND length:[200 TO 800] AND keyword:KW-0238 NOT ec:*', 40),
    ("receptor",        'reviewed:true AND existence:1 AND length:[300 TO 1024] AND keyword:KW-0675 NOT ec:*', 40),
]


def fetch_uniprot_batch(query: str, n: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "query":  query,
        "format": "json",
        "size":   min(n, 100),
        "fields": "accession,sequence,length",
    })
    url = f"{UNIPROT_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProteinFP/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        results = []
        for entry in data.get("results", []):
            uid = entry.get("primaryAccession", "")
            seq = entry.get("sequence", {}).get("value", "")
            length = entry.get("sequence", {}).get("length", 0)
            if uid and seq and length >= 60:
                results.append({"uniprot_id": uid, "sequence": seq, "length": length})
        return results
    except Exception as e:
        print(f"  ERROR fetching: {e}")
        return []


def main():
    # Load existing CSV
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found")
        sys.exit(1)

    existing_ids = set()
    existing_rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_rows.append(row)
            existing_ids.add(row.get("uniprot_id", ""))

    # Count existing non-enzymes
    ne_count = sum(1 for r in existing_rows if r.get("ec_class") == "non-enzyme")
    print(f"Existing dataset: {len(existing_rows)} rows, {ne_count} non-enzymes")
    print(f"Holdout IDs excluded: {len(HOLDOUT)}")
    print()

    # Fetch new non-enzymes
    new_rows = []
    seen_new = set()

    for family, query, target in FAMILY_QUERIES:
        print(f"Fetching {target} {family} proteins...", end=" ", flush=True)
        results = fetch_uniprot_batch(query, target * 2)  # fetch 2x, filter down

        added = 0
        for r in results:
            uid = r["uniprot_id"]
            seq = r["sequence"]
            if uid in existing_ids or uid in HOLDOUT or uid in seen_new:
                continue
            # Skip non-standard amino acids
            if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq.upper()):
                continue
            seen_new.add(uid)
            new_rows.append({
                "uniprot_id": uid,
                "sequence":   seq,
                "length":     r["length"],
                "ec_class":   "non-enzyme",
                "family":     family,
            })
            added += 1
            if added >= target:
                break

        print(f"got {added}")
        time.sleep(0.5)

    print(f"\nNew non-enzyme rows: {len(new_rows)}")

    # Ensure fieldnames include 'family' column
    if fieldnames and "family" not in fieldnames:
        fieldnames = list(fieldnames) + ["family"]

    # Write augmented CSV
    shutil.copy2(INPUT_CSV, INPUT_CSV.with_suffix(".csv.bak"))
    print(f"Backup: {INPUT_CSV.with_suffix('.csv.bak')}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in new_rows:
            writer.writerow(row)

    # Summary
    total_ne = ne_count + len(new_rows)
    print(f"\nAugmented CSV: {OUTPUT_CSV}")
    print(f"  Total rows     : {len(existing_rows) + len(new_rows)}")
    print(f"  Non-enzymes    : {total_ne}  (+{len(new_rows)} new)")
    print(f"  Enzyme classes : 800 each (unchanged)")
    print()

    # Length distribution of new non-enzymes
    import collections
    buckets = collections.Counter()
    for r in new_rows:
        l = int(r["length"])
        if l < 150:   buckets["<150"] += 1
        elif l < 300: buckets["150-300"] += 1
        elif l < 500: buckets["300-500"] += 1
        elif l < 700: buckets["500-700"] += 1
        else:         buckets["700+"] += 1
    print("New non-enzyme length distribution:")
    for k in ["<150","150-300","300-500","500-700","700+"]:
        print(f"  {k:10s}: {buckets[k]}")

    print()
    print("Now retrain:")
    print("  python pipeline/ml_ec_train_v2.py train \\")
    print("    --csv data/swissprot_curated_v4_augmented.csv \\")
    print("    --data-dir data/intermediate \\")
    print("    --model-dir models/ec_ensemble_v6 \\")
    print("    --target-per-class 600")


if __name__ == "__main__":
    main()