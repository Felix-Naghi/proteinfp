"""
add_100_to_ground_truth.py
──────────────────────────
Fetches ground truth from UniProt for all 100 proteins and adds
any missing ones to validation/new_entries.json.

Run from project root:
    python add_100_to_ground_truth.py

Only proteins NOT already in the file get added — safe to re-run.
"""

import json
import time
import urllib.request
import urllib.error
import shutil
from pathlib import Path

# ── Your full 100-protein list ────────────────────────────────────────────────
PROTEINS = [
    "P00533","P06213","P15056","P00519","P27361","Q02750","P45983","Q13153",
    "P06239","P08581","P04049","P00734","P42574","P55210","P29466","P08246",
    "P07711","P07858","P04637","P01106","P10275","P03372","Q01094","P17275",
    "P05412","P01100","P14921","P00441","P16083","P00387","P00352","P14550",
    "P11473","Q07869","P37231","P10827","P20393","P07900","P08238","P11142",
    "P04792","P02511","Q14524","P35561","P01116","P62834","P84095","P61586",
    "P60953","Q00987","O15151","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P68871","P69905","P02768","P11166","P38398","P51587","P12004","P02452",
    "P68032","P02144","Q9BYF1","P00918","P08473","P15144","P00488","P04180",
    "P01375","P05156","P46108","Q13480",
    # Extra kinases
    "P06493","P24941","P11802","P36507","P45985",
    # Proteases
    "P07339","P43235","P08311",
    # Phosphatases
    "P28562",
    # Transcription factors
    "P15923","P17542","P22415",
    # Metabolic
    "P00491","P15531","P07737",
    # Chaperones
    "P08107","P38646",
    # DNA repair
    "P20585","P52701",
    # GTPases
    "P62070","P55197",
    # Apoptosis
    "Q07812","P10415","Q16611", "P0DMV8", "Q06609",
    
]

# Deduplicate preserving order
seen = set()
PROTEINS_UNIQUE = []
for uid in PROTEINS:
    if uid not in seen:
        seen.add(uid)
        PROTEINS_UNIQUE.append(uid)

print(f"Total unique proteins: {len(PROTEINS_UNIQUE)}")

# ── Load existing entries ─────────────────────────────────────────────────────
NEW_ENTRIES_PATH = Path("validation/new_entries.json")

# Load existing VALIDATION_SET IDs so we don't duplicate those either
existing_ids = set()
try:
    import sys
    sys.path.insert(0, "validation")
    from run_validation import VALIDATION_SET
    for p in VALIDATION_SET:
        existing_ids.add(p["uniprot_id"])
    print(f"VALIDATION_SET: {len(existing_ids)} proteins already have ground truth")
except Exception as e:
    print(f"Warning: could not load VALIDATION_SET: {e}")

existing_entries = []
if NEW_ENTRIES_PATH.exists():
    existing_entries = json.loads(NEW_ENTRIES_PATH.read_text(encoding="utf-8"))
    for e in existing_entries:
        existing_ids.add(e.get("uniprot_id", ""))
    print(f"new_entries.json: {len(existing_entries)} existing entries")

# Backup
if NEW_ENTRIES_PATH.exists():
    bak = NEW_ENTRIES_PATH.with_suffix(".json.bak2")
    shutil.copy2(NEW_ENTRIES_PATH, bak)
    print(f"Backup -> {bak}")

to_fetch = [uid for uid in PROTEINS_UNIQUE if uid not in existing_ids]
print(f"Need to fetch ground truth for: {len(to_fetch)} proteins\n")

# ── Fetch from UniProt ────────────────────────────────────────────────────────
def fetch_uniprot(uid: str) -> dict | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "ProteinFP/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR fetching {uid}: {e}")
        return None


def parse_entry(uid: str, data: dict) -> dict:
    # Gene name
    gene = uid
    for g in data.get("genes", []):
        if "geneName" in g:
            gene = g["geneName"]["value"]
            break

    # Description
    desc = ""
    try:
        desc = (data["proteinDescription"]["recommendedName"]["fullName"]["value"])[:120]
    except Exception:
        pass

    # Organism
    organism = data.get("organism", {}).get("scientificName", "Homo sapiens")

    # EC numbers
    ecs = []
    try:
        for ec in (data["proteinDescription"]["recommendedName"].get("ecNumbers", [])):
            ecs.append(ec["value"])
    except Exception:
        pass
    # Also check catalytic activity comments
    if not ecs:
        for comment in data.get("comments", []):
            if comment.get("commentType") == "CATALYTIC ACTIVITY":
                try:
                    ec_val = comment["reaction"]["ecNumber"]
                    if ec_val not in ecs:
                        ecs.append(ec_val)
                except Exception:
                    pass

    ec = ecs[0] if ecs else ""
    is_enzyme = bool(ec)

    # GO terms
    go_mf, go_bp, go_cc = [], [], []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "GO":
            continue
        go_id = ref.get("id", "")
        aspect = next(
            (p["value"] for p in ref.get("properties", []) if p["key"] == "GoTerm"),
            ""
        )
        if aspect.startswith("F:") and len(go_mf) < 8:
            go_mf.append(go_id)
        elif aspect.startswith("P:") and len(go_bp) < 8:
            go_bp.append(go_id)
        elif aspect.startswith("C:") and len(go_cc) < 6:
            go_cc.append(go_id)

    # Active site residues from UniProt annotation
    active_residues = []
    for feature in data.get("features", []):
        if feature.get("type") in ("Active site", "Metal binding", "Binding site"):
            try:
                pos = feature["location"]["start"]["value"]
                if pos not in active_residues:
                    active_residues.append(pos)
            except Exception:
                pass

    # PPI partners from STRING-style comments (interaction section in UniProt)
    partners = []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "STRING":
            pass  # STRING IDs not gene names; skip
    # Use subunit comments for known partners
    for comment in data.get("comments", []):
        if comment.get("commentType") == "SUBUNIT":
            # Partners mentioned in free text — not easily parseable, skip
            pass

    # Assign category
    category = "other"
    gene_upper = gene.upper()
    if ec.startswith("2.7"):
        category = "kinase"
    elif ec.startswith("3.4"):
        category = "protease"
    elif ec.startswith("1."):
        category = "oxidoreductase"
    elif ec.startswith("2.") and not ec.startswith("2.7"):
        category = "transferase"
    elif ec.startswith("3.") and not ec.startswith("3.4"):
        category = "hydrolase"
    elif any(x in gene_upper for x in ["HSP", "DNAJ", "HSPA", "HSPB"]):
        category = "chaperone"
    elif any(x in gene_upper for x in ["BCL", "BAX", "BAK", "BID", "MCL"]):
        category = "apoptosis"
    elif any(x in gene_upper for x in ["TP53", "MYC", "ATF", "USF", "JUN", "FOS"]):
        category = "transcription_factor"
    elif any(x in gene_upper for x in ["MSH", "BRCA", "RAD"]):
        category = "dna_repair"
    elif any(x in gene_upper for x in ["RAS", "RAC", "RHO", "CDC42"]):
        category = "gtpase"

    return {
        "uniprot_id":            uid,
        "gene":                  gene,
        "description":           desc,
        "organism":              organism,
        "is_enzyme":             is_enzyme,
        "ec_number":             ec,
        "known_go_mf":           go_mf,
        "known_go_bp":           go_bp,
        "known_go_cc":           go_cc,
        "known_active_residues": active_residues[:10],
        "known_partners":        partners,
        "category":              category,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
new_entries = existing_entries.copy()
failed = []

for i, uid in enumerate(to_fetch):
    print(f"[{i+1}/{len(to_fetch)}] Fetching {uid}...", end=" ", flush=True)
    data = fetch_uniprot(uid)
    if data is None:
        failed.append(uid)
        print("FAILED")
        continue
    entry = parse_entry(uid, data)
    new_entries.append(entry)
    print(f"OK  {entry['gene']:10s}  enzyme={entry['is_enzyme']}  "
          f"EC={entry['ec_number'] or '-':12s}  "
          f"GO={len(entry['known_go_mf'])+len(entry['known_go_bp'])+len(entry['known_go_cc'])} terms")
    time.sleep(0.3)   # be polite to UniProt API

# ── Save ──────────────────────────────────────────────────────────────────────
NEW_ENTRIES_PATH.write_text(
    json.dumps(new_entries, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"\n{'='*60}")
print(f"  Added   : {len(new_entries) - len(existing_entries)} new proteins")
print(f"  Total   : {len(new_entries)} proteins in new_entries.json")
print(f"  Failed  : {len(failed)} {failed if failed else ''}")
print(f"\nNow run:")
print(f"  python validation/run_validation.py --score-only")
print(f"{'='*60}")