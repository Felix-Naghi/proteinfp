import json
from pathlib import Path

pairs = json.loads(Path("data/sim/validation_pairs.json").read_text())
missing = [p for p in pairs if not (Path("data/reports") / (p["uniprot_id"] + "_report.json")).exists()]

print("Missing reports:", len(missing))
for p in missing:
    print(f"  {p['uniprot_id']} {p['gene']}")