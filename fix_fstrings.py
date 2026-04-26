"""Compare v5 vs v6 — run from project root: python diff_v5_v6.py"""
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
gt = {}
try:
    from run_validation import VALIDATION_SET
    for p in VALIDATION_SET: gt[p["uniprot_id"]] = p
except Exception: pass
ne = Path("validation/new_entries.json")
if ne.exists():
    for p in json.loads(ne.read_text()):
        uid = p.get("uniprot_id", "")
        if uid and uid not in gt: gt[uid] = p

v5 = json.loads(Path("data/reports/validation/validation_report_v5.json").read_text())
v6 = json.loads(Path("data/reports/validation/validation_report_v6.json").read_text())
v5s = {s["uniprot_id"]: s for s in v5["scores"]}
v6s = {s["uniprot_id"]: s for s in v6["scores"]}

imp, wor, still_wrong = [], [], []
for uid, s6 in v6s.items():
    s5 = v5s.get(uid, {})
    c5 = s5.get("enzyme_correct", False)
    c6 = s6.get("enzyme_correct", False)
    g  = s6.get("gene", "?")
    is_enz = gt.get(uid, {}).get("is_enzyme", None)
    lbl = "enzyme" if is_enz else "non-enz"
    if not c5 and c6:
        imp.append((uid, g, lbl))
    elif c5 and not c6:
        wor.append((uid, g, lbl))
    elif not c5 and not c6:
        still_wrong.append((uid, g, lbl))

print(f"\nv5 → v6:  +{len(imp)} improved  -{len(wor)} worsened")
print(f"Still wrong: {len(still_wrong)}")
print()

if imp:
    print(f"IMPROVED ({len(imp)}):")
    for uid, g, lbl in imp:
        print(f"  {uid:<10} {g:<12} [{lbl}]")

if wor:
    print(f"\nWORSENED ({len(wor)}):")
    for uid, g, lbl in wor:
        print(f"  {uid:<10} {g:<12} [{lbl}]")

if still_wrong:
    print(f"\nSTILL WRONG ({len(still_wrong)}):")
    for uid, g, lbl in still_wrong:
        print(f"  {uid:<10} {g:<12} [{lbl}]")