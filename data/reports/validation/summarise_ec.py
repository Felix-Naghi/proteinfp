import json
import glob
import os
from collections import Counter

results = []
for f in glob.glob("data/intermediate/*_ec_ml.json"):
    uid = os.path.basename(f).replace("_ec_ml.json", "")
    with open(f) as fh:
        d = json.load(fh)
    is_enzyme   = d.get("is_enzyme", False)
    enzyme_conf = d.get("enzyme_confidence", 0)
    top         = d.get("top_prediction", {})
    top_ec      = top.get("ec_class", "?")
    top_ec_name = top.get("ec_name", "?")
    top_prob    = top.get("probability", 0)
    entropy     = d.get("ml_entropy", 0)
    results.append((uid, is_enzyme, enzyme_conf, top_ec, top_ec_name, top_prob, entropy))

results.sort(key=lambda x: x[0])

print(f"{'UniProt':<12} {'Enzyme?':<8} {'Confidence':<12} {'EC':<4} {'EC Name':<16} {'Prob':<8} {'Entropy'}")
print("-" * 72)
for uid, enz, conf, top_ec, top_name, top_prob, entropy in results:
    flag = "YES" if enz else "NO "
    print(f"{uid:<12} {flag:<8} {conf:<12.3f} {top_ec:<4} {top_name:<16} {top_prob:<8.3f} {entropy:.3f}")

print(f"\nTotal proteins : {len(results)}")
print(f"Enzymes        : {sum(1 for r in results if r[1])}")
print(f"Non-enzymes    : {sum(1 for r in results if not r[1])}")

ec_counts = Counter(r[3] for r in results if r[1])
names = {"1":"Oxidoreductase","2":"Transferase","3":"Hydrolase",
         "4":"Lyase","5":"Isomerase","6":"Ligase","7":"Translocase"}
print(f"\nEC class breakdown (enzymes only):")
for ec_class, count in sorted(ec_counts.items()):
    print(f"  EC {ec_class} ({names.get(ec_class,'?'):<16}): {count:>3} proteins")