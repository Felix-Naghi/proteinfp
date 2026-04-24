import urllib.request
import json
import glob
import os
import time

EC_NAMES = {
    "1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase",
    "4": "Lyase",          "5": "Isomerase",   "6": "Ligase",
    "7": "Translocase"
}

# Get all UniProt IDs we have ML predictions for
ml_files = sorted(glob.glob("data/intermediate/*_ec_ml.json"))
uids = [os.path.basename(f).replace("_ec_ml.json", "") for f in ml_files]

print(f"Fetching real EC annotations from UniProt for {len(uids)} proteins...")
print("(This may take a minute)\n")

true_ec_map = {}
for uid in uids:
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())

        # Extract EC numbers from recommendedName
        ec_numbers = []
        prot_desc = d.get("proteinDescription", {})
        rec_name  = prot_desc.get("recommendedName", {})
        for ec_entry in rec_name.get("ecNumbers", []):
            ec_numbers.append(ec_entry.get("value", ""))

        # Also check altNames
        for alt in prot_desc.get("alternativeNames", []):
            for ec_entry in alt.get("ecNumbers", []):
                ec_numbers.append(ec_entry.get("value", ""))

        if ec_numbers:
            # Take first digit of first EC number as the class
            true_ec_map[uid] = ec_numbers[0].split(".")[0]
        else:
            true_ec_map[uid] = "non-enzyme"

        time.sleep(0.2)  # be polite to UniProt API

    except Exception as e:
        true_ec_map[uid] = f"error: {e}"

# Now compare against ML predictions
correct, wrong, skipped = [], [], []

for ml_file in ml_files:
    uid = os.path.basename(ml_file).replace("_ec_ml.json", "")
    with open(ml_file) as f:
        ml = json.load(f)

    ml_ec   = (ml.get("top_prediction") or {}).get("ec_class", "?")
    ml_enz  = ml.get("is_enzyme", False)
    ml_conf = ml.get("enzyme_confidence", 0)
    true_ec = true_ec_map.get(uid, "unknown")

    if true_ec.startswith("error") or true_ec == "unknown":
        skipped.append((uid, ml_ec, ml_conf, true_ec))
        continue

    if true_ec == "non-enzyme":
        match = not ml_enz
        true_label = "non-enzyme"
    else:
        match = (ml_ec == true_ec)
        true_label = f"EC{true_ec}"

    row = (uid, true_label, ml_ec, ml_conf, match)
    if match:
        correct.append(row)
    else:
        wrong.append(row)

total = len(correct) + len(wrong)
acc   = len(correct) / total * 100 if total > 0 else 0

print(f"{'UniProt':<12} {'True EC':<16} {'ML EC':<18} {'Conf':<6} {'Result'}")
print("-" * 65)
for uid, true_ec, ml_ec, conf, match in sorted(correct + wrong, key=lambda x: x[0]):
    ml_name  = EC_NAMES.get(ml_ec, "Non-enzyme")[:13]
    result   = "CORRECT" if match else "WRONG"
    print(f"{uid:<12} {true_ec:<16} EC{ml_ec} {ml_name:<14} {conf:.2f}  {result}")

print(f"\n{'='*65}")
print(f"Total compared : {total}")
print(f"Correct        : {len(correct)}  ({acc:.1f}%)")
print(f"Wrong          : {len(wrong)}")
if skipped:
    print(f"Skipped        : {len(skipped)}")

if wrong:
    print(f"\nMistakes:")
    for uid, true_ec, ml_ec, conf, _ in wrong:
        ml_name   = EC_NAMES.get(ml_ec, "?")
        print(f"  {uid}: true={true_ec}  predicted=EC{ml_ec} ({ml_name})  conf={conf:.2f}")