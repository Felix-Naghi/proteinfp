import json
import requests
from pathlib import Path

# All 76 proteins minus the 15 already in ground truth
new_uids = [
    "P15056","P00519","P27361","Q02750","P45983","Q13153","P06239","P08581","P04049",
    "P55210","P29466","P08246","P07711","P07858","P01106","P10275","P03372","Q01094",
    "P17275","P05412","P01100","P14921","P00387","P00352","P14550","P11473","Q07869",
    "P37231","P10827","P20393","P08238","P11142","P04792","P02511","Q14524","P35561",
    "P62834","P84095","P61586","P60953","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P69905","P02768","P11166","P51587","P12004","P02452","P68032","P02144","P08473",
    "P15144","P00488","P04180","P01375","P05156","P46108","Q13480"
]

gt_path = Path("validation/ground_truth.json")
gt = json.loads(gt_path.read_text())
print(f"Existing ground truth: {len(gt)} proteins")

for uid in new_uids:
    if uid in gt:
        continue
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
        r = requests.get(url, timeout=15)
        data = r.json()
        
        # Get gene name
        gene = uid
        for g in data.get("genes", []):
            if "geneName" in g:
                gene = g["geneName"]["value"]
                break
        
        # Get GO terms
        go_terms = []
        for ref in data.get("uniProtKBCrossReferences", []):
            if ref["database"] == "GO":
                go_id = ref["id"]
                for prop in ref.get("properties", []):
                    if prop["key"] == "GoTerm":
                        go_terms.append({"id": go_id, "term": prop["value"]})
        
        # Get function description
        function_desc = ""
        for comment in data.get("comments", []):
            if comment["commentType"] == "FUNCTION":
                for text in comment.get("texts", []):
                    function_desc = text["value"][:200]
                    break
            if function_desc:
                break

        gt[uid] = {
            "gene_name": gene,
            "go_terms": go_terms[:20],
            "function_description": function_desc,
            "source": "uniprot_auto"
        }
        print(f"  Added {uid} ({gene}) - {len(go_terms)} GO terms")
    except Exception as e:
        print(f"  FAILED {uid}: {e}")

gt_path.write_text(json.dumps(gt, indent=2))
print(f"Done. Ground truth now has {len(gt)} proteins")
