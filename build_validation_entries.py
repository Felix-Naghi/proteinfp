import json, requests, sys
from pathlib import Path

new_uids = [
    "P15056","P00519","P27361","Q02750","P45983","Q13153","P06239","P08581","P04049",
    "P55210","P29466","P08246","P07711","P07858","P01106","P10275","P03372","Q01094",
    "P17275","P05412","P01100","P14921","P00387","P00352","P14550","P11473","Q07869",
    "P37231","P10827","P20393","P08238","P11142","P04792","P02511","Q14524","P35561",
    "P62834","P84095","P61586","P60953","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P69905","P02768","P11166","P51587","P12004","P02452","P68032","P02144","P08473",
    "P15144","P00488","P04180","P01375","P05156","P46108","Q13480"
]

ENZYME_GO = {"GO:0003824","GO:0016787","GO:0016301","GO:0016740","GO:0016829",
             "GO:0016853","GO:0016874","GO:0016491","GO:0004672","GO:0008233","GO:0061630"}

entries = []
for uid in new_uids:
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json", timeout=15)
        d = r.json()

        gene = uid
        for g in d.get("genes", []):
            if "geneName" in g:
                gene = g["geneName"]["value"]
                break

        go_mf, go_bp, go_cc = [], [], []
        is_enzyme = False
        for ref in d.get("uniProtKBCrossReferences", []):
            if ref["database"] != "GO":
                continue
            gid = ref["id"]
            for prop in ref.get("properties", []):
                if prop["key"] == "GoTerm":
                    aspect = prop["value"][0]
                    if aspect == "F": go_mf.append(gid)
                    elif aspect == "P": go_bp.append(gid)
                    elif aspect == "C": go_cc.append(gid)
                if prop["key"] == "GoTerm" and gid in ENZYME_GO:
                    is_enzyme = True
        is_enzyme = any(g in ENZYME_GO for g in go_mf)

        ec = ""
        for ref in d.get("uniProtKBCrossReferences", []):
            if ref["database"] == "EC":
                ec = ref["id"]
                break

        desc = ""
        for c in d.get("comments", []):
            if c["commentType"] == "FUNCTION":
                desc = c.get("texts", [{}])[0].get("value", "")[:120]
                break

        org = d.get("organism", {}).get("scientificName", "Homo sapiens")

        entry = {
            "uniprot_id": uid,
            "gene": gene,
            "description": desc or gene,
            "organism": org,
            "is_enzyme": is_enzyme,
            "ec_number": ec,
            "known_go_mf": go_mf[:6],
            "known_go_bp": go_bp[:6],
            "known_go_cc": go_cc[:4],
            "known_active_residues": [],
            "known_partners": [],
            "category": "other",
        }
        entries.append(entry)
        print(f"OK {uid} {gene} enzyme={is_enzyme} ec={ec} mf={len(go_mf)} bp={len(go_bp)}")
    except Exception as e:
        print(f"FAIL {uid}: {e}")

Path("validation/new_entries.json").write_text(json.dumps(entries, indent=2))
print(f"\nSaved {len(entries)} entries to validation/new_entries.json")
