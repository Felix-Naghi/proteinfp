import json
from pathlib import Path

# AZ13824374 physicochemical properties
# MW=592, logP=2.1, highly polar, good membrane permeability
# pIC50=8.2 (FRET), pIC50=6.2 (cellular NanoBRET)
# Designed for ATAD2 bromodomain acetyllysine pocket

drug = {
    "name":             "AZ13824374",
    "smiles":           "O=C1C2CN3CC(NC4=NN5N=C(C)C=CC5=N4)CCC3CC2N(C(=O)c2cc3n(CC(F)(C)C)ncc3nc2)C1",
    "molecular_weight": 592.7,
    "logP":             2.1,
    "pKa_basic":        6.8,
    "pKa_acidic":       12.0,
    "hbd":              2,
    "hba":              9,
    "psa":              112.4,
    "charge_at_pH74":   0.0,
    "rotatable_bonds":  5,
    "permeability":     "medium",
    "transporter":      None,
    "known_targets":    ["ATAD2"],
    "mechanism":        "Bromodomain inhibitor - acetyllysine competitive",
    "clinical_status":  "Research compound - antiproliferative in breast cancer models",
    "experimental_pIC50_FRET":    8.2,
    "experimental_pIC50_cellular": 6.2,
    "experimental_IC50_nM":       6.3,
}

out = Path("data/sim/molecule_AZ13824374_props.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(drug, indent=2))
print(f"Saved drug properties to {out}")
print(f"MW={drug['molecular_weight']}  logP={drug['logP']}  pKi_expected~8.2")
