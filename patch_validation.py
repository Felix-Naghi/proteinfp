import json, re
from pathlib import Path

new_entries = json.loads(Path('validation/new_entries.json').read_text())

# Assign proper categories
categories = {
    'BRAF':'kinase','ABL1':'kinase','MAPK3':'kinase','MAP2K1':'kinase',
    'MAPK8':'kinase','PAK1':'kinase','LCK':'kinase','MET':'kinase','RAF1':'kinase',
    'CASP7':'cysteine_protease','CASP1':'cysteine_protease',
    'ELANE':'serine_protease','CTSL':'cysteine_protease','CTSB':'cysteine_protease',
    'MYC':'transcription_factor','AR':'transcription_factor','ESR1':'transcription_factor',
    'E2F1':'transcription_factor','JUNB':'transcription_factor','JUN':'transcription_factor',
    'FOS':'transcription_factor','ETS1':'transcription_factor',
    'CYB5R3':'oxidoreductase','ALDH1A1':'oxidoreductase','AKR1A1':'oxidoreductase',
    'VDR':'nuclear_receptor','PPARA':'nuclear_receptor','PPARG':'nuclear_receptor',
    'THRA':'nuclear_receptor','NR1D1':'nuclear_receptor',
    'HSP90AB1':'chaperone','HSPA8':'chaperone','HSPB1':'chaperone','CRYAB':'chaperone',
    'SCN5A':'ion_channel','Kcnj2':'ion_channel',
    'RAP1A':'gtpase','RHOG':'gtpase','RHOA':'gtpase','CDC42':'gtpase',
    'RNF216':'ubiquitin_ligase','CDC26':'ubiquitin_ligase',
    'PTPN2':'phosphatase','PTPN6':'phosphatase','PTPN11':'phosphatase',
    'HBA1':'oxygen_transport','ALB':'transport','SLC2A1':'transport',
    'BRCA2':'dna_repair','PCNA':'dna_repair',
    'COL1A1':'structural','ACTC1':'structural','MB':'oxygen_transport',
    'MME':'metallopeptidase','ANPEP':'metallopeptidase',
    'F13A1':'other','LCAT':'lipid_metabolism',
    'TNF':'immune','CFI':'immune',
    'CRK':'adaptor','GAB1':'adaptor',
}

lines = []
for e in new_entries:
    cat = categories.get(e['gene'], 'other')
    line = f"""    {{
        "uniprot_id":  "{e['uniprot_id']}",
        "gene":        "{e['gene']}",
        "description": "{e['description'][:80].replace(chr(34), chr(39))}",
        "organism":    "{e['organism']}",
        "is_enzyme":   {str(e['is_enzyme']).lower()},
        "ec_number":   "{e['ec_number']}",
        "known_go_mf": {json.dumps(e['known_go_mf'][:4])},
        "known_go_bp": {json.dumps(e['known_go_bp'][:4])},
        "known_go_cc": {json.dumps(e['known_go_cc'][:3])},
        "known_active_residues": [],
        "known_partners":        [],
        "category":    "{cat}",
    }},"""
    lines.append(line)

# Read validation file
src = Path('validation/run_validation.py').read_text(encoding='utf-8')

# Insert before the closing bracket of VALIDATION_SET
insert_point = src.rfind(']', 0, src.find('QUICK_SET'))
new_entries_str = '\n' + '\n'.join(lines) + '\n'
new_src = src[:insert_point] + new_entries_str + src[insert_point:]

Path('validation/run_validation.py').write_text(new_src, encoding='utf-8')
print(f'Added {len(lines)} proteins to VALIDATION_SET')
print('Verifying...')
import ast
ast.parse(new_src)
print('Syntax OK')
