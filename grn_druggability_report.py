import json
from pathlib import Path

targets = [
    ('ATAD2',   'Q6PL18'),
    ('TOP2A',   'P11388'),
    ('STMN1',   'P16949'),
    ('SLC2A1',  'P11166'),
    ('CLSPN',   'Q9HAW4'),
    ('HELLS',   'Q9NRZ9'),
    ('LCN2',    'P80188'),
    ('CEACAM6', 'P40199'),
]

print('=' * 75)
print(f'  GRN TARGET DRUGGABILITY REPORT')
print(f'  Top regulators from PDAC tumor GRN x ProteinFP predictions')
print('=' * 75)
print(f'  {"Gene":<10} {"UniProt":<10} {"Category":<18} {"Top Pocket":<12} {"Druggability":<14} {"Active Sites"}')
print(f'  {"-"*10} {"-"*10} {"-"*18} {"-"*12} {"-"*14} {"-"*12}')

for gene, uid in targets:
    report_path = Path(f'data/reports/{uid}_report.json')
    if not report_path.exists():
        print(f'  {gene:<10} {uid:<10} NO REPORT')
        continue
    
    r = json.loads(report_path.read_text())
    
    # Category
    cat_path = Path(f'data/intermediate/{uid}_category.json')
    category = 'unknown'
    cat_conf = 0.0
    if cat_path.exists():
        cat = json.loads(cat_path.read_text())
        category = cat.get('top_category', 'unknown')
        cat_conf = cat.get('confidence', 0.0)

    # Best pocket
    pockets = r.get('binding_pockets', [])
    if pockets:
        top_pocket = pockets[0]
        vol        = top_pocket.get('volume_A3', 0)
        drug_score = top_pocket.get('druggability_score', 0)
        drug_class = top_pocket.get('druggability_class', '?')
    else:
        vol, drug_score, drug_class = 0, 0, 'none'

    # Active sites
    n_active = len(r.get('active_sites', []))
    
    # GO terms
    go_mf = r.get('go_terms_mf', [])
    top_go = go_mf[0]['go_name'][:30] if go_mf else 'none'

    print(f'  {gene:<10} {uid:<10} {category:<18} {vol:>6.0f}A3    '
          f'{drug_score:.2f} ({drug_class:<4})  {n_active} residues')
    print(f'  {"":<10} {"":<10} conf={cat_conf:.2f}            '
          f'Top GO: {top_go}')
    print()

print('=' * 75)
print('  PRIORITY RANKING (by druggability score):')
print('=' * 75)

scored = []
for gene, uid in targets:
    rp = Path(f'data/reports/{uid}_report.json')
    if not rp.exists():
        continue
    r = json.loads(rp.read_text())
    pockets = r.get('binding_pockets', [])
    if pockets:
        scored.append((gene, uid, pockets[0].get('druggability_score', 0),
                       pockets[0].get('volume_A3', 0),
                       pockets[0].get('druggability_class', '?')))

scored.sort(key=lambda x: -x[2])
for rank, (gene, uid, score, vol, cls) in enumerate(scored, 1):
    print(f'  {rank}. {gene:<10} druggability={score:.2f} ({cls})  vol={vol:.0f}A3')
