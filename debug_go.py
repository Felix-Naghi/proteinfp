import json
from pathlib import Path

proteins = {
    'P51587': ('BRCA2', ['GO:0043015','GO:0010484','GO:0010485','GO:0042802','GO:0002020','GO:0003697'],
                        ['GO:0007420','GO:0071479','GO:0090398','GO:0051298','GO:0030330','GO:0006302'],
                        ['GO:0033593','GO:0005813','GO:0000781','GO:0005829']),
    'P12004': ('PCNA',  ['GO:0003682','GO:0003684','GO:0032139','GO:0070182','GO:0030337','GO:0019899'],
                        ['GO:0006287','GO:0070301','GO:0034644','GO:0071466','GO:0006325','GO:0030855'],
                        ['GO:0005813','GO:0000785','GO:0000781','GO:0000307']),
    'P08473': ('MME',   ['GO:1901612','GO:0004175','GO:0008238','GO:0004181','GO:0004222','GO:0070012'],
                        ['GO:0021764','GO:0097242','GO:0150094','GO:0050435','GO:0002003','GO:0010815'],
                        ['GO:0030424','GO:0005903','GO:0009986','GO:0005813']),
    'Q13153': ('PAK1',  ['GO:0005524','GO:0005518','GO:0043015','GO:0042802','GO:0004672','GO:0019901'],
                        ['GO:0030036','GO:0006915','GO:0048754','GO:0016477','GO:0032869','GO:0009267'],
                        ['GO:0005884','GO:0030424','GO:0005911','GO:0005813']),
}

for uid, (gene, known_mf, known_bp, known_cc) in proteins.items():
    report_path = Path('data/reports') / f'{uid}_report.json'
    if not report_path.exists():
        print(f'{uid} {gene}: NO REPORT')
        continue
    r = json.loads(report_path.read_text())
    pred_mf = {t['go_id'] for t in r.get('go_terms_mf', [])}
    pred_bp = {t['go_id'] for t in r.get('go_terms_bp', [])}
    pred_cc = {t['go_id'] for t in r.get('go_terms_cc', [])}
    all_pred = pred_mf | pred_bp | pred_cc
    all_known = set(known_mf) | set(known_bp) | set(known_cc)
    hits = all_pred & all_known
    print(f'{uid} {gene}:')
    print(f'  Predicted: {len(all_pred)} GO terms total')
    print(f'  Known:     {len(all_known)} GO terms')
    print(f'  Hits:      {len(hits)} -> {sorted(hits)}')
    print(f'  Predicted MF: {sorted(pred_mf)[:8]}')
    print(f'  Known MF:     {sorted(known_mf)[:8]}')
    print()
