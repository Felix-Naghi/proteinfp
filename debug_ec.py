import json
from pathlib import Path

for uid, gene, expected in [
    ('P01116','KRAS', True),
    ('P42574','CASP3', True),
    ('P04637','TP53', False),
    ('P07900','HSP90', False),
    ('P06213','INSR', True),
]:
    ec = json.loads(Path('data/intermediate/' + uid + '_ec_prediction.json').read_text())
    print(uid + ' ' + gene + ' expected=' + str(expected))
    print('  is_enzyme=' + str(ec['is_enzyme']) + 
          ' conf=' + str(ec['enzyme_confidence']) + 
          ' non_enz=' + str(ec['non_enzyme_score']) +
          ' ml_prob=' + str(ec.get('ml_enzyme_prob', 'N/A')))
    active = json.loads(Path('data/intermediate/' + uid + '_active_sites.json').read_text())
    motifs = {m['motif_type'] for m in active.get('catalytic_motifs', [])}
    print('  motifs=' + str(sorted(motifs)))
    print()
