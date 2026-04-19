import json
from pathlib import Path

for uid, gene in [('P04637','TP53'),('P00533','EGFR'),('P07900','HSP90'),('P68871','HBB'),('P16083','NQO2'),('P01116','KRAS'),('P00734','F2'),('P00918','CA2')]:
    active = json.loads(Path('data/intermediate/' + uid + '_active_sites.json').read_text())
    found = {m['motif_type'] for m in active.get('catalytic_motifs', [])}
    print(uid + ' ' + gene + ': ' + str(sorted(found)))
