import json
from pathlib import Path

print('=== MDM4 pipeline state ===')
inter = Path('data/intermediate')
for mod in ['structure','physicochemical','homology','go_predictions','ppi']:
    p = inter / ('O15151_' + mod + '.json')
    if p.exists():
        d = json.loads(p.read_text())
        if mod == 'homology':
            print('  homology: blast_hits=' + str(len(d.get('blast_hits',[]))) + 
                  ' go_terms=' + str(len(d.get('all_go_terms',[]))))
        elif mod == 'ppi':
            print('  ppi: partners=' + str(len(d.get('partners',[]))))
        elif mod == 'structure':
            print('  structure: length=' + str(d.get('length',0)) + 
                  ' gene=' + d.get('gene_name','?'))
        else:
            print('  ' + mod + ': OK')
    else:
        print('  ' + mod + ': MISSING')

print()
print('=== CASP3 pipeline state ===')
for mod in ['structure','physicochemical','homology','go_predictions','ppi','active_sites']:
    p = inter / ('P42574_' + mod + '.json')
    if p.exists():
        d = json.loads(p.read_text())
        if mod == 'homology':
            print('  homology: blast_hits=' + str(len(d.get('blast_hits',[]))) + 
                  ' go_terms=' + str(len(d.get('all_go_terms',[]))))
        elif mod == 'ppi':
            print('  ppi: partners=' + str(len(d.get('partners',[]))))
        elif mod == 'active_sites':
            motifs = {m['motif_type'] for m in d.get('catalytic_motifs',[])}
            rns = [r['residue_number'] for r in d.get('active_residues',[]) 
                   if r.get('confidence') in ('HIGH','MEDIUM')]
            c163 = any(abs(r-163)<=3 for r in rns)
            c184 = any(abs(r-184)<=3 for r in rns)
            print('  active_sites: motifs=' + str(sorted(motifs)) + 
                  ' C163=' + str(c163) + ' C184=' + str(c184))
        elif mod == 'structure':
            print('  structure: length=' + str(d.get('length',0)) + 
                  ' gene=' + d.get('gene_name','?'))
        else:
            print('  ' + mod + ': OK')
    else:
        print('  ' + mod + ': MISSING')

print()
print('=== TP53 enzyme classification ===')
ec = json.loads(Path('data/intermediate/P04637_ec_prediction.json').read_text())
print('  is_enzyme=' + str(ec['is_enzyme']) + 
      ' conf=' + str(round(ec['enzyme_confidence'],2)) + 
      ' non_enz=' + str(round(ec['non_enzyme_score'],2)))
go = json.loads(Path('data/intermediate/P04637_go_predictions.json').read_text())
mf_ids = {p['go_id']: round(p['score'],2) for p in go.get('mf_predictions',[])}
print('  GO:0003700 score=' + str(mf_ids.get('GO:0003700', 'NOT FOUND')))
print('  GO:0003677 score=' + str(mf_ids.get('GO:0003677', 'NOT FOUND')))
