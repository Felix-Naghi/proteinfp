import json
from pathlib import Path

checks = {
    'P04637': ('TP53',  ['dna_binding_cluster', 'zinc_binding_cluster'], ['ghkl_atpase', 'flavin_binding', 'haem_binding']),
    'P00533': ('EGFR',  ['dfg_loop', 'hrd_catalytic_loop'],              ['haem_binding', 'ghkl_atpase']),
    'P07900': ('HSP90', ['ghkl_atpase'],                                  ['haem_binding', 'flavin_binding', 'dfg_loop']),
    'P68871': ('HBB',   ['haem_binding'],                                 ['dfg_loop', 'ghkl_atpase', 'flavin_binding']),
    'P16083': ('NQO2',  ['flavin_binding'],                               ['haem_binding', 'ghkl_atpase', 'dfg_loop']),
    'P01116': ('KRAS',  ['p_loop_walker_a'],                              ['haem_binding', 'flavin_binding', 'dfg_loop']),
    'P00734': ('F2',    ['serine_protease_triad'],                        ['haem_binding', 'flavin_binding', 'ghkl_atpase']),
    'P42574': ('CASP3', ['cysteine_protease_dyad'],                       ['haem_binding', 'flavin_binding', 'ghkl_atpase']),
    'P00918': ('CA2',   ['zinc_binding_cluster'],                         ['haem_binding', 'flavin_binding', 'dfg_loop']),
}

print('Motif detection check:')
print('=' * 65)
all_pass = True
for uid, (gene, should_have, should_not) in checks.items():
    path = Path('data/intermediate/' + uid + '_active_sites.json')
    if not path.exists():
        print(uid + ' ' + gene + ': MISSING active_sites.json')
        continue
    active = json.loads(path.read_text())
    found = {m['motif_type'] for m in active.get('catalytic_motifs', [])}
    missing  = [m for m in should_have if m not in found]
    unwanted = [m for m in should_not  if m in found]
    status = 'PASS' if not missing and not unwanted else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(uid + ' ' + gene + ' [' + status + '] total_motifs=' + str(len(found)))
    if missing:
        print('  MISSING:  ' + str(missing))
    if unwanted:
        print('  UNWANTED: ' + str(unwanted))
    if status == 'PASS':
        print('  motifs: ' + str(sorted(found)))

print()
print('Overall: ' + ('ALL PASS' if all_pass else 'SOME FAILURES - tighten thresholds'))
