import json

# 1. Enzyme classification diagnostic
print("=== 1. ENZYME CLASSIFICATION ===")
proteins = {
    'P00441': 'SOD1 (enzyme)',
    'P06213': 'INSR (enzyme)',
    'P00734': 'F2 (enzyme)',
    'Q00987': 'MDM2 (enzyme)',
    'Q9BYF1': 'ACE2 (enzyme)',
    'P07900': 'HSP90 (NON-enzyme)',
    'P04637': 'TP53 (NON-enzyme)',
    'P38398': 'BRCA1 (NON-enzyme)',
    'P68871': 'HBB (NON-enzyme)',
}
for uid, desc in proteins.items():
    ec = json.load(open('data/intermediate/' + uid + '_ec_prediction.json'))
    go = json.load(open('data/intermediate/' + uid + '_go_predictions.json'))
    mf_ids = {p['go_id'] for p in go.get('mf_predictions', [])}
    bp_ids = {p['go_id'] for p in go.get('bp_predictions', [])}
    print(uid + ' ' + desc)
    print('  is_enzyme=' + str(ec['is_enzyme']) + '  conf=' + str(round(ec['enzyme_confidence'],2)) + '  non_enz=' + str(round(ec['non_enzyme_score'],2)))
    print('  GO:0051082=' + str('GO:0051082' in mf_ids) + '  GO:0006457=' + str('GO:0006457' in bp_ids))
    print('  GO:0003700=' + str('GO:0003700' in mf_ids) + '  GO:0003677=' + str('GO:0003677' in mf_ids))
    print()

# 2. GO namespace routing
print("=== 2. GO NAMESPACE ROUTING ===")
checks = {
    'P00533': ('EGFR', ['GO:0007173','GO:0018108','GO:0008283']),
    'P06213': ('INSR', ['GO:0008286','GO:0046628']),
    'P00734': ('F2',   ['GO:0007596','GO:0030193']),
    'P68871': ('HBB',  ['GO:0015671','GO:0019430']),
}
for uid, (gene, known_bp) in checks.items():
    report = json.load(open('data/reports/' + uid + '_report.json'))
    pred_mf = {t['go_id'] for t in report.get('go_terms_mf', [])}
    pred_bp = {t['go_id'] for t in report.get('go_terms_bp', [])}
    print(uid + ' ' + gene)
    for gid in known_bp:
        if gid in pred_bp:
            status = 'CORRECT->BP'
        elif gid in pred_mf:
            status = 'MISROUTED->MF'
        else:
            status = 'MISSING'
        print('  ' + gid + ': ' + status)
    print()

# 3. New motif detection
print("=== 3. NEW MOTIF DETECTION ===")
motif_checks = {
    'P07900': 'HSP90 (want ghkl_atpase)',
    'P68871': 'HBB (want haem_binding)',
    'P16083': 'NQO2 (want flavin_binding)',
}
for uid, desc in motif_checks.items():
    active = json.load(open('data/intermediate/' + uid + '_active_sites.json'))
    motif_types = sorted({m['motif_type'] for m in active['catalytic_motifs']})
    print(uid + ' ' + desc)
    print('  ' + str(motif_types))
    print()

# 4. GO terms from new motifs
print("=== 4. NEW GO TERMS IN REPORTS ===")
go_checks = {
    'P07900': ('HSP90', ['GO:0005524','GO:0051082','GO:0042623','GO:0006457']),
    'P68871': ('HBB',   ['GO:0020037','GO:0019825','GO:0015671','GO:0005833']),
    'P16083': ('NQO2',  ['GO:0003955','GO:0010181','GO:0055114','GO:0016491']),
    'P38398': ('BRCA1', ['GO:0003684','GO:0006281','GO:0045739']),
}
for uid, (gene, expected) in go_checks.items():
    report = json.load(open('data/reports/' + uid + '_report.json'))
    all_pred = (
        {t['go_id'] for t in report.get('go_terms_mf',[])} |
        {t['go_id'] for t in report.get('go_terms_bp',[])} |
        {t['go_id'] for t in report.get('go_terms_cc',[])}
    )
    print(uid + ' ' + gene)
    for gid in expected:
        print('  ' + gid + ': ' + ('FOUND' if gid in all_pred else 'MISSING'))
    print()

# 5. Active site residues
print("=== 5. ACTIVE SITE RESIDUES ===")
site_checks = {
    'P07900': ('HSP90', [35, 83, 183]),
    'P68871': ('HBB',   [92]),
    'P16083': ('NQO2',  [103, 128]),
    'P38398': ('BRCA1', [1763, 1836]),
}
for uid, (gene, known) in site_checks.items():
    active = json.load(open('data/intermediate/' + uid + '_active_sites.json'))
    rns = [r['residue_number'] for r in active['active_residues']
           if r['confidence'] in ('HIGH','MEDIUM')]
    print(uid + ' ' + gene)
    for ka in known:
        found = any(abs(r-ka)<=3 for r in rns)
        print('  Residue ' + str(ka) + ': ' + ('FOUND' if found else 'MISSING'))
    print()

# 6. clean_ec.py source check
print("=== 6. CLEAN_EC SOURCE CHECK ===")
import inspect
from pipeline import clean_ec
src = inspect.getsource(clean_ec.predict_ec_number)
for phrase in ['transcription factor activity', 'DNA-binding transcription',
               'GO:0051082', 'GO:0006457', 'unfolded protein', 'ghkl_atpase',
               'haem_binding', 'flavin_binding']:
    print('  ' + repr(phrase) + ': ' + str(phrase in src))
