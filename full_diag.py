import json
import subprocess
import sys
from pathlib import Path

results = []

def check(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    results.append((status, name, detail))

# ── Load all intermediate data ─────────────────────────────────────────────────
inter = Path('data/intermediate')
reports = Path('data/reports')

def load(path):
    try:
        return json.loads(Path(path).read_text())
    except:
        return {}

# All 15 validation proteins
PROTEINS = {
    'P04637': {'gene':'TP53',  'is_enzyme':False, 'ec':'',          'known_mf':['GO:0003677','GO:0003700','GO:0046872'], 'known_bp':['GO:0006915','GO:0006974','GO:0045944'], 'known_cc':['GO:0005634','GO:0043234'], 'known_active':[176,179,248,273], 'known_partners':['MDM2','MDM4','ATM','CHEK2','EP300']},
    'P00533': {'gene':'EGFR',  'is_enzyme':True,  'ec':'2',         'known_mf':['GO:0004672','GO:0004714','GO:0005006'], 'known_bp':['GO:0007173','GO:0008283','GO:0018108'], 'known_cc':['GO:0005887','GO:0016020'], 'known_active':[837,855], 'known_partners':['GRB2','SOS1','PIK3R1','SHC1']},
    'P00441': {'gene':'SOD1',  'is_enzyme':True,  'ec':'1',         'known_mf':['GO:0004784','GO:0005507','GO:0008270'], 'known_bp':['GO:0019430','GO:0006801'], 'known_cc':['GO:0005737','GO:0005634'], 'known_active':[44,46,118], 'known_partners':['CCS','TNFRSF1A']},
    'P07900': {'gene':'HSP90', 'is_enzyme':False, 'ec':'',          'known_mf':['GO:0005524','GO:0051082','GO:0042623'], 'known_bp':['GO:0006457','GO:0051085'], 'known_cc':['GO:0005737','GO:0005634'], 'known_active':[35,83,183], 'known_partners':['CDC37','AHA1','HOP','CHIP']},
    'P06213': {'gene':'INSR',  'is_enzyme':True,  'ec':'2',         'known_mf':['GO:0004672','GO:0004713','GO:0005009'], 'known_bp':['GO:0008286','GO:0046628'], 'known_cc':['GO:0005887','GO:0005615'], 'known_active':[1131,1135,1136], 'known_partners':['IRS1','IRS2','GRB2','SHC1']},
    'P38398': {'gene':'BRCA1', 'is_enzyme':False, 'ec':'',          'known_mf':['GO:0003684','GO:0003723','GO:0004842'], 'known_bp':['GO:0006281','GO:0007131','GO:0045739'], 'known_cc':['GO:0005634','GO:0010369'], 'known_active':[1763,1836], 'known_partners':['BARD1','RAD51','TP53','ATM']},
    'P16083': {'gene':'NQO2',  'is_enzyme':True,  'ec':'1',         'known_mf':['GO:0003955','GO:0010181'], 'known_bp':['GO:0055114','GO:0042493'], 'known_cc':['GO:0005737'], 'known_active':[103,128], 'known_partners':['AHR']},
    'P00734': {'gene':'F2',    'is_enzyme':True,  'ec':'3',         'known_mf':['GO:0004252','GO:0005172'], 'known_bp':['GO:0007596','GO:0030193'], 'known_cc':['GO:0005576','GO:0072562'], 'known_active':[363,419,521], 'known_partners':['F5','F8','THBD']},
    'P68871': {'gene':'HBB',   'is_enzyme':False, 'ec':'',          'known_mf':['GO:0020037','GO:0019825'], 'known_bp':['GO:0015671','GO:0019430'], 'known_cc':['GO:0005833','GO:0031838'], 'known_active':[92], 'known_partners':['HBA1','HBA2']},
    'P00918': {'gene':'CA2',   'is_enzyme':True,  'ec':'4',         'known_mf':['GO:0004089','GO:0008270'], 'known_bp':['GO:0015701','GO:0001659'], 'known_cc':['GO:0005737','GO:0005829'], 'known_active':[94,96,119], 'known_partners':['SLC4A1','CA1']},
    'P01116': {'gene':'KRAS',  'is_enzyme':True,  'ec':'3',         'known_mf':['GO:0005525','GO:0003924','GO:0019003'], 'known_bp':['GO:0007165','GO:0008283'], 'known_cc':['GO:0016020','GO:0005737'], 'known_active':[10,12,13,16], 'known_partners':['BRAF','RAF1','SOS1','RALGDS']},
    'Q00987': {'gene':'MDM2',  'is_enzyme':True,  'ec':'2',         'known_mf':['GO:0061630','GO:0042802'], 'known_bp':['GO:0043066','GO:0051726'], 'known_cc':['GO:0005634','GO:0005737'], 'known_active':[305,308,319,322], 'known_partners':['TP53','MDM4','USP7','RB1']},
    'Q9BYF1': {'gene':'ACE2',  'is_enzyme':True,  'ec':'3',         'known_mf':['GO:0008237','GO:0008241','GO:0046872'], 'known_bp':['GO:0006508','GO:0010819'], 'known_cc':['GO:0016020','GO:0005615'], 'known_active':[374,378,402], 'known_partners':['TMPRSS2','AGT','SLC6A19']},
    'O15151': {'gene':'MDM4',  'is_enzyme':False, 'ec':'',          'known_mf':['GO:0061630','GO:0008270','GO:0004842'], 'known_bp':['GO:0043066','GO:0051726','GO:0006915'], 'known_cc':['GO:0005634','GO:0005737'], 'known_active':[460,463,466,469], 'known_partners':['MDM2','TP53','USP7']},
    'P42574': {'gene':'CASP3', 'is_enzyme':True,  'ec':'3',         'known_mf':['GO:0004197','GO:0008234','GO:0008233'], 'known_bp':['GO:0006915','GO:0043525'], 'known_cc':['GO:0005737','GO:0005829'], 'known_active':[163,184], 'known_partners':['CASP8','CASP9','XIAP','PARP1']},
}

print('=' * 65)
print('ProteinFP Diagnostic Test Suite')
print('=' * 65)

for uid, gt in PROTEINS.items():
    gene = gt['gene']
    report = load(reports / (uid + '_report.json'))
    ec_data = load(inter / (uid + '_ec_prediction.json'))
    active_data = load(inter / (uid + '_active_sites.json'))

    if not report:
        check(uid + ' report_exists', False, 'report JSON missing')
        continue

    # ── Enzyme classification ──────────────────────────────────────────────────
    pred_enzyme = report.get('is_enzyme', None)
    expected_enzyme = gt['is_enzyme']
    check(uid + ' ' + gene + ' enzyme_classification',
          pred_enzyme == expected_enzyme,
          'predicted=' + str(pred_enzyme) + ' expected=' + str(expected_enzyme) +
          ' conf=' + str(round(ec_data.get('enzyme_confidence',0),2)) +
          ' non_enz=' + str(round(ec_data.get('non_enzyme_score',0),2)))

    # ── EC class ───────────────────────────────────────────────────────────────
    if gt['ec']:
        pred_ec = str(report.get('ec_number', '')).strip()
        ec_ok = len(pred_ec) > 0 and pred_ec[0] == gt['ec']
        check(uid + ' ' + gene + ' ec_class',
              ec_ok,
              'predicted=' + pred_ec + ' expected_class=' + gt['ec'])

    # ── GO term recall ─────────────────────────────────────────────────────────
    pred_mf = {t['go_id'] for t in report.get('go_terms_mf', [])}
    pred_bp = {t['go_id'] for t in report.get('go_terms_bp', [])}
    pred_cc = {t['go_id'] for t in report.get('go_terms_cc', [])}

    for gid in gt['known_mf']:
        check(uid + ' ' + gene + ' GO_MF ' + gid,
              gid in pred_mf,
              'MISSING from MF (in BP=' + str(gid in pred_bp) + ')')

    for gid in gt['known_bp']:
        check(uid + ' ' + gene + ' GO_BP ' + gid,
              gid in pred_bp,
              'MISSING from BP (in MF=' + str(gid in pred_mf) + ')')

    for gid in gt['known_cc']:
        check(uid + ' ' + gene + ' GO_CC ' + gid,
              gid in pred_cc,
              'MISSING from CC (in MF=' + str(gid in pred_mf) + ')')

    # ── Active site recall (+-3 tolerance) ────────────────────────────────────
    pred_active = set()
    for r in report.get('active_sites', []):
        pred_active.add(r['residue_number'])
    inter_path = inter / (uid + '_active_sites.json')
    if inter_path.exists():
        full = load(inter_path)
        for r in full.get('active_residues', []):
            if r.get('confidence') in ('HIGH', 'MEDIUM'):
                pred_active.add(r['residue_number'])

    for ka in gt['known_active']:
        found = any(abs(pa - ka) <= 3 for pa in pred_active)
        check(uid + ' ' + gene + ' active_site_' + str(ka),
              found,
              'residue ' + str(ka) + ' not found within +-3 tolerance')

    # ── PPI partners ───────────────────────────────────────────────────────────
    pred_partners = {p['partner_name'].upper() for p in report.get('ppi_partners', [])}
    for partner in gt['known_partners']:
        check(uid + ' ' + gene + ' PPI_' + partner,
              partner.upper() in pred_partners,
              partner + ' not in predicted partners')

    # ── Motif detection ────────────────────────────────────────────────────────
    motif_types = {m['motif_type'] for m in active_data.get('catalytic_motifs', [])}
    if gene == 'HSP90':
        check(uid + ' ' + gene + ' motif_ghkl_atpase', 'ghkl_atpase' in motif_types,
              'GHKL ATPase motif not detected. Motifs found: ' + str(sorted(motif_types)))
    if gene == 'HBB':
        check(uid + ' ' + gene + ' motif_haem_binding', 'haem_binding' in motif_types,
              'Haem binding motif not detected. Motifs found: ' + str(sorted(motif_types)))
    if gene == 'NQO2':
        check(uid + ' ' + gene + ' motif_flavin_binding', 'flavin_binding' in motif_types,
              'Flavin binding motif not detected. Motifs found: ' + str(sorted(motif_types)))
    if gene in ('EGFR', 'INSR'):
        check(uid + ' ' + gene + ' motif_dfg_loop', 'dfg_loop' in motif_types,
              'DFG loop not detected. Motifs found: ' + str(sorted(motif_types)))
    if gene in ('F2', 'CASP3'):
        check(uid + ' ' + gene + ' motif_protease',
              'serine_protease_triad' in motif_types or 'cysteine_protease_dyad' in motif_types,
              'No protease motif detected')

    # ── Namespace routing ──────────────────────────────────────────────────────
    namespace_checks = {
        'GO:0007173': 'BP', 'GO:0018108': 'BP', 'GO:0008283': 'BP',
        'GO:0008286': 'BP', 'GO:0046628': 'BP', 'GO:0007596': 'BP',
        'GO:0015671': 'BP', 'GO:0019430': 'BP', 'GO:0006457': 'BP',
        'GO:0006915': 'BP', 'GO:0043066': 'BP', 'GO:0051726': 'BP',
    }
    for gid, expected_ns in namespace_checks.items():
        if gid in pred_mf:
            check(uid + ' ' + gene + ' namespace_' + gid, False,
                  gid + ' is in MF but should be ' + expected_ns)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
passes = [r for r in results if r[0] == 'PASS']
fails  = [r for r in results if r[0] == 'FAIL']

print('FAILURES (' + str(len(fails)) + '):')
print('-' * 65)
for status, name, detail in fails:
    print('FAIL  ' + name)
    if detail:
        print('      ' + detail)

print()
print('SUMMARY: ' + str(len(passes)) + ' passed, ' + str(len(fails)) + ' failed out of ' + str(len(results)) + ' total checks')
print()

# ── Source code checks ─────────────────────────────────────────────────────────
print('SOURCE CODE CHECKS:')
print('-' * 65)
import inspect
from pipeline import clean_ec, active_sites, deepfri_go, consensus

ec_src = inspect.getsource(clean_ec.predict_ec_number)
as_src = inspect.getsource(active_sites)
go_src = inspect.getsource(deepfri_go)
con_src = inspect.getsource(consensus._infer_ns)

checks = [
    ('clean_ec',   ec_src,  'ghkl_atpase',                   'GHKL motif in EC lookup'),
    ('clean_ec',   ec_src,  'haem_binding',                   'Haem motif in EC lookup'),
    ('clean_ec',   ec_src,  'flavin_binding',                  'Flavin motif in EC lookup'),
    ('clean_ec',   ec_src,  'non_enzyme_score >= 0.35',        'Non-enzyme threshold is 0.35'),
    ('clean_ec',   ec_src,  'dna-binding transcription factor','Specific TF check in non-enzyme'),
    ('active_sites',as_src, '_find_ghkl_atpase',               'GHKL motif detector exists'),
    ('active_sites',as_src, '_find_haem_binding',              'Haem motif detector exists'),
    ('active_sites',as_src, '_find_flavin_binding',            'Flavin motif detector exists'),
    ('active_sites',as_src, 'long_range',                      'Long-range triad detection'),
    ('deepfri_go', go_src,  'ghkl_atpase',                    'GHKL GO terms in deepfri_go'),
    ('deepfri_go', go_src,  'haem_binding',                    'Haem GO terms in deepfri_go'),
    ('deepfri_go', go_src,  'flavin_binding',                  'Flavin GO terms in deepfri_go'),
    ('consensus',  con_src, 'signaling',                       'signaling keyword in _infer_ns'),
    ('consensus',  con_src, 'phosphorylation',                 'phosphorylation keyword in _infer_ns'),
    ('consensus',  con_src, 'transport',                       'transport keyword in _infer_ns'),
]

for module, src, phrase, desc in checks:
    found = phrase.lower() in src.lower()
    print(('OK  ' if found else 'MISSING  ') + '[' + module + '] ' + desc)
