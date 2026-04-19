import json, pickle, numpy as np
from pathlib import Path

clf_path = Path('models/enzyme_classifier.pkl')
if not clf_path.exists():
    print('Enzyme classifier not found')
else:
    with open(clf_path, 'rb') as f:
        pkg = pickle.load(f)
    clf = pkg['classifier']
    print('ROC-AUC:', round(pkg.get('roc_auc', 0), 4))
    
    ground_truth = {
        'P04637': False, 'P00533': True,  'P00441': True,
        'P07900': False, 'P06213': True,  'P38398': False,
        'P16083': True,  'P00734': True,  'P68871': False,
        'P00918': True,  'P01116': True,  'Q00987': True,
        'Q9BYF1': True,  'O15151': False, 'P42574': True,
    }
    
    print()
    print('Protein   Expected  ML_prob  Decision  Correct')
    print('-' * 55)
    for uid, expected in ground_truth.items():
        esm2_path = Path('data/intermediate/' + uid + '_esm2.json')
        if not esm2_path.exists():
            print(uid + '  MISSING ESM2')
            continue
        esm2 = json.loads(esm2_path.read_text())
        emb = np.array(esm2.get('protein_embedding', []), dtype=np.float32)
        if len(emb) == 0:
            print(uid + '  NO EMBEDDING')
            continue
        prob = float(clf.predict_proba(emb.reshape(1,-1))[0][1])
        decision = prob >= 0.65
        correct = decision == expected
        print(uid + '  ' + str(expected)[:5] + '     ' + 
              str(round(prob,3)) + '   ' + str(decision)[:5] + '     ' + 
              ('OK' if correct else 'WRONG'))
