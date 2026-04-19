import numpy as np
import json
import pickle
import torch
from pathlib import Path
import sys
sys.path.insert(0, '.')
from train.train_go_classifier import predict_go_with_nn, load_go_classifier

model_path = Path('models/go_classifier.pkl')
model, package = load_go_classifier(model_path)
print('GO terms covered:', package['n_go_terms'])
print('Best train F1:', round(package['best_f1'], 4))
print()

# Test on validation proteins at different thresholds
test_proteins = {
    'P04637': {'gene':'TP53',  'known':['GO:0003677','GO:0003700','GO:0046872','GO:0006915','GO:0005634']},
    'P07900': {'gene':'HSP90', 'known':['GO:0005524','GO:0051082','GO:0006457','GO:0005737']},
    'P38398': {'gene':'BRCA1', 'known':['GO:0003684','GO:0006281','GO:0005634']},
    'P68871': {'gene':'HBB',   'known':['GO:0020037','GO:0015671','GO:0005833']},
    'P16083': {'gene':'NQO2',  'known':['GO:0003955','GO:0055114','GO:0005737']},
}

for threshold in [0.4, 0.5, 0.6]:
    print('=== Threshold:', threshold, '===')
    total_found = 0
    total_known = 0
    for uid, info in test_proteins.items():
        emb_path = Path('data/intermediate/' + uid + '_esm2.json')
        if not emb_path.exists():
            continue
        esm2 = json.loads(emb_path.read_text())
        emb = np.array(esm2['protein_embedding'], dtype=np.float32)
        preds = predict_go_with_nn(emb, model_path, threshold=threshold)
        pred_ids = {p['go_id'] for p in preds}
        known = set(info['known'])
        found = len(pred_ids & known)
        total_found += found
        total_known += len(known)
        recall = found / len(known)
        print(uid + ' ' + info['gene'] + ': ' + str(found) + '/' + str(len(known)) + 
              ' known terms found (' + str(round(recall*100)) + '%) | total predicted: ' + str(len(preds)))
    overall = total_found / max(total_known, 1)
    print('Overall recall: ' + str(round(overall*100, 1)) + '%')
    print()
