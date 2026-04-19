import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
from train.train_go_classifier import predict_go_with_nn, load_go_classifier

model_path = Path('models/go_classifier.pkl')

test_proteins = {
    'P04637': {'gene':'TP53',  'known':['GO:0003677','GO:0003700','GO:0046872','GO:0006915','GO:0005634']},
    'P07900': {'gene':'HSP90', 'known':['GO:0005524','GO:0051082','GO:0006457','GO:0005737']},
    'P38398': {'gene':'BRCA1', 'known':['GO:0003684','GO:0006281','GO:0005634']},
    'P68871': {'gene':'HBB',   'known':['GO:0020037','GO:0015671','GO:0005833']},
    'P16083': {'gene':'NQO2',  'known':['GO:0003955','GO:0055114','GO:0005737']},
}

for threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
    total_found = 0
    total_known = 0
    total_predicted = 0
    for uid, info in test_proteins.items():
        emb_path = Path('data/intermediate/' + uid + '_esm2.json')
        if not emb_path.exists():
            continue
        esm2 = json.loads(emb_path.read_text())
        emb = np.array(esm2['protein_embedding'], dtype=np.float32)
        preds = predict_go_with_nn(emb, model_path, threshold=threshold)
        pred_ids = {p['go_id'] for p in preds}
        found = len(pred_ids & set(info['known']))
        total_found += found
        total_known += len(info['known'])
        total_predicted += len(preds)
    recall = total_found / max(total_known, 1)
    avg_pred = total_predicted / len(test_proteins)
    print('Threshold ' + str(threshold) + ': recall=' + str(round(recall*100,1)) + '% avg_predictions=' + str(round(avg_pred,1)))
