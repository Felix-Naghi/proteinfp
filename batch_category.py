import json, pickle, numpy as np
from pathlib import Path

MODEL_DIR = Path('models/category_classifier')
clf        = pickle.load(open(MODEL_DIR / 'clf.pkl', 'rb'))
categories = json.loads((MODEL_DIR / 'categories.json').read_text())

inter = Path('data/intermediate')

uids = [
    'O15151','P00352','P00367','P00387','P00441','P00488','P00519','P00533',
    'P00734','P00748','P00749','P00918','P01031','P01100','P01106','P01116',
    'P01375','P02144','P02452','P02511','P02679','P02748','P02768','P02787',
    'P03372','P04049','P04180','P04637','P04792','P04899','P05156','P05412',
    'P06213','P06239','P07437','P07550','P07711','P07858','P07900','P08238',
    'P08246','P08473','P08581','P08603','P10275','P10827','P11021','P11142',
    'P11166','P11413','P11473','P12004','P14550','P14921','P15056','P15144',
    'P16083','P17275','P17706','P20393','P21554','P22309','P23588','P27361',
    'P28482','P29350','P29466','P32119','P35228','P35561','P37231','P38398',
    'P38646','P42574','P45983','P46108','P49841','P51587','P55210','P60953',
    'P61586','P62136','P62834','P62987','P63261','P68032','P68363','P68871',
    'P69905','P84095','Q00987','Q01094','Q02750','Q06124','Q07869','Q13153',
    'Q13480','Q14524','Q16539','Q8NHZ8','Q9BYF1','Q9NWF9'
]

done = 0
skipped = 0
for uid in uids:
    esm_path = inter / (uid + '_esm2.json')
    out_path = inter / (uid + '_category.json')
    if not esm_path.exists():
        skipped += 1
        continue
    esm2 = json.loads(esm_path.read_text())
    emb  = np.array(esm2['protein_embedding'], dtype=np.float32).reshape(1,-1)
    probs    = clf.predict_proba(emb)[0]
    top_idx  = int(probs.argmax())
    result = {
        'uniprot_id':   uid,
        'top_category': categories[top_idx],
        'confidence':   round(float(probs[top_idx]), 4),
        'all_scores':   {cat: round(float(p), 4) for cat, p in zip(categories, probs)}
    }
    out_path.write_text(json.dumps(result, indent=2))
    done += 1

print(f'Done: {done} category files saved, {skipped} skipped (no ESM-2)')
