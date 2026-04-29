import pickle, json, sys
import numpy as np
from pathlib import Path

MODEL_DIR = Path('models/category_classifier')


def predict_category(uniprot_id: str, esm2_result: dict) -> dict:
    clf        = pickle.load(open(MODEL_DIR / 'clf.pkl', 'rb'))
    categories = json.loads((MODEL_DIR / 'categories.json').read_text())

    emb      = np.array(esm2_result['protein_embedding'], dtype=np.float32).reshape(1, -1)
    probs    = clf.predict_proba(emb)[0]
    top_idx  = int(probs.argmax())

    return {
        'uniprot_id':   uniprot_id,
        'top_category': categories[top_idx],
        'confidence':   round(float(probs[top_idx]), 4),
        'all_scores':   {cat: round(float(p), 4) for cat, p in zip(categories, probs)}
    }


if __name__ == '__main__':
    uid       = sys.argv[1].upper()
    esm2_path = Path('data/intermediate') / (uid + '_esm2.json')

    if not esm2_path.exists():
        print('Run ESM-2 first: python pipeline/esm2_embeddings.py --uniprot ' + uid)
        sys.exit(1)

    esm2   = json.loads(esm2_path.read_text())
    result = predict_category(uid, esm2)

    print('Category  : ' + result['top_category'])
    print('Confidence: ' + str(result['confidence']))
    print('Top 5 scores:')
    for cat, score in sorted(result['all_scores'].items(), key=lambda x: -x[1])[:5]:
        print('  ' + cat.ljust(20) + str(score))

    out = Path('data/intermediate') / (uid + '_category.json')
    out.write_text(json.dumps(result, indent=2))
    print('Saved to ' + str(out))