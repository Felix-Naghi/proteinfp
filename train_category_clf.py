import csv, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CATEGORIES = [
    'ec_1','ec_2','ec_3','ec_4','ec_5','ec_6','ec_7',
    'gpcr','cytoskeletal','transmembrane','transcription',
    'ecm','transport','rna_binding','chaperone','immune'
]
label2idx = {c:i for i,c in enumerate(CATEGORIES)}

cache_dir = Path('data/category_cache')
model_dir = Path('models/category_classifier')
model_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
rows = []
with open('data/balanced_training.csv') as f:
    for row in csv.DictReader(f):
        rows.append(row)

# Load embeddings
X, y, ids = [], [], []
missing = 0
for row in rows:
    uid = row['uniprot_id']
    cat = row['category']
    npy = cache_dir / f'{uid}.npy'
    if not npy.exists():
        missing += 1
        continue
    idx = label2idx.get(cat, -1)
    if idx < 0:
        continue
    X.append(np.load(str(npy)))
    y.append(idx)
    ids.append(uid)

print(f'Loaded {len(X)} embeddings, {missing} missing')

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f'X shape: {X.shape}')

# Class distribution
from collections import Counter
dist = Counter(CATEGORIES[i] for i in y)
for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f'  {cat:<25} {count}')

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1,
    ))
])

print('Fitting...')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0))

with open(model_dir / 'clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
json.dump(CATEGORIES, open(model_dir / 'categories.json', 'w'))
print(f'Saved to {model_dir}')
