import json, pickle
from pathlib import Path
import numpy as np

pkg = pickle.loads(Path('models/go_classifier.pkl').read_bytes())
covered = set(pkg['go_terms'])

test_proteins = {
    'P12004': ['GO:0003682','GO:0003684','GO:0032139','GO:0070182','GO:0030337','GO:0019899',
               'GO:0006287','GO:0070301','GO:0034644','GO:0071466','GO:0006325','GO:0030855'],
    'P08473': ['GO:1901612','GO:0004175','GO:0008238','GO:0004181','GO:0004222','GO:0070012',
               'GO:0021764','GO:0097242','GO:0150094','GO:0050435','GO:0002003','GO:0010815'],
    'P51587': ['GO:0043015','GO:0010484','GO:0010485','GO:0042802','GO:0002020','GO:0003697',
               'GO:0007420','GO:0071479','GO:0090398','GO:0051298','GO:0030330','GO:0006302'],
}

for uid, known in test_proteins.items():
    in_model = [g for g in known if g in covered]
    missing  = [g for g in known if g not in covered]
    print(f'{uid}: {len(in_model)}/{len(known)} known terms are in the classifier')
    print(f'  Missing from model: {missing}')
    print()
