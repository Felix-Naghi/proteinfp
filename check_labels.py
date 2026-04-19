import json
from pathlib import Path

cache_dir = Path('models/embedding_cache')
cache_files = list(cache_dir.glob('swissprot_*.json'))

if not cache_files:
    print('No cache files found in ' + str(cache_dir))
else:
    for cache_file in sorted(cache_files):
        try:
            proteins = json.loads(cache_file.read_text())
            has_label = sum(1 for p in proteins if 'is_enzyme' in p)
            n_enzyme = sum(1 for p in proteins if p.get('is_enzyme') == True)
            n_non = sum(1 for p in proteins if p.get('is_enzyme') == False)
            pct = round(n_enzyme / max(has_label, 1) * 100, 1)
            print('File: ' + cache_file.name)
            print('  Total proteins:    ' + str(len(proteins)))
            print('  With enzyme label: ' + str(has_label))
            print('  Enzymes:           ' + str(n_enzyme) + ' (' + str(pct) + '%)')
            print('  Non-enzymes:       ' + str(n_non))
            print()
        except Exception as e:
            print('Could not read ' + cache_file.name + ': ' + str(e))
