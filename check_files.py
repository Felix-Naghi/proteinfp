from pathlib import Path

uid = 'Q6PL18'
inter = Path('data/intermediate')
files = list(inter.glob(f'{uid}_*.json'))
print(f'Files for {uid}:')
for f in sorted(files):
    print(f'  {f.name}')
