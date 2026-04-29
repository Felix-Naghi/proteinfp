import csv, time, json
import numpy as np
from pathlib import Path
import torch
import esm

BATCH = 4
SAVE_EVERY = 100
SEQ_MAX = 1022

rows = []
with open('data/balanced_training.csv') as f:
    for row in csv.DictReader(f):
        rows.append(row)
print(f'Total: {len(rows)}')

cache_dir = Path('data/category_cache')
cache_dir.mkdir(exist_ok=True)

done_path = cache_dir / 'done.json'
done = set(json.loads(done_path.read_text()) if done_path.exists() else '[]')
print(f'Already done: {len(done)}')

todo = [r for r in rows if r['uniprot_id'] not in done]
print(f'Remaining: {len(todo)}')

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.cuda().eval()
bc = alphabet.get_batch_converter()

t0 = time.time()
for i in range(0, len(todo), BATCH):
    batch = todo[i:i+BATCH]
    data  = [(r['uniprot_id'], r['sequence'][:SEQ_MAX]) for r in batch]
    try:
        _, _, tokens = bc(data)
        tokens = tokens.cuda()
        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)
        for j, r in enumerate(batch):
            L   = min(len(r['sequence']), SEQ_MAX)
            emb = out['representations'][33][j,1:L+1,:].mean(0).cpu().numpy()
            np.save(str(cache_dir / f"{r['uniprot_id']}.npy"), emb)
            done.add(r['uniprot_id'])
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f'OOM at batch {i}, skipping')
        torch.cuda.empty_cache()
        continue

    if i % SAVE_EVERY == 0:
        done_path.write_text(json.dumps(list(done)))
        rate = (i+BATCH) / (time.time()-t0)
        eta  = (len(todo)-i) / max(rate,0.01)
        print(f'  {i}/{len(todo)} | {rate:.1f} seq/s | ETA {eta:.0f}s')

done_path.write_text(json.dumps(list(done)))
print(f'Done. {len(done)} embeddings saved.')
