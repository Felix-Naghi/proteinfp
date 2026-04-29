import json, pickle
import numpy as np
from pathlib import Path

def chunk_and_embed(uid, sequence, domains, model, bc, device):
    SEQ_MAX = 1022
    chunks = []

    # Always include start of sequence
    chunks.append((0, min(SEQ_MAX, len(sequence))))

    # Add each domain as its own chunk with 50aa padding either side
    for d in domains:
        start = max(0, d['start'] - 50)
        end   = min(len(sequence), d['end'] + 50)
        # If chunk is longer than SEQ_MAX, trim to SEQ_MAX centered on domain
        if end - start > SEQ_MAX:
            mid   = (d['start'] + d['end']) // 2
            start = max(0, mid - SEQ_MAX // 2)
            end   = min(len(sequence), start + SEQ_MAX)
        chunks.append((start, end))

    # Deduplicate overlapping chunks
    chunks = sorted(set(chunks))

    import torch
    all_residue_embs = {}  # residue_number -> embedding

    for chunk_start, chunk_end in chunks:
        chunk_seq = sequence[chunk_start:chunk_end]
        data = [(uid, chunk_seq)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)

        L   = len(chunk_seq)
        emb = out['representations'][33][0, 1:L+1, :].cpu().numpy()

        for i, res_emb in enumerate(emb):
            global_pos = chunk_start + i + 1  # 1-based
            if global_pos not in all_residue_embs:
                all_residue_embs[global_pos] = res_emb

    # Mean pool all residue embeddings for protein embedding
    all_embs = np.array(list(all_residue_embs.values()))
    protein_emb = all_embs.mean(axis=0)

    # High attention residues from contact-like variance
    variances = all_embs.var(axis=1)
    threshold = np.percentile(variances, 75)
    func_res  = [pos for pos, var in zip(all_residue_embs.keys(), variances)
                 if var >= threshold]

    return protein_emb, all_residue_embs, func_res


# Load model once
import torch
import esm as esm_lib

print('Loading ESM-2...')
model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = model.to(device).eval()
bc     = alphabet.get_batch_converter()
print(f'  Loaded on {device}')

inter = Path('data/intermediate')

long_uids = [
    'P02751','Q14524','P38398','P01031','P02452',
    'P51587','P08581','P06213','P08603','P00533','P35228','P00519'
]

for uid in long_uids:
    esm_path = inter / (uid + '_esm2.json')
    hom_path = inter / (uid + '_homology.json')

    if not hom_path.exists():
        print(f'{uid}: no homology, skipping')
        continue

    esm_data = json.loads(esm_path.read_text())
    hom_data = json.loads(hom_path.read_text())
    sequence = ''.join([aa for aa in esm_data.get('sequence', '')])

    # Get sequence from structure if not in esm2
    if not sequence:
        struct_path = inter / (uid + '_structure.json')
        if struct_path.exists():
            sequence = json.loads(struct_path.read_text()).get('sequence', '')

    if not sequence:
        print(f'{uid}: no sequence found, skipping')
        continue

    domains  = hom_data.get('interpro_domains', [])
    print(f'{uid}: {len(sequence)} aa, {len(domains)} domains')

    prot_emb, res_embs, func_res = chunk_and_embed(
        uid, sequence, domains, model, bc, device
    )

    # Update ESM-2 JSON with improved embeddings
    esm_data['protein_embedding']              = prot_emb.tolist()
    esm_data['predicted_functional_residues']  = sorted(func_res)
    esm_data['sequence_length']                = len(sequence)
    esm_data['chunked']                        = True
    esm_data['n_chunks']                       = len(domains) + 1

    esm_path.write_text(json.dumps(esm_data, indent=2))
    print(f'  Updated {uid}_esm2.json — {len(func_res)} functional residues across full sequence')

    torch.cuda.empty_cache()

print('Done.')
