import requests, json, time
from pathlib import Path

# Map gene names to UniProt IDs
TOP_GENES = [
    'ATAD2', 'TOP2A', 'STMN1', 'SLC2A1', 'MKI67',
    'CLSPN', 'AGR2', 'CEACAM6', 'LCN2', 'HELLS',
    'KIAA0101', 'CENPW', 'CST6', 'IFI27', 'MT2A'
]

print('Fetching UniProt IDs for top GRN regulators...')
results = {}

for gene in TOP_GENES:
    url = 'https://rest.uniprot.org/uniprotkb/search'
    r = requests.get(url, params={
        'query': f'gene:{gene} AND organism_id:9606 AND reviewed:true',
        'fields': 'accession,gene_names,protein_name,length',
        'format': 'json',
        'size': 1
    }, timeout=10)
    
    if r.status_code == 200:
        entries = r.json().get('results', [])
        if entries:
            uid  = entries[0]['primaryAccession']
            name = entries[0].get('proteinDescription', {}) \
                             .get('recommendedName', {}) \
                             .get('fullName', {}).get('value', '')
            length = entries[0].get('sequence', {}).get('length', 0)
            results[gene] = {'uniprot_id': uid, 'name': name, 'length': length}
            print(f'  {gene:<15} {uid}  {length} aa  {name[:50]}')
        else:
            print(f'  {gene:<15} NOT FOUND')
    time.sleep(0.2)

Path('data/grn/intermediate/top_regulator_ids.json').write_text(
    json.dumps(results, indent=2)
)
print(f'Saved {len(results)} UniProt IDs')
