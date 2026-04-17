import requests, json, time
ticket = requests.post('https://search.foldseek.com/api/ticket',
    files={'q': ('test.pdb', open('data/structures/P04637.pdb','rb'), 'application/octet-stream')},
    data={'mode':'3diaa', 'database[]':['pdb100']},
    timeout=30).json()
tid = ticket.get('id','')
print('Ticket:', tid)
time.sleep(35)
r = requests.get('https://search.foldseek.com/api/result/' + tid + '/0',
    params={'format':'json','limit':1}, timeout=30).json()
results = r.get('results', [])
if results:
    aligns = results[0].get('alignments', [[]])[0]
    if aligns:
        print('Keys:', list(aligns[0].keys()))
        print('Hit:', json.dumps(aligns[0], indent=2))
