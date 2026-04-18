import json

ground_truth = {
    'P04637': False, 'P00533': True,  'P00441': True,  'P07900': False,
    'P06213': True,  'P38398': False, 'P16083': True,  'P00734': True,
    'P68871': False, 'P00918': True,  'P01116': True,  'Q00987': True,
    'Q9BYF1': True,
}

print('Enzyme classification errors:')
for uid, expected in ground_truth.items():
    ec = json.load(open('data/intermediate/' + uid + '_ec_prediction.json'))
    predicted = ec['is_enzyme']
    conf = ec['enzyme_confidence']
    non_enz = ec['non_enzyme_score']
    status = 'OK' if predicted == expected else 'WRONG'
    if status == 'WRONG':
        print(uid + ' expected=' + str(expected) + ' got=' + str(predicted) + ' conf=' + str(round(conf,2)) + ' non_enz=' + str(round(non_enz,2)))
