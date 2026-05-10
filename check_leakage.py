import pandas as pd

df = pd.read_csv('data/swissprot_curated_v4_augmented.csv')

print(f"Total training proteins: {len(df)}")
print()

# Check all 100 validation proteins
VALIDATION = [
    "P00533","P06213","P15056","P00519","P27361","Q02750","P45983","Q13153",
    "P06239","P08581","P04049","P00734","P42574","P55210","P29466","P08246",
    "P07711","P07858","P04637","P01106","P10275","P03372","Q01094","P17275",
    "P05412","P01100","P14921","P00441","P16083","P00387","P00352","P14550",
    "P11473","Q07869","P37231","P10827","P20393","P07900","P08238","P11142",
    "P04792","P02511","Q14524","P35561","P01116","P62834","P84095","P61586",
    "P60953","Q00987","O15151","Q9NWF9","Q8NHZ8","P17706","P29350","Q06124",
    "P68871","P69905","P02768","P11166","P38398","P51587","P12004","P02452",
    "P68032","P02144","Q9BYF1","P00918","P08473","P15144","P00488","P04180",
    "P01375","P05156","P46108","Q13480","P06493","P24941","P11802","P36507",
    "P45985","P07339","P43235","P08311","P28562","P15923","P17542","P22415",
    "P00491","P15531","P07737","P08107","P38646","P20585","P52701","P62070",
    "P55197","Q07812","P10415","Q16611",
]
    
leaked = []
for uid in VALIDATION:
    rows = df[df['uniprot_id'] == uid]
    if len(rows):
        ec = rows.iloc[0]['ec_class']
        leaked.append((uid, ec))
        print(f"  LEAKED: {uid}  ec_class={ec}")

print()
print(f"Total leaked into training: {len(leaked)} / {len(VALIDATION)}")