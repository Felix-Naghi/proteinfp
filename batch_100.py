import subprocess, sys
from pathlib import Path

PROTEINS = [
    "P00533","P06213","P15056","P00519","P27361","Q02750","P45983","Q13153","P06239","P08581","P04049",
    "P00734","P42574","P55210","P29466","P08246","P07711","P07858",
    "P04637","P01106","P10275","P03372","Q01094","P17275","P05412","P01100","P14921",
    "P00441","P16083","P00387","P00352","P14550",
    "P11473","Q07869","P37231","P10827","P20393",
    "P07900","P08238","P11142","P04792","P02511",
    "Q14524","P35561",
    "P01116","P62834","P84095","P61586","P60953",
    "Q00987","O15151","Q9NWF9","Q8NHZ8",
    "P17706","P29350","Q06124",
    "P68871","P69905","P02768","P11166",
    "P38398","P51587","P12004",
    "P02452","P68032","P02144",
    "Q9BYF1","P00918","P08473","P15144",
    "P00488","P04180","P01375","P05156",
    "P46108","Q13480",
        # Kinases
    "P06493",  # CDK1
    "P24941",  # CDK2
    "P11802",  # CDK4
    "P36507",  # MAP2K2
    "P45985",  # MAP2K4
    # Proteases
    "P07339",  # CTSD (Cathepsin D)
    "P43235",  # CTSK (Cathepsin K)
    "P08311",  # CTSG (Cathepsin G)
    # Phosphatases
    "P17706",  # PTPN2 (already in your list - skip)
    "P28562",  # PTPN7
    # Transcription factors
    "P15923",  # MYC
    "P17542",  # ATF4
    "P22415",  # USF1
    # Metabolic enzymes
    "P00491",  # PNP (Purine nucleoside phosphorylase)
    "P15531",  # NME1
    "P07737",  # PFN1 (Profilin)
    # Chaperones
    "P08107",  # HSPA1A
    "P38646",  # HSPA9
    # DNA repair
    "P20585",  # MSH3
    "P52701",  # MSH6
    # GTPases
    "P62070",  # RRAS2
    "P55197",  # MLF2
    # Apoptosis
    "Q07812",  # BAX
    "P10415",  # BCL2
    "Q16611",  # BAK1

]

seen = set()
unique = []
for uid in PROTEINS:
    if uid not in seen:
        seen.add(uid)
        unique.append(uid)

print(f"Running {len(unique)} proteins (~{len(unique)*12//60}h {len(unique)*12%60}m estimated)")

for i, uid in enumerate(unique):
    print(f"\n[{i+1}/{len(unique)}] {uid}")
    result = subprocess.run(
        [sys.executable, "-m", "proteinfp.cli", "--uniprot", uid],
        cwd=Path(__file__).parent
    )
    if result.returncode != 0:
        print(f"  FAILED: {uid}")