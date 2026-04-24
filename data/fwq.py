import pandas as pd
import numpy as np

# ── FILE PATHS ─────────────────────────────────────────
enz_path = r"C:\Users\adria\Documents\proteinFP\data\swissprot_enzymes.tsv"
nonenz_path = r"C:\Users\adria\Documents\proteinFP\data\non_enzymes.tsv"

# ── LOAD TSV FILES ─────────────────────────────────────
enz = pd.read_csv(enz_path, sep="\t")
nonenz = pd.read_csv(nonenz_path, sep="\t")

# ── STANDARDIZE COLUMNS ────────────────────────────────
enz = enz.rename(columns={
    "Entry": "uniprot_id",
    "Sequence": "sequence",
    "EC number": "ec_class"
})

nonenz = nonenz.rename(columns={
    "Entry": "uniprot_id",
    "Sequence": "sequence"
})

# Assign non-enzyme label
nonenz["ec_class"] = "non-enzyme"

# ── CLEAN EC CLASSES ───────────────────────────────────
def simplify_ec(ec):
    if pd.isna(ec) or ec == "" or ec == "non-enzyme":
        return "non-enzyme"
    return str(ec).split(".")[0]

enz["ec_class"] = enz["ec_class"].apply(simplify_ec)

# Keep only valid EC classes 1–7
enz = enz[enz["ec_class"].isin(list("1234567"))]

# ── MERGE DATASETS ─────────────────────────────────────
df = pd.concat([enz, nonenz], ignore_index=True)

# Add binary label
df["is_enzyme"] = df["ec_class"].apply(lambda x: 0 if x == "non-enzyme" else 1)

print("Before balancing:")
print(df["ec_class"].value_counts())

# ── TARGET SIZE ────────────────────────────────────────
TARGET_TOTAL = 2000
TARGET_NON_ENZ = TARGET_TOTAL // 2   # 1000
TARGET_ENZ = TARGET_TOTAL // 2       # 1000

# ── SAMPLE NON-ENZYMES ─────────────────────────────────
nonenz_sample = df[df["is_enzyme"] == 0].sample(
    n=min(TARGET_NON_ENZ, len(df[df["is_enzyme"] == 0])),
    random_state=42
)

# ── BALANCE EC CLASSES (STRONGER MODEL) ────────────────
enz_df = df[df["is_enzyme"] == 1]

classes = list("1234567")
per_class = TARGET_ENZ // len(classes)  # ~142 each

enz_samples = []

for c in classes:
    subset = enz_df[enz_df["ec_class"] == c]
    if len(subset) == 0:
        continue

    sample_n = min(per_class, len(subset))
    sampled = subset.sample(n=sample_n, random_state=42)
    enz_samples.append(sampled)

enz_sample = pd.concat(enz_samples)

# ── COMBINE FINAL DATASET ──────────────────────────────
final_df = pd.concat([enz_sample, nonenz_sample])
final_df = final_df.sample(frac=1, random_state=42)  # shuffle

print("\nAfter balancing:")
print(final_df["ec_class"].value_counts())
print(f"Total samples: {len(final_df)}")

# ── SAVE ──────────────────────────────────────────────
output_path = "data/training_balanced_2000.csv"
final_df[["uniprot_id", "sequence", "ec_class"]].to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")