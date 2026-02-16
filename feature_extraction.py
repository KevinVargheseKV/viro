import pandas as pd
import itertools
import numpy as np

# ---------- SETTINGS ----------

INPUT_FILE = "balanced_viro_dataset.csv"   # your 0 & 1 dataset
OUTPUT_FILE = "final_feature_dataset.csv"

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
dipeptides = [''.join(p) for p in itertools.product(amino_acids, repeat=2)]

# ---------- FEATURE FUNCTIONS ----------

def compute_aac(seq):
    seq = seq.upper()
    length = len(seq)
    return [seq.count(aa) / length for aa in amino_acids]

def compute_dipeptide(seq):
    seq = seq.upper()
    length = len(seq)
    dipep_count = dict.fromkeys(dipeptides, 0)

    for i in range(len(seq) - 1):
        dipep = seq[i:i+2]
        if dipep in dipep_count:
            dipep_count[dipep] += 1

    return [dipep_count[d] / (length - 1) for d in dipeptides]

# ---------- LOAD DATA ----------

df = pd.read_csv(INPUT_FILE)

features = []

print("Extracting features...")

for i, row in df.iterrows():
    v_seq = row["Virus_Sequence"]
    h_seq = row["Human_Sequence"]
    label = row["Label"]

    v_aac = compute_aac(v_seq)
    v_dipep = compute_dipeptide(v_seq)

    h_aac = compute_aac(h_seq)
    h_dipep = compute_dipeptide(h_seq)

    row_features = v_aac + v_dipep + h_aac + h_dipep + [label]
    features.append(row_features)

    if i % 100 == 0:
        print(f"Processed {i}/{len(df)}")

# ---------- SAVE DATA ----------

columns = (
    [f"v_AAC_{aa}" for aa in amino_acids] +
    [f"v_DI_{d}" for d in dipeptides] +
    [f"h_AAC_{aa}" for aa in amino_acids] +
    [f"h_DI_{d}" for d in dipeptides] +
    ["Label"]
)

out_df = pd.DataFrame(features, columns=columns)
out_df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE!")
print("Final feature dataset shape:", out_df.shape)
