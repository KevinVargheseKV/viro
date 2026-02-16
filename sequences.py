import pandas as pd
import requests
import time

INPUT_FILE = "protein_dataset.csv"
OUTPUT_FILE = "positive_sequence_dataset.csv"
FAILED_FILE = "failed_ids.csv"

HUMAN_COL = "source_desc"
VIRUS_COL = "target"

headers = {"User-Agent": "Mozilla/5.0"}

def fetch_sequence(uniprot_id, retries=3):
    uniprot_id = str(uniprot_id).strip()
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10, headers=headers)
            if r.status_code == 200 and r.text.startswith(">"):
                return "".join(r.text.split("\n")[1:])
        except:
            time.sleep(2)
    return None

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

dataset = []
failed = []

print("Fetching sequences from UniProt...")

for i, row in df.iterrows():
    h_id = row[HUMAN_COL]
    v_id = row[VIRUS_COL]

    h_seq = fetch_sequence(h_id)
    v_seq = fetch_sequence(v_id)

    if h_seq and v_seq:
        dataset.append([v_seq, h_seq, 1])
    else:
        failed.append([h_id, v_id])

    if i % 50 == 0:
        print(f"Processed {i}/{len(df)}")

out_df = pd.DataFrame(
    dataset,
    columns=["Virus_Sequence", "Human_Sequence", "Label"]
)

out_df.to_csv(OUTPUT_FILE, index=False)
pd.DataFrame(failed, columns=["Human_UniProt", "Virus_UniProt"]).to_csv(FAILED_FILE, index=False)

print("\nDONE!")
print("Final dataset shape:", out_df.shape)
print("Failed IDs:", len(failed))
