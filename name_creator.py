import pandas as pd
import requests
import time


# -----------------------------------------
# Fetch HUMAN protein name + sequence
# -----------------------------------------
def fetch_protein_data(uniprot_id):

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            print(f"Failed: {uniprot_id}")
            return None, None

        fasta = r.text.split("\n")

        header = fasta[0]
        sequence = ''.join(fasta[1:])

        # Extract protein name
        # Example header:
        # >sp|P62258|1433E_HUMAN 14-3-3 protein epsilon OS=Homo sapiens ...

        name_part = header.split("|")[2]
        protein_name = name_part.split(" OS=")[0]

        # Clean unwanted tag
        protein_name = protein_name.replace("_HUMAN", "")

        return protein_name, sequence

    except Exception as e:
        print(f"Error: {uniprot_id} → {e}")
        return None, None


# -----------------------------------------
# Fetch VIRUS protein name
# -----------------------------------------
def fetch_virus_name(uniprot_id):

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return "Unknown Virus Protein"

        header = r.text.split("\n")[0]

        name_part = header.split("|")[2]
        virus_name = name_part.split(" OS=")[0]

        return virus_name

    except:
        return "Unknown Virus Protein"


# -----------------------------------------
# BUILD HUMAN SEQUENCE DATABASE
# -----------------------------------------
def build_human_database():

    print("\nLoading protein_dataset.csv ...")

    df = pd.read_csv("protein_dataset.csv")

    if "source_desc" not in df.columns:
        raise Exception("Column 'source_desc' not found!")

    human_ids = df["source_desc"].dropna().unique()

    print(f"Total Unique Human Proteins: {len(human_ids)}\n")

    data = []

    for i, hid in enumerate(human_ids, 1):

        print(f"[{i}/{len(human_ids)}] Fetching {hid}")

        name, seq = fetch_protein_data(hid)

        if seq:
            data.append([hid, name, seq])

        # Avoid UniProt rate limiting
        time.sleep(0.5)

    # Save CSV
    out = pd.DataFrame(
        data,
        columns=[
            "human_uniprot",
            "protein_name",
            "sequence"
        ]
    )

    out.to_csv("human_sequences.csv", index=False)

    print("\nSaved updated human_sequences.csv ✅")
    print(f"Total stored sequences: {len(out)}")


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":

    print("\n--- BUILDING HUMAN PROTEIN DATABASE ---\n")

    build_human_database()

    # Example virus name test
    test_id = input(
        "\nEnter a Virus UniProt ID to test name fetch (or press Enter to skip): "
    ).strip()

    if test_id:
        vname = fetch_virus_name(test_id)
        print(f"\nVirus Protein Name: {vname}")
