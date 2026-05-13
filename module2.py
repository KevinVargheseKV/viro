import numpy as np
import joblib
import pandas as pd
from feature_calculator import extract_features
from uniprot_api import fetch_sequence_from_uniprot
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------
# Load trained Random Forest model
# ---------------------------------------------
model = joblib.load("random_forest.pkl")


# ---------------------------------------------
# Load Human Sequences + Names
# ---------------------------------------------
def load_human_database():

    df = pd.read_csv("human_sequences.csv")

    required_cols = [
        "human_uniprot",
        "protein_name",
        "sequence"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Column '{col}' missing!")

    seq_dict = dict(
        zip(df["human_uniprot"], df["sequence"])
    )

    name_dict = dict(
        zip(df["human_uniprot"], df["protein_name"])
    )

    return seq_dict, name_dict


# ---------------------------------------------
# Load Virus Name from Dataset
# ---------------------------------------------
def get_virus_name(virus_id):

    df = pd.read_csv("protein_dataset.csv")

    match = df[df["target"] == virus_id]

    if len(match) == 0:
        return "Unknown Virus Protein"

    return match.iloc[0]["target_desc"]


# ---------------------------------------------
# Ranking Function
# ---------------------------------------------
def rank_interactions(virus_id, top_n=10):

    # Fetch virus sequence
    virus_seq = fetch_sequence_from_uniprot(virus_id)

    if virus_seq is None:
        raise Exception("Virus sequence not found!")

    virus_feat = extract_features(virus_seq)

    # Load human DB
    human_seq_dict, human_name_dict = load_human_database()

    results = []

    for human_id, human_seq in human_seq_dict.items():

        try:
            human_feat = extract_features(human_seq)

            pair_features = np.concatenate(
                [virus_feat, human_feat]
            ).reshape(1, -1)

            prob = model.predict_proba(pair_features)[0][1]

            results.append((human_id, prob))

        except Exception as e:
            print(f"Skipped {human_id} → {e}")
            continue

    ranked = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n], human_name_dict


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":

    print("\n--- VIRO++ MODULE 2 : Named Interaction Ranker ---\n")

    virus_id = input("Enter VIRUS UniProt ID: ").strip()

    try:
        # Get virus name
        virus_name = get_virus_name(virus_id)

        # Rank humans
        top_results, name_dict = rank_interactions(
            virus_id,
            top_n=10
        )

        print(f"\nVirus Protein: {virus_name}")
        print("\n--- TOP PREDICTED HUMAN INTERACTIONS ---\n")

        for rank, (hid, score) in enumerate(top_results, 1):

            pname = name_dict.get(hid, "Unknown Protein")

            print(f"{rank}. {pname} → {round(score,4)}")

    except Exception as e:
        print("Error:", e)
