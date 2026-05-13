import numpy as np
import joblib
from feature_calculator import extract_features
from uniprot_api import fetch_sequence_from_uniprot
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load trained Random Forest model
model = joblib.load("random_forest.pkl")

def predict_interaction(human_id, virus_id):

    # Fetch sequences using UniProt API
    human_seq = fetch_sequence_from_uniprot(human_id)
    virus_seq = fetch_sequence_from_uniprot(virus_id)

    # Feature extraction
    human_feat = extract_features(human_seq)
    virus_feat = extract_features(virus_seq)

    # Concatenate in correct order → [Virus | Human]
    pair_features = np.concatenate([virus_feat, human_feat]).reshape(1, -1)

    # Prediction
    prob = model.predict_proba(pair_features)[0][1]
    pred = model.predict(pair_features)[0]

    return pred, prob


if __name__ == "__main__":

    print("\n--- VIRO++ MODULE 1 : Interaction Predictor ---\n")

    human_id = input("Enter HUMAN UniProt ID: ").strip()
    virus_id = input("Enter VIRUS UniProt ID: ").strip()

    try:
        prediction, probability = predict_interaction(human_id, virus_id)

        print("\n--- RESULT ---")
        print("Interaction Probability:", round(probability, 4))

        if prediction == 1:
            print("Prediction: INTERACT ✅")
        else:
            print("Prediction: NON-INTERACT ❌")

    except Exception as e:
        print("Error:", e)
