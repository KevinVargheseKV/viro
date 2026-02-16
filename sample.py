import pandas as pd
import random

# 1. Load your current dataset (the one with Virus_Sequence, Human_Sequence, Label)
df = pd.read_csv('positive_sequence_dataset.csv')

# 2. Get lists of unique sequences
all_virus_seqs = df['Virus_Sequence'].unique().tolist()
all_human_seqs = df['Human_Sequence'].unique().tolist()

# Create a set of existing positive pairs for quick lookup
positive_pairs = set(zip(df['Virus_Sequence'], df['Human_Sequence']))

negative_samples = []
target_count = len(df)  # We want a 1:1 ratio (equal number of 0s and 1s)

print(f"Generating {target_count} negative samples...")

while len(negative_samples) < target_count:
    # Pick a random virus and a random human sequence
    v_seq = random.choice(all_virus_seqs)
    h_seq = random.choice(all_human_seqs)
    
    # Check if this pair is already a known positive or already in our negatives
    if (v_seq, h_seq) not in positive_pairs:
        # Add to negatives list with Label 0
        negative_samples.append({
            'Virus_Sequence': v_seq,
            'Human_Sequence': h_seq,
            'Label': 0
        })

# 3. Combine Positives and Negatives
neg_df = pd.DataFrame(negative_samples)
balanced_df = pd.concat([df, neg_df], ignore_index=True)

# 4. Shuffle the dataset so the model doesn't see all 1s followed by all 0s
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# 5. Save the new balanced dataset
balanced_df.to_csv('balanced_viro_dataset.csv', index=False)
print("Success! 'balanced_viro_dataset.csv' created with 50% Positives and 50% Negatives.")