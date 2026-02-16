import numpy as np
from collections import Counter
from itertools import product

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def compute_aac(sequence):
    sequence = sequence.upper()
    length = len(sequence)
    counts = Counter(sequence)
    return np.array([counts.get(aa, 0) / length for aa in AMINO_ACIDS])

def compute_dpc(sequence):
    sequence = sequence.upper()
    length = len(sequence) - 1
    dipeptides = [a + b for a, b in product(AMINO_ACIDS, repeat=2)]
    counts = Counter(sequence[i:i+2] for i in range(len(sequence)-1))
    return np.array([counts.get(dp, 0) / length for dp in dipeptides])

def extract_features(sequence):
    return np.concatenate([compute_aac(sequence), compute_dpc(sequence)])

def clean_fasta(seq):
    lines = seq.strip().splitlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")])
