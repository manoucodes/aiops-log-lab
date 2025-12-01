from pathlib import Path
import ast

import numpy as np
import pandas as pd

from config import BGL_SEQUENCE_DIR, BGL_FEATURE_DIR

SEQ_FILE = BGL_SEQUENCE_DIR / "bgl_sequences.csv"
FEATURE_FILE = BGL_FEATURE_DIR / "bgl_features.csv"


def load_sequences(seq_file: Path) -> pd.DataFrame:
    print(f"Chargement des séquences depuis {seq_file} …")
    df = pd.read_csv(seq_file)
    # transforme la string "[1, 2, 3]" en vraie liste [1, 2, 3]
    df["EventSequence"] = df["EventSequence"].apply(ast.literal_eval)
    return df


def build_bow_features(df_seq: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Bag-of-events sur les EventSequence."""
    sequences = df_seq["EventSequence"].tolist()

    # on infère le nombre d'événements différents à partir du max des IDs
    max_event_id = max(max(seq) for seq in sequences)
    num_events = max_event_id + 1
    print(f"→ Nombre d'événements uniques (dimension): {num_events}")

    num_seqs = len(sequences)
    X = np.zeros((num_seqs, num_events), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for eid in seq:
            X[i, eid] += 1.0  # comptage brut

    if normalize:
        # normalisation par la somme (pour que chaque ligne = distribution de probas)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # évite division par zéro
        X = X / row_sums
        print("→ Features normalisées (fréquences).")

    # on construit un DataFrame avec des colonnes e_0, e_1, ...
    feature_cols = [f"e_{i}" for i in range(num_events)]
    df_feat = pd.DataFrame(X, columns=feature_cols)

    # on garde SeqId et Label à côté
    df_feat.insert(0, "SeqId", df_seq["SeqId"].values)
    df_feat["Label"] = df_seq["Label"].values.astype(int)

    return df_feat


def main():
    BGL_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    df_seq = load_sequences(SEQ_FILE)

    print("Construction des features (bag-of-events)…")
    df_feat = build_bow_features(df_seq, normalize=True)

    print(f"Sauvegarde dans {FEATURE_FILE} …")
    df_feat.to_csv(FEATURE_FILE, index=False)
    print("Terminé ✅")


if __name__ == "__main__":
    main()