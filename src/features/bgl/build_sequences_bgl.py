import ast
import pandas as pd
from collections import Counter
from config import BGL_PARSED_DIR, BGL_SEQUENCE_DIR

STRUCTURED_FILE = BGL_PARSED_DIR / "BGL.log_structured.csv"
SEQUENCE_FILE = BGL_SEQUENCE_DIR / "bgl_sequences.csv"

# Liste des severities considérées comme anomalies
ANOMALY_LEVELS = ["ERROR", "FATAL", "SEVERE", "FAILURE"]

# Taille de fenêtre
WINDOW_SIZE = 50

def extract_label(df):
    """
    Construit une colonne Label : 0 = normal, 1 = anomalie,
    déduite depuis EventTemplate (plus stable que Content).
    """
    sev = df["EventTemplate"].str.extract(
        r" (INFO|WARN|WARNING|ERROR|FATAL|SEVERE|FAILURE) "
    )[0].fillna("INFO")

    df["Label"] = sev.isin(ANOMALY_LEVELS).astype(int)
    return df


def encode_event_ids(df):
    """
    Transforme chaque EventId en entier, car les modèles attendent des ints.
    """
    uniq = sorted(df["EventId"].unique())
    mapping = {eid: i for i, eid in enumerate(uniq)}
    print(f" → {len(mapping)} événements uniques")

    df["EventIdInt"] = df["EventId"].map(mapping)
    return df, mapping


def build_sliding_sequences(df, window_size):
    """
    Construire des séquences glissantes sur EventIdInt.
    Label de la séquence = max(label dans la fenêtre).
    """
    events = df["EventIdInt"].tolist()
    labels = df["Label"].tolist()

    seq_ids = []
    seqs = []
    seq_labels = []

    for i in range(len(events) - window_size + 1):
        window = events[i:i + window_size]
        window_labels = labels[i:i + window_size]

        seq_ids.append(i)
        seqs.append(window)
        seq_labels.append(max(window_labels))

    print(f" → Séquences générées : {len(seqs)}")

    return pd.DataFrame({
        "SeqId": seq_ids,
        "EventSequence": seqs,
        "Label": seq_labels
    })


def main():
    print("Chargement du structured log…")
    df = pd.read_csv(STRUCTURED_FILE)
    print(f"Lignes totales : {df.shape[0]}")

    print("Détection des anomalies via la sévérité…")
    df = extract_label(df)
    print(df["Label"].value_counts())

    print("Encodage des EventId…")
    df, mapping = encode_event_ids(df)

    print(f"Construction des séquences glissantes (taille {WINDOW_SIZE})…")
    sequences_df = build_sliding_sequences(df, WINDOW_SIZE)

    BGL_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    sequences_df.to_csv(SEQUENCE_FILE, index=False)

    print(f"Sauvegardé → {SEQUENCE_FILE}")
    print("Terminé !")


if __name__ == "__main__":
    main()