import ast
import pandas as pd
from collections import Counter
from config import HDFS_SEQUENCE_DIR, HDFS_FEATURE_DIR

SEQ_LABELED_FILE = HDFS_SEQUENCE_DIR / "hdfs_sequences_labeled.csv"
FEATURE_FILE = HDFS_FEATURE_DIR / "hdfs_features.csv"


def main():
    df = pd.read_csv(SEQ_LABELED_FILE)

    # EventSequence est une string, on la convertit en liste Python
    df["EventSequence"] = df["EventSequence"].apply(ast.literal_eval)

    # construire le vocabulaire d'événements
    all_events = sorted({e for seq in df["EventSequence"] for e in seq})
    print("Nombre d'events uniques :", len(all_events))

    rows = []
    for _, row in df.iterrows():
        counts = Counter(row["EventSequence"])

        feature_row = {
            "BlockId": row["BlockId"],
            "Label": row["Label"],
        }
        for eid in all_events:
            feature_row[f"cnt_{eid}"] = counts.get(eid, 0)

        rows.append(feature_row)

    feat_df = pd.DataFrame(rows)
    print("Shape features :", feat_df.shape)

    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURE_FILE, index=False)


if __name__ == "__main__":
    main()