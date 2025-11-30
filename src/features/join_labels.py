import pandas as pd
from config import HDFS_SEQUENCE_DIR, HDFS_RAW_DIR

SEQUENCES_FILE = HDFS_SEQUENCE_DIR / "hdfs_sequences.csv"
LABEL_FILE = HDFS_RAW_DIR / "anomaly_label.csv"
OUTPUT_FILE = HDFS_SEQUENCE_DIR / "hdfs_sequences_labeled.csv"


def main():
    seq = pd.read_csv(SEQUENCES_FILE)
    labels = pd.read_csv(LABEL_FILE)

    print("Séquences :", seq.shape[0])
    print("Labels    :", labels.shape[0])

    df = seq.merge(labels, on="BlockId", how="inner")

    print("Après merge :", df.shape[0])
    print("Répartition des labels :")
    print(df["Label"].value_counts())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()