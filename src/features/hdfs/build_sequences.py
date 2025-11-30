import re
import pandas as pd
from pathlib import Path
from config import HDFS_PARSED_DIR, HDFS_SEQUENCE_DIR

STRUCTURED_FILE = HDFS_PARSED_DIR / "HDFS.log_structured.csv"
SEQUENCES_FILE = HDFS_SEQUENCE_DIR / "hdfs_sequences.csv"

BLOCK_REGEX = re.compile(r"blk_-?\d+")

def extract_block_id(row):
    params = str(row.get("ParameterList", ""))
    m = BLOCK_REGEX.search(params)
    if m:
        return m.group(0)

    content = str(row.get("Content", ""))
    m = BLOCK_REGEX.search(content)
    if m:
        return m.group(0)

    return None


def main():
    df = pd.read_csv(STRUCTURED_FILE)
    print("Lignes totales dans structured :", df.shape[0])

    df["BlockId"] = df.apply(extract_block_id, axis=1)

    nb_missing = df["BlockId"].isna().sum()
    print("Lignes sans BlockId :", nb_missing)

    df = df.dropna(subset=["BlockId"])
    print("Lignes avec BlockId :", df.shape[0])

    df["Timestamp"] = df["Date"].astype(str) + df["Time"].astype(str)
    df = df.sort_values(["BlockId", "Timestamp"])

    # séquence + timestamp de bloc
    grouped = df.groupby("BlockId")
    seqs = grouped["EventId"].apply(list).reset_index().rename(columns={"EventId": "EventSequence"})

    # on prend le premier timestamp pour le bloc
    block_ts = grouped["Timestamp"].min().reset_index().rename(columns={"Timestamp": "BlockTimestamp"})

    seqs = seqs.merge(block_ts, on="BlockId")

    print("Nombre de BlockId (séquences) :", seqs.shape[0])

    SEQUENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    seqs.to_csv(SEQUENCES_FILE, index=False)


if __name__ == "__main__":
    main()