from pathlib import Path
from logparser import Drain
from config import HDFS_RAW_DIR, HDFS_PARSED_DIR, HDFS_LOG_FORMAT

LOG_FILE_NAME = "HDFS.log"  # à adapter au vrai nom du fichier LogHub

def main():
    HDFS_PARSED_DIR.mkdir(parents=True, exist_ok=True)

    parser = Drain.LogParser(
        indir=str(HDFS_RAW_DIR),
        outdir=str(HDFS_PARSED_DIR),
        log_format=HDFS_LOG_FORMAT,
        depth=4,      # profondeur de l'arbre Drain
        st=0.5,       # seuil de similarité
        rex=[]        # éventuelles regex pour prétraiter
    )

    parser.parse(LOG_FILE_NAME)


if __name__ == "__main__":
    main()