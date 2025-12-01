from pathlib import Path
from logparser.Drain import LogParser
from config import BGL_RAW_DIR, BGL_PARSED_DIR, BGL_LOG_FORMAT

LOG_FILE_NAME = "BGL.log"  # à adapter

# regex de prétraitement pour Drain
REX = [
    r'\d+\.\d+\.\d+\.\d+',   # IPs
    r'\d+',                  # nombres
]

def main():
    # Crée le dossier de sortie si besoin
    BGL_PARSED_DIR.mkdir(parents=True, exist_ok=True)

    parser = LogParser(
        indir=str(BGL_RAW_DIR),
        outdir=str(BGL_PARSED_DIR),
        log_format=BGL_LOG_FORMAT,
        depth=4,      # profondeur de l'arbre Drain
        st=0.5,       # seuil de similarité
        rex=REX
    )

    parser.parse(LOG_FILE_NAME)


if __name__ == "__main__":
    main()