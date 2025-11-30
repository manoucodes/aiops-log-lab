from pathlib import Path
from logparser import Drain
from config import BGL_RAW_DIR, BGL_PARSED_DIR, BGL_LOG_FORMAT

LOG_FILE_NAME = "BGL.log"  # à adapter

def main():
    BGL_PARSED_DIR.mkdir(parents=True, exist_ok=True)

    parser = Drain.LogParser(
        indir=str(BGL_RAW_DIR),
        outdir=str(BGL_PARSED_DIR),
        log_format=BGL_LOG_FORMAT,
        depth=4,
        st=0.5,
        rex=[]
    )

    parser.parse(LOG_FILE_NAME)


if __name__ == "__main__":
    main()