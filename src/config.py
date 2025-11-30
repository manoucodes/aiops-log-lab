from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"

HDFS_RAW_DIR = RAW_DIR / "hdfs"
BGL_RAW_DIR = RAW_DIR / "bgl"

HDFS_PARSED_DIR = PARSED_DIR / "hdfs"
BGL_PARSED_DIR = PARSED_DIR / "bgl"

HDFS_LOG_FORMAT = ""
BGL_LOG_FORMAT = ""