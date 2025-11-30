from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SEQUENCE_DIR = DATA_DIR / "sequences"
FEATURE_DIR = DATA_DIR / "features"

HDFS_RAW_DIR = RAW_DIR / "hdfs"
BGL_RAW_DIR = RAW_DIR / "bgl"

HDFS_PARSED_DIR = PARSED_DIR / "hdfs"
BGL_PARSED_DIR = PARSED_DIR / "bgl"

HDFS_SEQUENCE_DIR = SEQUENCE_DIR / "hdfs"
BGL_SEQUENCE_DIR = SEQUENCE_DIR / "bgl"

HDFS_FEATURE_DIR = FEATURE_DIR / "hdfs"
BGL_FEATURE_DIR = FEATURE_DIR / "bgl"

HDFS_LOG_FORMAT = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
BGL_LOG_FORMAT = ""