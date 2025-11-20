import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Date ranges
MIN_DATE = "2016-10-01"
MAX_DATE = "2018-08-31"

TRAIN_VAL_END = "2018-04-30"
BACKTEST_START = "2018-05-01"
BACKTEST_END = "2018-07-31"
TEST_FINAL = "2018-08-01"
