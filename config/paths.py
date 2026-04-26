import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT, "Data")
MARKET_DATA_DIR = os.path.join(
    ROOT, "Data", "preprocessed_continuous_intraday_prices_and_volume.db"
)
LOGS_DIR = os.path.join(ROOT, "LOGS")
