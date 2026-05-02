import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT, "Data")
MARKET_DATA_DIR = os.path.join(
    ROOT, "Data", "preprocessed_continuous_intraday_prices_and_volume.db"
)
CONCATENATED_RAW_MARKET_DATA = os.path.join(
    DATA_DIR, "Transactions", "concatenated_table.csv"
)
INTERMEDIATE_MARKET_DATA = os.path.join(
    DATA_DIR, "Transactions", "quarterhourly_price_analysis_table_5min.csv"
)
INITIALLY_PREPROCESSED_MARKET_DATA = os.path.join(
    DATA_DIR, "quarterhourly_preprocessed_dataset_5min.csv"
)
LOGS_DIR = os.path.join(ROOT, "LOGS")
os.makedirs(
    LOGS_DIR, exist_ok=True
)  # create here to avoid calling it multiple times in other scripts

TIMING_RESULTS_DIR = os.path.join(ROOT, "TIMING_RESULTS")
os.makedirs(
    TIMING_RESULTS_DIR, exist_ok=True
)  # create here to avoid calling it multiple times in other scripts

BENCHMARK_RESULTS_DIR = os.path.join(ROOT, "BENCHMARK_FORECASTING_SIMULATION_RESULTS")
MODEL_RESULTS_DIR = os.path.join(ROOT, "FORECASTING_SIMULATION_RESULTS")
RAW_MODEL_RESULTS_DIR = "FORECASTING_SIMULATION_RESULTS"
