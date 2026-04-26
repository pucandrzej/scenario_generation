import sys
import pandas as pd
import subprocess
import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Redirect all stdout/stderr to file
sys.stdout = open("STRATEGIES_TEST_RESULTS.txt", "w")
sys.stderr = sys.stdout
parser.add_argument(
    "--weighting_type",
    default="kernel",
    help="Choose the weighting type to calculate the results for.",
)
args = parser.parse_args()

calibration_files = [
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_median.csv",
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_median.csv",
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_bands_risk_seeking.csv",
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_bands_risk_seeking.csv",
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_bands_risk_averse.csv",
    "PAPER_GRADE_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_bands_risk_averse.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_median.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_median.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_bands_risk_seeking.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_bands_risk_seeking.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_False_0_bands_risk_averse.csv",
    "PAPER_GRADE_NO_SCENARIOS_FILTER_FROM_PICKLE_grid_search_trading_strategy_measures_True_-1_bands_risk_averse.csv"
]


def parse_file_flags(filename):

    flags = {}

    if "False_0" in filename:
        flags["one_sided"] = False
        flags["direction"] = None

    elif "True_-1" in filename:
        flags["one_sided"] = True
        flags["direction"] = -1

    if "_median" in filename:
        flags["model"] = "median"

    elif "_bands_" in filename:
        flags["model"] = "bands"

    if "risk_seeking" in filename:
        flags["band_type"] = "risk_seeking"

    elif "risk_averse" in filename:
        flags["band_type"] = "risk_averse"

    else:
        flags["band_type"] = None

    return flags


for file in calibration_files:

    print("\n" + "=" * 80)
    print(f"RUNNING MODELS FROM: {file}")
    print("=" * 80)

    df = pd.read_csv(file, index_col=0)

    weighting_type_df = df[df["weights"] == args.weighting_type]

    if args.weighting_type == "_": # hack to make the simulation run for best static strategy parameters
        weighting_type_df[['param2', 'param3']] = (
            weighting_type_df[['param2', 'param3']]
            .replace('_', np.nan)
            )
        weighting_type_df["threshold"] = 'mae'
        weighting_type_df["weights"] = 'mae'

    idx = weighting_type_df.groupby("model_setting")["Sortino_ratio"].idxmax()
    best_rows = weighting_type_df.loc[idx]

    flags = parse_file_flags(file)

    for _, row in best_rows.iterrows():

        cmd = [
            "python",
            "trading_strategies_simulation.py",
            "--model", flags["model"],
            "--run_type", "test",
            "--weights_method", str(row["weights"]),
            "--distribution_param", str(row["param2"]),
            "--lambda_parameter", str(row["param3"]),
            "--trust_threshold", str(row["threshold"]),
            "--underlying_model", str(row["model_setting"]),
            "--underlying_model_column", str(row["model"]),
        ]

        if flags["one_sided"]:
            cmd.append("--one_sided")
            cmd.extend(["--direction", str(flags["direction"])])

        if flags["model"] == "bands" and flags["band_type"]:
            cmd.extend(["--band_type", flags["band_type"]])
            cmd.extend(["--scp", str(row["param1"])])
        print("\nCOMMAND:")
        print(" ".join(cmd))
        print("-" * 80)

        subprocess.run(cmd, check=True)


print("\nALL RUNS FINISHED")
