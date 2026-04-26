"""Runner of the forecasting simulation for all the deliveries"""

import os
import time
import subprocess

import sys

import argparse

from config.paths import LOGS_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_delivery", default=0, help="Start of the simulated deliveries"
)
parser.add_argument(
    "--end_delivery", default=96, help="End of the simulated deliveries"
)
parser.add_argument(
    "--models", default=["kernel_hr_naive_mult"], help="Models to simulate."
)
parser.add_argument(
    "--calibration_window_len",
    default=182,
    help="For every date consider a historical results from a calibration window.",
)
parser.add_argument(
    "--special_results_directory",
    default=None,
    help="Running on WCSS Wroclaw University of Science and Technology supercomputers requires us to save the results in dedicated path.",
)

probab_approaches = ["hist_insample", "weather_scenarios"]

wasserstein_stopping_crits = [
    True,
    False,
]

senarios_sampling_methods = [
    "dual_coeff",
    None,
]

required_scenarios_list = [28, 182, None]

repeated_required_scenarios_list = [
    val
    for val in required_scenarios_list
    for _ in range(len(wasserstein_stopping_crits))
]

print(
    wasserstein_stopping_crits * len(required_scenarios_list),
    senarios_sampling_methods * len(required_scenarios_list),
    repeated_required_scenarios_list,
)

args = parser.parse_args()
for model in args.models:
    start = args.start_delivery
    joblist = []
    sys.stderr = open(
        os.path.join(
            LOGS_DIR,
            f"TOTAL_SIMU_ERR_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}.txt",
        ),
        "w",
    )
    sys.stdout = open(
        os.path.join(
            LOGS_DIR,
            f"TOTAL_SIMU_LOG_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}.txt",
        ),
        "w",
    )

    for probab_approach in probab_approaches:
        for shift_trade in [
            6
        ]:  # delivery time - shift_trade is the trade time, 7 because we want to forecast the last interval at 35 to 30min before the delivery
            for delivery_time in range(int(start), int(args.end_delivery)):
                trade_time = delivery_time * 3 + 8 * 12 - shift_trade
                processes = 32

                for (
                    wasserstein_stopping_crit,
                    scenarios_sampling_method,
                    required_scenarios,
                ) in zip(
                    wasserstein_stopping_crits * len(required_scenarios_list),
                    senarios_sampling_methods * len(required_scenarios_list),
                    repeated_required_scenarios_list,
                ):
                    joblist.append(
                        [
                            sys.executable,
                            "-m",
                            "Forecasting.path_forecasting",
                            "--trade_time",
                            str(trade_time),
                            "--delivery_time",
                            str(delivery_time),
                            "--calibration_window_len",
                            str(args.calibration_window_len),
                            "--processes",
                            str(processes),
                            "--probab_approach",
                            probab_approach,
                        ]
                        + ["--scenarios_sampling_method", scenarios_sampling_method]
                        * (scenarios_sampling_method is not None)
                        + ["--required_scenarios", str(required_scenarios)]
                        * (required_scenarios is not None)
                        + ["--wasserstein_stopping_crit"] * wasserstein_stopping_crit
                        + [
                            "--special_results_directory",
                            str(args.special_results_directory),
                        ]
                        * (args.special_results_directory is not None)
                    )

    invoked = 0
    stack = []
    ts = time.time()
    concurrent = 1
    while invoked < len(joblist):
        while len(stack) == concurrent:
            for no, p in enumerate(stack):
                if p.poll() is not None:
                    stack.pop(no)
                    break
            time.sleep(1)
        line = joblist[invoked]
        print(f"running job {invoked + 1} of {len(joblist)}: {joblist[invoked]}")
        stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
        stack[-1].wait()  # wait for the process to finish
        invoked += 1
