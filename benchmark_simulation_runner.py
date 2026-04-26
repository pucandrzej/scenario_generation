"""
Script to run the simulation for the required configuration of distances before trading and forecast and different deliveries
"""

import sys
import time
import subprocess

import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_delivery", default=0, help="Start of the simulated deliveries"
)

parser.add_argument(
    "--end_delivery", default=96, help="End of the simulated deliveries"
)

parser.add_argument(
    "--calibration_window_len",
    default=182,
    help="For every date consider a historical results from a calibration window.",
)

processes = 32
required_scenarios_list = [28, 182, None]

args = parser.parse_args()

start = args.start_delivery
joblist = []
sys.stderr = open(
    f"TOTAL_SIMU_ERR_{start}_{args.end_delivery}_benchmark_{args.calibration_window_len}.txt",
    "w",
)
sys.stdout = open(
    f"TOTAL_SIMU_LOG_{start}_{args.end_delivery}_benchmark_{args.calibration_window_len}.txt",
    "w",
)
for shift_trade in [
    6
]:  # delivery time - shift_trade is the trade time, 7 because we want to forecast the last interval at 35 to 30min before the delivery
    for delivery_time in range(int(start), int(args.end_delivery)):
        trade_time = delivery_time * 3 + 8 * 12 - shift_trade

        for required_scenarios in required_scenarios_list:
            joblist.append(
                [
                    sys.executable,
                    "-m",
                    "naive_path_forecasting.py",
                    "--trade_time",
                    str(trade_time),
                    "--delivery_time",
                    str(delivery_time),
                    "--calibration_window_len",
                    str(args.calibration_window_len),
                    "--processes",
                    str(processes),
                ]
                + ["--required_scenarios", str(required_scenarios)]
                * (required_scenarios is not None)
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
