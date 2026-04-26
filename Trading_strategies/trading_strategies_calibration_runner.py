"""
Script to run the simulation for the required configuration of distances before trading and forecast and different deliveries
"""

import time
import subprocess

import sys

joblist = []
sys.stderr = open(
    "TRADING_CALIBRATION_ERR.txt",
    "w",
)
sys.stdout = open(
    "TRADING_CALIBRATION_OUT.txt",
    "w",
)

model = "bands"

for direction, one_sided in zip([-1], [True]):
    joblist.append(
        [
            "C:/Users/riczi/Studies/Continuous_market_analysis/contmarket311/Scripts/python.exe",
            "trading_strategies_simulation.py",
        ]
        + ["--direction", str(direction)] * (direction is not None)
        + ["--one_sided"] * one_sided
        + ["--model", model]
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
    print(
        f"running job {invoked + 1} of {len(joblist)}: {joblist[invoked]}", flush=True
    )
    stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
    stack[-1].wait()  # wait for the process to finish
    invoked += 1
