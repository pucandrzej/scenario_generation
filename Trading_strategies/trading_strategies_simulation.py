import os
import pickle
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from multiprocessing import Pool
import plotly.graph_objects as go

from calibration_config import bands_grid_config, median_grid_config
from utils import (
    # STRATEGY QUALITY MEASURES UTILS
    rtp,
    hhi,
    gini,
    topk_contribution,
    profit,
    mdd,
    avg_dd,
    downside_std,
    win_rate,
    # BANDS UTILS
    vanilla_band,
    weighted_band,
    # WEIGHTS UTILS
    weighted_median,
    compute_weights,
    # DEVEL TRAJECTORIES PLOTS UTILS
    add_curve,
    # NOVEL PROBABILISTIC FORECAST MEASURES UTILS
    weighted_classification_accuracy,
    probabilistic_weighted_classification_accuracy,
    # STRATEGY UTILS
    get_trust_threshold,
)

# script parameters
DEV_PLOTS = False
NO_PARALLEL = 32

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="median",
    help="Select the strategies model: bands, median, naive_1, naive_30, crystal_ball, opposite_crystal_ball in wca and pwca, where wca and pwca are measures and not strategies.",
)
parser.add_argument(
    "--underlying_model",
    default=None,
    help="Select the model for price forecasts: _____None____, _hist_insample_None_True_dual_coeff or _weather_scenarios_None_True_dual_coeff.",
)
parser.add_argument(
    "--underlying_model_column",
    default=None,
    help="Select the MULTI_prediction or benchmark_prediction.",
)
parser.add_argument(
    "--run_type",
    default="calibration",
    help="Select the run type: calibration or test.",
)
parser.add_argument(
    "--scp",
    default=np.nan,
    help="If test run is selected and model is bands we must set the scp for bands.",
)
parser.add_argument(
    "--distribution_param",
    default=np.nan,
    help="If test run is selected we must set the distribution p parameter.",
)
parser.add_argument(
    "--lambda_parameter",
    default=np.nan,
    help="If test run is selected we must set the lambda exponential path history impact damping parameter.",
)
parser.add_argument(
    "--trust_threshold",
    default=None,
    help="If test run is selected we must set the method of certainty threshold selection.",
)
parser.add_argument(
    "--weights_method",
    default=None,
    help="Method of calculating the weights: kernel or mae.",
)
parser.add_argument(
    "--results_dir",
    default="D:/SIMULATION_RESULTS/hist_dirs",
    help="The main directory where the results are stored.",
)
parser.add_argument(
    "--direction",
    default=0,
    help="For one sided strategy we need to specify the direction.",
)
parser.add_argument(
    "--one_sided",
    default=False,
    action="store_true",
    help="Whether to run a speculative two sided strategy or one sided strategies.",
)
parser.add_argument(
    "--band_type",
    default="risk_seeking",
    help="Risk seeking or risk averse band trading. In risk seeking type we enter at max.",
)
parser.add_argument(
    "--calibration_pickle_path",
    default=None,
    help="Path to a pickle containing calibration results - if passed, the calibration run won't be recalculated but rather the csv report of calibration generated based on this pickle.",
)
args = parser.parse_args()

if args.underlying_model is None:
    models = [
        "_hist_insample_None_True_dual_coeff",
        "_weather_scenarios_None_True_dual_coeff",
        "_____None____",
        "_hist_insample_None_False_None",
        "_weather_scenarios_None_False_None",
    ]
else:
    models = [args.underlying_model]

if args.underlying_model_column is None:
    columns_names = [
        "MULTI_prediction",
        "MULTI_prediction",
        "benchmark_predictionMULTI_prediction",
        "MULTI_prediction",
    ]
else:
    columns_names = [args.underlying_model_column]


def naive_1(y_actual):
    """
    Naive that always sells in 1st step for seller and sells in 1st step and buys at last step for speculator
    """

    if float(args.direction) == -1 and args.one_sided:
        profit = y_actual[0]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    elif float(args.direction) == 0 and not args.one_sided:
        profit = y_actual[0] - y_actual[-1]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit
    else:
        raise ValueError("Buyer is not implemented.")

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit


def naive_30(y_actual):
    """
    Naive that always sells in last step for seller and sells in last step and buys at first step for speculator
    """

    if float(args.direction) == -1 and args.one_sided:
        profit = y_actual[-1]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    elif float(args.direction) == 0 and not args.one_sided:
        profit = y_actual[-1] - y_actual[0]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit
    else:
        raise ValueError("Buyer is not implemented.")

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit


def one_sided_bands_strategy(
    y_actual,
    y_forecast,
    scp,
    p,
    lambda_=0.25,
    trust_threshold_method="3sigma",
    weights_method="kernel",
    direction=float(args.direction),
):
    if DEV_PLOTS:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape

    if direction == -1:
        if args.band_type == "risk_seeking":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="upper"
            )  # taking max of upper band maximizing the max expected price
        elif args.band_type == "risk_averse":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="lower"
            )  # taking max of lower band maximizing the min expected price
    elif direction == 1:
        if args.band_type == "risk_seeking":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="lower"
            )  # taking min of upper band minimizing the min expected price
        elif args.band_type == "risk_averse":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="upper"
            )  # taking min of upper band minimizing the max expected price

    # initial plan of trading
    argmax = int(np.argmax(band))
    argmin = int(np.argmin(band))

    if direction == 1:  # we buy - we want to buy at min price possible
        planned_entry = argmin
    elif direction == -1:  # we sell - we want to sell at max price possible
        planned_entry = argmax

    initial_planned_entry = planned_entry

    basic_profit = y_actual[initial_planned_entry]
    if direction == 1:
        best_profit = np.min(y_actual)
        worst_profit = np.max(y_actual)
    elif direction == -1:
        best_profit = np.max(y_actual)
        worst_profit = np.min(y_actual)

    played = False

    profit = 0  # profit declaration

    # iterate over t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = (
            np.median(forecast_so_far, axis=1) - price_so_far
        )  # calc errors between median of trajectories and price observed so far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )

        w = compute_weights(
            forecast_so_far,
            price_so_far,
            nonzero_mae,
            p,
            lambda_,
            weights_method,
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            break

        if direction == -1:
            if args.band_type == "risk_seeking":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")
            elif args.band_type == "risk_averse":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
        elif direction == 1:
            if args.band_type == "risk_seeking":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
            elif args.band_type == "risk_averse":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")

        if DEV_PLOTS:
            if t == 0:
                for fore_idx in range(np.shape(y_forecast)[1]):
                    add_curve(
                        fig,
                        x,
                        y_forecast[:, fore_idx],
                        f"{fore_idx} forecast path",
                        "grey",
                    )
                add_curve(fig, x, band, "Band", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(fig, x[T - len(cond_band) :], cond_band, f"Band {t}", "red")

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_band))
        rel_argmin = int(np.argmin(cond_band))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        if direction == 1:
            desired_entry = new_argmin
        elif direction == -1:
            desired_entry = new_argmax

        if planned_entry > t:
            planned_entry_profit = cond_band[planned_entry - t - 1]
        elif planned_entry == t:
            planned_entry_profit = y_actual[planned_entry]

        if planned_entry > t:
            desired_entry_profit = cond_band[desired_entry - t - 1]
        elif planned_entry == t:
            desired_entry_profit = y_actual[desired_entry]

        # we shift the entering of position if we see more profit from changing it
        if desired_entry_profit - trust_threshold > planned_entry_profit:
            planned_entry = desired_entry

        # entry logic: if not in position and planned entry is now -> enter
        if planned_entry == t:
            played = True
            profit = y_actual[planned_entry]
            break

    # force an action at the end of the path if action was not performed in course of the path
    if not played:
        played = True
        profit = y_actual[-1]

    return profit, basic_profit, played, best_profit, worst_profit


def two_sided_bands_strategy(
    y_actual,
    y_forecast,
    scp,
    p,
    lambda_=0.25,
    trust_threshold_method="3sigma",
    weights_method="kernel",
):
    if DEV_PLOTS:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape

    if args.band_type == "risk_seeking":
        max_band = vanilla_band(y_forecast, scp=scp, band_type="upper")
        min_band = vanilla_band(y_forecast, scp=scp, band_type="lower")
    elif args.band_type == "risk_averse":
        max_band = vanilla_band(y_forecast, scp=scp, band_type="lower")
        min_band = vanilla_band(y_forecast, scp=scp, band_type="upper")

    # initial plan of trading
    argmax = int(np.argmax(max_band))
    argmin = int(np.argmin(min_band))

    if argmin > argmax:
        planned_direction = -1
        planned_entry = argmax
        planned_exit = argmin
    else:
        planned_direction = 1
        planned_entry = argmin
        planned_exit = argmax

    direction = planned_direction

    initial_planned_entry = planned_entry
    initial_planned_exit = planned_exit

    basic_profit = (
        y_actual[initial_planned_exit] - y_actual[initial_planned_entry]
    ) * direction
    best_profit = np.max(y_actual) - np.min(y_actual)
    worst_profit = -best_profit

    # indicator if we are in position already
    in_position = False
    played = False

    # entry price and index
    entry_price = None

    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = np.median(forecast_so_far, axis=1) - price_so_far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )
        w = compute_weights(
            forecast_so_far,
            price_so_far,
            nonzero_mae,
            p,
            lambda_,
            weights_method,
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            # no future points; if in position close at last observed price
            if in_position:
                exit_price = y_actual[t]
                profit += (exit_price - entry_price) * direction
                in_position = False
            break

        if args.band_type == "risk_seeking":
            cond_max_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")
            cond_min_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
        elif args.band_type == "risk_averse":
            cond_max_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
            cond_min_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")

        if DEV_PLOTS:
            if t == 0:
                add_curve(fig, x, max_band, "Max Band", "blue")
                add_curve(fig, x, min_band, "Min Band", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig, x[T - len(cond_max_band) :], cond_max_band, f"Max Band {t}", "red"
            )
            add_curve(
                fig, x[T - len(cond_min_band) :], cond_min_band, f"Min Band {t}", "red"
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_max_band))
        rel_argmin = int(np.argmin(cond_min_band))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        # desired trading plan from conditional medians
        if new_argmin > new_argmax:
            desired_direction = -1
            desired_entry = new_argmax
            desired_exit = new_argmin
        else:
            desired_direction = 1
            desired_entry = new_argmin
            desired_exit = new_argmax

        if planned_entry > t:
            if (
                planned_direction == -1
            ):  # if we short we go from entry on max band to exit on min band
                planned_entry_profit = (
                    cond_min_band[planned_exit - t - 1]
                    - cond_max_band[planned_entry - t - 1]
                )
            else:  # if we long we go from entry on min band to exit on max band
                planned_entry_profit = (
                    cond_max_band[planned_exit - t - 1]
                    - cond_min_band[planned_entry - t - 1]
                )
        elif planned_entry == t:
            if (
                planned_direction == -1
            ):  # if we short we go from entry on max band to exit on min band
                planned_entry_profit = (
                    cond_min_band[planned_exit - t - 1] - y_actual[planned_entry]
                )
            else:  # if we long we go from entry on min band to exit on max band
                planned_entry_profit = (
                    cond_max_band[planned_exit - t - 1] - y_actual[planned_entry]
                )

        if desired_entry > t:
            if desired_direction == -1:
                desired_entry_profit = (
                    cond_min_band[desired_exit - t - 1]
                    - cond_max_band[desired_entry - t - 1]
                )
            else:
                desired_entry_profit = (
                    cond_max_band[desired_exit - t - 1]
                    - cond_min_band[desired_entry - t - 1]
                )
        elif desired_entry == t:
            if desired_direction == -1:
                desired_entry_profit = (
                    cond_min_band[desired_exit - t - 1] - y_actual[desired_entry]
                )
            else:
                desired_entry_profit = (
                    cond_max_band[desired_exit - t - 1] - y_actual[desired_entry]
                )

        # we shift the entering of position if we see more profit from changing it
        if (
            desired_exit != desired_entry
            and not in_position
            and desired_entry_profit * desired_direction - trust_threshold
            > planned_entry_profit * direction
        ):
            planned_entry = desired_entry
            planned_exit = desired_exit
            planned_direction = desired_direction

        # entry logic: if not in position and planned entry is now or in past -> enter
        if (not in_position) and (planned_entry == t):
            entry_price = y_actual[t]
            exit_index = planned_exit
            in_position = True
            played = True
            direction = planned_direction  # commit to direction at entry time

        if in_position:
            # check whether taking profit based on current weighted median and observed errors is profitable
            if (
                direction == -1
                and (y_actual[t] - entry_price) * direction
                > (min(cond_min_band) - entry_price) * direction + trust_threshold
            ) or (
                direction == 1
                and (y_actual[t] - entry_price) * direction
                > (max(cond_max_band) - entry_price) * direction + trust_threshold
            ):
                exit_price = y_actual[t]
                profit += (exit_price - entry_price) * direction
                in_position = False
                break

            # if planned exit is now -> check whether it is worth waiting and if not exit, otherwise update the exit time
            if exit_index == t:
                if (
                    direction == -1
                    and (y_actual[t] - entry_price) * direction
                    > (min(cond_min_band) - entry_price) * direction - trust_threshold
                ) or (
                    direction == 1
                    and (y_actual[t] - entry_price) * direction
                    > (max(cond_max_band) - entry_price) * direction - trust_threshold
                ):
                    exit_price = y_actual[t]
                    profit += (exit_price - entry_price) * direction
                    in_position = False
                    break
                else:
                    if direction == -1:
                        exit_index = new_argmin
                    elif direction == 1:
                        exit_index = new_argmax

    # end loop: if still in position close at last observation
    if in_position:
        exit_price = y_actual[-1]
        profit += (exit_price - entry_price) * direction

    return profit, basic_profit, played, best_profit, worst_profit


def one_sided_median_trading_strategy(
    y_actual,
    y_forecast,
    p=2,
    lambda_=0.25,
    trust_threshold_method="3sigma",
    weights_method="kernel",
    direction=float(args.direction),
):
    """
    Minimal dynamic evolution-tracking.
    - initial entry/exit from unconditional median across paths
    - at each time t, compute path weights based on observed history
    - compute weighted medians for future times and replan entry/exit
    - simulate immediate fills: enter when planned_entry <= t, exit when planned_exit <= t
    - if direction flips while in position, close and flip immediately
    - returns profit (single number).

    y_actual: shape (T,)
    y_forecast: shape (T, Npaths)
    """

    if DEV_PLOTS:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))
    argmin = int(np.argmin(central))

    if direction == 1:  # we buy - we want to buy at min price possible
        planned_entry = argmin
    elif direction == -1:  # we sell - we want to sell at max price possible
        planned_entry = argmax

    initial_planned_entry = planned_entry

    basic_profit = y_actual[initial_planned_entry]
    if direction == 1:
        best_profit = np.min(y_actual)
        worst_profit = np.max(y_actual)
    elif direction == -1:
        best_profit = np.max(y_actual)
        worst_profit = np.min(y_actual)

    # indicator if we are in position already
    played = False
    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = np.median(forecast_so_far, axis=1) - price_so_far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )

        w = compute_weights(
            forecast_so_far, price_so_far, nonzero_mae, p, lambda_, weights_method
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            break

        cond_medians = np.empty(future_count)
        for idx, s in enumerate(range(t + 1, T)):
            vals = y_forecast[s, :]
            cond_medians[idx] = weighted_median(vals, w)

        if DEV_PLOTS:
            if t == 0:
                add_curve(fig, x, central, "Median", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig,
                x[T - len(cond_medians) :],
                cond_medians,
                f"Conditional medians {t}",
                "red",
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_medians))
        rel_argmin = int(np.argmin(cond_medians))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        if direction == 1:
            desired_entry = new_argmin
        elif direction == -1:
            desired_entry = new_argmax

        if planned_entry > t:
            planned_entry_profit = cond_medians[planned_entry - t - 1]
        elif planned_entry == t:
            planned_entry_profit = y_actual[planned_entry]

        if desired_entry > t:
            desired_entry_profit = cond_medians[desired_entry - t - 1]
        elif desired_entry == t:
            desired_entry_profit = y_actual[desired_entry]

        # we shift the entering of position if we see more profit from changing it
        if desired_entry_profit - trust_threshold > planned_entry_profit:
            planned_entry = desired_entry

        # entry logic: if not in position and planned entry is now -> enter
        if planned_entry == t:
            played = True
            profit = y_actual[planned_entry]
            break

    # force an action at the end of the path if action was not performed in course of the path
    if not played:
        played = True
        profit = y_actual[-1]

    # no trade executed
    return profit, basic_profit, played, best_profit, worst_profit


def two_sided_median_trading_strategy(
    y_actual,
    y_forecast,
    p=2,
    lambda_=0.25,
    trust_threshold_method="3sigma",
    weights_method="kernel",
):
    """
    Minimal dynamic evolution-tracking.
    - initial entry/exit from unconditional median across paths
    - at each time t, compute path weights based on observed history
    - compute weighted medians for future times and replan entry/exit
    - simulate immediate fills: enter when planned_entry <= t, exit when planned_exit <= t
    - if direction flips while in position, close and flip immediately
    - returns profit (single number).

    y_actual: shape (T,)
    y_forecast: shape (T, Npaths)
    """

    if DEV_PLOTS:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))
    argmin = int(np.argmin(central))

    if argmin > argmax:
        planned_direction = -1
        planned_entry = argmax
        planned_exit = argmin
    else:
        planned_direction = 1
        planned_entry = argmin
        planned_exit = argmax

    direction = planned_direction

    initial_planned_entry = planned_entry
    initial_planned_exit = planned_exit

    basic_profit = (
        y_actual[initial_planned_exit] - y_actual[initial_planned_entry]
    ) * direction
    best_profit = np.max(y_actual) - np.min(y_actual)
    worst_profit = -best_profit  # for speculator the worst profit is - best profit

    # indicator if we are in position already
    in_position = False
    played = False

    # entry price and index
    entry_price = None

    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = np.median(forecast_so_far, axis=1) - price_so_far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )
        w = compute_weights(
            forecast_so_far,
            price_so_far,
            nonzero_mae,
            p,
            lambda_,
            weights_method,
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            # no future points; if in position close at last observed price
            if in_position:
                exit_price = y_actual[t]
                profit = (exit_price - entry_price) * direction
                in_position = False
            break

        cond_medians = np.empty(future_count)
        for idx, s in enumerate(range(t + 1, T)):
            vals = y_forecast[s, :]
            cond_medians[idx] = weighted_median(vals, w)

        if DEV_PLOTS:
            if t == 0:
                add_curve(fig, x, central, "Median", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig,
                x[T - len(cond_medians) :],
                cond_medians,
                f"Conditional medians {t}",
                "red",
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_medians))
        rel_argmin = int(np.argmin(cond_medians))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        # desired trading plan from conditional medians
        if new_argmin > new_argmax:
            desired_direction = -1
            desired_entry = new_argmax
            desired_exit = new_argmin
        else:
            desired_direction = 1
            desired_entry = new_argmin
            desired_exit = new_argmax

        # prepare the planned profit: in case the planned entry is at t we already know the price
        if planned_entry > t:
            planned_entry_profit = (
                cond_medians[planned_exit - t - 1] - cond_medians[planned_entry - t - 1]
            )
        elif planned_entry == t:
            planned_entry_profit = (
                cond_medians[planned_exit - t - 1] - y_actual[planned_entry]
            )

        if desired_entry > t:
            desired_entry_profit = (
                cond_medians[desired_exit - t - 1] - cond_medians[desired_entry - t - 1]
            )
        elif desired_entry == t:
            desired_entry_profit = (
                cond_medians[desired_exit - t - 1] - y_actual[desired_entry]
            )

        # we shift the entering of position if we see more profit from changing it
        if (
            desired_exit != desired_entry
            and not in_position
            and desired_entry_profit * desired_direction - trust_threshold
            > planned_entry_profit * direction
        ):
            planned_entry = desired_entry
            planned_exit = desired_exit
            planned_direction = desired_direction

        # entry logic: if not in position and planned entry is now or in past -> enter
        if (not in_position) and (planned_entry == t):
            entry_price = y_actual[t]
            exit_index = planned_exit
            in_position = True
            played = True
            direction = planned_direction  # commit to direction at entry time

        if in_position:
            # check whether taking profit based on current weighted median and observed errors is profitable
            if (
                direction == -1
                and (y_actual[t] - entry_price) * direction
                > (min(cond_medians) - entry_price) * direction + trust_threshold
            ) or (
                direction == 1
                and (y_actual[t] - entry_price) * direction
                > (max(cond_medians) - entry_price) * direction + trust_threshold
            ):
                exit_price = y_actual[t]
                profit = (exit_price - entry_price) * direction
                in_position = False
                break

            # if planned exit is now -> check whether it is worth waiting and if not exit, otherwise update the exit time
            if exit_index == t:
                if (
                    direction == -1
                    and (y_actual[t] - entry_price) * direction
                    > (min(cond_medians) - entry_price) * direction - trust_threshold
                ) or (
                    direction == 1
                    and (y_actual[t] - entry_price) * direction
                    > (max(cond_medians) - entry_price) * direction - trust_threshold
                ):
                    exit_price = y_actual[t]
                    profit = (exit_price - entry_price) * direction
                    in_position = False
                    break
                else:
                    if direction == -1:
                        exit_index = new_argmin
                    elif direction == 1:
                        exit_index = new_argmax

    # end loop: if still in position close at last observation
    if in_position:
        exit_price = y_actual[-1]
        profit = (exit_price - entry_price) * direction

    return profit, basic_profit, played, best_profit, worst_profit


def iterate_over_probab_results_and_prepare_measure(inp):
    measure_func = inp[0]
    dir_name = inp[1]
    column_name = inp[2]
    scp = inp[3]
    p = inp[4]
    lambda_ = inp[5]
    trust_threshold_method = inp[6]
    weights_method = inp[7]

    measure_values_delivery = []

    for counter, daily_file in enumerate(
        [
            f_name
            for f_name in os.listdir(f"{args.results_dir}/{dir_name}")
            if f_name.startswith(f"{args.run_type}_")
        ]
    ):
        measure_values_day = []

        df = pd.read_csv(f"{args.results_dir}/{dir_name}/{daily_file}", index_col=0)
        actual = df["actual"].values
        if args.model in ["wca", "pwca"]:
            naive = df["naive"].values
        fore = df[
            [
                c
                for c in df.columns
                if c.startswith(column_name) and "base_path" not in c
            ]
        ].values

        if args.model == "median":
            measure_values_day.append(
                measure_func(
                    actual, fore, p, lambda_, trust_threshold_method, weights_method
                )
            )
        elif args.model == "bands":
            measure_values_day.append(
                measure_func(
                    actual,
                    fore,
                    scp,
                    p,
                    lambda_,
                    trust_threshold_method,
                    weights_method,
                )
            )
        elif args.model in ["naive_1", "naive_30"]:
            measure_values_day.append(measure_func(actual))
        elif args.model == "wca":
            measure_values_day.append(measure_func(actual, fore, naive))
        elif args.model == "pwca":
            measure_values_day.append(measure_func(actual, fore, naive))

        measure_values_delivery.append(measure_values_day)

    return measure_values_delivery


if __name__ == "__main__":
    results = {}
    if args.run_type == "calibration":  # calibration on calibration window data
        if args.model == "bands":
            grid_config = bands_grid_config
        elif args.model == "median":
            grid_config = median_grid_config
        else:
            raise ValueError(f"No calibration implemented for model type {args.model}")
        p_list = grid_config["p_list"]
        lambda_list = grid_config["lambda_list"]
        trust_threshold_method = grid_config["trust_threshold_method"]
        parameter_method_1 = grid_config["parameter_method_1"]
        parameter_method_2 = grid_config["parameter_method_2"]
        scp_list = grid_config["scp"]
        grid = list(
            itertools.product(
                scp_list,
                p_list,
                lambda_list,
                trust_threshold_method,
                parameter_method_1,
            )
        ) + list(
            itertools.product(
                scp_list,
                [np.nan],
                [np.nan],
                trust_threshold_method,
                parameter_method_2,
            )
        )
    elif args.run_type == "test":  # test on test window data
        scp_list = [float(args.scp)]
        p_list = [float(args.distribution_param)]
        lambda_list = [float(args.lambda_parameter)]
        trust_threshold_method = [args.trust_threshold]
        parameter_method = [args.weights_method]
        grid = list(
            itertools.product(
                scp_list, p_list, lambda_list, trust_threshold_method, parameter_method
            )
        )
    if args.one_sided:
        if args.model == "bands":
            func = one_sided_bands_strategy
        elif args.model == "median":
            func = one_sided_median_trading_strategy
        elif args.model == "naive_1":
            func = naive_1
        elif args.model == "naive_30":
            func = naive_30
        elif args.model == "wca":
            func = weighted_classification_accuracy
        elif args.model == "pwca":
            func = probabilistic_weighted_classification_accuracy
    else:
        if args.model == "bands":
            func = two_sided_bands_strategy
        elif args.model == "median":
            func = two_sided_median_trading_strategy
        elif args.model == "naive_1":
            func = naive_1
        elif args.model == "naive_30":
            func = naive_30

    calibration_pickle_name = f"results_gridsearch_{args.one_sided}_{args.direction}_{args.model}_{args.band_type}_{datetime.now().strftime('%Y-%m-%d %H;%M;%S')}.pkl"

    if args.calibration_pickle_path is None:
        for model, column_name in zip(models, columns_names):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing {model}, {column_name}",
                flush=True,
            )

            delivery_directories = [
                d for d in os.listdir(f"{args.results_dir}") if d.endswith(model)
            ]

            for parameter_tuple in grid:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    processing {parameter_tuple}",
                    flush=True,
                )

                inputlist = [
                    [
                        func,
                        delivery_dir,
                        column_name,
                        parameter_tuple[0],
                        parameter_tuple[1],
                        parameter_tuple[2],
                        parameter_tuple[3],
                        parameter_tuple[4],
                    ]
                    for delivery_dir in delivery_directories
                ]
                with Pool(processes=NO_PARALLEL) as p:
                    parallel_results = p.map(
                        iterate_over_probab_results_and_prepare_measure, inputlist
                    )

                results[parameter_tuple + (model, column_name)] = np.array(
                    parallel_results
                )

        # save results to pickle file
        if args.run_type == "calibration":
            with open(
                calibration_pickle_name,
                "wb",
            ) as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(
            args.calibration_pickle_path,
            "rb",
        ) as f:
            results = pickle.load(f)

    if args.model in ["wca", "pwca"]:
        stability_measures = {}

        for (k, arr), params in zip(results.items(), results.keys()):
            stability_measures[params] = []
            # sum over axis=0, take last index along axis=-1, then cumsum
            y = np.cumsum(np.sum(arr, axis=0)[:, -1, :], axis=0) / 1000

            measure_per_day = arr[:, :, 0, 0].T.reshape(-1)
            ref_measure_per_day = arr[:, :, 0, 1].T.reshape(-1)

            stability_measures[params].append(np.mean(measure_per_day))
            ref_params = tuple(
                ["_" for p in range(np.shape(grid)[1])]
                + list(params[np.shape(grid)[1] :])
            )
            stability_measures[ref_params] = []
            stability_measures[ref_params].append(np.mean(ref_measure_per_day))

        df = pd.DataFrame(
            [
                (a, b, c, d, e, f, *vals)
                for (a, b, c, d, e, f), vals in stability_measures.items()
            ],
            columns=[
                "param1",
                "param2",
                "threshold",
                "weights",
                "model_setting",
                "model",
                "measure_value",
            ],
        )

        print(df.to_string())

    else:
        fig = go.Figure()

        stability_measures = {}
        stability_measures_reference = {}

        for (k, arr), params in zip(results.items(), results.keys()):
            stability_measures[params] = []
            # sum over axis=0, take last index along axis=-1, then cumsum
            try:
                y = np.cumsum(np.sum(arr, axis=0)[:, -1, :], axis=0) / 1000
            except:
                breakpoint()
                print(" ")

            pnl_per_delivery_and_day = arr[:, :, 0, 0].T.reshape(-1)
            action_per_delivery_and_day = arr[:, :, 0, 2].T.reshape(-1)
            ref_pnl_per_delivery_and_day = arr[:, :, 0, 1].T.reshape(-1)
            best_pnl_per_delivery_and_day = arr[:, :, 0, 3].T.reshape(-1)
            worst_pnl_per_delivery_and_day = arr[:, :, 0, 4].T.reshape(-1)

            stability_measures[params].append(mdd(pnl_per_delivery_and_day))
            stability_measures[params].append(avg_dd(pnl_per_delivery_and_day))
            stability_measures[params].append(
                downside_std(pnl_per_delivery_and_day, one_sided=args.one_sided)
            )
            stability_measures[params].append(np.std(pnl_per_delivery_and_day))
            stability_measures[params].append(win_rate(pnl_per_delivery_and_day))
            stability_measures[params].append(profit(pnl_per_delivery_and_day))
            stability_measures[params].append(hhi(pnl_per_delivery_and_day))
            stability_measures[params].append(gini(pnl_per_delivery_and_day))
            stability_measures[params].append(
                topk_contribution(pnl_per_delivery_and_day)
            )
            stability_measures[params].append(
                rtp(
                    pnl_per_delivery_and_day,
                    best_pnl_per_delivery_and_day,
                    worst_pnl_per_delivery_and_day,
                    one_sided=args.one_sided,
                )
            )
            stability_measures[params].append(
                1
                - np.sum(action_per_delivery_and_day) / len(action_per_delivery_and_day)
            )
            stability_measures[params].append(np.mean(pnl_per_delivery_and_day))
            stability_measures[params].append(profit(best_pnl_per_delivery_and_day))
            stability_measures[params].append(profit(worst_pnl_per_delivery_and_day))
            stability_measures[params].append(
                downside_std(best_pnl_per_delivery_and_day, one_sided=args.one_sided)
            )
            stability_measures[params].append(
                downside_std(worst_pnl_per_delivery_and_day, one_sided=args.one_sided)
            )

            x = np.arange(len(y))  # forecast days

            if args.run_type == "test":
                fig.add_trace(
                    go.Scatter(x=x, y=y[:, 0], mode="lines", name=f"strategy {k}")
                )

            if args.model == "bands":  # for bands we want to save every basic SCP level
                ref_params = tuple(
                    [params[0]]
                    + ["_" for p in range(1, np.shape(grid)[1])]
                    + list(params[np.shape(grid)[1] :])
                )
            else:
                ref_params = tuple(
                    ["_" for p in range(np.shape(grid)[1])]
                    + list(params[np.shape(grid)[1] :])
                )

            if ref_params not in stability_measures_reference:
                stability_measures_reference[ref_params] = []

                stability_measures_reference[ref_params].append(
                    mdd(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    avg_dd(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    downside_std(ref_pnl_per_delivery_and_day, one_sided=args.one_sided)
                )
                stability_measures_reference[ref_params].append(
                    np.std(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    win_rate(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    hhi(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    gini(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    topk_contribution(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    rtp(
                        ref_pnl_per_delivery_and_day,
                        best_pnl_per_delivery_and_day,
                        worst_pnl_per_delivery_and_day,
                        one_sided=args.one_sided,
                    )
                )
                stability_measures_reference[ref_params].append(0)
                stability_measures_reference[ref_params].append(
                    np.mean(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(best_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(worst_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    downside_std(
                        best_pnl_per_delivery_and_day, one_sided=args.one_sided
                    )
                )
                stability_measures_reference[ref_params].append(
                    downside_std(
                        worst_pnl_per_delivery_and_day, one_sided=args.one_sided
                    )
                )

                if args.run_type == "test":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y[:, 1],
                            mode="lines",
                            name=f"baseline strategy {ref_params}",
                        )
                    )

        if args.run_type == "test":
            # Labels and style
            fig.update_layout(
                title="PROBAB MEDIAN TRADING profit and loss (PnL)",
                xaxis_title="forecast days",
                yaxis_title="PROBAB MEDIAN TRADING<br>profit and loss (PnL) [1000 EUR/MWh]",
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
                ),
                font=dict(size=10),
            )

            fig.write_html(
                f"strategies_{args.run_type}_{args.one_sided}_{args.direction}_{args.model}.html"
            )
            fig.show()

        # save the table with the measures
        for k in stability_measures_reference.keys():
            stability_measures[k] = stability_measures_reference[k]
        df = pd.DataFrame(
            [
                (a, b, c, d, e, f, g, *vals)
                for (a, b, c, d, e, f, g), vals in stability_measures.items()
            ],
            columns=[
                "param1",
                "param2",
                "param3",
                "threshold",
                "weights",
                "model_setting",
                "model",
                "MDD",
                "avg D",
                "std_minus",
                "std",
                "win_rate",
                "profit",
                "hhi",
                "gini",
                "topk",
                "rtp",
                "no_action_perc",
                "profit_per_action",
                "crystal_profit",
                "noncrystal_profit",
                "crystal_std_minus",
                "noncrystal_std_minus",
            ],
        )

        if (
            float(args.direction) == 0 or float(args.direction) == -1
        ):  # RATIO if we maximize the profit and minimize risk
            df["Sortino_ratio"] = df["profit"] / df["std_minus"]
            df = df.sort_values("Sortino_ratio", ascending=False)
        elif (
            float(args.direction) == 1
        ):  # PRODUCT if we minimize the profit and minimize risk
            df["Sortino_product"] = df["profit"] * df["std_minus"]
            df = df.sort_values("Sortino_product", ascending=True)

        if args.run_type == "calibration":
            df.to_csv(
                f"FROM_PICKLE_grid_search_trading_strategy_measures_{args.one_sided}_{args.direction}_{args.model}.csv"
            )
        else:
            df.to_csv(
                f"test_trading_strategy_measures_{args.underlying_model}_{args.underlying_model_column}_{args.one_sided}_{args.direction}_{args.model}_{args.band_type}.csv"
            )
            print(df.to_string())
