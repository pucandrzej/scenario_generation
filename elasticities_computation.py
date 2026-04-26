import os
import sqlite3
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import timedelta

from config.paths import DATA_DIR, MARKET_DATA_DIR
from config.test_calibration_validation import (
    required_start,
    required_end,
)
from utils import fill_march_dst_daily, check_for_missing_data

DEVEL_PLOTS = False


def compute_transformed_supply_elasticity(
    df_curve,
    P_clearing,
    P_auction,
    q,
    forecasting_date,
    P_min=-3000,
    side_supply="Sell",
    side_demand="Purchase",
):
    """
    - df_curve: DataFrame with columns ["Price","Volume","Sale/Purchase"].
    - P_clearing: observed clearing price P_{DA} (intraday price).
    - q: volume offset for finite-difference.
    - P_min: floor price (e.g. -3000).
    Returns: elasticity and shows 3-panel plots.
    """

    # Split curves
    df_sup = df_curve[df_curve["Sale/Purchase"] == side_supply].copy()
    df_dem = df_curve[df_curve["Sale/Purchase"] == side_demand].copy()

    if len(df_sup) == 0 or len(df_dem) == 0:
        return np.nan  # no offers - time change

    # Sort by Price for price→volume mappings
    df_sup_price = df_sup.sort_values("Price").reset_index(drop=True)
    df_dem_price = df_dem.sort_values("Price").reset_index(drop=True)

    # Build price->volume functions
    price2vol_sup = lambda P: np.interp(
        P, df_sup_price["Price"], df_sup_price["Volume"]
    )
    price2vol_dem = lambda P: np.interp(
        P, df_dem_price["Price"], df_dem_price["Volume"]
    )

    # Compute key volumes
    q0 = price2vol_sup(P_auction)  # original clearing volume from supply side
    DEM_inelastic = price2vol_dem(P_min)  # inelastic demand volume at floor price

    # Define transformed supply S_trans(P) returning volume
    def S_trans(P):
        return price2vol_sup(P) + DEM_inelastic - price2vol_dem(P)

    q_implied = S_trans(
        P_clearing
    )  # implied volume corresponding to the last known continuous market price (naive)

    # Build inversion arrays from breakpoints
    # Collect unique price levels from supply and demand
    P_vals = np.unique(
        np.concatenate([df_sup_price["Price"].values, df_dem_price["Price"].values])
    )

    # Compute transformed volumes at each price
    V_vals = np.array(
        [price2vol_sup(P) + DEM_inelastic - price2vol_dem(P) for P in P_vals]
    )

    # Sort by volume to ensure monotonicity
    idx = np.argsort(V_vals)
    V_sorted = V_vals[idx]
    P_sorted = P_vals[idx]

    # Now invert: given z, find P via interpolation on (V_sorted, P_sorted)
    def sup_trans_inv(z):
        return np.interp(z, V_sorted, P_sorted)

    # (c) Slope segment around q0
    P_plus = sup_trans_inv(q_implied + q)
    P_minus = sup_trans_inv(q_implied - q)

    # 8) Compute elasticity: dP/dz ≈ (P_plus - P_minus)/(2*q)
    elasticity = (P_plus - P_minus) / (2 * q)

    if DEVEL_PLOTS:
        # Create subplot figure
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "(a) Original curves",
                "(b) Transformed curves",
                "(c) Slope coefficient",
            ],
        )

        # -----------------------------------------------------------
        # (a) Original inverse curves
        # -----------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=df_sup_price["Volume"],
                y=df_sup_price["Price"],
                mode="lines",
                name="SUP_WS",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_dem_price["Volume"],
                y=df_dem_price["Price"],
                mode="lines",
                name="DEM_WS",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[min(df_sup_price["Volume"]), max(df_sup_price["Volume"])],
                y=[P_auction, P_auction],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[q0],
                y=[P_auction],
                mode="markers",
                marker=dict(color="black", size=8),
                name="Auction clearing point",
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title="Volume [MWh]", row=1, col=1)
        fig.update_yaxes(title="Price [€]", row=1, col=1)

        # -----------------------------------------------------------
        # (b) Transformed supply + inelastic demand
        # -----------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=V_sorted,
                y=P_sorted,
                mode="lines",
                name="Transformed supply",
                line=dict(color="blue"),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[DEM_inelastic, DEM_inelastic],
                y=[min(P_sorted), max(P_sorted)],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="DEM_inelastic",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[min(V_sorted), max(V_sorted)],
                y=[P_auction, P_auction],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title="Volume [MWh]", row=1, col=2)
        fig.update_yaxes(title="Price [€]", row=1, col=2)

        # -----------------------------------------------------------
        # (c) Slope segment around q0
        # -----------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=V_sorted,
                y=P_sorted,
                mode="lines",
                name="Transformed supply",
                line=dict(color="blue"),
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[q_implied - q, q_implied + q],
                y=[P_minus, P_plus],
                mode="lines",
                name=f"Slope segment Δ={q}",
                line=dict(color="green"),
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[q_implied - q, q_implied, q_implied + q],
                y=[P_minus, P_clearing, P_plus],
                mode="markers",
                marker=dict(color="black", size=8),
                name="Slope points",
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[min(V_sorted), max(V_sorted)],
                y=[P_auction, P_auction],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[q_implied, q_implied],
                y=[min(P_sorted), max(P_sorted)],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[DEM_inelastic, DEM_inelastic],
                y=[min(P_sorted), max(P_sorted)],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="DEM_inelastic",
            ),
            row=1,
            col=3,
        )

        fig.update_xaxes(title="Volume [MWh]", row=1, col=3)
        fig.update_yaxes(title="Price [€]", row=1, col=3)

        # -----------------------------------------------------------
        # Layout
        # -----------------------------------------------------------
        fig.update_layout(showlegend=True, title="Supply/Demand Curve Transformations")

        fig.show()

    return elasticity


forecasting_horizon = 31  # 31 5min intervals before the delivery

first_training_date = required_start
last_forecasting_date = required_end - timedelta(days=1)

for delivery_time in tqdm(range(96)):
    last_trade_time = (
        delivery_time * 3 + 8 * 12 - 6
    )  # 8*12 is 8 hours each containing 12 5min periods, -6 as we are trading up to 30min before the delivery

    information_shift = forecasting_horizon + 1

    first_trade_time = last_trade_time - information_shift

    con = sqlite3.connect(MARKET_DATA_DIR)
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {last_trade_time} AND Time >= '2018-12-31 16:00:00' AND Day >= 61;"  # load only the data required for simu, so up to trade time
    daily_data = pd.read_sql(sql_str, con)[
        [str(i) for i in range(192)] + ["288"]
    ].to_numpy()  # column 288 contains weekday no. indicators
    daily_data = np.reshape(
        daily_data, (np.shape(daily_data)[0] // last_trade_time, last_trade_time, 193)
    )

    last_known_id_prices = daily_data[:, -information_shift, delivery_time]

    # for each day required in the simulation extract outages known before 12:00 d-1 and outages reported between 12:00 d-1 and time of forecasting
    elasticities_per_date = []
    last_known_prices = []
    actual_datetimes = []

    for forecasting_date_idx, forecasting_date in enumerate(
        pd.date_range(first_training_date, last_forecasting_date)
    ):
        naive_datetime = (forecasting_date - timedelta(days=1)).replace(
            hour=16
        ) + timedelta(minutes=5 * first_trade_time)
        actual_datetimes.append(naive_datetime)

        last_known_id_price = last_known_id_prices[forecasting_date_idx]
        last_known_prices.append(last_known_id_price)

        df = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "Intraday_Auction",
                "Aggregated curves",
                f"{forecasting_date.year}",
                f"intraday_auction_aggregated_curves_15-call_germany_{forecasting_date.strftime('%Y%m%d')}.csv",
            ),
            skiprows=1,
            dtype={"Hour": str},
        )

        # in 2019 the volume was not scaled to 15min, but rather reported as 1h value - we correct that, so that it is coherent with 2020 values of volume
        if forecasting_date.year == 2019:
            df["Volume"] = df["Volume"] / 4

        # handle the duplicated hours by dropping the 2nd one
        if "3A" in df["Hour"].values:
            df.loc[df.Hour == "3A", "Hour"] = "3"
            df = df[df.Hour != "3B"]

        # change the dtype of Hour column
        df["Hour"] = df["Hour"].astype(int)

        # extract demanded delivery
        df = df[
            (df["Hour"] == delivery_time // 4 + 1)
            & (df["Quarter hour"] == delivery_time % 4 + 1)
        ]

        # load the intraday auction clearing price
        intraday_auction_prices = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "ID_auction_preprocessed",
                "ID_auction_price_2018-2020_preproc.csv",
            ),
            index_col=0,
        )
        intraday_auction_prices.index = pd.to_datetime(intraday_auction_prices.index)
        index_hour = int(delivery_time * 0.25)
        index_minute = int((delivery_time * 0.25 - index_hour) * 4 * 15)
        ID_price = intraday_auction_prices[
            (intraday_auction_prices.index.hour == index_hour)
            & (intraday_auction_prices.index.minute == index_minute)
            & (intraday_auction_prices.index.date == forecasting_date.date())
        ]["price"].values[0]

        all_volume_shifts = []
        for volume_delta in [
            500,
            1000,
            2000,
        ]:  # volume deltas to calculate the slope between
            all_volume_shifts.append(
                compute_transformed_supply_elasticity(
                    df,
                    last_known_id_price,
                    ID_price,
                    volume_delta,
                    forecasting_date,
                    P_min=-3000,
                )
            )

        elasticities_per_date.append(all_volume_shifts)

    df_elasticities_per_date = pd.DataFrame(elasticities_per_date)
    df_elasticities_per_date.index = actual_datetimes
    df_elasticities_per_date["naive_used"] = last_known_prices
    df_elasticities_per_date = fill_march_dst_daily(
        df_elasticities_per_date, 0
    )  # handle the missing data from March dst change
    print(f"Calculated the elasticities for {delivery_time}.")
    check_for_missing_data(
        df_elasticities_per_date,
        (min(df_elasticities_per_date.index) + timedelta(days=1)).normalize(),
        (max(df_elasticities_per_date.index) + timedelta(days=1)).normalize(),
        freq="1d",  # we use max of index here as we sometimes have elasticities for first deliveries of the day where naive is taken from prev. day
    )
    df_elasticities_per_date.to_csv(
        os.path.join(DATA_DIR, "Elasticities", f"{delivery_time}_elasticities.csv")
    )
