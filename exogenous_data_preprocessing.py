import os
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.paths import DATA_DIR

DEVEL_PLOTS = False
required_start = datetime(
    2019, 1, 1
)  # we cannot go further back due to the lack of Load forecasts in several days before 2019-01-02
required_end = datetime(2021, 1, 1)

border_ctys = [
    "DE",
    "AT",
    "BE",
    "CZ",
    "CH",
    "DK1",
    "DK2",
    "FR",
    "NL",
    "NO2",
    "PL",
    "SE4",
]  # DE and all the borders of DE


def fill_march_dst(df, col):
    """get the dst gaps"""
    dst_gaps = df[(df.index >= required_start) & df[col].isna()].index
    print(f"DST GAPS: {dst_gaps} {col}")

    # Fill missing values using average of Hour 2 and Hour 3 (before and after gap)
    for ts in dst_gaps:
        before = ts - pd.Timedelta(hours=1)
        after = ts + pd.Timedelta(hours=1)
        if before in df.index and after in df.index:
            df = df.copy(deep=True)
            df.loc[ts] = (df.loc[before] + df.loc[after]) / 2

    return df


def fill_march_dst_daily(df, col):
    """get the dst gaps in daily granularity series (corresponding to one specific delivery from each day)"""
    dst_gaps = df[(df.index >= required_start) & df[col].isna()].index
    print(f"DST GAPS: {dst_gaps} {col}")

    # Fill missing values using average of Hour 2 and Hour 3 (before and after gap)
    for ts in dst_gaps:
        before = ts - pd.Timedelta(days=1)
        after = ts + pd.Timedelta(days=1)
        if before in df.index and after in df.index:
            df = df.copy(deep=True)
            df.loc[ts] = (df.loc[before] + df.loc[after]) / 2

    return df


def check_for_missing_data(df, required_start, required_end, freq):
    matching_len = len(df[df.index >= required_start]) == len(
        pd.date_range(required_start, required_end, inclusive="left", freq=freq)
    )
    any_nans = df[df.index >= required_start].isnull().values.any()
    if not matching_len or any_nans:
        raise ValueError("Missing data found!")


###################################################################################
print("Processing the hourly border and DE d-a price data...")

all_ctys_dfs = []

for cty in border_ctys:
    da_price = []
    for year in range(2018, 2021):
        df = pd.read_csv(
            os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", f"DA_{cty}_{year}.csv"),
            na_values=["n/e"],
            index_col=0,
        )
        da_column = [c for c in df.columns if "Day-ahead" in c][0]
        if df.index.name != "MTU (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        df.rename(columns={da_column: cty}, inplace=True)
        df = df[[cty]]

        new_datetimes = []
        for dat in df.index:
            new_datetimes.append(dat.split(" - ")[0])

        try:
            df.index = pd.to_datetime(
                new_datetimes,
                format="%d/%m/%Y %H:%M:%S",
                dayfirst=False,
                yearfirst=False,
            )
        except Exception as err:
            print(
                f"Exception: {err}. Failed to format the index as datetime using %d/%m/%Y %H:%M:%S;\ntrying %d.%m.%Y %H:%M instead."
            )
            df.index = pd.to_datetime(
                new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
            )

        da_price.append(df[[cty]])

    da_price_all_years = pd.concat(da_price, axis=0)
    da_price_all_years_dst_corrected = fill_march_dst(
        da_price_all_years[~da_price_all_years.index.duplicated()], col=cty
    )

    # for PL we need to change the currency to EUR for data till 2019.11.19 (inclusively)
    if cty == "PL":
        pln_eur = pd.read_csv(
            os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", "PLN_EUR_2018_2019.csv"), index_col=0
        )["Price"]
        pln_eur.index = pd.to_datetime(pln_eur.index, format="%d-%m-%Y")
        pln_eur_resampled = pln_eur.sort_index().resample("1h").ffill()
        da_price_all_years_dst_corrected.loc[
            da_price_all_years_dst_corrected.index < datetime(2019, 11, 20), "PL"
        ] = (
            da_price_all_years_dst_corrected.loc[
                da_price_all_years_dst_corrected.index < datetime(2019, 11, 20), "PL"
            ]
            * pln_eur_resampled.loc[pln_eur_resampled.index < datetime(2019, 11, 20)]
        )

    if cty != "DE":
        da_price_all_years_dst_corrected[f"DE-{cty}"] = (
            da_price_all_years_dst_corrected[cty] - all_ctys_dfs[0]["DE"]
        )

    all_ctys_dfs.append(da_price_all_years_dst_corrected)

all_de_border_prices = pd.concat(all_ctys_dfs, axis=1)

all_de_border_prices.to_csv(
    os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", "DE_and_all_DE_borders_hourly_day_ahead_prices.csv")
)

check_for_missing_data(all_de_border_prices, required_start, required_end, freq="1h")

if (
    len(all_de_border_prices[all_de_border_prices.index >= required_start])
    != len(pd.date_range(required_start, required_end, inclusive="left", freq="1h"))
    or all_de_border_prices[all_de_border_prices.index >= required_start]
    .isnull()
    .values.any()
):
    raise

###################################################################################
print("Processing the q-hrly DE day-ahead prices...")

cty = "DE"
da_price = []
for year in range(2018, 2021):
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", f"DA_{cty}_Q_{year}.csv"),
        na_values=["n/e"],
        index_col=0,
    )
    da_column = [c for c in df.columns if "Day-ahead" in c][0]
    if df.index.name != "MTU (CET/CEST)":
        raise ValueError(f"Unexpected index name {df.index.name}")
    df.rename(columns={da_column: cty}, inplace=True)
    df = df[[cty]]

    new_datetimes = []
    for dat in df.index:
        new_datetimes.append(dat.split(" - ")[0])

    try:
        df.index = pd.to_datetime(
            new_datetimes, format="%d/%m/%Y %H:%M:%S", dayfirst=False, yearfirst=False
        )
    except Exception as err:
        print(
            f"Exception: {err}. Failed to format the index as datetime using %d/%m/%Y %H:%M:%S;\ntrying %d.%m.%Y %H:%M instead."
        )
        df.index = pd.to_datetime(
            new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
        )

    da_price.append(df[[cty]])

da_price_all_years = pd.concat(da_price, axis=0)
da_price_all_years_dst_corrected = fill_march_dst(
    da_price_all_years[~da_price_all_years.index.duplicated()], col=cty
)

da_price_all_years_dst_corrected.to_csv(
    os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", f"{cty}_quarterhourly_day_ahead_prices.csv")
)

check_for_missing_data(
    da_price_all_years_dst_corrected, required_start, required_end, freq="15min"
)

###################################################################################
print("Processing the intraday auction data...")
total_df = pd.concat(
    [
        pd.read_csv(
            "Data/ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2018.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
        pd.read_csv(
            "Data/ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2019.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
        pd.read_csv(
            "Data/ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2020.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
    ]
).iloc[:, :-8]
# drop the B set of DST deliveries
total_df = total_df[[c for c in total_df.columns if "Hour 3B" not in c]]
new_df = []
for idx in total_df.index:
    new_df.append(
        pd.DataFrame(
            columns=["price"],
            index=pd.date_range(idx, idx + timedelta(minutes=95 * 15), freq="15min"),
            data=total_df.loc[idx].to_numpy(),
        )
    )
df = fill_march_dst(pd.concat(new_df).sort_index(), col="price")
df.to_csv("Data/ID_auction_preprocessed/ID_auction_price_2018-2020_preproc.csv")

check_for_missing_data(df, required_start, required_end, freq="15min")

###################################################################################
print("Processing the hourly physical exchange data...")

all_ctys_exchange = []
for cty in border_ctys:
    if cty == "DE":
        continue

    exchange = []
    for year in range(2018, 2021):
        df = pd.read_csv(
            os.path.join(DATA_DIR, "Crossborder", f"crossborder_de_{cty.lower()}_{year}.csv"),
            na_values=["n/e"],
            index_col=0,
        )

        if df.index.name != "Time (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        de_export_column = [c for c in df.columns if "BZN|DE-LU > " in c][0]
        de_import_column = [c for c in df.columns if "> BZN|DE-LU" in c][0]
        df.rename(
            columns={
                de_export_column: f"{cty}_import",
                de_import_column: f"{cty}_export",
            },
            inplace=True,
        )

        df = (df[f"{cty}_import"] - df[f"{cty}_export"]).to_frame(name=cty)

        new_datetimes = []
        for dat in df.index:
            new_datetimes.append(dat.split(" - ")[0])

        try:
            df.index = pd.to_datetime(
                new_datetimes,
                format="%d/%m/%Y %H:%M:%S",
                dayfirst=False,
                yearfirst=False,
            )
        except Exception as err:
            print(
                f"Exception: {err}. Failed to format the index as datetime using %d/%m/%Y %H:%M:%S;\ntrying %d.%m.%Y %H:%M instead."
            )
            df.index = pd.to_datetime(
                new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
            )

        exchange.append(df)

    exchange_all_years = pd.concat(exchange, axis=0)
    exchange_all_years_dst_corrected = fill_march_dst(
        exchange_all_years[~exchange_all_years.index.duplicated()], col=cty
    )

    exchange_all_years_dst_corrected_resampled = (
        exchange_all_years_dst_corrected.resample("1h").mean()
    )

    all_ctys_exchange.append(exchange_all_years_dst_corrected_resampled)

all_de_border_exchanges = pd.concat(all_ctys_exchange, axis=1)

all_de_border_exchanges = all_de_border_exchanges.drop(columns="NO2")

all_de_border_exchanges.to_csv(os.path.join(DATA_DIR, "Crossborder", "DE_physexc.csv"))

check_for_missing_data(all_de_border_exchanges, required_start, required_end, freq="1h")

###################################################################################
print("Processing the hourly commercial scheduled exchange data...")

all_ctys_exchange = []
for cty in border_ctys:
    if cty in ["DE", "BE", "NO2"]:
        continue

    exchange = []
    for year in range(2018, 2021):
        df = pd.read_csv(
            os.path.join(DATA_DIR, "Crossborder", f"commex_de_{cty.lower()}_{year}.csv"),
            na_values=["n/e"],
            index_col=0,
        )

        if df.index.name != "Time (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        de_export_column = [c for c in df.columns if "BZN|DE-LU > " in c][0]
        de_import_column = [c for c in df.columns if "> BZN|DE-LU" in c][0]
        df.rename(
            columns={
                de_export_column: f"{cty}_import",
                de_import_column: f"{cty}_export",
            },
            inplace=True,
        )

        df = (df[f"{cty}_import"] - df[f"{cty}_export"]).to_frame(name=cty)

        new_datetimes = []
        for dat in df.index:
            new_datetimes.append(dat.split(" - ")[0])

        try:
            df.index = pd.to_datetime(
                new_datetimes,
                format="%d/%m/%Y %H:%M:%S",
                dayfirst=False,
                yearfirst=False,
            )
        except:
            df.index = pd.to_datetime(
                new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
            )

        exchange.append(df)

    exchange_all_years = pd.concat(exchange, axis=0)
    exchange_all_years_dst_corrected = fill_march_dst(
        exchange_all_years[~exchange_all_years.index.duplicated()], col=cty
    )

    exchange_all_years_dst_corrected_resampled = (
        exchange_all_years_dst_corrected.resample("1h").mean()
    )

    all_ctys_exchange.append(exchange_all_years_dst_corrected_resampled)

all_de_border_exchanges = pd.concat(all_ctys_exchange, axis=1)

all_de_border_exchanges.to_csv(os.path.join(DATA_DIR, "Crossborder", "DE_commex.csv"))

check_for_missing_data(all_de_border_exchanges, required_start, required_end, freq="1h")

###################################################################################
print("Processing the load data...")
load = []
for year in range(2018, 2021):
    load.append(
        pd.read_csv(
            os.path.join(DATA_DIR, "Load", f"Total Load - Day Ahead _ Actual_{year}01010000-{year + 1}01010000.csv"),
            na_values=["n/e"],
        )
    )

load = pd.concat(load, ignore_index=True)

new_datetimes = []
for dat in load["Time (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

load_df = pd.DataFrame()
load_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
load_df["Actual"] = load["Actual Total Load [MW] - BZN|DE-LU"]
load_df["Forecast"] = load["Day-ahead Total Load Forecast [MW] - BZN|DE-LU"]
load_df.index = pd.to_datetime(load_df["Time from"])
load_df.drop(columns="Time from", inplace=True)

# remove October dst change impact
load_df = fill_march_dst(load_df[~load_df.index.duplicated()], col="Actual")

load_df.to_csv("Data/Load/Load_2018-2020.csv")

check_for_missing_data(load_df, required_start, required_end, freq="15min")

###################################################################################
print("Processing the generation data...")
gen = []
for year in range(2018, 2021):
    gen.append(pd.read_csv(os.path.join(DATA_DIR, "Generation", f"generation_{year}.csv"), na_values=["n/e"]))

gen = pd.concat(gen, ignore_index=True)[
    [
        "MTU",
        "Solar  - Actual Aggregated [MW]",
        "Wind Offshore  - Actual Aggregated [MW]",
        "Wind Onshore  - Actual Aggregated [MW]",
    ]
]

new_datetimes = []
for dat in gen["MTU"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_df = pd.DataFrame()
gen_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_df["SPV"] = gen["Solar  - Actual Aggregated [MW]"]
gen_df["W"] = (
    gen["Wind Offshore  - Actual Aggregated [MW]"]
    + gen["Wind Onshore  - Actual Aggregated [MW]"]
)
gen_df.index = pd.to_datetime(gen_df["Time from"])
gen_df.drop(columns="Time from", inplace=True)

gen_fore = []
for year in range(2018, 2021):
    gen_fore.append(
        pd.read_csv(os.path.join(DATA_DIR, "Generation", f"generation_fore_{year}.csv"), na_values=["n/e"])
    )

gen_fore = pd.concat(gen_fore, ignore_index=True)[
    [
        "MTU (CET/CEST)",
        "Generation - Solar  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Solar [MW] Intraday / BZN|DE-LU",
        "Generation - Wind Offshore [MW] Intraday / BZN|DE-LU",
        "Generation - Wind Onshore [MW] Intraday / BZN|DE-LU",
    ]
]

new_datetimes = []
for dat in gen_fore["MTU (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_fore_df = pd.DataFrame()
gen_fore_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_fore_df["SPV DA"] = gen_fore["Generation - Solar  [MW] Day Ahead/ BZN|DE-LU"]
gen_fore_df["W DA"] = (
    gen_fore["Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU"]
    + gen_fore["Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU"]
)
gen_fore_df["SPV ID"] = gen_fore["Generation - Solar [MW] Intraday / BZN|DE-LU"]
gen_fore_df["W ID"] = (
    gen_fore["Generation - Wind Offshore [MW] Intraday / BZN|DE-LU"]
    + gen_fore["Generation - Wind Onshore [MW] Intraday / BZN|DE-LU"]
)
gen_fore_df.index = pd.to_datetime(gen_fore_df["Time from"])
gen_fore_df.drop(columns="Time from", inplace=True)

gen_df = pd.concat([gen_df, gen_fore_df], axis=1)

gen_df = fill_march_dst(gen_df[~gen_df.index.duplicated()], col="SPV")

check_for_missing_data(gen_df, required_start, required_end, freq="15min")

gen_df.to_csv("Data/Generation/Generation_2018-2020.csv")

###################################################################################
print("Preparing the market elasticity...")


def compute_transformed_supply_elasticity(
    df_curve,
    P_clearing,
    P_auction,
    q,
    forecasting_date,
    P_min=-3000,  # TODO confirm it is -3000
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

first_training_date = datetime(year=2019, month=1, day=1)
last_forecasting_date = datetime(year=2020, month=12, day=31)

for delivery_time in tqdm(range(96)):
    last_trade_time = (
        delivery_time * 3 + 8 * 12 - 6
    )  # 8*12 is 8 hours each containing 12 5min periods, -6 as we are trading up to 30min before the delivery

    information_shift = forecasting_horizon + 1

    first_trade_time = last_trade_time - information_shift

    con = sqlite3.connect(os.path.join(DATA_DIR, "preprocessed_continuous_intraday_prices_and_volume.db"))
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
            os.path.join(DATA_DIR, "Intraday_Auction", "Aggregated curves", f"{forecasting_date.year}", f"intraday_auction_aggregated_curves_15-call_germany_{forecasting_date.strftime('%Y%m%d')}.csv"),
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
            os.path.join(DATA_DIR, "ID_auction_preprocessed", "ID_auction_price_2018-2020_preproc.csv"),
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
