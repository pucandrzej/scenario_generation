import pandas as pd
from config.test_calibration_validation import (
    required_start,
)


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
