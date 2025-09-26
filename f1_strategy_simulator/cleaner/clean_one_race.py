# Refactor this code to make it cleaner, remove redundancy, and follow best practices

import pandas as pd
from fastf1.core import Session
from fastf1.core import Laps
from f1_strategy_simulator.common.enums import TrackStatus


def clean_race_data(session: Session, driver: str) -> pd.DataFrame:
    """
    Fetches and cleans FastF1 lap data for a given race and driver.

    Args:
        year: Race year
        event: Race name (e.g., 'Monaco')
        session_type: 'R' (race), 'Q' (qualifying), 'FP1', etc.
        driver: Optional; if None, fetch all drivers

    Returns:
        pd.DataFrame with cleaned lap data
    """

    if driver:
        laps = session.laps.pick_drivers(driver)
    else:
        laps = session.laps  # all drivers

    df = pd.DataFrame(
        {
            "driver": laps["Driver"],
            "lap_number": laps["LapNumber"],
            "lap_time_s": laps["LapTime"].dt.total_seconds(),
            "stint": laps["Stint"],
            "compound": laps["Compound"],
            "tyre_life": laps["TyreLife"],
            "position": laps["Position"],
            "pit_stop_in_lap": laps["PitInTime"].notna(),
            "pit_stop_out_lap": laps["PitOutTime"].notna(),
        }
    )

    # Expand TrackStatus into binary columns
    df = _expand_track_statuses(laps, df)

    # Approximate missing lap times using telemetry data
    df = _approximate_empty_lap_times(laps, df)

    return df


def _expand_track_statuses(laps: Laps, df: pd.DataFrame) -> pd.DataFrame:
    """Expand TrackStatus strings like '12' or '6' into binary columns."""
    # Expand TrackStatus into binary columns
    status_dummies = TrackStatus.expand_statuses(laps["TrackStatus"])

    # Concatenate side by side
    df = pd.concat([df, status_dummies], axis=1)

    return df


def _approximate_empty_lap_times(laps: Laps, df: pd.DataFrame) -> pd.DataFrame:
    """Approximate missing lap times using telemetry data."""

    laps = laps.copy()  # Create a copy to avoid SettingWithCopyWarning
    laps.loc[:, "LapTimeApprox"] = laps["LapTime"]

    # Replace NaN lap times with telemetry-based estimates
    for i, lap in laps.iterlaps():  # iterlaps() yields (index, Lap) pairs
        if pd.isna(lap.LapTime):
            telem = lap.get_telemetry()
            laps.loc[i, "LapTimeApprox"] = telem["Time"].max() - telem["Time"].min()

    df["lap_time_approx_s"] = laps["LapTimeApprox"].dt.total_seconds()

    return df
