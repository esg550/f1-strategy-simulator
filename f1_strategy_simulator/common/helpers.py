import fastf1
from fastf1.core import Session

import numpy as np
import scipy.stats as stats
import pandas as pd
import logging

from f1_strategy_simulator.cleaner import clean_one_race
import f1_strategy_simulator.common.names as n
import f1_strategy_simulator.common.constants as c


def calculate_tyre_degradation(
    driver: str, race: str, year: int, compound: str
) -> float:
    """
    Calculate the tyre degradation for a given driver, race, year, and
    compound. The function will use the latest race without rain before
    the specified year to perform the analysis.
    Args:
        driver (str): Driver code (e.g., 'HAM' for Lewis Hamilton)
        race (str): Race name (e.g., 'Monaco')
        year (int): Year of the simulated tyre degradation
        compound (str): Tyre compound to analyze (e.g., 'MEDIUM')
    Returns:
        float: Average tyre degradation in seconds per lap
    """
    cleaned_session = get_last_race_without_rain(
        race=race, year=year - 1, driver=driver
    )

    compound_laps = _filter_valid_laps(cleaned_session, compound)

    # group by stints
    stints = compound_laps.groupby("stint")

    slopes = [
        stats.linregress(
            stint_data["lap_number"].astype(float),
            stint_data["lap_time_approx_s"].apply(lambda x: x.total_seconds()),
        ).slope
        for _, stint_data in stints
        if len(stint_data) >= 3
    ]

    if not slopes:
        logging.warning(
            f"No valid stints for driver={driver}, race={race}, year={year}, "
            f"compound={compound}"
        )
        return np.nan

    return np.mean(slopes) + c.FUEL_CORRECTION_SECONDS


def _filter_valid_laps(df: pd.DataFrame, compound: str) -> pd.DataFrame:
    """Filters laps for a given compound, excluding pit laps,
    first lap, and race interruptions."""
    mask = (
        (df["compound"] == compound)
        & (~df["pit_stop_in_lap"])
        & (~df["pit_stop_out_lap"])
        & (df["lap_number"] != 1)
        & (df["RED_FLAG"] == 0)
        & (df["YELLOW"] == 0)
        & (df["SAFETY_CAR"] == 0)
        & (df["VIRTUAL_SAFETY_CAR"] == 0)
        & (df["VSC_ENDING"] == 0)
    )
    return df[mask].copy()


def get_last_race_without_rain(race: str, year: int, driver: str) -> Session:
    """
    Finds the last requested race before the given year that was not
    affected by rain and returns cleaned race data for the specified driver.
    Args:
        race (str): Race name (e.g. 'Monaco')
        year (int): Year to start the search from
        driver (str): Driver code (e.g., 'HAM' for Lewis Hamilton)
    Returns:
        Session: Cleaned race data for the specified driver
    """
    wet_compounds = ["INTERMEDIATE", "WET"]
    max_year_limit = 1950  # Formula 1 started in 1950
    while year >= max_year_limit:
        session_previous_year = fastf1.get_session(year, race, n.RACE)
        session_previous_year.load()
        driver_laps_previous_year = session_previous_year.laps
        if driver_laps_previous_year["Compound"].isin(wet_compounds).any():
            year -= 1
        else:
            cleaned_session = clean_one_race.clean_race_data(
                session=session_previous_year, driver=driver
            ).reset_index(drop=True)
            return cleaned_session

    raise ValueError(
        f"No suitable race found without rain for race={race}, "
        f"driver={driver} starting from year={year}."
    )


def calculate_avg_lap_time(race: str, year: int, compound: str, driver: str) -> float:
    """
    Calculate the average lap time for a given driver, race, year, and compound.
    Args:
        race (str): Race name (e.g., 'Monaco')
        year (int): Year of the simulated average lap time
        compound (str): Tyre compound to analyze (e.g., 'HARD', 'MEDIUM', 'SOFT')
        driver (str): Driver code (e.g., 'HAM' for Lewis Hamilton)
    Returns:
        float: Average lap time in seconds for the specified compound and driver
    """
    cleaned_session = get_last_race_without_rain(
        race=race, year=year - 1, driver=driver
    )

    stint_laps = cleaned_session[cleaned_session["compound"] == compound]
    avg_lap_time = (
        stint_laps["lap_time_approx_s"].apply(lambda x: x.total_seconds()).mean()
    )
    return avg_lap_time


# Enum helpers
def enum_from_race_name(enum_cls, race: str):
    try:
        return enum_cls[race.replace(" ", "_").upper()].value
    except KeyError:
        raise ValueError(f"Race not found: {race}")
