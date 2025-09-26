import fastf1
from fastf1.core import Session

import numpy as np
import scipy.stats as stats
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

    # filter only the compound and remove safety car/VSC/red flag laps
    mask = (
        (cleaned_session["compound"] == compound)
        & (cleaned_session["RED_FLAG"] == 0)
        & (cleaned_session["YELLOW"] == 0)
        & (cleaned_session["SAFETY_CAR"] == 0)
        & (cleaned_session["VIRTUAL_SAFETY_CAR"] == 0)
        & (cleaned_session["VSC_ENDING"] == 0)
    )
    compound_laps = cleaned_session[mask].copy()

    # remove in- and out-laps explicitly
    compound_laps = compound_laps[
        (~compound_laps["pit_stop_in_lap"])
        & (~compound_laps["pit_stop_out_lap"])
        & (compound_laps["lap_number"] != 1.0)
    ]

    # group by stints
    stints = compound_laps.groupby("stint")

    stint_deg = {}
    for stint_number, stint_data in stints:
        if len(stint_data) < 3:
            # too few laps to fit a line
            continue
        slope, _, _, _, _ = stats.linregress(
            stint_data["lap_number"], stint_data["lap_time_approx_s"]
        )
        stint_deg[stint_number] = slope

    if len(stint_deg) == 0:
        logging.warning(
            f"""No valid stint data for driver={driver}, race={race},
              year={year}, compound={compound}"""
        )
        avg_deg = np.nan
    else:
        avg_deg = np.mean(list(stint_deg.values()))

    fuel_corrected_avg_deg = avg_deg + c.FUEL_CORRECTION_SECONDS

    return fuel_corrected_avg_deg


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
        f"""No suitable race found without rain for race={race},
          driver={driver} starting from year={year}."""
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
    avg_lap_time = stint_laps["lap_time_approx_s"].mean()
    return avg_lap_time
