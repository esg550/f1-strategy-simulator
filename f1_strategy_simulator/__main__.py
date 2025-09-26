# Refactor this code to make it cleaner, remove redundancy, and follow best practices

import fastf1
import f1_strategy_simulator.common.helpers as helpers
import f1_strategy_simulator.common.names as n
from f1_strategy_simulator.common.enums import NumberOfLaps, PitStopTimeLoss
from f1_strategy_simulator.cleaner.clean_one_race import clean_race_data

def simulate_race(driver: str, race: str, year: int, strategies: list[dict]) -> str:
    """


    Args:
        driver (str): Driver code (e.g., 'HAM' for Lewis Hamilton)
        race (str): Country name of race (e.g., 'Monaco')
        year (int): Year of the simulated tyre degradation
        strategies (list(dict)): List of strategies to analyze. Each strategy is a dict with keys:
            - 'stop_laps': List of laps where pit stops occur
            - 'compounds': List of tyre compounds used in the strategy (e.g., ['SOFT', 'HARD'])
    """
    session = fastf1.get_session(year=year, gp=race, identifier=n.RACE)
    session.load()

    stints = len(strategies[0]['stop_laps']) + 1
    number_of_laps = NumberOfLaps.from_name(race)
    pit_stop_time_loss = PitStopTimeLoss.from_name(race)

    # clean the race data
    cleaned_race = clean_race_data(session=session, driver=driver)

    # calculate tyre degradation for each compound
    tyre_degs, avg_lap_times = calculate_tyre_degradation_and_avg_lap_times(driver, race, year, strategies)

    total_race_times = {}
    for strategy in strategies:
        total_race_time = 0.0

        stop_laps = strategy["stop_laps"]
        compounds = strategy["compounds"]

        # build stint lengths
        stint_lengths = []
        prev = 0
        for stop in stop_laps:
            stint_lengths.append(stop - prev)
            prev = stop
        stint_lengths.append(number_of_laps - prev)  # final stint to end

        # now multiply avg lap times by stint lengths
        for compound, stint_len in zip(compounds, stint_lengths):
            current_lap_time = avg_lap_times[strategy["name"]][compound]
            for lap in range(stint_len):
                total_race_time += current_lap_time
                next_lap_time = current_lap_time + tyre_degs[strategy["name"]][compound]
                current_lap_time = next_lap_time

        total_race_time += pit_stop_time_loss * (stints - 1)
        total_race_times[strategy["name"]] = total_race_time

    return total_race_times

def calculate_tyre_degradation_and_avg_lap_times(driver, race, year, strategies):
    tyre_degs = {}
    avg_lap_times = {}
    compounds_cache = {}

    for strategy in strategies:
        strategy_name = strategy['name']
        tyre_degs[strategy_name] = {}
        avg_lap_times[strategy_name] = {}

        for compound in set(strategy['compounds']):
            if compound not in compounds_cache:
                compounds_cache[compound] = {
                    "tyre_deg": helpers.calculate_tyre_degradation(driver=driver, race=race, year=year, compound=compound),
                    "avg_lap_time": helpers.calculate_avg_lap_time(race=race, year=year, compound=compound, driver=driver)
                }

            tyre_degs[strategy_name][compound] = compounds_cache[compound]["tyre_deg"]
            avg_lap_times[strategy_name][compound] = compounds_cache[compound]["avg_lap_time"]

    return tyre_degs, avg_lap_times


if __name__ == "__main__":
    simulate_race(driver="VER", race="Netherlands", year=2024, strategies=[{"name": "two-stop", "stop_laps": [15, 30], "compounds": ["MEDIUM", "HARD", "HARD"]},
                                                                           {"name": "one-late-stop", "stop_laps": [40], "compounds": ["HARD", "MEDIUM"]},
                                                                           {"name": "one-early-stop", "stop_laps": [20], "compounds": ["MEDIUM", "HARD"]}])