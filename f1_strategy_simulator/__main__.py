import fastf1
import f1_strategy_simulator.common.helpers as helpers
import f1_strategy_simulator.common.names as n
from f1_strategy_simulator.common.enums import NumberOfLaps, PitStopTimeLoss
from dataclasses import dataclass


@dataclass
class Strategy:
    name: str
    stop_laps: list[int]
    compounds: list[str]


def simulate_race(
    driver: str, race: str, year: int, strategies: list[Strategy]
) -> dict[str, float]:
    """
    Simulates the race length in seconds for each given strategy
    based on tyre degradation and average lap times.

    Args:
        driver (str): Driver code (e.g., 'HAM' for Lewis Hamilton)
        race (str): Country name of race (e.g., 'Monaco')
        year (int): Year of the simulated tyre degradation
        strategies (list(Strategy)): List of strategies to analyze. Each strategy is a
            Strategy dataclass with the following fields:
            - name: Name of the strategy (e.g., 'two-stop')
            - stop_laps: List of laps where pit stops occur
            - compounds: List of tyre compounds used in the strategy
                 (e.g., ['SOFT', 'HARD'])
    """
    session = fastf1.get_session(year=year, gp=race, identifier=n.RACE)
    session.load()

    number_of_laps = NumberOfLaps.from_name(race)
    pit_stop_time_loss = PitStopTimeLoss.from_name(race)

    # calculate tyre degradation for each compound
    tyre_degs, avg_lap_times = calculate_tyre_degradation_and_avg_lap_times(
        driver, race, year, strategies
    )

    total_race_times = {}
    for strategy in strategies:
        total_race_time = 0.0

        stop_laps = strategy.stop_laps
        compounds = strategy.compounds

        # build stint lengths
        stint_lengths = [
            stop - prev for prev, stop in zip([0] + stop_laps[:-1], stop_laps)
        ]
        stint_lengths.append(
            number_of_laps - stop_laps[-1] if stop_laps else number_of_laps
        )

        # now multiply avg lap times by stint lengths
        for i, (compound, stint_len) in enumerate(zip(compounds, stint_lengths)):
            total_stint_time = calculate_total_stint_time(
                avg_lap_times[strategy.name][compound],
                tyre_deg=tyre_degs[strategy.name][compound],
                stint_len=stint_len,
            )

            total_race_time += total_stint_time
            # add pit stop time loss if not the final stint
            if i != len(stint_lengths) - 1:
                total_race_time += pit_stop_time_loss
        total_race_times[strategy.name] = total_race_time

    return total_race_times


def calculate_tyre_degradation_and_avg_lap_times(
    driver: str, race: str, year: int, strategies: list[Strategy]
) -> tuple[dict, dict]:
    tyre_degs: dict[str, dict[str, float]] = {}
    avg_lap_times: dict[str, dict[str, float]] = {}

    compounds_cache = {}

    for strategy in strategies:
        strategy_name = strategy.name
        tyre_degs[strategy_name] = {}
        avg_lap_times[strategy_name] = {}

        for compound in set(strategy.compounds):
            if compound not in compounds_cache:
                compounds_cache[compound] = {
                    n.TYRE_DEG: helpers.calculate_tyre_degradation(
                        driver=driver, race=race, year=year, compound=compound
                    ),
                    n.AVG_LAP_TIME: helpers.calculate_avg_lap_time(
                        race=race, year=year, compound=compound, driver=driver
                    ),
                }

            tyre_degs[strategy_name][compound] = compounds_cache[compound][n.TYRE_DEG]
            avg_lap_times[strategy_name][compound] = compounds_cache[compound][
                n.AVG_LAP_TIME
            ]

    return tyre_degs, avg_lap_times


def calculate_total_stint_time(
    avg_lap_time: float, tyre_deg: float, stint_len: int
) -> float:
    return stint_len * avg_lap_time + tyre_deg * (stint_len * (stint_len - 1) / 2)


if __name__ == "__main__":
    simulate_race(
        driver="VER",
        race="Netherlands",
        year=2024,
        strategies=[
            Strategy(
                name="two-stop",
                stop_laps=[15, 30],
                compounds=["MEDIUM", "HARD", "HARD"],
            ),
            Strategy(
                name="one-late-stop",
                stop_laps=[40],
                compounds=["HARD", "MEDIUM"],
            ),
            Strategy(
                name="one-early-stop",
                stop_laps=[20],
                compounds=["MEDIUM", "HARD"],
            ),
            Strategy(  # illegal, but for comparison
                name="no-stop",
                stop_laps=[],
                compounds=["HARD"],
            ),
        ],
    )
