import pandas as pd

from enum import Enum


class TrackStatus(Enum):
    GREEN = 1
    YELLOW = 2
    SAFETY_CAR = 4
    RED_FLAG = 5
    VIRTUAL_SAFETY_CAR = 6
    VSC_ENDING = 7

    @classmethod
    def expand_statuses(cls, series: pd.Series) -> pd.DataFrame:
        """Expand TrackStatus strings like '12' or '6' into binary columns."""
        # make sure we're working with strings
        as_str = series.astype(str).fillna("")

        result = pd.DataFrame(index=series.index)

        for status in cls:  # iterate over Enum members
            result[status.name] = as_str.apply(
                lambda code: str(status.value) in code
            ).astype(int)

        return result


class PitStopTimeLoss(Enum):
    """Estimated time loss for pit stops including entry and exit time loss."""

    BAHRAIN = 25.0
    SAUDI_ARABIA = 21.0
    AUSTRALIA = 18.0
    JAPAN = 23.0
    CHINA = 23.0
    MIAMI = 23.0
    EMILIA_ROMAGNA = 30.0
    MONACO = 24.0
    CANADA = 24.0
    SPAIN = 22.0
    AUSTRIA = 22.0
    GREAT_BRITAIN = 29.0
    HUNGARY = 22.0
    BELGIUM = 23.0
    NETHERLANDS = 21.0
    ITALY = 24.0
    AZERBAIJAN = 20.0
    SINGAPORE = 29.0
    UNITED_STATES = 24.0
    MEXICO = 23.0
    BRAZIL = 25.0
    LAS_VEGAS = 21.0
    QATAR = 23.0
    ABU_DHABI = 22.0

    @classmethod
    def from_name(cls, race: str) -> float:
        race_key = race.replace(" ", "_").upper()
        try:
            return cls[race_key].value
        except KeyError:
            raise ValueError(f"Race not found: {race}")


class NumberOfLaps(Enum):
    BAHRAIN = 57
    SAUDI_ARABIA = 50
    AUSTRALIA = 58
    JAPAN = 53
    CHINA = 56
    MIAMI = 57
    EMILIA_ROMAGNA = 63
    MONACO = 78
    CANADA = 70
    SPAIN = 66
    AUSTRIA = 71
    GREAT_BRITAIN = 52
    HUNGARY = 70
    BELGIUM = 44
    NETHERLANDS = 72
    ITALY = 53
    AZERBAIJAN = 51
    SINGAPORE = 61
    UNITED_STATES = 56
    MEXICO = 71
    BRAZIL = 71
    LAS_VEGAS = 50
    QATAR = 57
    ABU_DHABI = 58

    @classmethod
    def from_name(cls, race: str) -> int:
        race_key = race.replace(" ", "_").upper()
        try:
            return cls[race_key].value
        except KeyError:
            raise ValueError(f"Race not found: {race}")
