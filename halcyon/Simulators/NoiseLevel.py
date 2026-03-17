from enum import Enum


class NoiseLevel(str, Enum):
    """Class to keep track of noise levels in data"""

    NONOISE = "_nonoise_"
    NOEXPNOISE = "_noexpnoise_"
    ALLNOISE = "_"
