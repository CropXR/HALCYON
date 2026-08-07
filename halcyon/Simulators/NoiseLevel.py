from enum import Enum


class NoiseLevel(str, Enum):
    """Class to keep track of noise levels in data. For each noise level points to correct settings file."""

    LOW = "gnw_low_noise.txt"
    MEDIUM = "gnw_medium_noise.txt"
    HIGH = "gnw_high_noise.txt"
