from enum import Enum


class NoiseLevel(str, Enum):
    LOW = "gnw_low_noise.txt"
    MEDIUM = "gnw_medium_noise.txt"
    HIGH = "gnw_high_noise.txt"
