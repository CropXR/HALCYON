from enum import Enum


class NoiseLevel(str, Enum):
    NONOISE = "_nonoise_"
    NOEXPNOISE = "_noexpnoise_"
    ALLNOISE = "_"
