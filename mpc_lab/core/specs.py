from typing import Tuple
import numpy as np

class SpaceSpec(dict):
    """Simple mapping-like spec: names, units, lower, upper."""
    names: Tuple[str, ...]
    units: Tuple[str, ...]
    lower: np.ndarray  # (n,)
    upper: np.ndarray  # (n,)
StateSpec = SpaceSpec
InputSpec = SpaceSpec
