from typing import Protocol, Tuple, Optional
import numpy as np

class Trajectory(Protocol):
    """Time-indexed sequences with interpolation & time-shift."""
    def T(self) -> int: ...
    def dt(self) -> float: ...
    def states(self) -> np.ndarray: ...      # (T+1, n_x)
    def inputs(self) -> np.ndarray: ...      # (T,   n_u)
    def at(self, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...  # (x_k, u_k)
    def shift(self) -> "Trajectory": ...
