from typing import Protocol
import numpy as np

class LinearizationProvider(Protocol):
    def get(self, x_bar: np.ndarray, u_bar: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class MPCLinear(Protocol):
    def set_linearizer(self, provider: LinearizationProvider) -> None: ...
