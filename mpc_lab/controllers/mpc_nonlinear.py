from typing import Protocol
import numpy as np

class SystemModel(Protocol):
    def f_disc(self, x: np.ndarray, u: np.ndarray, p: dict | None = None) -> np.ndarray: ...

class MPCNonlinear(Protocol):
    def set_model(self, model: SystemModel) -> None: ...
