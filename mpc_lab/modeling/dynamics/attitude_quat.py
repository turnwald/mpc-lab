from typing import Protocol, Optional, Dict, Any, Tuple
import numpy as np
from ...core.specs import StateSpec, InputSpec

class SystemModel(Protocol):
    state_spec: StateSpec
    input_spec: InputSpec
    dt: float
    def f_disc(self, x: np.ndarray, u: np.ndarray, p: Optional[Dict[str, Any]] = None) -> np.ndarray: ...
    def jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
    def f_cont(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...
