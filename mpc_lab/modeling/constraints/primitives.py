from typing import Protocol, Optional, Dict, Any, Tuple
import numpy as np

class Constraint(Protocol):
    name: str
    def g(self, x: np.ndarray, u: np.ndarray, k: int | None = None, p: Optional[Dict[str, Any]] = None) -> np.ndarray: ...
    def jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
