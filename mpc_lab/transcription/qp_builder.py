from typing import Protocol, Optional, Dict, Any, Tuple
import numpy as np

class OCPProblem(dict):
    """Container for CasADi graphs and metadata (builder output)."""
    ...

class QPBuilder(Protocol):
    def build(self,
              A_seq: np.ndarray, B_seq: np.ndarray, d_seq: np.ndarray,
              Q: np.ndarray, R: np.ndarray, P: np.ndarray,
              x_bounds: tuple[np.ndarray, np.ndarray],
              u_bounds: tuple[np.ndarray, np.ndarray],
              linear_ineq: Optional[Dict[str, np.ndarray]] = None,
              soften: Optional[Dict[str, float]] = None,
              ) -> OCPProblem: ...
