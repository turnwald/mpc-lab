from typing import Protocol, Optional, Dict, Any, Tuple, TypedDict
import numpy as np

class MPCInfo(TypedDict):
    obj: float
    status: str
    iters: int
    solve_ms: float
    violated: bool
    extras: Dict[str, Any]

class MPCBase(Protocol):
    def build(self) -> None: ...
    def prepare(self, ref_traj: Any | None = None) -> None: ...
    def compute_control(self,
                        x_now: np.ndarray,
                        t_now: float,
                        ref: Optional[Dict[str, np.ndarray]] = None,
                        warmstart: Optional[Dict[str, np.ndarray]] = None
                        ) -> tuple[np.ndarray, MPCInfo]: ...
