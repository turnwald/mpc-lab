from typing import Protocol, Optional, Dict, Any, Tuple
import numpy as np

class SystemModel(Protocol):
    def f_disc(self, x: np.ndarray, u: np.ndarray, p: Optional[Dict[str, Any]] = None) -> np.ndarray: ...

class Constraint(Protocol):
    def g(self, x: np.ndarray, u: np.ndarray, k: int | None = None, p: Optional[Dict[str, Any]] = None) -> np.ndarray: ...

class StageCost(Protocol):
    def l(self, x: np.ndarray, u: np.ndarray, r: Optional[Dict[str, np.ndarray]] = None) -> float: ...

class TerminalCost(Protocol):
    def m(self, x: np.ndarray, r: Optional[Dict[str, np.ndarray]] = None) -> float: ...

class OCPProblem(dict):
    ...

class NLPBuilder(Protocol):
    def build(self,
              model: SystemModel,
              N: int,
              costs: tuple[StageCost, TerminalCost],
              constraints: tuple[Constraint, ...],
              x0_param: bool = True,
              soften: Optional[Dict[str, float]] = None,
              ) -> OCPProblem: ...
