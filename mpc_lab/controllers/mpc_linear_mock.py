import numpy as np
from typing import Protocol, Optional
from .base import MPCBase, MPCInfo
from ..constants import SolveStatus
from ..transcription.qp_builder_mock import MockQPBuilder, MockQPFacade

class LinearizationProvider(Protocol):
    def get(self, x_bar: np.ndarray, u_bar: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class DummyLinearizationProvider:
    def __init__(self, nx: int, nu: int, N: int):
        self.nx, self.nu, self.N = nx, nu, N
    def get(self, x_bar: np.ndarray, u_bar: np.ndarray, N: int):
        A = np.tile(np.eye(self.nx)[None,:,:], (N,1,1))
        B = np.tile(np.zeros((self.nx, self.nu))[None,:,:], (N,1,1))
        d = np.zeros((N, self.nx))
        return A, B, d

class MPCLinearMock(MPCBase):
    """Minimal runnable linear MPC mock using the mock QP pipeline."""
    def __init__(self, nx: int = 3, nu: int = 2, N: int = 10):
        self.nx, self.nu, self.N = nx, nu, N
        self.provider: Optional[LinearizationProvider] = None
        self.builder = MockQPBuilder()
        self.solver = MockQPFacade()
        self._built = False
    def set_linearizer(self, provider: LinearizationProvider) -> None:
        self.provider = provider
    def build(self) -> None:
        assert self.provider is not None, "Linearization provider not set"
        A,B,d = self.provider.get(np.zeros(self.nx), np.zeros(self.nu), self.N)
        Q = np.eye(self.nx); R = np.eye(self.nu); P = np.eye(self.nx)
        x_bounds = ( -1e3*np.ones(self.nx), 1e3*np.ones(self.nx) )
        u_bounds = ( -1e3*np.ones(self.nu), 1e3*np.ones(self.nu) )
        prob = self.builder.build(A,B,d,Q,R,P,x_bounds,u_bounds)
        self.solver.compile(prob)
        self._built = True
    def prepare(self, ref_traj=None) -> None:
        pass
    def compute_control(self, x_now: np.ndarray, t_now: float, ref=None, warmstart=None):
        assert self._built, "Call build() first"
        sol = self.solver.solve(params={})
        u_cmd = sol['u'][0] if sol['u'].shape[0] > 0 else np.zeros(self.nu)
        info: MPCInfo = dict(
            obj=float(sol.get('obj', 0.0)),
            status=str(sol.get('status', SolveStatus.SUCCESS)),
            iters=int(sol.get('iters', 0)),
            solve_ms=0.1,
            violated=False,
            extras={}
        )
        return u_cmd, info
