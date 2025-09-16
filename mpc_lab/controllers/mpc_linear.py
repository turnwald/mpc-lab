import numpy as np
from typing import Optional, Tuple, Dict
from .base import MPCBase, MPCInfo
from ..transcription.qp_builder import LinearMPCQPBuilder
from ..solvers.qp_facade import QPSolverFacade

class LinearizationProvider:
    """Provides (A,B,d) sequences for LTV over the horizon."""
    def __init__(self, model, N: int):
        self.model = model; self.N = N
    def get(self, x_bar: np.ndarray, u_bar: np.ndarray):
        nx, nu, N = self.model.nx, self.model.nu, self.N
        A = np.zeros((N, nx, nx)); B = np.zeros((N, nx, nu)); d = np.zeros((N, nx))
        xk = x_bar.copy()
        for k in range(N):
            Ak, Bk = self.model.jacobians(xk, u_bar)
            fxu = self.model.f_disc(xk, u_bar)
            A[k], B[k] = Ak, Bk
            d[k] = fxu - Ak @ xk - Bk @ u_bar
            xk = fxu
        return A, B, d

class MPCLinear(MPCBase):
    def __init__(self, model, N: int, Q: np.ndarray, R: np.ndarray, P: Optional[np.ndarray] = None, solver: str = "qpoases"):
        self.model = model
        self.N = int(N)
        self.nx, self.nu = model.nx, model.nu
        self.Q = Q; self.R = R; self.P = (P if P is not None else Q)
        self.solver_name = solver
        self.builder = LinearMPCQPBuilder()
        self.prob = None
        self.solver = None
        self.xl = -1e3*np.ones(self.nx); self.xu = 1e3*np.ones(self.nx)
        self.ul = -1.0*np.ones(self.nu); self.uu =  1.0*np.ones(self.nu)
        self.xr = np.zeros((self.N+1, self.nx))
        self.ur = np.zeros((self.N, self.nu))
        self.linprov = LinearizationProvider(model, self.N)

    def set_bounds(self, x_bounds: Tuple[np.ndarray, np.ndarray], u_bounds: Tuple[np.ndarray, np.ndarray]):
        self.xl, self.xu = x_bounds; self.ul, self.uu = u_bounds

    def set_references(self, xr: np.ndarray, ur: np.ndarray):
        assert xr.shape == (self.N+1, self.nx)
        assert ur.shape == (self.N, self.nu)
        self.xr, self.ur = xr, ur

    def build(self):
        self.prob = self.builder.build(N=self.N, nx=self.nx, nu=self.nu, solver=self.solver_name)
        self.solver = QPSolverFacade(self.prob)

    def compute_control(self, x_now: np.ndarray, t_now: float, ref: Dict[str, np.ndarray] | None = None, warmstart: Dict[str, np.ndarray] | None = None):
        if ref is not None:
            if "xr" in ref: self.xr = ref["xr"]
            if "ur" in ref: self.ur = ref["ur"]
        A,B,d = self.linprov.get(x_now, np.zeros(self.nu))
        # flatten horizon stacks to match (N*nx, nx/nu) and (N*nx, 1)
        A_flat = np.vstack(A)                  # (N*nx, nx)
        B_flat = np.vstack(B)                  # (N*nx, nu)
        d_flat = np.vstack(d).reshape(-1, 1)   # (N*nx, 1)

        params = {
            "A": A_flat,
            "B": B_flat,
            "d": d_flat,
            "x0": x_now,
            "xr": self.xr,
            "ur": self.ur,
            "Q": self.Q, "R": self.R, "P": self.P,
            "xl": self.xl, "xu": self.xu, "ul": self.ul, "uu": self.uu
        }

        sol = self.solver.solve(params)
        u_cmd = sol["u"][0]
        info: MPCInfo = dict(obj=float(sol.get("obj",0.0)), status=sol.get("status","Solve_Success"),
                             iters=int(sol.get("iters",0)), solve_ms=0.0, violated=False, extras={})
        return u_cmd, info
