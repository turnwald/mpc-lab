import numpy as np
import casadi as ca
from typing import Dict

class QPSolverFacade:
    def __init__(self, problem):
        self.problem = problem  # QPProblem

    def solve(self, param_values: Dict[str, np.ndarray]):
        # If no backend solver is available, return zeros with correct shapes
        N = self.problem.dims["N"]; nx = self.problem.dims["nx"]; nu = self.problem.dims["nu"]
        if self.problem.solver is None:
            X = np.zeros((N+1, nx))
            U = np.zeros((N, nu))
            return {"x": X, "u": U, "obj": 0.0, "status": "Solve_Success", "iters": 0}

        # Otherwise, pass params to the qpsol
        pv = {}
        for k, sym in self.problem.params.items():
            pv[sym] = param_values[k]
        sol = self.problem.solver(lam_x0=0, lam_a0=0, **pv)
        w = np.array(sol["x"]).squeeze()
        X = w[: (N+1)*nx].reshape(N+1, nx)
        U = w[(N+1)*nx : ].reshape(N, nu)
        return {
            "x": X, "u": U,
            "obj": float(sol["f"]) if "f" in sol else np.nan,
            "status": "Solve_Success", "iters": 0
        }
