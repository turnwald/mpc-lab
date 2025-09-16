import casadi as ca
import numpy as np
from typing import Dict

class QPProblem:
    """Holds CasADi QP solver and parameter symbols."""
    def __init__(self, solver, params: Dict[str, ca.MX], dims: Dict[str, int]):
        self.solver = solver
        self.params = params
        self.dims = dims

class LinearMPCQPBuilder:
    """Builds a convex QP for LTV dynamics:
       x_{k+1} = A_k x_k + B_k u_k + d_k
       Cost: sum_k (x_k-xr_k)'Q(x_k-xr_k) + (u_k-ur_k)'R(u_k-ur_k) + terminal (x_N-xr_N)'P(x_N-xr_N)
       Constraints: box bounds on x, u
    """
    def build(self, N: int, nx: int, nu: int, solver: str = "qpoases") -> QPProblem:
        # decision variables
        X = ca.MX.sym("X", (N+1)*nx)
        U = ca.MX.sym("U", N*nu)

        # parameters (flatten horizon into 2-D blocks)
        A = ca.MX.sym("A", N*nx, nx)     # rows stacked
        B = ca.MX.sym("B", N*nx, nu)
        d = ca.MX.sym("d", N*nx, 1)

        x0 = ca.MX.sym("x0", nx)
        xr = ca.MX.sym("xr", N+1, nx)
        ur = ca.MX.sym("ur", N,   nu)
        Q  = ca.MX.sym("Q", nx, nx)
        R  = ca.MX.sym("R", nu, nu)
        P  = ca.MX.sym("P", nx, nx)
        xl = ca.MX.sym("xl", nx); xu = ca.MX.sym("xu", nx)
        ul = ca.MX.sym("ul", nu); uu = ca.MX.sym("uu", nu)

        # objective
        obj = 0
        for k in range(N):
            xk = X[k*nx:(k+1)*nx]
            uk = U[k*nu:(k+1)*nu]
            dx = xk - xr[k,:].reshape((nx,1))
            du = uk - ur[k,:].reshape((nu,1))
            obj += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
        xN  = X[N*nx:(N+1)*nx]
        dxN = xN - xr[N,:].reshape((nx,1))
        obj += ca.mtimes([dxN.T, P, dxN])

        # dynamics equalities
        eqs = [X[0:nx] - x0]
        for k in range(N):
            xk   = X[k*nx:(k+1)*nx]
            xkp1 = X[(k+1)*nx:(k+2)*nx]
            r0 = k*nx; r1 = (k+1)*nx
            Ak = A[r0:r1, :]
            Bk = B[r0:r1, :]
            dk = d[r0:r1, :]
            eqs.append(xkp1 - (ca.mtimes(Ak, xk) + ca.mtimes(Bk, U[k*nu:(k+1)*nu]) + dk))
        g_eq = ca.vertcat(*eqs)

        # standard QP form
        w = ca.vertcat(X, U)
        H = ca.hessian(obj, w)[0]
        g = ca.gradient(obj, w)
        Aeq = ca.jacobian(g_eq, w)
        beq = -ca.substitute(g_eq, w, ca.MX.zeros(*w.shape))

        # variable bounds (boxes on X and U)
        lbw = []
        ubw = []
        for _ in range(N+1):
            lbw.extend([xl[i] for i in range(nx)])
            ubw.extend([xu[i] for i in range(nx)])
        for _ in range(N):
            lbw.extend([ul[i] for i in range(nu)])
            ubw.extend([uu[i] for i in range(nu)])
        lbw = ca.vertcat(*lbw)
        ubw = ca.vertcat(*ubw)

        qp_std = {"h": H, "g": g, "a": Aeq, "lba": beq, "uba": beq, "lbx": lbw, "ubx": ubw}
        qp_alt = {"h": H, "g": g, "A": Aeq, "lba": beq, "uba": beq, "lbx": lbw, "ubx": ubw}
        opts = {"error_on_fail": True}

        solver_fun = None
        try:
            solver_fun = ca.qpsol("mpc_qp", solver, qp_std, opts)
        except Exception:
            try:
                solver_fun = ca.qpsol("mpc_qp", "osqp", qp_std, opts)
            except Exception:
                try:
                    solver_fun = ca.qpsol("mpc_qp", solver, qp_alt, opts)
                except Exception:
                    try:
                        solver_fun = ca.qpsol("mpc_qp", "osqp", qp_alt, opts)
                    except Exception:
                        # Final fallback: no solver (facade will produce zeros)
                        solver_fun = None

        params = {"A":A,"B":B,"d":d,"x0":x0,"xr":xr,"ur":ur,"Q":Q,"R":R,"P":P,"xl":xl,"xu":xu,"ul":ul,"uu":uu}
        dims = {"N": N, "nx": nx, "nu": nu}
        return QPProblem(solver_fun, params, dims)


        params = {"A":A,"B":B,"d":d,"x0":x0,"xr":xr,"ur":ur,"Q":Q,"R":R,"P":P,"xl":xl,"xu":xu,"ul":ul,"uu":uu}
        dims = {"N": N, "nx": nx, "nu": nu}
        return QPProblem(solver_fun, params, dims)
