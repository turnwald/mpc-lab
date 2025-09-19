import casadi as ca
from .solver_probe import make_solver
import math


def build_rover_nmpc_tv(model, N: int, weights=None, bounds=None,
                        solver_opts=None, state_box=None, linear_state_ineq=None):
    """
    Rover NMPC with time-varying refs.
    p = [x0(3); xref_stack(3*(N+1))].

    New:
      state_box: dict with optional 'lbx' and 'ubx' of shape (nx,1) or (nx,)
      linear_state_ineq: dict with 'H' (m x nx) and 'h' (m x 1); enforces H x_k <= h for k=0..N
    """
    w = {"w_pos": 10.0, "w_heading": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"v_min": -0.8, "v_max": 1.0, "om_min": -2.0, "om_max": 2.0}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu
    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    p = ca.MX.sym("p", nx + 3*(N+1), 1)
    x0 = p[0:nx, :]
    xref_stack = p[nx:nx+3*(N+1), :]

    def xr_k(k): return xref_stack[3*k:3*(k+1), :]

    J = ca.MX(0)
    g_eq = []      # equalities (dynamics + initial condition)
    g_ineq = []    # inequalities (linear state sets)

    # Initial condition
    g_eq.append(X[:, 0] - x0)

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        xr = xr_k(k)

        pos_err = xk[0:2, :] - xr[0:2, :]
        th_err = xk[2, 0] - xr[2, 0]
        J += w["w_pos"]*ca.dot(pos_err, pos_err) + w["w_heading"]*(th_err*th_err) + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            du = uk - U[:, k-1]
            J += w["wdu"] * ca.dot(du, du)

        # Dynamics
        g_eq.append(X[:, k+1] - model.step(xk, uk))

        # Linear state set: H x_k <= h
        if linear_state_ineq is not None:
            H = ca.DM(linear_state_ineq["H"])  # (m x nx)
            h = ca.DM(linear_state_ineq["h"]).reshape((-1,1))  # (m x 1)
            g_ineq.append(H @ xk - h)

    # Terminal cost
    xN = X[:, N]
    xrN = xr_k(N)
    pos_errN = xN[0:2, :] - xrN[0:2, :]
    th_errN = xN[2, 0] - xrN[2, 0]
    J += w["w_pos"]*ca.dot(pos_errN, pos_errN) + w["w_heading"]*(th_errN*th_errN)

    # Pack decision variables
    W = ca.vertcat(ca.reshape(X, nx*(N+1), 1), ca.reshape(U, nu*N, 1))

    # Variable bounds (default unbounded)
    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)

    # State box bounds (per state, all stages)
    if state_box is not None:
        lbx_vec = state_box.get("lbx", None)
        ubx_vec = state_box.get("ubx", None)
        if lbx_vec is not None:
            lbx_vec = ca.DM(lbx_vec).reshape((nx,1))
        if ubx_vec is not None:
            ubx_vec = ca.DM(ubx_vec).reshape((nx,1))
        for k in range(N+1):
            for i in range(nx):
                idx = i + nx*k
                if lbx_vec is not None and not math.isinf(lbx_vec[i,0]):
                    lbw[idx] = float(lbx_vec[i,0])
                if ubx_vec is not None and not math.isinf(ubx_vec[i,0]):
                    ubw[idx] = float(ubx_vec[i,0])


    # Input bounds
    u_off = nx*(N+1)
    for k in range(N):
        lbw[u_off + 0 + 2*k] = float(b["v_min"])
        ubw[u_off + 0 + 2*k] = float(b["v_max"])
        lbw[u_off + 1 + 2*k] = float(b["om_min"])
        ubw[u_off + 1 + 2*k] = float(b["om_max"])

    # Constraints vector and bounds
    g = ca.vertcat(*(g_eq + g_ineq)) if len(g_ineq)>0 else ca.vertcat(*g_eq)
    lbg = ca.DM.zeros(g.shape)
    ubg = ca.DM.zeros(g.shape)
    if len(g_ineq) > 0:
        # First |g_eq| are equalities -> [0,0]; the rest are inequalities -> (-inf, 0]
        neq = sum(gi.size1() for gi in g_eq)
        nineq = sum(gi.size1() for gi in g_ineq)
        lbg = ca.vertcat(ca.DM.zeros(neq,1), (-ca.inf)*ca.DM.ones(nineq,1))
        ubg = ca.vertcat(ca.DM.zeros(neq,1), ca.DM.zeros(nineq,1))

    prob = {"x": W, "f": J, "g": g, "p": p}
    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, xref_seq: ca.DM) -> ca.DM:
        return ca.vertcat(x0_dm, ca.reshape(xref_seq, 3*(N+1), 1))

    def solve_one_step(x0_dm: ca.DM, xref_seq: ca.DM, u_init=None):
        X0 = ca.repmat(x0_dm, 1, N+1)
        W0 = ca.DM.zeros(W.shape)
        W0[0:nx*(N+1)] = ca.reshape(X0, nx*(N+1), 1)
        if u_init is not None:
            u_init = ca.DM(u_init)
            W0[nx*(N+1):] = ca.reshape(u_init, nu*N, 1)

        sol = solver(x0=W0, p=pack_p(x0_dm, xref_seq),
                     lbx=ca.DM(lbw), ubx=ca.DM(ubw), lbg=lbg, ubg=ubg)
        Wopt = sol["x"]
        Uopt = ca.reshape(Wopt[nx*(N+1):], nu, N)
        Xopt = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
        return Uopt[:, 0], Xopt, Uopt, sol

    return {"solver": solver, "plugin": plugin,
            "pack_p": pack_p, "solve_one_step": solve_one_step,
            "nx": nx, "nu": nu, "N": N}
