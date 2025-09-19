# mpc_lab/nmpc/build_rover_tv.py
import math
import casadi as ca
from .solver_probe import make_solver
from mpc_lab.constraints.boxes import apply_state_bounds, apply_input_bounds
from mpc_lab.constraints.linear_sets import add_stagewise_ineq

def build_rover_nmpc_tv(model, N: int, weights=None, bounds=None,
                        solver_opts=None, state_box=None, linear_state_ineq=None):
    """
    Rover NMPC with time-varying references and constraints via helpers.

    Parameters
    ----------
    model : RoverUnicycleModel-like with .nx, .nu, .dt, .step(x,u)
    N     : horizon length
    weights : dict {w_pos, w_heading, wu, wdu}
    bounds  : dict {v_min, v_max, om_min, om_max}
    solver_opts : dict passed to nlpsol
    state_box : optional dict {"lbx": (nx,1), "ubx": (nx,1)} applied to X[:,k] for k=0..N
    linear_state_ineq : optional {"H": (m x nx), "h": (m x 1)} enforcing H x_k <= h for k=0..N

    Returns
    -------
    dict with keys: solver, plugin, pack_p, solve_one_step, nx, nu, N
    """
    w = {"w_pos": 10.0, "w_heading": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"v_min": -0.8, "v_max": 1.0, "om_min": -2.0, "om_max": 2.0}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu
    assert nx == 3 and nu == 2, "Rover builder expects (nx=3, nu=2)."

    # Decision variables
    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    # Parameters: x0 and stacked refs (3*(N+1))
    p = ca.MX.sym("p", nx + 3*(N+1), 1)
    x0 = p[0:nx, :]
    xref_stack = p[nx:nx+3*(N+1), :]
    def xr_k(k): return xref_stack[3*k:3*(k+1), :]

    # Objective and constraints
    J = ca.MX(0)
    g_eq = []
    g_ineq = []

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

        # Stagewise linear state inequalities: H x_k <= h
        if linear_state_ineq is not None:
            H = linear_state_ineq["H"]
            h = linear_state_ineq["h"]
            add_stagewise_ineq(g_ineq, X, H, h, N, slc=None)
            # Only add once outside loop; break to avoid duplicate appends
            linear_state_ineq = None  # sentinel to avoid re-appending

    # Terminal cost
    xN = X[:, N]
    xrN = xr_k(N)
    pos_errN = xN[0:2, :] - xrN[0:2, :]
    th_errN = xN[2, 0] - xrN[2, 0]
    J += w["w_pos"]*ca.dot(pos_errN, pos_errN) + w["w_heading"]*(th_errN*th_errN)

    # Pack decision variables
    W = ca.vertcat(ca.reshape(X, nx*(N+1), 1), ca.reshape(U, nu*N, 1))

    # Variable bounds
    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)

    # Apply state box bounds (if provided)
    if state_box is not None:
        lbx_vec = state_box.get("lbx", None)
        ubx_vec = state_box.get("ubx", None)
        apply_state_bounds(lbw, ubw, nx, N, lbx_vec=lbx_vec, ubx_vec=ubx_vec)

    # Apply input bounds
    u_off = nx*(N+1)
    apply_input_bounds(lbw, ubw, u_off, nu, N,
                       umin_vec=[b["v_min"], b["om_min"]],
                       umax_vec=[b["v_max"], b["om_max"]])

    # Constraint vector and bounds
    g = ca.vertcat(*(g_eq + g_ineq)) if len(g_ineq)>0 else ca.vertcat(*g_eq)
    if len(g_ineq) == 0:
        lbg = ca.DM.zeros(g.shape)
        ubg = ca.DM.zeros(g.shape)
    else:
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
