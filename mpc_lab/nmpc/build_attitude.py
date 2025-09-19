import casadi as ca
from .solver_probe import make_solver
from .costs import attitude_cost_sign_invariant, du_penalty

def build_attitude_nmpc(model, N: int, weights=None, bounds=None, solver_opts=None):
    """Construct NMPC for the quaternion attitude model.
    Returns dict with {solver, plugin, pack_p, solve_one_step, nx, nu, N}.
    """
    w = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": None}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu

    # Decision variables
    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    # Parameters p = [x0(7); qref(4); wref(3)]
    p = ca.MX.sym("p", nx + 4 + 3, 1)
    x0   = p[0:nx, :]
    qref = p[nx:nx+4, :]
    wref = p[nx+4:nx+7, :]

    J = ca.MX(0)
    g_list = []

    # Initial condition
    g_list.append(X[:, 0] - x0)

    for k in range(N):
        qk = X[0:4, k]
        wk = X[4:7, k]
        uk = U[:, k]

        att_err = attitude_cost_sign_invariant(qk, qref)
        J += w["wq"]*att_err + w["ww"]*ca.dot(wk - wref, wk - wref) + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            J += du_penalty(uk, U[:, k-1], w["wdu"])

        x_next = model.step(X[:, k], uk)
        g_list.append(X[:, k+1] - x_next)

    # Terminal cost
    qN = X[0:4, N]
    wN = X[4:7, N]
    att_err_N = attitude_cost_sign_invariant(qN, qref)
    J += w["wqN"]*att_err_N + w["ww"]*ca.dot(wN - wref, wN - wref)

    # Pack decision variables (column-major)
    W = ca.vertcat(ca.reshape(X, nx*(N+1), 1), ca.reshape(U, nu*N, 1))

    # Variable bounds
    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)

    # Optional omega bounds
    if b["w_abs_max"] is not None:
        wmax = float(b["w_abs_max"])
        for k in range(N+1):
            for i in range(3):
                idx = (i+4) + nx*k
                lbw[idx] = -wmax
                ubw[idx] =  wmax

    # Torque bounds
    u_off = nx*(N+1)
    for k in range(N):
        for i in range(nu):
            idx = u_off + i + nu*k
            lbw[idx] = float(b["u_min"])
            ubw[idx] = float(b["u_max"])

    g = ca.vertcat(*g_list)
    lbg = ca.DM.zeros(g.shape)
    ubg = ca.DM.zeros(g.shape)

    prob = {"x": W, "f": J, "g": g, "p": p}

    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, qref_dm: ca.DM, wref_dm: ca.DM) -> ca.DM:
        return ca.vertcat(x0_dm, qref_dm, wref_dm)

    def solve_one_step(x0_dm: ca.DM, qref_dm: ca.DM, wref_dm: ca.DM, u_init=None):
        # Build an initial guess that avoids undefined Jacobians:
        # - replicate x0 across the horizon
        # - ensure quaternion entries are normalized in the guess
        X0 = ca.repmat(x0_dm, 1, N+1)
        q0 = X0[0:4, 0]
        q0 = q0 / max(float(ca.norm_2(q0)), 1e-8)
        X0[0:4, :] = ca.repmat(q0, 1, N+1)

        W0 = ca.DM.zeros(W.shape)
        W0[0:nx*(N+1)] = ca.reshape(X0, nx*(N+1), 1)

        if u_init is not None:
            u_init = ca.DM(u_init)
            W0[nx*(N+1):] = ca.reshape(u_init, nu*N, 1)
        # else zeros are fine

        sol = solver(x0=W0, p=pack_p(x0_dm, qref_dm, wref_dm),
                     lbx=ca.DM(lbw), ubx=ca.DM(ubw), lbg=lbg, ubg=ubg)
        Wopt = sol["x"]
        Uopt = ca.reshape(Wopt[nx*(N+1):], nu, N)
        Xopt = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
        return Uopt[:, 0], Xopt, Uopt, sol

    return {
        "solver": solver, "plugin": plugin,
        "pack_p": pack_p, "solve_one_step": solve_one_step,
        "nx": nx, "nu": nu, "N": N
    }
