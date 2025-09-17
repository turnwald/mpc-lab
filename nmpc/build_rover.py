import casadi as ca
from .solver_probe import make_solver

def build_rover_nmpc(model, N: int, weights=None, bounds=None, solver_opts=None):
    """Construct NMPC for the rover (unicycle) model with constant reference.
    Returns dict with {solver, plugin, pack_p, solve_one_step, nx, nu, N}.
    """
    w = {"w_pos": 10.0, "w_heading": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"v_min": -0.8, "v_max": 1.0, "om_min": -2.0, "om_max": 2.0}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu

    # Decision variables
    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    # Parameters p = [x0(3); xref; yref; thref]
    p = ca.MX.sym("p", nx + 3, 1)
    x0    = p[0:nx, :]
    xref  = p[nx + 0, 0]
    yref  = p[nx + 1, 0]
    thref = p[nx + 2, 0]

    J = ca.MX(0)
    g_list = []

    # Initial condition
    g_list.append(X[:, 0] - x0)

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        pos_err = ca.vertcat(xk[0,0] - xref, xk[1,0] - yref)
        th_err  = xk[2,0] - thref

        J += w["w_pos"]*ca.dot(pos_err, pos_err) + w["w_heading"]*(th_err*th_err) + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            du = uk - U[:, k-1]
            J += w["wdu"] * ca.dot(du, du)

        x_next = model.step(xk, uk)
        g_list.append(X[:, k+1] - x_next)

    # Terminal cost (same form)
    xN = X[:, N]
    pos_errN = ca.vertcat(xN[0,0] - xref, xN[1,0] - yref)
    th_errN  = xN[2,0] - thref
    J += w["w_pos"]*ca.dot(pos_errN, pos_errN) + w["w_heading"]*(th_errN*th_errN)

    # Pack decision variables
    W = ca.vertcat(ca.reshape(X, nx*(N+1), 1), ca.reshape(U, nu*N, 1))

    # Variable bounds
    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)

    # Input bounds
    u_off = nx*(N+1)
    for k in range(N):
        # v
        lbw[u_off + 0 + 2*k] = float(b["v_min"])
        ubw[u_off + 0 + 2*k] = float(b["v_max"])
        # omega
        lbw[u_off + 1 + 2*k] = float(b["om_min"])
        ubw[u_off + 1 + 2*k] = float(b["om_max"])

    g = ca.vertcat(*g_list)
    lbg = ca.DM.zeros(g.shape)
    ubg = ca.DM.zeros(g.shape)

    prob = {"x": W, "f": J, "g": g, "p": p}

    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, xref_val: float, yref_val: float, thref_val: float) -> ca.DM:
        return ca.vertcat(x0_dm, ca.DM([xref_val, yref_val, thref_val]).reshape((3,1)))

    def solve_one_step(x0_dm: ca.DM, xref_val: float, yref_val: float, thref_val: float, u_init=None):
        W0 = ca.DM.zeros(W.shape)
        if u_init is not None:
            u_init = ca.DM(u_init)
            W0[nx*(N+1):] = ca.reshape(u_init, nu*N, 1)
        sol = solver(x0=W0, p=pack_p(x0_dm, xref_val, yref_val, thref_val),
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
