import casadi as ca
from .solver_probe import make_solver

def build_rover_nmpc_tv(model, N: int, weights=None, bounds=None, solver_opts=None):
    """Time-varying reference NMPC for rover: refs per stage in parameters.
    p = [x0(3); Xref(3*(N+1))]
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
    Xref = p[nx:, :]

    def xref_k(k):
        base = 3*k
        return Xref[base:base+3, :]

    J = ca.MX(0)
    g_list = []
    g_list.append(X[:,0] - x0)

    for k in range(N):
        xk = X[:,k]
        uk = U[:,k]
        xrk = xref_k(k)
        pos_err = xk[0:2, :] - xrk[0:2, :]
        th_err  = xk[2,0] - xrk[2,0]

        J += w["w_pos"]*ca.dot(pos_err, pos_err) + w["w_heading"]*(th_err*th_err) + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            du = uk - U[:, k-1]
            J += w["wdu"] * ca.dot(du, du)

        x_next = model.step(xk, uk)
        g_list.append(X[:,k+1] - x_next)

    xN = X[:,N]; xNr = xref_k(N)
    pos_errN = xN[0:2, :] - xNr[0:2, :]
    th_errN  = xN[2,0] - xNr[2,0]
    J += w["w_pos"]*ca.dot(pos_errN, pos_errN) + w["w_heading"]*(th_errN*th_errN)

    W = ca.vertcat(ca.reshape(X, nx*(N+1),1), ca.reshape(U, nu*N,1))

    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)
    u_off = nx*(N+1)
    for k in range(N):
        lbw[u_off + 0 + 2*k] = float(b["v_min"])
        ubw[u_off + 0 + 2*k] = float(b["v_max"])
        lbw[u_off + 1 + 2*k] = float(b["om_min"])
        ubw[u_off + 1 + 2*k] = float(b["om_max"])

    g = ca.vertcat(*g_list)
    lbg = ca.DM.zeros(g.shape); ubg = ca.DM.zeros(g.shape)
    prob = {"x": W, "f": J, "g": g, "p": p}
    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, Xref_seq: ca.DM) -> ca.DM:
        assert Xref_seq.shape[0] == 3
        return ca.vertcat(x0_dm, ca.reshape(Xref_seq, 3*(N+1), 1))

    def solve_one_step(x0_dm: ca.DM, Xref_seq: ca.DM, W0=None):
        if W0 is None:
            X0 = ca.repmat(x0_dm, 1, N+1)
            W0 = ca.vertcat(ca.reshape(X0, nx*(N+1),1), ca.DM.zeros(nu*N,1))
        sol = solver(x0=W0, p=pack_p(x0_dm, Xref_seq),
                     lbx=ca.DM(lbw), ubx=ca.DM(ubw), lbg=lbg, ubg=ubg)
        Wopt = sol["x"]
        Uopt = ca.reshape(Wopt[nx*(N+1):], nu, N)
        Xopt = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
        return Uopt[:,0], Xopt, Uopt, Wopt, sol

    return {"solver": solver, "plugin": plugin, "pack_p": pack_p,
            "solve_one_step": solve_one_step, "nx": nx, "nu": nu, "N": N}
