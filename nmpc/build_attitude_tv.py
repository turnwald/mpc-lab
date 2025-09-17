import casadi as ca
from .solver_probe import make_solver
from .costs import attitude_cost_sign_invariant, du_penalty

def build_attitude_nmpc_tv(model, N: int, weights=None, bounds=None, solver_opts=None):
    """Time-varying reference NMPC for attitude: refs per stage in parameters.
    p = [x0(7); qref(4*(N+1)); wref(3*(N+1))]
    """
    w = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": None}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu

    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    p = ca.MX.sym("p", nx + 4*(N+1) + 3*(N+1), 1)
    x0 = p[0:nx, :]
    off = nx
    qref = p[off:off+4*(N+1), :]; off += 4*(N+1)
    wref = p[off:off+3*(N+1), :]

    def qref_k(k):
        return qref[4*k:4*(k+1), :]
    def wref_k(k):
        return wref[3*k:3*(k+1), :]

    J = ca.MX(0)
    g_list = []
    g_list.append(X[:, 0] - x0)

    for k in range(N):
        qk = X[0:4, k]
        wk = X[4:7, k]
        uk = U[:, k]
        qrk = qref_k(k)
        wrk = wref_k(k)

        J += w["wq"]*attitude_cost_sign_invariant(qk, qrk) \
           + w["ww"]*ca.dot(wk - wrk, wk - wrk) \
           + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            J += du_penalty(uk, U[:, k-1], w["wdu"])
        x_next = model.step(X[:, k], uk)
        g_list.append(X[:, k+1] - x_next)

    # terminal cost uses k=N ref
    qN = X[0:4, N]; wN = X[4:7, N]
    qNr = qref_k(N); wNr = wref_k(N)
    J += w["wqN"]*attitude_cost_sign_invariant(qN, qNr) + w["ww"]*ca.dot(wN - wNr, wN - wNr)

    W = ca.vertcat(ca.reshape(X, nx*(N+1),1), ca.reshape(U, nu*N,1))

    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)
    if b["w_abs_max"] is not None:
        wmax = float(b["w_abs_max"])
        for k in range(N+1):
            for i in range(3):
                idx = (i+4) + nx*k
                lbw[idx] = -wmax
                ubw[idx] =  wmax
    u_off = nx*(N+1)
    for k in range(N):
        for i in range(nu):
            idx = u_off + i + nu*k
            lbw[idx] = float(b["u_min"])
            ubw[idx] = float(b["u_max"])

    g = ca.vertcat(*g_list)
    lbg = ca.DM.zeros(g.shape); ubg = ca.DM.zeros(g.shape)

    prob = {"x": W, "f": J, "g": g, "p": p}
    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, qref_seq: ca.DM, wref_seq: ca.DM) -> ca.DM:
        assert qref_seq.shape[0] == 4 and wref_seq.shape[0] == 3
        return ca.vertcat(x0_dm, ca.reshape(qref_seq, 4*(N+1),1), ca.reshape(wref_seq, 3*(N+1),1))

    def solve_one_step(x0_dm: ca.DM, qref_seq: ca.DM, wref_seq: ca.DM, W0=None):
        if W0 is None:
            X0 = ca.repmat(x0_dm, 1, N+1)
            q0 = X0[0:4, 0]; q0 = q0 / max(float(ca.norm_2(q0)), 1e-8)
            X0[0:4, :] = ca.repmat(q0, 1, N+1)
            W0 = ca.vertcat(ca.reshape(X0, nx*(N+1),1), ca.DM.zeros(nu*N,1))
        sol = solver(x0=W0, p=pack_p(x0_dm, qref_seq, wref_seq),
                     lbx=ca.DM(lbw), ubx=ca.DM(ubw), lbg=lbg, ubg=ubg)
        Wopt = sol["x"]
        Uopt = ca.reshape(Wopt[nx*(N+1):], nu, N)
        Xopt = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
        return Uopt[:,0], Xopt, Uopt, Wopt, sol

    return {"solver": solver, "plugin": plugin, "pack_p": pack_p,
            "solve_one_step": solve_one_step, "nx": nx, "nu": nu, "N": N}
