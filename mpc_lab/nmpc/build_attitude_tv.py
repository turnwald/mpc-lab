# mpc_lab/nmpc/build_attitude_tv.py
import casadi as ca
from .solver_probe import make_solver
from mpc_lab.constraints.boxes import apply_state_bounds

def build_attitude_nmpc_tv(model, N: int, weights=None, bounds=None, solver_opts=None):
    """
    Attitude NMPC (quaternion) with time-varying references.
    Enforces input (torque) bounds and optional body-rate bounds via state boxes.
    Parameters
    ----------
    model: has nx=7 (q (4), w (3)), nu=3; step(x,u)
    p = [x0(7); qref_stack(4*(N+1)); wref_stack(3*(N+1))]
    """
    w = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    if weights: w.update(weights)
    b = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": None}
    if bounds: b.update(bounds)

    nx, nu = model.nx, model.nu
    assert nx == 7 and nu == 3, "Attitude builder expects (nx=7, nu=3)."

    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    # parameters
    p = ca.MX.sym("p", nx + 4*(N+1) + 3*(N+1), 1)
    x0 = p[0:nx, :]
    qref_stack = p[nx:nx+4*(N+1), :]
    wref_stack = p[nx+4*(N+1):, :]

    def qref_k(k): return qref_stack[4*k:4*(k+1), :]
    def wref_k(k): return wref_stack[3*k:3*(k+1), :]

    J = ca.MX(0)
    g = []  # equalities only (dynamics, initial condition)

    # Initial condition
    g.append(X[:, 0] - x0)

    def q_cost(q, qref):
        # Sign-invariant alignment: 1 - (q·qref)^2
        c = ca.dot(q, qref)
        return 1.0 - c*c

    for k in range(N):
        qk = X[0:4, k]
        wk = X[4:7, k]
        uk = U[:, k]

        qr = qref_k(k)
        wr = wref_k(k)

        J += w["wq"]*q_cost(qk, qr) + w["ww"]*ca.dot(wk - wr, wk - wr) + w["wu"]*ca.dot(uk, uk)
        if k > 0:
            du = uk - U[:, k-1]
            J += w["wdu"] * ca.dot(du, du)

        # Dynamics
        x_next = model.step(X[:, k], uk)
        g.append(X[:, k+1] - x_next)

    # Terminal cost
    qN = X[0:4, N]; wN = X[4:7, N]
    qNr = qref_k(N); wNr = wref_k(N)
    J += w["wqN"]*q_cost(qN, qNr) + w["ww"]*ca.dot(wN - wNr, wN - wNr)

    # Pack decision variables
    W = ca.vertcat(ca.reshape(X, nx*(N+1), 1), ca.reshape(U, nu*N, 1))

    # Variable bounds
    lbw = [-ca.inf]*(nx*(N+1) + nu*N)
    ubw = [ ca.inf]*(nx*(N+1) + nu*N)

    # Input (torque) bounds on U
    u_off = nx*(N+1)
    for k in range(N):
        for i in range(nu):
            lbw[u_off + i + nu*k] = float(b["u_min"])
            ubw[u_off + i + nu*k] = float(b["u_max"])

    # Optional body-rate bounds via state box on indices 4:7
    if b.get("w_abs_max", None) is not None:
        wmax = float(b["w_abs_max"])
        # Build lbx/ubx for all 7 states, set only w-components; others remain ±inf
        lbx = [ -ca.inf ]*nx; ubx = [ ca.inf ]*nx
        for i in range(4,7):
            lbx[i] = -wmax; ubx[i] = wmax
        apply_state_bounds(lbw, ubw, nx, N, lbx_vec=lbx, ubx_vec=ubx)

    g_vec = ca.vertcat(*g)
    lbg = ca.DM.zeros(g_vec.shape)
    ubg = ca.DM.zeros(g_vec.shape)

    prob = {"x": W, "f": J, "g": g_vec, "p": p}
    default_opts = {"ipopt.print_level": 0, "print_time": 0}
    if solver_opts: default_opts.update(solver_opts)
    solver, plugin = make_solver(prob, default_opts)

    def pack_p(x0_dm: ca.DM, qref_seq: ca.DM, wref_seq: ca.DM) -> ca.DM:
        return ca.vertcat(x0_dm,
                          ca.reshape(qref_seq, 4*(N+1), 1),
                          ca.reshape(wref_seq, 3*(N+1), 1))

    def solve_one_step(x0_dm: ca.DM, qref_seq: ca.DM, wref_seq: ca.DM, u_init=None):
        X0 = ca.repmat(x0_dm, 1, N+1)
        W0 = ca.DM.zeros(W.shape)
        W0[0:nx*(N+1)] = ca.reshape(X0, nx*(N+1), 1)
        if u_init is not None:
            u_init = ca.DM(u_init)
            W0[nx*(N+1):] = ca.reshape(u_init, nu*N, 1)
        sol = solver(x0=W0, p=pack_p(x0_dm, qref_seq, wref_seq),
                     lbx=ca.DM(lbw), ubx=ca.DM(ubw), lbg=lbg, ubg=ubg)
        Wopt = sol["x"]
        Uopt = ca.reshape(Wopt[nx*(N+1):], nu, N)
        Xopt = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
        return Uopt[:, 0], Xopt, Uopt, sol

    return {"solver": solver, "plugin": plugin,
            "pack_p": pack_p, "solve_one_step": solve_one_step,
            "nx": nx, "nu": nu, "N": N}
