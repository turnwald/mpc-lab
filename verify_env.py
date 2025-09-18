import casadi as ca
import numpy as np

print("== Verifying CasADi nlpsol plugins (probing) ==")

def probe_nlpsol(name: str) -> bool:
    try:
        x = ca.MX.sym("x")
        nlp = {"x": x, "f": x, "g": []}
        ca.nlpsol("probe", name, nlp)
        return True
    except Exception:
        return False

available = {}
for s in ("ipopt", "fatrop", "sqpmethod"):
    ok = probe_nlpsol(s)
    available[s] = ok
    print(f"{s:10s}: {ok}")

if not any(available.values()):
    raise SystemExit("No usable CasADi nlpsol plugin found (need at least one of ipopt/fatrop/sqpmethod).")

# Minimal NMPC smoke test on a double integrator to confirm end-to-end
print("\n== Running minimal NMPC smoke test (double integrator) ==")
dt = 0.1
nx, nu, N = 2, 1, 10

def f_disc(x, u):
    # x = [p, v]; u = [a]
    return ca.vertcat(x[0] + dt * x[1], x[1] + dt * u[0])

# Decision vars
X = ca.MX.sym("X", (N+1)*nx)
U = ca.MX.sym("U", N*nu)
w = ca.vertcat(X, U)

# Params
x0 = ca.MX.sym("x0", nx, 1)           # (nx,1)
xr = ca.MX.sym("xr", N+1, nx)         # (N+1,nx)
ur = ca.MX.sym("ur", N, nu)           # (N,nu)
Q  = ca.MX.sym("Q", nx, nx)
R  = ca.MX.sym("R", nu, nu)
P  = ca.MX.sym("P", nx, nx)
p  = ca.vertcat(ca.vec(x0), ca.vec(xr), ca.vec(ur), ca.vec(Q), ca.vec(R), ca.vec(P))

# Objective
obj = 0
for k in range(N):
    xk = X[k*nx:(k+1)*nx]
    uk = U[k*nu:(k+1)*nu]
    dx = xk - ca.reshape(xr[k,:], (nx,1))
    du = uk - ca.reshape(ur[k,:], (nu,1))
    obj += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
xN  = X[N*nx:(N+1)*nx]
dxN = xN - ca.reshape(xr[N,:], (nx,1))
obj += ca.mtimes([dxN.T, P, dxN])

# Dynamics constraints
eqs = [ X[0:nx] - x0 ]  # fixed: no reshape, both are (nx,)
for k in range(N):
    xk   = X[k*nx:(k+1)*nx]
    uk   = U[k*nu:(k+1)*nu]
    xkp1 = X[(k+1)*nx:(k+2)*nx]
    x_next = f_disc(xk, uk)
    # with this
    eqs.append(xkp1 - x_next)
    g = ca.vertcat(*eqs)

# Bounds
w_l = ca.DM.zeros(w.shape); w_u = ca.DM.zeros(w.shape)
xl = np.array([-5, -5]); xu = np.array([5, 5])
ul = np.array([-2.0]);    uu = np.array([2.0])

for k in range(N+1):
    off = k*nx
    w_l[off:off+nx] = ca.DM(xl)
    w_u[off:off+nx] = ca.DM(xu)
ubase = (N+1)*nx
for k in range(N):
    off = ubase + k*nu
    w_l[off:off+nu] = ca.DM(ul)
    w_u[off:off+nu] = ca.DM(uu)

g_l = ca.DM.zeros(g.shape); g_u = ca.DM.zeros(g.shape)

# Numeric params
x0_val = ca.DM([2.0, -1.0])
xr_val = ca.DM.zeros(N+1, nx)
ur_val = ca.DM.zeros(N, nu)
Q_val  = ca.DM(np.diag([10.0, 1.0]))
R_val  = ca.DM(np.diag([0.1]))
P_val  = Q_val

p_val = ca.vertcat(ca.vec(x0_val), ca.vec(xr_val), ca.vec(ur_val), ca.vec(Q_val), ca.vec(R_val), ca.vec(P_val))

# Build and solve NLP with the best available solver
for solver_name in ("ipopt", "fatrop", "sqpmethod"):
    if available.get(solver_name):
        S = ca.nlpsol("solver", solver_name, {"x": w, "f": obj, "g": g, "p": p}, {"print_time": False})
        break

sol = S(x0=ca.DM.zeros(w.shape), lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u, p=p_val)
w_opt = sol["x"]
u0 = w_opt[(N+1)*nx : (N+1)*nx + nu]
print("u0:", np.array(u0.full()).ravel())
print("OK: NMPC smoke test passed.")
