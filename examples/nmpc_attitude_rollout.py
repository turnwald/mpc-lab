"""Closed-loop rollout for attitude NMPC with warm-start."""
import math
import casadi as ca
from models.attitude_quat import AttitudeQuatModel
from nmpc.build_attitude import build_attitude_nmpc

def shift_u_warm_start(Uopt: ca.DM) -> ca.DM:
    """Shift optimal input sequence forward by one and append last value."""
    nu, N = int(Uopt.size1()), int(Uopt.size2())
    if N == 0:
        return Uopt
    Us = ca.DM(nu, N)
    if N > 1:
        Us[:, 0:N-1] = Uopt[:, 1:N]
    Us[:, N-1] = Uopt[:, N-1]
    return Us

def main():
    dt = 0.1
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    N = 20
    weights = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    bounds  = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0}
    nmpc = build_attitude_nmpc(model, N, weights=weights, bounds=bounds)

    # Initial state
    x = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))
    # Reference: 45 deg yaw step
    ang = math.radians(45.0) * 0.5
    qref = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
    wref = ca.DM([0,0,0]).reshape((3,1))

    # Rollout
    T = 40  # steps
    Uws = None
    hist = []
    for t in range(T):
        u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x, qref, wref, u_init=Uws)
        # apply first control to the model (simulate plant = model)
        x = model.step(x, u0)
        # warm-start for next iteration
        Uws = shift_u_warm_start(Uopt)
        hist.append((float(t*dt), x, u0))

    q = x[0:4, :]
    w = x[4:7, :]
    print("Final q:", q.T)
    print("||q||:", float(ca.norm_2(q)))
    print("Final ω:", w.T)

if __name__ == "__main__":
    main()
