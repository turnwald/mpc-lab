# examples/nmpc_attitude_tracking_constrained.py
# Run from repo root:
#   python examples/nmpc_attitude_tracking_constrained.py
#import os, sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import casadi as ca
import matplotlib.pyplot as plt

from mpc_lab.models.attitude_quat import AttitudeQuatModel
from mpc_lab.nmpc.build_attitude_tv import build_attitude_nmpc_tv

# Prefer the improved time-varying reference if available
try:
    from mpc_lab.utils.refs import attitude_yaw_trapezoid
except Exception:
    attitude_yaw_trapezoid = None

def _fallback_attitude_yaw_slew(dt: float, yaw_final_deg: float):
    """Constant quaternion target (zero omega)."""
    ang = math.radians(yaw_final_deg) * 0.5
    q = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
    def window(t0: float, N: int):
        return ca.repmat(q, 1, N+1), ca.DM.zeros(3, N+1)
    return window

def shift_u_warm_start(Uopt: ca.DM) -> ca.DM:
    nu, N = int(Uopt.size1()), int(Uopt.size2())
    Us = ca.DM(nu, N)
    if N > 1:
        Us[:, 0:N-1] = Uopt[:, 1:N]
    Us[:, N-1] = Uopt[:, N-1] if N>0 else 0
    return Us

def main():
    dt = 0.1
    N  = 20
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))

    # Constraints: torque and body-rate bounds
    bounds = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0}
    weights = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}

    nmpc = build_attitude_nmpc_tv(model, N, weights=weights, bounds=bounds)

    # Reference: yaw trapezoid (if available) or constant target
    if attitude_yaw_trapezoid is not None:
        yaw_traj = attitude_yaw_trapezoid(dt, total_time=5.0,
                                          yaw_final_deg=45.0, max_rate_dps=30.0, max_accel_dps2=60.0)
        def get_refs(t0): return yaw_traj(t0, N)
    else:
        yaw_slew = _fallback_attitude_yaw_slew(dt, yaw_final_deg=45.0)
        def get_refs(t0): return yaw_slew(t0, N)

    # Closed-loop rollout
    x = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))
    T = 40
    Uws = None
    times, W_hist, U_hist = [], [], []

    for t in range(T):
        qref_seq, wref_seq = get_refs(t*dt)
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, qref_seq, wref_seq, u_init=Uws)
        times.append(t*dt)
        W_hist.append(x[4:7, :])  # body rates
        U_hist.append(u0)         # torque
        x = model.step(x, u0)
        Uws = shift_u_warm_start(Uopt)

    # Plots with constraint overlays
    t = times
    wx = [float(w[0,0]) for w in W_hist]
    wy = [float(w[1,0]) for w in W_hist]
    wz = [float(w[2,0]) for w in W_hist]
    tx = [float(u[0,0]) for u in U_hist]
    ty = [float(u[1,0]) for u in U_hist]
    tz = [float(u[2,0]) for u in U_hist]

    wmax = bounds["w_abs_max"]
    umax = bounds["u_max"]; umin = bounds["u_min"]

    # Body rates with ±wmax bounds
    plt.figure()
    plt.plot(t, wx, label='ωx'); plt.plot(t, wy, label='ωy'); plt.plot(t, wz, label='ωz')
    if wmax is not None:
        plt.plot(t, [ wmax]*len(t), '--', label='ω ≤ wmax'); plt.plot(t, [-wmax]*len(t), '--', label='ω ≥ -wmax')
    plt.xlabel("time [s]"); plt.ylabel("ω [rad/s]"); plt.title("Body rates with bounds")
    plt.legend()

    # Torques with ±u bounds
    plt.figure()
    plt.plot(t, tx, label='τx'); plt.plot(t, ty, label='τy'); plt.plot(t, tz, label='τz')
    plt.plot(t, [umax]*len(t), '--', label='τ ≤ umax'); plt.plot(t, [umin]*len(t), '--', label='τ ≥ umin')
    plt.xlabel("time [s]"); plt.ylabel("τ [Nm]"); plt.title("Torque with bounds")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
