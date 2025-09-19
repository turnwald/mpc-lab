"""Attitude tracking with time-varying references + plotting (constant target demo)."""
import casadi as ca
from models.attitude_quat import AttitudeQuatModel
from nmpc.build_attitude_tv import build_attitude_nmpc_tv
from utils.refs import attitude_yaw_slew
from utils.plotting import plot_attitude_rollout

def shift_u_warm_start(Uopt: ca.DM) -> ca.DM:
    nu, N = int(Uopt.size1()), int(Uopt.size2())
    Us = ca.DM(nu, N)
    if N > 1:
        Us[:, 0:N-1] = Uopt[:, 1:N]
    Us[:, N-1] = Uopt[:, N-1] if N>0 else 0
    return Us

def main():
    dt = 0.1
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    N = 20
    nmpc = build_attitude_nmpc_tv(model, N,
                                  weights={"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3},
                                  bounds={"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0})

    # Target: 45° yaw (constant across horizon in this simple demo)
    qref_full, wref_full = attitude_yaw_slew(dt, 1000, yaw_deg_final=45.0)

    # Initial state
    x = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))

    # Closed-loop rollout
    T = 40
    Uws = None
    times, Q_hist, W_hist, U_hist = [], [], [], []

    for t in range(T):
        qref_seq = qref_full[:, 0:N+1]
        wref_seq = wref_full[:, 0:N+1]
        u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x, qref_seq, wref_seq, u_init=Uws)

        times.append(t*dt)
        Q_hist.append(x[0:4, :])
        W_hist.append(x[4:7, :])
        U_hist.append(u0)

        x = model.step(x, u0)
        Uws = shift_u_warm_start(Uopt)

    plot_attitude_rollout(times, Q_hist, W_hist, U_hist)

if __name__ == "__main__":
    main()
