import math, os
import casadi as ca
from mpc_lab.models.attitude_quat import AttitudeQuatModel
from mpc_lab.nmpc.build_attitude_tv import build_attitude_nmpc_tv
from mpc_lab.utils.trajectory import yaw_slew_profile
from mpc_lab.utils.logging import RolloutLogger
from mpc_lab.utils.plotting import plot_attitude_rollout

def shift_warm(Wopt: ca.DM, nx: int, nu: int, N: int) -> ca.DM:
    """Shift the whole decision vector by one step (X,U) for warm-start."""
    X = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
    U = ca.reshape(Wopt[nx*(N+1):], nu, N)
    # shift U forward, keep last
    if N > 1:
        U[:, 0:N-1] = U[:, 1:N]
    # copy last
    U[:, N-1] = U[:, N-1]
    return ca.vertcat(ca.reshape(X, nx*(N+1),1), ca.reshape(U, nu*N,1))

def main():
    dt = 0.1
    model_ctrl = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    # Plant with inertia mismatch (10% heavier on Jz) and constant disturbance torque
    model_plant = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.077))
    tau_dist = ca.DM([0.0, 0.0, 0.01]).reshape((3,1))

    N = 20
    nmpc = build_attitude_nmpc_tv(model_ctrl, N,
                                  weights={"wq": 60.0, "wqN": 120.0, "ww": 1.0, "wu": 1e-2, "wdu": 5e-3},
                                  bounds={"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0})
    total_time = 6.0
    T = int(round(total_time / dt))
    qref_seq, wref_seq = yaw_slew_profile(total_time, dt, yaw_final_deg=60.0)

    # Initial
    x = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))

    # Logger
    out_dir = os.path.join("/mnt/data", "mpc_lab_outputs")
    log_path = os.path.join(out_dir, "attitude_tv_rollout.csv")
    logger = RolloutLogger(log_path, header=["t","qw","qx","qy","qz","wx","wy","wz","u1","u2","u3"])

    W0 = None
    for k in range(T):
        # slice horizon refs (k..k+N)
        qref_h = qref_seq[:, k:k+N+1] if k+N <= qref_seq.size2()-1 else qref_seq[:, -N-1:]
        wref_h = wref_seq[:, k:k+N+1] if k+N <= wref_seq.size2()-1 else wref_seq[:, -N-1:]
        u0, Xopt, Uopt, Wopt, sol = nmpc["solve_one_step"](x, qref_h, wref_h, W0=W0)
        # apply to plant: mismatch + disturbance
        x = model_plant.step(x, u0 + tau_dist)
        # warm-start
        W0 = shift_warm(Wopt, nmpc["nx"], nmpc["nu"], nmpc["N"])
        # log
        logger.log([k*dt, float(x[0,0]), float(x[1,0]), float(x[2,0]), float(x[3,0]),
                    float(x[4,0]), float(x[5,0]), float(x[6,0]), float(u0[0,0]), float(u0[1,0]), float(u0[2,0])])

    csv_out = logger.save_csv()
    png1, png2 = plot_attitude_rollout(csv_out, os.path.join(out_dir, "attitude_tv_rollout.png"))
    print("Saved:", csv_out, png1, png2)

if __name__ == "__main__":
    main()
