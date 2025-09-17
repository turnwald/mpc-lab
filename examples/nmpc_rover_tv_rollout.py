import os
import casadi as ca
from mpc_lab.models.rover_unicycle import RoverUnicycleModel
from mpc_lab.nmpc.build_rover_tv import build_rover_nmpc_tv
from mpc_lab.utils.trajectory import rover_line_trajectory
from mpc_lab.utils.logging import RolloutLogger
from mpc_lab.utils.plotting import plot_rover_rollout

def shift_warm(Wopt: ca.DM, nx: int, nu: int, N: int) -> ca.DM:
    X = ca.reshape(Wopt[0:nx*(N+1)], nx, N+1)
    U = ca.reshape(Wopt[nx*(N+1):], nu, N)
    if N > 1:
        U[:, 0:N-1] = U[:, 1:N]
    U[:, N-1] = U[:, N-1]
    return ca.vertcat(ca.reshape(X, nx*(N+1),1), ca.reshape(U, nu*N,1))

def main():
    dt = 0.1
    model_ctrl = RoverUnicycleModel(dt=dt)
    # Plant with actuation gain error and small bias (model mismatch + disturbance)
    class RoverPlant(RoverUnicycleModel):
        def step(self, x, u):
            # gain error on v and omega, plus small constant bias
            u_plant = ca.vertcat(1.1*u[0,0] + 0.02, 0.95*u[1,0] - 0.01).reshape((2,1))
            return super().step(x, u_plant)
    model_plant = RoverPlant(dt=dt)

    N = 25
    nmpc = build_rover_nmpc_tv(model_ctrl, N,
                               weights={"w_pos": 12.0, "w_heading": 1.0, "wu": 5e-3, "wdu": 1e-3},
                               bounds={"v_min": -0.6, "v_max": 1.0, "om_min": -1.5, "om_max": 1.5})
    total_time = 8.0
    T = int(round(total_time / dt))
    Xref_seq = rover_line_trajectory(total_time, dt, x_goal=4.0, y_goal=1.0, th_goal=0.0)

    # Initial
    x = ca.DM([0,0,0]).reshape((3,1))

    out_dir = os.path.join("/mnt/data", "mpc_lab_outputs")
    log_path = os.path.join(out_dir, "rover_tv_rollout.csv")
    logger = RolloutLogger(log_path, header=["t","x","y","theta","v","omega"])

    W0 = None
    for k in range(T):
        Xref_h = Xref_seq[:, k:k+N+1] if k+N <= Xref_seq.size2()-1 else Xref_seq[:, -N-1:]
        u0, Xopt, Uopt, Wopt, sol = nmpc["solve_one_step"](x, Xref_h, W0=W0)
        x = model_plant.step(x, u0)
        W0 = shift_warm(Wopt, nmpc["nx"], nmpc["nu"], nmpc["N"])
        logger.log([k*dt, float(x[0,0]), float(x[1,0]), float(x[2,0]), float(u0[0,0]), float(u0[1,0])])

    csv_out = logger.save_csv()
    png1, png2 = plot_rover_rollout(csv_out, os.path.join(out_dir, "rover_tv_rollout.png"))
    print("Saved:", csv_out, png1, png2)

if __name__ == "__main__":
    main()
