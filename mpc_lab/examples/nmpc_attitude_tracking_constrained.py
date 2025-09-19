
import casadi as ca
from mpc_lab.models.attitude_quat import AttitudeQuatModel
from mpc_lab.nmpc.build_attitude_tv import build_attitude_nmpc_tv
from mpc_lab.refs.attitude import attitude_yaw_trapezoid
from mpc_lab.solvers.warmstarts import shift_u
from mpc_lab.plotting.attitude import plot_omega_with_bounds, plot_torque_with_bounds

def main():
    dt = 0.1; N = 20
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    bounds = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0}
    weights = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    nmpc = build_attitude_nmpc_tv(model, N, weights=weights, bounds=bounds)
    yaw_traj = attitude_yaw_trapezoid(dt, total_time=5.0, yaw_final_deg=45.0, max_rate_dps=30.0, max_accel_dps2=60.0)

    x = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))
    T = 40
    Uws = None
    t, wx, wy, wz, tx, ty, tz = [], [], [], [], [], [], []

    for k in range(T):
        qref_seq, wref_seq = yaw_traj(k*dt, N)
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, qref_seq, wref_seq, u_init=Uws)
        t.append(k*dt)
        wx.append(float(x[4,0])); wy.append(float(x[5,0])); wz.append(float(x[6,0]))
        tx.append(float(u0[0,0])); ty.append(float(u0[1,0])); tz.append(float(u0[2,0]))
        x = model.step(x, u0)
        Uws = shift_u(Uopt)

    plot_omega_with_bounds(t, wx, wy, wz, wmax=bounds["w_abs_max"], show=True)
    plot_torque_with_bounds(t, tx, ty, tz, umin=bounds["u_min"], umax=bounds["u_max"], show=True)

if __name__ == "__main__":
    main()
