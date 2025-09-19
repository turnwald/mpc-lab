
import casadi as ca
from mpc_lab.models.rover_unicycle import RoverUnicycleModel
from mpc_lab.nmpc.build_rover_tv import build_rover_nmpc_tv
from mpc_lab.refs.rover import rover_waypoints_constant_speed
from mpc_lab.plotting.rover import plot_path_with_constraints, plot_inputs_with_bounds
from mpc_lab.solvers.warmstarts import shift_u

def main():
    dt = 0.1; N = 25
    model = RoverUnicycleModel(dt=dt)
    weights = {"w_pos": 8.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3}
    bounds = {"v_min": -0.5, "v_max": 1.0, "om_min": -1.5, "om_max": 1.5}
    nmpc = build_rover_nmpc_tv(model, N, weights=weights, bounds=bounds)

    waypoints = [(0,0), (1.5,0.2), (3.0,0.0)]
    traj = rover_waypoints_constant_speed(dt, waypoints, v_des=0.6)

    x = ca.DM([0, 0, 0]).reshape((3,1))
    T = 60
    Uws = None
    t, xs, ys, ths, vs, oms = [], [], [], [], [], []

    for k in range(T):
        xref_seq = traj(k*dt, N)
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, xref_seq, u_init=Uws)
        t.append(k*dt); xs.append(float(x[0,0])); ys.append(float(x[1,0])); ths.append(float(x[2,0]))
        vs.append(float(u0[0,0])); oms.append(float(u0[1,0]))
        x = model.step(x, u0)
        Uws = shift_u(Uopt)

    plot_path_with_constraints(xs, ys, waypoints=waypoints,
                               box=(-1,4,-1,1), corridor={"ylim": (-1,1)}, show=True)
    plot_inputs_with_bounds(t, vs, oms, vmin=bounds["v_min"], vmax=bounds["v_max"],
                            omin=bounds["om_min"], omax=bounds["om_max"], show=True)

if __name__ == "__main__":
    main()
