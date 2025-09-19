# examples/nmpc_rover_tracking_constrained.py
# Run from repo root:
#   python examples/nmpc_rover_tracking_constrained.py


import casadi as ca
import matplotlib.pyplot as plt
import inspect

from mpc_lab.models.rover_unicycle import RoverUnicycleModel
from mpc_lab.nmpc.build_rover_tv import build_rover_nmpc_tv

# Prefer improved waypoint reference if available
try:
    from mpc_lab.utils.refs import rover_waypoints_constant_speed
except Exception:
    rover_waypoints_constant_speed = None

def _fallback_rover_line_to(dt: float, waypoints, v_des: float):
    """Hold last waypoint as constant pose target with zero heading."""
    last = waypoints[-1]
    def window(t0: float, N: int):
        xr = ca.DM([last[0], last[1], 0.0]).reshape((3,1))
        return ca.repmat(xr, 1, N+1)
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
    N  = 25
    model = RoverUnicycleModel(dt=dt)

    # Input bounds
    bounds = {"v_min": -0.5, "v_max": 1.0, "om_min": -1.5, "om_max": 1.5}
    weights = {"w_pos": 8.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3}

    # State constraints
    state_box = {"lbx": [-1, -1, -ca.inf], "ubx": [4, 1, ca.inf]}
    H = ca.DM([[0,  1, 0],   # +y <= 1
               [0, -1, 0]])  # -y <= 1
    h = ca.DM([[1],[1]])
    linear_state_ineq = {"H": H, "h": h}

    # Build NMPC, passing state constraints only if supported by this builder version
    build_sig = inspect.signature(build_rover_nmpc_tv)
    kwargs = dict(weights=weights, bounds=bounds)
    if "state_box" in build_sig.parameters:
        kwargs["state_box"] = state_box
    else:
        print("[info] build_rover_nmpc_tv has no 'state_box' parameter; skipping box enforcement.")
    if "linear_state_ineq" in build_sig.parameters:
        kwargs["linear_state_ineq"] = linear_state_ineq
    else:
        print("[info] build_rover_nmpc_tv has no 'linear_state_ineq' parameter; skipping corridor enforcement.")

    nmpc = build_rover_nmpc_tv(model, N, **kwargs)

    # Time-varying waypoint reference (or fallback)
    waypoints = [(0,0), (1.5,0.2), (3.0,0.0)]
    if rover_waypoints_constant_speed is not None:
        traj = rover_waypoints_constant_speed(dt, waypoints, v_des=0.6)
    else:
        traj = _fallback_rover_line_to(dt, waypoints, v_des=0.6)

    # Closed-loop rollout
    x = ca.DM([0, 0, 0]).reshape((3,1))
    T = 60
    Uws = None
    times, X_hist, U_hist = [], [], []

    for t in range(T):
        xref_seq = traj(t*dt, N)
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, xref_seq, u_init=Uws)
        times.append(t*dt)
        X_hist.append(x)
        U_hist.append(u0)
        x = model.step(x, u0)
        Uws = shift_u_warm_start(Uopt)

    # Plots with constraint overlays
    t = times
    xs = [float(xk[0,0]) for xk in X_hist]
    ys = [float(xk[1,0]) for xk in X_hist]
    th= [float(xk[2,0]) for xk in X_hist]
    v = [float(u[0,0]) for u in U_hist]
    om= [float(u[1,0]) for u in U_hist]

    # Path with corridor and box overlays
    plt.figure()
    plt.plot(xs, ys, label="path")
    xmin, xmax = min(xs)-0.5, max(xs)+0.5
    # Corridor lines y=±1
    plt.plot([xmin, xmax], [ 1,  1], '--', label='corridor y=±1')
    plt.plot([xmin, xmax], [-1, -1], '--')
    # State box rectangle: [-1,4] x [-1,1]
    box_x = [-1, 4, 4, -1, -1]
    box_y = [-1, -1, 1, 1, -1]
    plt.plot(box_x, box_y, ':', label='state box')
    # Waypoints
    wx, wy = zip(*waypoints)
    plt.plot(wx, wy, 'o', label='waypoints')
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Rover path with constraints")
    plt.legend()

    # Heading
    plt.figure()
    plt.plot(t, th)
    plt.xlabel("time [s]"); plt.ylabel("theta [rad]"); plt.title("Heading")

    # Inputs with bounds
    plt.figure()
    plt.plot(t, v,  label='v')
    plt.plot(t, [bounds["v_max"]]*len(t), '--', label='v ≤ vmax')
    plt.plot(t, [bounds["v_min"]]*len(t), '--', label='v ≥ vmin')
    plt.xlabel("time [s]"); plt.ylabel("v [m/s]"); plt.title("Linear speed with bounds")
    plt.legend()

    plt.figure()
    plt.plot(t, om, label='omega')
    plt.plot(t, [bounds["om_max"]]*len(t), '--', label='ω ≤ ωmax')
    plt.plot(t, [bounds["om_min"]]*len(t), '--', label='ω ≥ ωmin')
    plt.xlabel("time [s]"); plt.ylabel("omega [rad/s]"); plt.title("Angular speed with bounds")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
