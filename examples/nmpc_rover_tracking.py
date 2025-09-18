"""Rover tracking with time-varying references + plotting (constant target demo)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import casadi as ca
from models.rover_unicycle import RoverUnicycleModel
from nmpc.build_rover_tv import build_rover_nmpc_tv
from utils.refs import rover_line_to
from utils.plotting import plot_rover_rollout

def shift_u_warm_start(Uopt: ca.DM) -> ca.DM:
    nu, N = int(Uopt.size1()), int(Uopt.size2())
    Us = ca.DM(nu, N)
    if N > 1:
        Us[:, 0:N-1] = Uopt[:, 1:N]
    Us[:, N-1] = Uopt[:, N-1] if N>0 else 0
    return Us

def main():
    dt = 0.1
    model = RoverUnicycleModel(dt=dt)
    N = 25
    nmpc = build_rover_nmpc_tv(model, N,
                               weights={"w_pos": 8.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3},
                               bounds={"v_min": -0.5, "v_max": 1.0, "om_min": -1.5, "om_max": 1.5})

    x = ca.DM([0, 0, 0]).reshape((3,1))
    T = 60
    xref_full = rover_line_to(dt, 1000, 3.0, 0.0, 0.0)

    Uws = None
    times, X_hist, U_hist = [], [], []

    for t in range(T):
        xref_seq = xref_full[:, 0:N+1]
        u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x, xref_seq, u_init=Uws)

        times.append(t*dt)
        X_hist.append(x)
        U_hist.append(u0)

        x = model.step(x, u0)
        Uws = shift_u_warm_start(Uopt)

    plot_rover_rollout(times, X_hist, U_hist)

if __name__ == "__main__":
    main()
