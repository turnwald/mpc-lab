"""Closed-loop rollout for rover NMPC with warm-start."""
import casadi as ca
from mpc_lab.models.rover_unicycle import RoverUnicycleModel
from mpc_lab.nmpc.build_rover import build_rover_nmpc

def shift_u_warm_start(Uopt: ca.DM) -> ca.DM:
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
    model = RoverUnicycleModel(dt=dt)
    N = 25
    weights = {"w_pos": 8.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3}
    bounds  = {"v_min": -0.5, "v_max": 1.0, "om_min": -1.5, "om_max": 1.5}
    nmpc = build_rover_nmpc(model, N, weights=weights, bounds=bounds)

    # Start pose
    x = ca.DM([0, 0, 0]).reshape((3,1))
    # Goal
    xref, yref, thref = 3.0, 0.0, 0.0

    T = 60
    Uws = None
    for t in range(T):
        u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x, xref, yref, thref, u_init=Uws)
        x = model.step(x, u0)
        Uws = shift_u_warm_start(Uopt)

    print("Final state:", x.T)

if __name__ == "__main__":
    main()
