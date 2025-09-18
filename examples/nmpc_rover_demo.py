import casadi as ca
from models.rover_unicycle import RoverUnicycleModel
from nmpc.build_rover import build_rover_nmpc

def main():
    dt = 0.1
    model = RoverUnicycleModel(dt=dt)
    N = 30

    weights = {"w_pos": 10.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3}
    bounds  = {"v_min": -0.5, "v_max": 0.8, "om_min": -1.5, "om_max": 1.5}
    nmpc = build_rover_nmpc(model, N, weights=weights, bounds=bounds)

    # Start at origin, point right, go to (2, 0) heading 0
    x0 = ca.DM([0, 0, 0]).reshape((3,1))
    xref, yref, thref = 2.0, 0.0, 0.0

    u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x0, xref, yref, thref)
    print("Plugin:", nmpc["plugin"])
    print("u0:", u0)
    print("x_N:", Xopt[:, -1])

if __name__ == "__main__":
    main()
