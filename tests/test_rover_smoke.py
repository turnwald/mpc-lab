import casadi as ca
from mpc_lab.models.rover_unicycle import RoverUnicycleModel
from mpc_lab.nmpc.build_rover import build_rover_nmpc

def test_rover_smoke():
    dt = 0.1
    model = RoverUnicycleModel(dt=dt)
    N = 15
    nmpc = build_rover_nmpc(model, N,
                            weights={"w_pos": 5.0, "w_heading": 0.5, "wu": 1e-2, "wdu": 1e-3},
                            bounds={"v_min": -0.5, "v_max": 0.8, "om_min": -1.5, "om_max": 1.5})
    x0 = ca.DM([0, 0, 0]).reshape((3,1))
    xref, yref, thref = 1.0, 0.0, 0.0
    u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x0, xref, yref, thref)
    assert Xopt.shape == (3, N+1)
    assert Uopt.shape == (2, N)
