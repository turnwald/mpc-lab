import math
import casadi as ca
from mpc_lab.models.attitude_quat import AttitudeQuatModel
from mpc_lab.nmpc.build_attitude import build_attitude_nmpc

def test_attitude_smoke():
    dt = 0.1
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    N = 10
    nmpc = build_attitude_nmpc(model, N,
                               weights={"wq": 20.0, "wqN": 50.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3},
                               bounds={"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0})
    x0 = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))
    ang = math.radians(20.0) * 0.5
    qref = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
    wref = ca.DM([0,0,0]).reshape((3,1))
    u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x0, qref, wref)
    # Basic sanity checks
    assert Xopt.shape == (7, N+1)
    assert Uopt.shape == (3, N)
    assert float(ca.norm_2(Xopt[0:4, -1])) == 1.0  # quaternion normalized
