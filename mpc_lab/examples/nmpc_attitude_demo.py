import math
import casadi as ca

from nmpc.build_attitude import build_attitude_nmpc
from models.attitude_quat import AttitudeQuatModel

def main():
    dt = 0.1
    model = AttitudeQuatModel(dt=dt, J_diag=(0.05, 0.06, 0.07))
    N = 20

    weights = {"wq": 50.0, "wqN": 100.0, "ww": 1.0, "wu": 1e-2, "wdu": 1e-3}
    bounds  = {"u_min": -0.2, "u_max": 0.2, "w_abs_max": 2.0}
    nmpc = build_attitude_nmpc(model, N, weights=weights, bounds=bounds)

    x0 = ca.DM([1,0,0,0, 0,0,0]).reshape((7,1))
    ang = math.radians(30.0) * 0.5
    qref = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
    wref = ca.DM([0,0,0]).reshape((3,1))

    u0, Xopt, Uopt, sol = nmpc["solve_one_step"](x0, qref, wref)
    print("Plugin:", nmpc["plugin"])
    print("u0:", u0)
    print("q_N:", Xopt[0:4, -1])
    print("omega_N:", Xopt[4:7, -1])

if __name__ == "__main__":
    main()
