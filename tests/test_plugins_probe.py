import casadi as ca
from mpc_lab.nmpc.solver_probe import make_solver

def test_plugins_probe():
    # Trivial NLP: min 0 s.t. x == 0
    x = ca.MX.sym("x", 1, 1)
    g = x
    f = ca.MX(0)
    prob = {"x": x, "f": f, "g": g}
    solver, plugin = make_solver(prob, {"ipopt.print_level": 0, "print_time": 0})
    sol = solver(x0=ca.DM([1]), lbg=ca.DM([0]), ubg=ca.DM([0]))
    assert plugin in ["ipopt", "fatrop", "sqpmethod"]
    assert float(sol["x"][0]) == 0.0
