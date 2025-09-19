
import casadi as ca

def step_closed_loop(nmpc, model, x: ca.DM, refs: tuple, u_ws=None):
    if isinstance(refs, tuple) and len(refs) == 2:
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, refs[0], refs[1], u_init=u_ws)
    else:
        u0, Xopt, Uopt, _ = nmpc["solve_one_step"](x, refs[0], u_init=u_ws)
    x_next = model.step(x, u0)
    return x_next, u0, Xopt, Uopt
