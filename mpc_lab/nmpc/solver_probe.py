import casadi as ca

def make_solver(problem_dict, opts=None):
    """Probe available NLP solvers in a preferred order and return (solver, plugin_name)."""
    opts = {} if opts is None else dict(opts)
    for plugin in ["ipopt", "fatrop", "sqpmethod"]:
        try:
            solver = ca.nlpsol("solver", plugin, problem_dict, opts)
            return solver, plugin
        except Exception:
            continue
    raise RuntimeError("No suitable NLP solver plugin available.")
