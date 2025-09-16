import numpy as np

def rollout(model, controller, x0: np.ndarray, steps: int, ref_fun=None):
    nx, nu = model.nx, model.nu
    X = np.zeros((steps+1, nx)); U = np.zeros((steps, nu))
    X[0] = x0
    for k in range(steps):
        ref = None if ref_fun is None else ref_fun(k)
        u, info = controller.compute_control(X[k], k*model.dt, ref=ref)
        U[k] = u
        X[k+1] = model.f_disc(X[k], U[k])
    return X, U
