import numpy as np
from mpc_lab.modeling.diffdrive import DiffDriveModel
from mpc_lab.controllers.mpc_linear import MPCLinear

def make_refs(N, nx, nu, goal):
    xr = np.tile(goal, (N+1,1))
    ur = np.zeros((N,nu))
    return xr, ur

def main():
    dt = 0.1
    model = DiffDriveModel(dt=dt)
    N = 20
    Q = np.diag([10.0, 10.0, 2.0])
    R = np.diag([0.1, 0.1])
    P = np.diag([20.0, 20.0, 3.0])
    ctrl = MPCLinear(model, N=N, Q=Q, R=R, P=P, solver="qpoases")
    ctrl.set_bounds(x_bounds=(np.array([-10,-10,-np.pi]), np.array([10,10,np.pi])),
                    u_bounds=(np.array([-1.0,-1.0]), np.array([1.0,1.0])))

    x0 = np.array([0.0, 0.0, 0.0])
    goal = np.array([2.0, 2.0, 0.0])
    xr, ur = make_refs(N, model.nx, model.nu, goal)
    ctrl.set_references(xr, ur)
    ctrl.build()

    steps = 60
    X = [x0]; U = []
    xk = x0.copy()
    for k in range(steps):
        xr, ur = make_refs(N, model.nx, model.nu, goal)
        u, info = ctrl.compute_control(xk, k*dt, ref={"xr": xr, "ur": ur})
        xk = model.f_disc(xk, u)
        X.append(xk); U.append(u)
    X = np.array(X); U = np.array(U)
    print("Final state:", X[-1])
    print("First 5 controls:\\n", U[:5])

if __name__ == "__main__":
    main()
