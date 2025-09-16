import numpy as np
import pytest
casadi = pytest.importorskip("casadi")

from mpc_lab.modeling.diffdrive import DiffDriveModel
from mpc_lab.controllers.mpc_linear import MPCLinear

def test_linear_mpc_runs_one_step():
    dt = 0.1
    model = DiffDriveModel(dt=dt)
    N = 10
    Q = np.eye(model.nx)
    R = 0.1*np.eye(model.nu)
    ctrl = MPCLinear(model, N=N, Q=Q, R=R, solver="qpoases")
    ctrl.set_bounds(x_bounds=(np.array([-10,-10,-np.pi]), np.array([10,10,np.pi])),
                    u_bounds=(np.array([-1.0,-1.0]), np.array([1.0,1.0])))
    xr = np.tile(np.array([1.0, 1.0, 0.0]), (N+1,1))
    ur = np.zeros((N, model.nu))
    ctrl.set_references(xr, ur)
    ctrl.build()
    x0 = np.zeros(model.nx)
    u, info = ctrl.compute_control(x0, 0.0)
    assert u.shape == (model.nu,)
