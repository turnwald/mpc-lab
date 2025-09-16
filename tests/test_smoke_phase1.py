import numpy as np
from mpc_lab.controllers.mpc_linear_mock import MPCLinearMock, DummyLinearizationProvider

def test_smoke_mock_linear_mpc():
    nx, nu, N = 3, 2, 8
    ctrl = MPCLinearMock(nx=nx, nu=nu, N=N)
    ctrl.set_linearizer(DummyLinearizationProvider(nx, nu, N))
    ctrl.build()
    u, info = ctrl.compute_control(np.zeros(nx), 0.0)
    assert u.shape == (nu,)
    assert 'status' in info and 'obj' in info
    assert info['status'] in ("Solve_Success","Success")
