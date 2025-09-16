"""Phase 1 smoke example: build mock linear MPC and compute one control."""
import numpy as np
from mpc_lab.controllers.mpc_linear_mock import MPCLinearMock, DummyLinearizationProvider

def main():
    nx, nu, N = 3, 2, 10
    ctrl = MPCLinearMock(nx=nx, nu=nu, N=N)
    ctrl.set_linearizer(DummyLinearizationProvider(nx, nu, N))
    ctrl.build()
    x0 = np.zeros(nx)
    u, info = ctrl.compute_control(x0, t_now=0.0)
    print("u_cmd:", u, "info:", info)

if __name__ == "__main__":
    main()
