import numpy as np
from typing import Tuple

class DiffDriveModel:
    """Discrete kinematic diff-drive model.
    State x = [x, y, theta], input u = [v, omega].
    x_{k+1} = x_k + dt * [ v*cos(theta), v*sin(theta), omega ]
    """
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.nx, self.nu = 3, 2

    def f_disc(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x0, y, th = float(x[0]), float(x[1]), float(x[2])
        v, om   = float(u[0]), float(u[1])
        dt = self.dt
        return np.array([
            x0 + dt * v * np.cos(th),
            y  + dt * v * np.sin(th),
            th + dt * om
        ], dtype=float)

    def jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        th = float(x[2]); v = float(u[0]); dt = self.dt
        A = np.eye(3)
        A[0,2] = -dt * v * np.sin(th)
        A[1,2] =  dt * v * np.cos(th)
        B = np.zeros((3,2))
        B[0,0] = dt * np.cos(th)
        B[1,0] = dt * np.sin(th)
        B[2,1] = dt
        return A, B
