import casadi as ca

class RoverUnicycleModel:
    """Unicycle kinematics model for a simple rover.

    State x = [X, Y, Theta]^T (3x1)
    Input u = [v, omega]^T (2x1)
    """
    nx: int = 3
    nu: int = 2

    def __init__(self, dt: float):
        self.dt = float(dt)

    def f(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Continuous-time kinematics."""
        X = x[0, 0]
        Y = x[1, 0]
        Th = x[2, 0]
        v = u[0, 0]
        om = u[1, 0]
        xdot = ca.vertcat(v*ca.cos(Th), v*ca.sin(Th), om)
        return xdot

    def step(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Discrete RK4 step to keep consistency with attitude model."""
        h = self.dt
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5*h*k1, u)
        k3 = self.f(x + 0.5*h*k2, u)
        k4 = self.f(x + h*k3, u)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
