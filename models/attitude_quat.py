import casadi as ca

def _skew(v: ca.MX) -> ca.MX:
    """Return skew-symmetric matrix of v (3x1) -> (3x3)."""
    v1, v2, v3 = v[0, 0], v[1, 0], v[2, 0]
    return ca.vertcat(
        ca.horzcat(ca.DM(0), -v3,      v2),
        ca.horzcat(     v3 , ca.DM(0), -v1),
        ca.horzcat(    -v2 ,     v1 ,  ca.DM(0))
    )

def _omega_mat(w: ca.MX) -> ca.MX:
    """Quaternion kinematics matrix Omega(ω) for scalar-first quaternion.
    w: (3,1) -> returns (4,4)
    """
    wx, wy, wz = w[0,0], w[1,0], w[2,0]
    return ca.vertcat(
        ca.horzcat( ca.DM(0), -wx,     -wy,     -wz),
        ca.horzcat(      wx , ca.DM(0),  wz,     -wy),
        ca.horzcat(      wy ,    -wz , ca.DM(0),  wx),
        ca.horzcat(      wz ,     wy ,   -wx , ca.DM(0))
    )

class AttitudeQuatModel:
    """Rigid-body attitude model with quaternion q (scalar-first) and body rates ω.

    State x = [q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z]^T (7x1)
    Input u = τ (3x1) body torque
    """
    nx: int = 7
    nu: int = 3

    def __init__(self, dt: float, J_diag=(0.05, 0.06, 0.07)):
        self.dt = float(dt)
        Jx, Jy, Jz = J_diag
        self.J  = ca.DM([[Jx, 0, 0],[0, Jy, 0],[0, 0, Jz]])
        self.Ji = ca.DM([[1.0/Jx, 0, 0],[0, 1.0/Jy, 0],[0, 0, 1.0/Jz]])

    def f(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Continuous-time dynamics xdot = f(x,u). Shapes: x(7,1), u(3,1)."""
        q = x[0:4, :]
        w = x[4:7, :]
        qdot = 0.5 * (_omega_mat(w) @ q)
        wdot = self.Ji @ (u - _skew(w) @ (self.J @ w))
        return ca.vertcat(qdot, wdot)

    def step(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Discrete RK4 step with in-place quaternion renormalization. Shapes preserved."""
        h = self.dt
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5*h*k1, u)
        k3 = self.f(x + 0.5*h*k2, u)
        k4 = self.f(x + h*k3, u)
        x_next = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # Normalize quaternion
        q_next = x_next[0:4, :]
        nrm = ca.norm_2(q_next)
        q_next = q_next / ca.fmax(nrm, 1e-8)
        return ca.vertcat(q_next, x_next[4:7, :])
