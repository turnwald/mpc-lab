import math
import casadi as ca

def apply_state_bounds(lbw, ubw, nx, N, lbx_vec=None, ubx_vec=None):
    """Apply per-state bounds across all stages to variable bounds arrays."""
    if lbx_vec is not None:
        lbx_vec = ca.DM(lbx_vec).reshape((nx,1))
    if ubx_vec is not None:
        ubx_vec = ca.DM(ubx_vec).reshape((nx,1))
    for k in range(N+1):
        for i in range(nx):
            idx = i + nx*k
            if lbx_vec is not None:
                v = float(lbx_vec[i,0])
                if not math.isinf(v):
                    lbw[idx] = v
            if ubx_vec is not None:
                v = float(ubx_vec[i,0])
                if not math.isinf(v):
                    ubw[idx] = v

def apply_input_bounds(lbw, ubw, u_off, nu, N, umin_vec, umax_vec):
    """Apply per-input bounds across all control stages."""
    if not isinstance(umin_vec, (list, tuple)):
        umin_vec = [umin_vec]*nu
    if not isinstance(umax_vec, (list, tuple)):
        umax_vec = [umax_vec]*nu
    for k in range(N):
        for i in range(nu):
            idx = u_off + i + nu*k
            umin = float(umin_vec[i]); umax = float(umax_vec[i])
            if not math.isinf(umin):
                lbw[idx] = umin
            if not math.isinf(umax):
                ubw[idx] = umax
