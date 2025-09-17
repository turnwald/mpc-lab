import casadi as ca

def attitude_cost_sign_invariant(q: ca.MX, qref: ca.MX) -> ca.MX:
    """Return 1 - (q^T qref)^2, invariant to q ≡ -q."""
    c = ca.dot(q, qref)
    return 1.0 - (c*c)

def du_penalty(u_k: ca.MX, u_km1: ca.MX, w: float) -> ca.MX:
    """Weighted squared input rate cost."""
    du = u_k - u_km1
    return w * ca.dot(du, du)
