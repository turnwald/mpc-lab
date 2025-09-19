
import casadi as ca

def add_stagewise_ineq(g_ineq_list, X, H, h, N, slc=None):
    H = ca.DM(H)
    h = ca.DM(h).reshape((-1,1))
    for k in range(N+1):
        xk = X[:, k]
        if slc is not None:
            xk = xk[slc[0]:slc[1], :]
        g_ineq_list.append(H @ xk - h)
