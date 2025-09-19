
import casadi as ca

def shift_u(Uopt: ca.DM) -> ca.DM:
    nu, N = int(Uopt.size1()), int(Uopt.size2())
    Us = ca.DM(nu, N)
    if N > 1:
        Us[:, 0:N-1] = Uopt[:, 1:N]
    Us[:, N-1] = Uopt[:, N-1] if N>0 else 0
    return Us
