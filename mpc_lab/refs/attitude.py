
import math
import casadi as ca

def attitude_yaw_trapezoid(dt: float, total_time: float,
                           yaw_final_deg: float, max_rate_dps: float, max_accel_dps2: float):
    yaw_f = math.radians(yaw_final_deg)
    w_max = math.radians(max_rate_dps)
    a_max = math.radians(max_accel_dps2)

    t_acc = w_max / max(a_max, 1e-9)
    s_acc = 0.5 * a_max * t_acc**2
    if 2*s_acc >= abs(yaw_f):
        t_acc = math.sqrt(abs(yaw_f)/max(a_max, 1e-9))
        t_flat = 0.0
        w_max_eff = a_max * t_acc
    else:
        rem = abs(yaw_f) - 2*s_acc
        t_flat = rem / max(w_max, 1e-9)
        w_max_eff = w_max
    sign = 1.0 if yaw_f >= 0 else -1.0

    def yaw_rate(t):
        if t < 0: return 0.0
        if t < t_acc: return sign * a_max * t
        if t < t_acc + t_flat: return sign * w_max_eff
        if t < 2*t_acc + t_flat: return sign * (w_max_eff - a_max*(t - t_acc - t_flat))
        return 0.0

    def yaw_angle(t):
        if t <= 0: return 0.0
        if t <= t_acc:
            return 0.5*sign*a_max*t*t
        if t <= t_acc + t_flat:
            return sign*(0.5*a_max*t_acc*t_acc + w_max_eff*(t - t_acc))
        if t <= 2*t_acc + t_flat:
            tau = t - (t_acc + t_flat)
            return sign*(0.5*a_max*t_acc*t_acc + w_max_eff*t_flat + w_max_eff*tau - 0.5*a_max*tau*tau)
        return yaw_f

    def window(t0: float, N: int):
        qs, ws = [], []
        for k in range(N+1):
            t = t0 + k*dt
            yaw = yaw_angle(t)
            wz  = yaw_rate(t)
            half = 0.5*yaw
            q = ca.DM([math.cos(half), 0.0, 0.0, math.sin(half)]).reshape((4,1))
            w = ca.DM([0.0, 0.0, wz]).reshape((3,1))
            qs.append(q); ws.append(w)
        return ca.hcat(qs), ca.hcat(ws)

    return window
