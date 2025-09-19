import math
import casadi as ca

def attitude_yaw_slew(dt: float, horizon_steps: int, yaw_deg_final: float):
    """Generate (qref_seq, wref_seq) for a yaw slew to a fixed target yaw.
    Currently constant across the horizon; easy to extend to profiles.
    """
    ang = math.radians(yaw_deg_final) * 0.5
    q = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
    qref_seq = ca.repmat(q, 1, horizon_steps+1)
    wref_seq = ca.DM.zeros(3, horizon_steps+1)
    return qref_seq, wref_seq

def rover_line_to(dt: float, horizon_steps: int, x_goal: float, y_goal: float, th_goal: float):
    """Generate a constant target pose over the horizon (placeholder for path/waypoints)."""
    xr = ca.DM([x_goal, y_goal, th_goal]).reshape((3,1))
    xref_seq = ca.repmat(xr, 1, horizon_steps+1)
    return xref_seq


def attitude_yaw_trapezoid(dt: float, total_time: float,
                           yaw_final_deg: float, max_rate_dps: float, max_accel_dps2: float):
    """
    Returns callables that give horizon windows:
      window(t0, N) -> (qref_seq: (4,N+1), wref_seq: (3,N+1))
    Slew about body z with trapezoidal yaw rate profile.
    """
    yaw_f = math.radians(yaw_final_deg)
    w_max = math.radians(max_rate_dps)
    a_max = math.radians(max_accel_dps2)

    # Compute trapezoid durations
    t_acc = w_max / a_max
    s_acc = 0.5 * a_max * t_acc**2
    if 2*s_acc >= abs(yaw_f):  # triangle profile
        t_acc = math.sqrt(abs(yaw_f)/a_max)
        t_flat = 0.0
        w_max_eff = a_max * t_acc
    else:
        rem = abs(yaw_f) - 2*s_acc
        t_flat = rem / w_max
        w_max_eff = w_max
    sign = 1.0 if yaw_f >= 0 else -1.0

    def yaw_rate(t):
        if t < 0: return 0.0
        if t < t_acc: return sign * a_max * t
        if t < t_acc + t_flat: return sign * w_max_eff
        if t < 2*t_acc + t_flat: return sign * (w_max_eff - a_max*(t - t_acc - t_flat))
        return 0.0

    def yaw_angle(t):
        # integrate piecewise analytically
        if t <= 0: return 0.0
        if t <= t_acc:
            return 0.5*sign*a_max*t**2
        if t <= t_acc + t_flat:
            return sign*(0.5*a_max*t_acc**2 + w_max_eff*(t - t_acc))
        if t <= 2*t_acc + t_flat:
            tau = t - (t_acc + t_flat)
            return sign*(0.5*a_max*t_acc**2 + w_max_eff*t_flat + w_max_eff*tau - 0.5*a_max*tau**2)
        return yaw_f

    def window(t0: float, N: int):
        qs, ws = [], []
        for k in range(N+1):
            t = t0 + k*dt
            yaw = yaw_angle(t)
            wy  = 0.0  # only z-rate
            wx  = 0.0
            wz  = yaw_rate(t)
            # scalar-first quaternion for yaw about z
            half = 0.5*yaw
            q = ca.DM([math.cos(half), 0.0, 0.0, math.sin(half)]).reshape((4,1))
            w = ca.DM([wx, wy, wz]).reshape((3,1))
            qs.append(q); ws.append(w)
        qref_seq = ca.hcat(qs)
        wref_seq = ca.hcat(ws)
        return qref_seq, wref_seq

    return window


def rover_waypoints_constant_speed(dt: float, waypoints: list, v_des: float):
    """
    piecewise-linear path through waypoints: [(x0,y0),(x1,y1),...].
    Heading = segment direction. Uniform time at speed v_des.

    Returns: window(t0, N) -> xref_seq (3, N+1)
    """
    # Precompute segment lengths and times
    segs = []
    t_cum = [0.0]
    for i in range(len(waypoints)-1):
        x0,y0 = waypoints[i]
        x1,y1 = waypoints[i+1]
        dx, dy = x1-x0, y1-y0
        L = math.hypot(dx, dy)
        T = L / max(v_des, 1e-6)
        th = math.atan2(dy, dx) if L>0 else (segs[-1][4] if segs else 0.0)
        segs.append((x0,y0,x1,y1,th,L,T))
        t_cum.append(t_cum[-1] + T)
    total_T = t_cum[-1]

    def sample(t):
        if t <= 0.0: x,y,th = waypoints[0][0], waypoints[0][1], (segs[0][4] if segs else 0.0); return x,y,th
        if t >= total_T: x,y = waypoints[-1]; th = (segs[-1][4] if segs else 0.0); return x,y,th
        # find segment
        for i in range(len(segs)):
            if t_cum[i] <= t <= t_cum[i+1]:
                t0 = t_cum[i]
                x0,y0,x1,y1,th,L,T = segs[i]
                s = (t - t0)/max(T, 1e-9)
                x = x0 + s*(x1-x0); y = y0 + s*(y1-y0)
                return x,y,th
        return waypoints[-1][0], waypoints[-1][1], (segs[-1][4] if segs else 0.0)

    def window(t0: float, N: int):
        xs = []
        for k in range(N+1):
            t = t0 + k*dt
            x,y,th = sample(t)
            xs.append(ca.DM([x,y,th]).reshape((3,1)))
        return ca.hcat(xs)

    return window
