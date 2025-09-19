
import casadi as ca
import math

def rover_waypoints_constant_speed(dt: float, waypoints: list, v_des: float):
    segs = []
    t_cum = [0.0]
    for i in range(len(waypoints)-1):
        x0,y0 = waypoints[i]
        x1,y1 = waypoints[i+1]
        dx, dy = x1-x0, y1-y0
        L = math.hypot(dx, dy)
        T = L / max(v_des, 1e-9)
        th = math.atan2(dy, dx) if L>0 else (segs[-1][4] if segs else 0.0)
        segs.append((x0,y0,x1,y1,th,L,T))
        t_cum.append(t_cum[-1] + T)
    total_T = t_cum[-1] if segs else 0.0

    def sample(t):
        if not segs or t <= 0.0:
            th0 = segs[0][4] if segs else 0.0
            x0,y0 = waypoints[0]
            return x0,y0,th0
        if t >= total_T:
            xN,yN = waypoints[-1]
            thN = segs[-1][4] if segs else 0.0
            return xN,yN,thN
        for i in range(len(segs)):
            if t_cum[i] <= t <= t_cum[i+1]:
                x0,y0,x1,y1,th,L,T = segs[i]
                s = (t - t_cum[i])/max(T, 1e-9)
                return x0 + s*(x1-x0), y0 + s*(y1-y0), th
        return waypoints[-1][0], waypoints[-1][1], (segs[-1][4] if segs else 0.0)

    def window(t0: float, N: int):
        xs = []
        for k in range(N+1):
            t = t0 + k*dt
            x,y,th = sample(t)
            xs.append(ca.DM([x,y,th]).reshape((3,1)))
        return ca.hcat(xs)

    return window
