import math
import casadi as ca

def yaw_slew_profile(total_time: float, dt: float, yaw_final_deg: float):
    """Return (q_seq, w_seq) over timesteps 0..T for a smooth yaw slew.
    - q_seq: (4, T+1) DM, scalar-first quaternion
    - w_seq: (3, T+1) DM, desired body rates (zero here for simplicity)
    """
    T = int(round(total_time / dt))
    yaw_final = math.radians(yaw_final_deg)
    # Smooth step via cosine (zero vel at ends)
    q_list = []
    w_list = []
    for k in range(T+1):
        s = k / T if T > 0 else 1.0
        s_smooth = 0.5 - 0.5*math.cos(math.pi*s)  # 0→1 with zero slope at ends
        yaw = s_smooth * yaw_final
        ang = 0.5 * yaw
        q = ca.DM([math.cos(ang), 0, 0, math.sin(ang)]).reshape((4,1))
        q_list.append(q)
        w_list.append(ca.DM.zeros(3,1))
    q_seq = ca.hcat(q_list)
    w_seq = ca.hcat(w_list)
    return q_seq, w_seq

def rover_line_trajectory(total_time: float, dt: float, x_goal: float, y_goal: float, th_goal: float=0.0):
    """Return reference states (x,y,theta) over 0..T that move along a straight line to the goal with smooth timing."""
    T = int(round(total_time / dt))
    q_list = []
    for k in range(T+1):
        s = k / T if T > 0 else 1.0
        s_smooth = 0.5 - 0.5*math.cos(math.pi*s)
        x = s_smooth * x_goal
        y = s_smooth * y_goal
        th = s_smooth * th_goal
        q_list.append(ca.DM([x, y, th]).reshape((3,1)))
    Xref = ca.hcat(q_list)
    return Xref

def rover_circle_trajectory(total_time: float, dt: float, radius: float=2.0, period: float=10.0):
    """Circular reference for rover: returns (x,y,theta) over time."""
    T = int(round(total_time / dt))
    Xref = ca.DM.zeros(3, T+1)
    for k in range(T+1):
        t = k*dt
        theta = 2*math.pi * (t/period)
        Xref[0,k] = radius*math.cos(theta)
        Xref[1,k] = radius*math.sin(theta)
        Xref[2,k] = theta + math.pi/2  # heading tangent to circle
    return Xref
