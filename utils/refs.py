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
