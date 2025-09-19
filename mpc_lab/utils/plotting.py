import casadi as ca
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")  # always render to files

def plot_attitude_rollout(times, Q_hist, W_hist, U_hist):
    """times: list[float]; Q_hist: list[DM(4,1)]; W_hist: list[DM(3,1)]; U_hist: list[DM(3,1)]"""
    t = times
    qw = [float(q[0,0]) for q in Q_hist]
    qx = [float(q[1,0]) for q in Q_hist]
    qy = [float(q[2,0]) for q in Q_hist]
    qz = [float(q[3,0]) for q in Q_hist]
    wx = [float(w[0,0]) for w in W_hist]
    wy = [float(w[1,0]) for w in W_hist]
    wz = [float(w[2,0]) for w in W_hist]
    tx = [float(u[0,0]) for u in U_hist]
    ty = [float(u[1,0]) for u in U_hist]
    tz = [float(u[2,0]) for u in U_hist]

    plt.figure()
    plt.plot(t, qw); plt.plot(t, qx); plt.plot(t, qy); plt.plot(t, qz)
    plt.xlabel("time [s]"); plt.ylabel("quaternion"); plt.title("q(t)")

    plt.figure()
    plt.plot(t, wx); plt.plot(t, wy); plt.plot(t, wz)
    plt.xlabel("time [s]"); plt.ylabel("ω [rad/s]"); plt.title("omega(t)")

    plt.figure()
    plt.plot(t, tx); plt.plot(t, ty); plt.plot(t, tz)
    plt.xlabel("time [s]"); plt.ylabel("τ [Nm]"); plt.title("u(t)")

    plt.show()

def plot_rover_rollout(times, X_hist, U_hist):
    t = times
    x = [float(xk[0,0]) for xk in X_hist]
    y = [float(xk[1,0]) for xk in X_hist]
    th = [float(xk[2,0]) for xk in X_hist]
    v = [float(uk[0,0]) for uk in U_hist]
    om = [float(uk[1,0]) for uk in U_hist]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Rover path")

    plt.figure()
    plt.plot(t, th)
    plt.xlabel("time [s]"); plt.ylabel("theta [rad]"); plt.title("Heading")

    plt.figure()
    plt.plot(t, v); plt.plot(t, om)
    plt.xlabel("time [s]"); plt.ylabel("inputs"); plt.title("v, omega")

    plt.show()
