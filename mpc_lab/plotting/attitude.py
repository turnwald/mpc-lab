
import matplotlib.pyplot as plt

def plot_omega_with_bounds(t, wx, wy, wz, wmax=None, show=True, save=None):
    plt.figure()
    plt.plot(t, wx, label='ωx'); plt.plot(t, wy, label='ωy'); plt.plot(t, wz, label='ωz')
    if wmax is not None:
        plt.plot(t, [ wmax]*len(t), '--', label='ω bound'); plt.plot(t, [-wmax]*len(t), '--')
    plt.xlabel("time [s]"); plt.ylabel("ω [rad/s]"); plt.title("Body rates with bounds")
    plt.legend()
    if save: plt.savefig(save)
    if show: plt.show()

def plot_torque_with_bounds(t, tx, ty, tz, umin, umax, show=True, save=None):
    plt.figure()
    plt.plot(t, tx, label='τx'); plt.plot(t, ty, label='τy'); plt.plot(t, tz, label='τz')
    plt.plot(t, [umax]*len(t), '--', label='τ bounds'); plt.plot(t, [umin]*len(t), '--')
    plt.xlabel("time [s]"); plt.ylabel("τ [Nm]"); plt.title("Torque with bounds")
    plt.legend()
    if save: plt.savefig(save)
    if show: plt.show()
