
import matplotlib.pyplot as plt

def plot_path_with_constraints(xs, ys, waypoints=None, box=None, corridor=None, show=True, save=None):
    plt.figure()
    plt.plot(xs, ys, label="path")
    xmin, xmax = min(xs)-0.5, max(xs)+0.5
    if corridor is not None:
        ylim = corridor.get("ylim", None)
        if ylim is not None:
            ylow, yhigh = ylim
            plt.plot([xmin, xmax], [yhigh, yhigh], '--', label=f'corridor y={yhigh}')
            plt.plot([xmin, xmax], [ylow,  ylow ], '--', label=f'corridor y={ylow}')
    if box is not None:
        xlow,xhigh,ylow,yhigh = box
        box_x = [xlow, xhigh, xhigh, xlow, xlow]
        box_y = [ylow, ylow,  yhigh, yhigh, ylow]
        plt.plot(box_x, box_y, ':', label='state box')
    if waypoints is not None and len(waypoints)>0:
        wx, wy = zip(*waypoints)
        plt.plot(wx, wy, 'o', label='waypoints')
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Rover path with constraints")
    plt.legend()
    if save: plt.savefig(save)
    if show: plt.show()

def plot_inputs_with_bounds(t, v, om, vmin, vmax, omin, omax, show=True, save=None):
    plt.figure()
    plt.plot(t, v,  label='v')
    plt.plot(t, [vmax]*len(t), '--', label='v bounds')
    plt.plot(t, [vmin]*len(t), '--')
    plt.xlabel("time [s]"); plt.ylabel("v [m/s]"); plt.title("Linear speed with bounds")
    plt.legend()
    if save: plt.savefig(save)

    plt.figure()
    plt.plot(t, om, label='omega')
    plt.plot(t, [omax]*len(t), '--', label='ω bounds')
    plt.plot(t, [omin]*len(t), '--')
    plt.xlabel("time [s]"); plt.ylabel("omega [rad/s]"); plt.title("Angular speed with bounds")
    plt.legend()
    if save:
        base, ext = (save.rsplit('.',1)+['png'])[:2]
        plt.savefig(f"{base}_omega.{ext}")
    if show: plt.show()
