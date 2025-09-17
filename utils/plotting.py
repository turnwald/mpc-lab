import csv
from pathlib import Path
import matplotlib.pyplot as plt

def plot_attitude_rollout(csv_path: str, out_png: str):
    xs = []
    qw, qx, qy, qz = [], [], [], []
    wx, wy, wz = [], [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["t"]))
            qw.append(float(row["qw"])); qx.append(float(row["qx"])); qy.append(float(row["qy"])); qz.append(float(row["qz"]))
            wx.append(float(row["wx"])); wy.append(float(row["wy"])); wz.append(float(row["wz"]))

    # Figure 1: quaternion components
    plt.figure()
    plt.plot(xs, qw, label="qw")
    plt.plot(xs, qx, label="qx")
    plt.plot(xs, qy, label="qy")
    plt.plot(xs, qz, label="qz")
    plt.xlabel("t [s]"); plt.ylabel("q components"); plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    # Figure 2: body rates
    out2 = str(Path(out_png).with_name(Path(out_png).stem + "_rates.png"))
    plt.figure()
    plt.plot(xs, wx, label="wx")
    plt.plot(xs, wy, label="wy")
    plt.plot(xs, wz, label="wz")
    plt.xlabel("t [s]"); plt.ylabel("ω [rad/s]"); plt.legend()
    plt.savefig(out2, bbox_inches="tight")
    plt.close()

    return out_png, out2

def plot_rover_rollout(csv_path: str, out_png: str):
    xs = []
    Xs, Ys, Ths = [], [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["t"]))
            Xs.append(float(row["x"]))
            Ys.append(float(row["y"]))
            Ths.append(float(row["theta"]))

    # Figure 1: trajectory XY
    plt.figure()
    plt.plot(Xs, Ys, label="trajectory")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.axis("equal"); plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    # Figure 2: heading vs time
    out2 = str(Path(out_png).with_name(Path(out_png).stem + "_theta.png"))
    plt.figure()
    plt.plot(xs, Ths, label="theta")
    plt.xlabel("t [s]"); plt.ylabel("θ [rad]"); plt.legend()
    plt.savefig(out2, bbox_inches="tight")
    plt.close()

    return out_png, out2
