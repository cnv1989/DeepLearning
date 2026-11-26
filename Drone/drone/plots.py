"""Charts of a flight log: the main way to understand what happened in a flight.

Real drone developers spend a lot of time on exactly these plots (PX4's
"Flight Review", ArduPilot's "Mission Planner" log viewer): setpoint vs actual
for every loop, estimates vs sensor readings, motor outputs, battery.
"""

import os

import matplotlib

matplotlib.use("Agg")  # draw to files; no screen needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from drone.rotations import euler_to_rotation  # noqa: E402

AXES = ["x", "y", "z"]
ANGLES = ["roll", "pitch", "yaw"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


def plot_flight(log, path, title="Flight"):
    """Overview: position, attitude, motors and battery against time."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    t = log.time

    for i, (axis, color) in enumerate(zip(AXES, ["C0", "C1", "C2"])):
        axs[0].plot(t, log.position[:, i], color=color, label=f"{axis} actual")
        axs[0].plot(t, log.setpoint_position[:, i], "--", color=color, alpha=0.7, label=f"{axis} setpoint")
    axs[0].set_ylabel("position (m)")
    axs[0].legend(ncol=3, fontsize=8)

    for i, (angle, color) in enumerate(zip(ANGLES[:2], ["C0", "C1"])):
        axs[1].plot(t, np.degrees(log.euler[:, i]), color=color, label=f"{angle} actual")
        axs[1].plot(t, np.degrees(log.setpoint_euler[:, i]), "--", color=color, alpha=0.7, label=f"{angle} setpoint")
    axs[1].set_ylabel("angle (deg)")
    axs[1].legend(ncol=2, fontsize=8)

    for m in range(4):
        axs[2].plot(t, log.motor_speeds[:, m], label=f"motor {m}")
    axs[2].set_ylabel("motor speed (rad/s)")
    axs[2].legend(ncol=4, fontsize=8)

    axs[3].plot(t, log.battery_voltage, color="C3")
    axs[3].set_ylabel("battery (V)", color="C3")
    charge = axs[3].twinx()
    charge.plot(t, log.battery_charge * 100, color="C4")
    charge.set_ylabel("charge (%)", color="C4")
    axs[3].set_xlabel("time (s)")

    _add_mode_bands(axs[0], log)
    fig.suptitle(title)
    _save(fig, path)


def _add_mode_bands(ax, log):
    """Shade the background by flight mode."""
    colors = {"takeoff": "#dfe9f7", "mission": "#e3f4e1", "land": "#fbe7d5"}
    if len(log.mode) == 0:
        return
    start = 0
    for i in range(1, len(log.mode) + 1):
        if i == len(log.mode) or log.mode[i] != log.mode[start]:
            color = colors.get(str(log.mode[start]))
            if color:
                ax.axvspan(log.time[start], log.time[i - 1], color=color, zorder=0)
                ax.text(log.time[start], ax.get_ylim()[1], f" {log.mode[start]}", va="top", fontsize=8)
            start = i


def plot_estimation(log, path, title="State estimation"):
    """How well the estimators track the truth, compared with raw GPS."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    t = log.time
    for i, axis in enumerate(AXES):
        ax = axs[i, 0]
        if len(log.gps_time):
            ax.plot(log.gps_time, log.gps_position[:, i], ".", color="0.6", ms=3, label="GPS reading")
        ax.plot(t, log.position[:, i], color="k", lw=1.5, label="true")
        ax.plot(t, log.est_position[:, i], color="C1", lw=1, label="Kalman estimate")
        ax.set_ylabel(f"{axis} (m)")
        if i == 0:
            ax.legend(fontsize=8)

    for i, angle in enumerate(ANGLES):
        ax = axs[i, 1]
        true = np.degrees(log.euler[:, i])
        estimate = np.degrees(log.est_euler[:, i])
        if angle == "yaw":
            error = (estimate - true + 180) % 360 - 180
            ax.plot(t, error, color="C1")
            ax.set_ylabel("yaw error (deg)")
        else:
            ax.plot(t, true, color="k", lw=1.5, label="true")
            ax.plot(t, estimate, color="C1", lw=1, label="complementary filter")
            ax.set_ylabel(f"{angle} (deg)")
            if i == 0:
                ax.legend(fontsize=8)
    axs[2, 0].set_xlabel("time (s)")
    axs[2, 1].set_xlabel("time (s)")
    fig.suptitle(title)
    _save(fig, path)


def plot_3d(log, path, waypoints=None, title="Flight path"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*log.position.T, color="C0", label="flown path")
    ax.plot(*log.setpoint_position.T, "--", color="C1", alpha=0.7, label="setpoint")
    if waypoints is not None:
        w = np.array(waypoints)
        ax.scatter(*w.T, color="C3", s=40, label="waypoints")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.set_zlabel("z up (m)")
    ax.legend()
    ax.set_title(title)
    _save(fig, path)


def plot_map(grid, path, raw_path=None, smooth_path=None, flown=None, title="Path planning"):
    """Top-down view: obstacles, the planned paths and the flown track."""
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [grid.origin[0], grid.origin[0] + grid.blocked.shape[0] * grid.resolution,
              grid.origin[1], grid.origin[1] + grid.blocked.shape[1] * grid.resolution]
    ax.imshow(grid.blocked.T, origin="lower", extent=extent, cmap="Greys", alpha=0.35)
    for x0, y0, x1, y1 in grid.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.3"))
    if raw_path is not None:
        ax.plot(*np.array(raw_path).T, ".-", color="C1", ms=3, lw=1, label="A* path (grid cells)")
    if smooth_path is not None:
        ax.plot(*np.array(smooth_path).T, "o-", color="C3", lw=2, label="after shortcutting")
    if flown is not None:
        ax.plot(flown[:, 0], flown[:, 1], color="C0", lw=1.5, label="flown")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title + "  (light grey: obstacles grown by the safety margin)")
    _save(fig, path)


def animate(log, path, params, every=10, waypoints=None):
    """A 3D animation of the drone's body (arms and nose) along its path, saved as a GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    lo, hi = log.position.min(axis=0) - 1, log.position.max(axis=0) + 1
    span = (hi - lo).max()
    centre = (hi + lo) / 2
    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(0, max(hi[2], 1))
    if waypoints is not None:
        ax.scatter(*np.array(waypoints).T, color="C3", s=20)
    trail, = ax.plot([], [], [], color="C0", lw=1)
    arm_a, = ax.plot([], [], [], color="k", lw=3)
    arm_b, = ax.plot([], [], [], color="k", lw=3)
    nose, = ax.plot([], [], [], color="C3", lw=3)
    arms = params.motor_positions * 3  # drawn 3x larger so it's visible
    frames = range(0, len(log.time), every)

    def draw(k):
        p = log.position[k]
        R = euler_to_rotation(*log.euler[k])
        corners = [p + R @ np.array([x, y, 0.0]) for x, y in arms]
        arm_a.set_data_3d(*np.array([corners[0], corners[1]]).T)
        arm_b.set_data_3d(*np.array([corners[2], corners[3]]).T)
        nose.set_data_3d(*np.array([p, p + R @ np.array([1.5 * arms[0, 0], 0, 0])]).T)
        trail.set_data_3d(*log.position[: k + 1].T)
        ax.set_title(f"t = {log.time[k]:.1f} s   mode: {log.mode[k]}")
        return trail, arm_a, arm_b, nose

    animation = FuncAnimation(fig, draw, frames=frames, interval=50)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved {path}")
