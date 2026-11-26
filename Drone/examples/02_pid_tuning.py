"""Example 2: tuning PID controllers. What too little, good and too much gain look like.

    python examples/02_pid_tuning.py

The drone hovers at x = 0, then is asked to move to x = 3 m. The same step is
flown with three sets of gains. Guide: docs/08-control.md
"""

import _setup  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drone.control import ControllerGains, Setpoint  # noqa: E402
from drone.dynamics import State  # noqa: E402
from drone.simulation import SimConfig, Simulator  # noqa: E402

GAIN_SETS = {
    "sluggish (low P)": ControllerGains(position_p=np.array([0.3, 0.3, 1.5])),
    "default": ControllerGains(),
    "too aggressive (high P, no I)": ControllerGains(position_p=np.array([4.0, 4.0, 1.5]),
                                                       velocity_p=np.array([8.0, 8.0, 4.0]),
                                                       velocity_i=np.array([0.0, 0.0, 2.0]),
                                                       attitude_p=np.array([3.0, 3.0, 3.0])),
}


def step(time):
    target = [0.0, 0.0, 2.0] if time < 1.0 else [3.0, 0.0, 2.0]
    return Setpoint(np.array(target))


fig, (ax_x, ax_pitch) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
print(f"{'gains':<32}{'rise time (10-90%)':>20}{'overshoot':>12}{'settled (±5 cm)':>18}")
for name, gains in GAIN_SETS.items():
    simulator = Simulator(gains=gains, config=SimConfig(duration=10, use_estimator=False),
                          initial_state=State(position=np.array([0.0, 0.0, 2.0])))
    log = simulator.run(setpoint_function=step)
    t, x = log.time, log.position[:, 0]

    after = t >= 1.0
    rise_start = t[after][np.argmax(x[after] >= 0.3)]
    rise_end = t[after][np.argmax(x[after] >= 2.7)]
    overshoot = max(0.0, x.max() - 3.0)
    outside = np.where(np.abs(x - 3.0) > 0.05)[0]
    still_moving = len(outside) and outside[-1] == len(t) - 1
    settled = "not in 10 s" if still_moving else f"{t[outside[-1]] - 1.0:.1f} s"
    print(f"{name:<32}{rise_end - rise_start:>18.2f} s{overshoot * 100:>9.0f} cm{settled:>18}")

    ax_x.plot(t, x, label=name)
    ax_pitch.plot(t, np.degrees(log.euler[:, 1]), label=name)

ax_x.axhline(3.0, color="k", ls="--", lw=0.8)
ax_x.set_ylabel("x position (m)")
ax_x.legend()
ax_pitch.set_ylabel("pitch (deg)")
ax_pitch.set_xlabel("time (s)")
fig.suptitle("Position step response with three sets of gains")
fig.tight_layout()
path = f"{_setup.OUTPUT}/02_pid_tuning.png"
fig.savefig(path, dpi=110)
print(f"saved {path}")
