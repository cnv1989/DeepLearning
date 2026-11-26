"""Example 6: the real world fights back. Wind, the integral term, and a low battery.

    python examples/06_wind_and_battery.py

Part 1 hovers in a 4 m/s wind, with and without the velocity loop's integral
term. Part 2 starts a long mission with a nearly empty battery to trigger
the low-battery failsafe. Guides: docs/08-control.md, docs/05-battery.md
"""

import _setup  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drone.control import ControllerGains, Setpoint  # noqa: E402
from drone.dynamics import State  # noqa: E402
from drone.mission import Mission  # noqa: E402
from drone.simulation import SimConfig, Simulator  # noqa: E402

# ---- Part 1: wind ----
WIND = np.array([4.0, 0.0, 0.0])  # blowing toward +x (east)
fig, axs = plt.subplots(2, 1, figsize=(10, 7))
print("Part 1: hovering at (0, 0, 3) in a 4 m/s wind with gusts")
for name, velocity_i in [("with integral (default)", ControllerGains().velocity_i),
                         ("without integral (I = 0)", np.zeros(3))]:
    simulator = Simulator(gains=ControllerGains(velocity_i=velocity_i),
                          config=SimConfig(duration=20, wind=WIND, gust_strength=1.0),
                          initial_state=State(position=np.array([0.0, 0.0, 3.0])))
    log = simulator.run(setpoint_function=lambda t: Setpoint(np.array([0.0, 0.0, 3.0])))
    late = log.time > 10
    print(f"  {name:<26} average drift downwind {log.position[late, 0].mean():5.2f} m, "
          f"average pitch {np.degrees(log.euler[late, 1]).mean():5.1f}°")
    axs[0].plot(log.time, log.position[:, 0], label=name)
axs[0].axhline(0, color="k", lw=0.8, ls="--")
axs[0].set_ylabel("x (m), downwind drift")
axs[0].set_xlabel("time (s)")
axs[0].legend()
axs[0].set_title("Hovering in wind: the integral term removes the steady offset")

# ---- Part 2: low battery failsafe ----
print("\nPart 2: a long mission starting at 23% battery (failsafe lands at 20%)")
route = [[20, 0, 3], [20, 20, 3], [0, 20, 3], [0, 0, 3]] * 3
simulator = Simulator(config=SimConfig(duration=120, battery_charge=0.23))
log = simulator.run(Mission(route, takeoff_altitude=3.0, cruise_speed=3.0))
for time, event in log.events:
    print(f"  {time:5.1f} s  {event}")
axs[1].plot(log.time, log.battery_charge * 100, color="C4", label="charge (%)")
axs[1].axhline(20, color="C3", ls="--", lw=0.8)
axs[1].set_ylabel("battery charge (%)")
height = axs[1].twinx()
height.plot(log.time, log.position[:, 2], color="C0")
height.set_ylabel("altitude (m)", color="C0")
axs[1].set_xlabel("time (s)")
axs[1].set_title("Low-battery failsafe: the drone lands where it is at 20%")
fig.tight_layout()
path = f"{_setup.OUTPUT}/06_wind_and_battery.png"
fig.savefig(path, dpi=110)
print(f"saved {path}")
