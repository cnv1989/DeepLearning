"""Example 4: a complete autonomous flight. Take off, fly a route, land.

    python examples/04_waypoint_mission.py            (add --gif for an animation)

Everything is in the loop: physics, motors, battery, noisy sensors,
estimation, the cascaded controller and the flight-mode state machine.
Guides: docs/09-navigation-and-planning.md, docs/10-flight-modes.md,
docs/11-the-simulation-loop.md
"""

import sys

import _setup  # noqa: F401
import numpy as np

from drone.mission import Mission
from drone.params import QuadParams
from drone.plots import animate, plot_3d, plot_flight
from drone.simulation import SimConfig, Simulator

WAYPOINTS = [[6, 0, 3], [6, 6, 4], [0, 6, 3], [0, 0, 2]]

mission = Mission(WAYPOINTS, takeoff_altitude=2.0, cruise_speed=2.0, yaw_along_path=True)
simulator = Simulator(config=SimConfig(duration=60))
log = simulator.run(mission)

print("Flight events")
for time, event in log.events:
    print(f"  {time:5.1f} s  {event}")
in_mission = log.mode == "mission"
error = np.linalg.norm(log.position - log.setpoint_position, axis=1)[in_mission]
path_length = np.sum(np.linalg.norm(np.diff(log.position, axis=0), axis=1))
print(f"\nflew {path_length:.1f} m in {log.time[-1]:.1f} s")
print(f"tracking error during the mission: mean {error.mean():.2f} m, worst {error.max():.2f} m")
print(f"landed at {np.round(log.position[-1], 2)} (last waypoint {WAYPOINTS[-1][:2]})")
print(f"battery used: {log.battery_charge[0] - log.battery_charge[-1]:.1%}")

plot_flight(log, f"{_setup.OUTPUT}/04_mission_flight.png", "Waypoint mission")
plot_3d(log, f"{_setup.OUTPUT}/04_mission_3d.png", WAYPOINTS)
if "--gif" in sys.argv:
    animate(log, f"{_setup.OUTPUT}/04_mission.gif", QuadParams(), every=25, waypoints=WAYPOINTS)
