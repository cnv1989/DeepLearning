"""Example 5: planning a route around obstacles with A*, then flying it.

    python examples/05_path_planning.py

Guide: docs/09-navigation-and-planning.md
"""

import _setup  # noqa: F401
import numpy as np

from drone.dynamics import State
from drone.mission import Mission
from drone.planning import OccupancyGrid, astar, shortcut_path
from drone.plots import plot_map
from drone.simulation import SimConfig, Simulator

ALTITUDE = 2.5
START, GOAL = (1.0, 1.0), (18.0, 18.0)

grid = OccupancyGrid(width=20, height=20, resolution=0.25)
grid.add_box(4, 0, 6, 12)  # walls and buildings
grid.add_box(9, 7, 11, 20)
grid.add_box(13, 3, 17, 6)
grid.add_box(13.5, 11, 20, 13)
grid.add_box(0, 15, 6, 16.5)

drawn_grid = OccupancyGrid(20, 20, 0.25)
drawn_grid.obstacles = list(grid.obstacles)
grid.inflate(0.6)  # keep 0.6 m from every obstacle
drawn_grid.blocked = grid.blocked

raw_path = astar(grid, START, GOAL)
smooth_path = shortcut_path(grid, raw_path)
raw_length = np.sum(np.linalg.norm(np.diff(raw_path, axis=0), axis=1))
print(f"A* path: {len(raw_path)} cells, {raw_length:.1f} m")
print(f"after shortcutting: {len(smooth_path)} waypoints")

waypoints = [[x, y, ALTITUDE] for x, y in smooth_path[1:]]
simulator = Simulator(config=SimConfig(duration=90), initial_state=State(position=np.array([*START, 0.0])))
log = simulator.run(Mission(waypoints, takeoff_altitude=ALTITUDE, cruise_speed=2.5))


def distance_to_box(x, y, box):
    """0 inside the box; otherwise the straight-line distance to its nearest point."""
    x0, y0, x1, y1 = box
    return np.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


airborne = log.position[log.position[:, 2] > 0.3]
clearance = [min(distance_to_box(x, y, box) for box in grid.obstacles) for x, y, _ in airborne]
print(f"flight time {log.time[-1]:.1f} s | closest approach to an obstacle: {min(clearance):.2f} m "
      f"(planned margin 0.6 m)")
plot_map(drawn_grid, f"{_setup.OUTPUT}/05_path_planning.png", raw_path, smooth_path, log.position)
