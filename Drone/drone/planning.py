"""Path planning and waypoint following.

Two layers, as in real autonomous drones:

  PLANNING (A*)       Find a collision-free route through a map of obstacles.
                      Runs once, before or occasionally during the flight.

  FOLLOWING           Turn that route into a smooth, moving position setpoint
  (WaypointFollower)  for the flight controller, many times per second.

The map is an occupancy grid: the area is divided into square cells, each
either free or blocked. Planning is in 2D at a fixed altitude, which is how
many survey and delivery drones fly.
"""

import heapq

import numpy as np

from drone.control import Setpoint


class OccupancyGrid:
    """A 2D map: cells are free (False) or blocked (True)."""

    def __init__(self, width, height, resolution=0.5, origin=(0.0, 0.0)):
        self.resolution = resolution  # metres per cell
        self.origin = np.array(origin, dtype=float)  # world (x, y) of cell (0, 0)
        self.blocked = np.zeros((int(round(width / resolution)), int(round(height / resolution))), dtype=bool)
        self.obstacles = []  # rectangles, kept for drawing

    def add_box(self, x_min, y_min, x_max, y_max):
        """Block a rectangle (world coordinates in metres)."""
        self.obstacles.append((x_min, y_min, x_max, y_max))
        i0, j0 = self.to_cell((x_min, y_min))
        i1, j1 = self.to_cell((x_max, y_max))
        self.blocked[max(i0, 0) : i1 + 1, max(j0, 0) : j1 + 1] = True

    def inflate(self, radius):
        """Grow every obstacle by `radius` metres, so paths keep a safety margin.

        Planning for a point is easy; planning for a drone 0.5 m wide is the same
        problem once obstacles are made 0.25 m bigger in every direction.
        """
        cells = int(np.ceil(radius / self.resolution))
        width, height = self.blocked.shape
        # Pad with free space so shifting never wraps obstacles around the map's edges.
        padded = np.pad(self.blocked, cells)
        grown = np.zeros_like(self.blocked)
        for di in range(-cells, cells + 1):
            for dj in range(-cells, cells + 1):
                if di * di + dj * dj <= cells * cells:
                    grown |= padded[cells + di : cells + di + width, cells + dj : cells + dj + height]
        self.blocked = grown

    def to_cell(self, xy):
        i, j = np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        return int(i), int(j)

    def to_world(self, cell):
        return self.origin + (np.array(cell) + 0.5) * self.resolution

    def is_free(self, cell):
        i, j = cell
        inside = 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1]
        return inside and not self.blocked[i, j]


def astar(grid, start_xy, goal_xy):
    """A* search: the shortest path between two points on the grid, avoiding blocked cells.

    A* explores cells in order of  f = g + h:
        g  the length of the best path found so far from the start
        h  a guess of the remaining distance (here: straight-line distance)
    Because the guess never overestimates, the first path to reach the goal
    is the shortest. Returns a list of world (x, y) points, or None.
    """
    start, goal = grid.to_cell(start_xy), grid.to_cell(goal_xy)
    if not grid.is_free(start) or not grid.is_free(goal):
        return None

    def distance(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_heap = [(distance(start, goal), 0.0, start)]
    best_cost = {start: 0.0}
    came_from = {}

    while open_heap:
        _, cost, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return [grid.to_world(c) for c in reversed(path)]
        if cost > best_cost[cell]:
            continue  # an outdated entry; a cheaper route to this cell was found later

        for di, dj in neighbours:
            neighbour = (cell[0] + di, cell[1] + dj)
            if not grid.is_free(neighbour):
                continue
            # Don't cut diagonally between two blocked cells.
            if di and dj and not (grid.is_free((cell[0] + di, cell[1])) and grid.is_free((cell[0], cell[1] + dj))):
                continue
            new_cost = cost + distance(cell, neighbour)
            if new_cost < best_cost.get(neighbour, np.inf):
                best_cost[neighbour] = new_cost
                came_from[neighbour] = cell
                heapq.heappush(open_heap, (new_cost + distance(neighbour, goal), new_cost, neighbour))
    return None


def line_is_free(grid, a, b):
    """True if the straight line from a to b stays in free cells (checked every quarter cell)."""
    steps = max(1, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / (grid.resolution / 4))))
    return all(grid.is_free(grid.to_cell(a + (b - a) * t)) for t in np.linspace(0, 1, steps + 1))


def shortcut_path(grid, path):
    """Remove unnecessary waypoints: jump straight to the furthest point still in view.

    Grid paths zig-zag along cell edges; flying them literally is slow and jerky.
    """
    if path is None or len(path) <= 2:
        return path
    simplified, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_is_free(grid, path[i], path[j]):
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


class WaypointFollower:
    """Turns a list of 3D waypoints into a smoothly moving setpoint.

    Rather than jumping the setpoint to the next waypoint (the drone would
    sprint, overshoot, and tilt hard), a "carrot" slides along each leg and
    the drone chases it. The carrot's velocity is also passed on as
    feedforward, so the controller doesn't wait to fall behind before moving.

    The carrot's speed follows a trapezoidal profile on every leg:

        speed
        cruise ┤    ┌────────────┐
               │   ╱              ╲        speed up at max_acceleration,
               │  ╱                ╲       cruise, then slow down just in
             0 ┼─┴──────────────────┴──    time to stop at the waypoint
                 start            waypoint

    Stopping at each waypoint means the drone never overshoots a corner,
    which matters when the corner is next to a wall.
    """

    def __init__(self, waypoints, cruise_speed=2.0, max_acceleration=1.5, acceptance_radius=0.3,
                 yaw_along_path=False):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.cruise_speed = cruise_speed
        self.max_acceleration = max_acceleration
        self.acceptance_radius = acceptance_radius
        self.yaw_along_path = yaw_along_path
        self.index = 1  # the waypoint currently being flown toward
        self.carrot = self.waypoints[0].copy()
        self.speed = 0.0  # current carrot speed
        self.yaw = 0.0

    @property
    def finished(self):
        return self.index >= len(self.waypoints)

    def update(self, position, dt):
        if self.finished:
            return Setpoint(self.waypoints[-1].copy(), self.yaw)

        target = self.waypoints[self.index]
        to_target = target - self.carrot
        distance = np.linalg.norm(to_target)

        # Fastest speed from which we can still stop at the waypoint: v² = 2·a·d.
        stopping_speed = np.sqrt(2 * self.max_acceleration * distance)
        self.speed = min(self.speed + self.max_acceleration * dt, self.cruise_speed, stopping_speed)
        step = self.speed * dt

        if distance > step and step > 0:
            direction = to_target / distance
            self.carrot = self.carrot + direction * step
            velocity = direction * self.speed
            if self.yaw_along_path and np.linalg.norm(direction[:2]) > 0.1:
                self.yaw = float(np.arctan2(direction[1], direction[0]))
        else:
            self.carrot = target.copy()
            self.speed = 0.0
            velocity = np.zeros(3)

        # Move on once both the carrot and the drone have reached the waypoint.
        if distance <= step and np.linalg.norm(position - target) < self.acceptance_radius:
            self.index += 1
        return Setpoint(self.carrot.copy(), self.yaw, velocity)
