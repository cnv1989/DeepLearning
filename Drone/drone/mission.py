"""Flight modes: the sequence of a whole flight, as a state machine.

    DISARMED ──arm──► TAKEOFF ──at altitude──► MISSION ──done──► LAND ──on ground──► LANDED
                                                  │                 ▲
                                                  └──battery low────┘   (failsafe)

Every mode produces a position setpoint for the flight controller. Real
autopilots have many more modes (manual, position hold, return to launch,
orbit...), but each is the same idea: a rule for choosing the setpoint, and
rules for when to switch.
"""

from enum import Enum

import numpy as np

from drone.control import Setpoint
from drone.planning import WaypointFollower


class Mode(Enum):
    DISARMED = "disarmed"  # motors off
    TAKEOFF = "takeoff"  # climb straight up
    MISSION = "mission"  # follow the waypoints
    LAND = "land"  # descend slowly where we are
    LANDED = "landed"  # touched down; motors off


class Mission:
    def __init__(self, waypoints, takeoff_altitude=2.0, cruise_speed=2.0, climb_speed=1.0,
                 land_speed=0.5, low_battery=0.2, yaw_along_path=False):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.takeoff_altitude = takeoff_altitude
        self.cruise_speed = cruise_speed
        self.climb_speed = climb_speed
        self.land_speed = land_speed
        self.low_battery = low_battery  # land early below this state of charge
        self.yaw_along_path = yaw_along_path

        self.mode = Mode.DISARMED
        self.follower = None
        self.hold = None  # (x, y) to take off or land at
        self.target_altitude = 0.0
        self.events = []  # (time, message), for the flight log

    def _switch(self, mode, time, reason):
        self.events.append((time, f"{self.mode.value} -> {mode.value}: {reason}"))
        self.mode = mode

    def arm(self, position, time=0.0):
        self.hold = np.asarray(position[:2], dtype=float)
        self.target_altitude = position[2]
        self._switch(Mode.TAKEOFF, time, "armed")

    @property
    def motors_on(self):
        return self.mode in (Mode.TAKEOFF, Mode.MISSION, Mode.LAND)

    def update(self, estimate, battery_charge, on_ground, time, dt):
        """Pick the setpoint for this moment, switching modes when their job is done."""
        if self.mode in (Mode.MISSION, Mode.TAKEOFF) and battery_charge < self.low_battery:
            self.hold = estimate.position[:2].copy()
            self.target_altitude = estimate.position[2]
            self._switch(Mode.LAND, time, f"battery low ({battery_charge:.0%})")

        if self.mode == Mode.TAKEOFF:
            # Raise the target altitude gradually, for a steady climb.
            self.target_altitude = min(self.target_altitude + self.climb_speed * dt, self.takeoff_altitude)
            if estimate.position[2] > self.takeoff_altitude - 0.15:
                start = np.array([*estimate.position[:2], self.takeoff_altitude])
                self.follower = WaypointFollower([start, *self.waypoints], self.cruise_speed,
                                                 yaw_along_path=self.yaw_along_path)
                self._switch(Mode.MISSION, time, "reached takeoff altitude")
            return Setpoint(np.array([*self.hold, self.target_altitude]),
                            velocity=np.array([0.0, 0.0, self.climb_speed]))

        if self.mode == Mode.MISSION:
            setpoint = self.follower.update(estimate.position, dt)
            if self.follower.finished:
                self.hold = self.waypoints[-1][:2].copy()
                self.target_altitude = setpoint.position[2]
                self._switch(Mode.LAND, time, "mission complete")
            return setpoint

        if self.mode == Mode.LAND:
            self.target_altitude -= self.land_speed * dt
            if on_ground:
                self._switch(Mode.LANDED, time, "touched down")
            yaw = self.follower.yaw if self.follower else 0.0
            return Setpoint(np.array([*self.hold, self.target_altitude]), yaw,
                            velocity=np.array([0.0, 0.0, -self.land_speed]))

        return Setpoint(estimate.position.copy())  # disarmed or landed: nothing to do
