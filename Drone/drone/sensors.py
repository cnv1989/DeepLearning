"""Sensors: what the drone can actually measure about itself.

The flight software never knows the true state; it only gets these readings,
each imperfect in its own way:

    Sensor          Measures                          Rate     Weakness
    ─────────────   ───────────────────────────────   ──────   ───────────────────────────
    Gyroscope       body rotation rates (rad/s)       fast     slowly drifting bias
    Accelerometer   specific force (m/s²), body       fast     noisy; can't tell tilt from
                                                               acceleration
    Magnetometer    Earth's magnetic field, body      medium   disturbed by nearby metal
    Barometer       altitude from air pressure        medium   noisy, drifts with weather
    GPS             position and velocity, world      slow     metres of error; delayed

The gyroscope and accelerometer together form the IMU (inertial measurement
unit). No single sensor is good enough alone; state estimation (estimation.py)
combines them.

Key idea, the accelerometer: it measures "specific force", everything
except gravity. Sitting still on a table it reads 9.81 m/s² *upward* (the
table pushing up). In free fall it reads zero. That's why, when not
accelerating, it shows which way is up and therefore how the drone is tilted.
"""

from dataclasses import dataclass

import numpy as np

from drone.dynamics import GRAVITY_VECTOR

# Direction of Earth's magnetic field in the world frame: mostly north (+y), dipping down.
MAGNETIC_FIELD_WORLD = np.array([0.0, 0.6, -0.8])


@dataclass
class SensorNoise:
    """Noise levels (one standard deviation), typical of cheap drone sensors."""
    gyro: float = 0.005  # rad/s
    gyro_bias: float = 0.01  # rad/s, constant offset fixed at power-on
    accel: float = 0.1  # m/s²
    magnetometer: float = 0.02  # fraction of field strength
    barometer: float = 0.15  # m
    gps_position: float = 0.6  # m (horizontal and vertical)
    gps_velocity: float = 0.1  # m/s


@dataclass
class SensorRates:
    """How often each sensor produces a reading, in Hz."""
    imu: float = 500.0
    magnetometer: float = 50.0
    barometer: float = 50.0
    gps: float = 10.0


class Sensors:
    """Produces noisy readings from the true state, each sensor at its own rate."""

    def __init__(self, noise=None, rates=None, seed=0):
        self.noise = noise or SensorNoise()
        self.rates = rates or SensorRates()
        self.random = np.random.default_rng(seed)
        self.gyro_bias = self.random.normal(0.0, self.noise.gyro_bias, 3)
        self.restart_clock()

    def restart_clock(self):
        """Start the slow sensors' schedules again from time 0."""
        self._next_reading_time = {"magnetometer": 0.0, "barometer": 0.0, "gps": 0.0}

    def _gaussian(self, std, size=3):
        return self.random.normal(0.0, std, size)

    def _due(self, sensor, time):
        """True if `sensor` should produce a reading at this time."""
        if time + 1e-9 >= self._next_reading_time[sensor]:
            self._next_reading_time[sensor] += 1.0 / getattr(self.rates, sensor)
            return True
        return False

    def read(self, state, acceleration, time):
        """All readings available at this time. Slower sensors return None between readings."""
        rotation_world_to_body = state.rotation().T

        # IMU (every step)
        gyro = state.body_rates + self.gyro_bias + self._gaussian(self.noise.gyro)
        specific_force = rotation_world_to_body @ (acceleration - GRAVITY_VECTOR)
        accel = specific_force + self._gaussian(self.noise.accel)

        readings = {"gyro": gyro, "accel": accel, "magnetometer": None, "barometer": None, "gps": None}

        if self._due("magnetometer", time):
            field = rotation_world_to_body @ MAGNETIC_FIELD_WORLD
            readings["magnetometer"] = field + self._gaussian(self.noise.magnetometer)
        if self._due("barometer", time):
            readings["barometer"] = state.position[2] + self._gaussian(self.noise.barometer, 1)[0]
        if self._due("gps", time):
            readings["gps"] = {
                "position": state.position + self._gaussian(self.noise.gps_position),
                "velocity": state.velocity + self._gaussian(self.noise.gps_velocity),
            }
        return readings
