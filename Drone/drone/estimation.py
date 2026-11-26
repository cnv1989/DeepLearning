"""State estimation: combining noisy sensors into one best guess of the state.

Two estimators run side by side, as in many simple flight controllers:

  AttitudeEstimator  (complementary filter)
      Which way is the drone pointing?  Gyro + accelerometer + magnetometer.

  PositionEstimator  (Kalman filter)
      Where is it, and how fast is it moving?  Accelerometer + GPS + barometer.

Both rest on the same idea. One sensor is smooth but drifts over time
(integrating the gyro or accelerometer); another is noisy but right on
average (accelerometer tilt, magnetometer, GPS, barometer). Trust the smooth
one over short times and the stable one over long times, and you get the
best of both.

Real autopilots (PX4's EKF2, ArduPilot's EKF3) use one large Extended Kalman
Filter for everything, which also estimates sensor biases and wind. The two
filters here are the same ideas in their simplest form.
"""

from dataclasses import dataclass, field

import numpy as np

from drone.dynamics import GRAVITY_VECTOR
from drone.rotations import euler_to_quat, euler_to_rotation, quat_to_euler, wrap_angle


@dataclass
class Estimate:
    """The flight software's belief about the state: what the controller uses."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    body_rates: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def attitude(self):
        return euler_to_quat(*self.euler)

    def rotation(self):
        return euler_to_rotation(*self.euler)


def estimate_from_true_state(state):
    """A perfect "estimate" straight from the true state: for learning control on its own."""
    return Estimate(state.position.copy(), state.velocity.copy(),
                    quat_to_euler(state.attitude), state.body_rates.copy())


# ----------------------------------------------------------------------------- #
# Attitude: complementary filter
# ----------------------------------------------------------------------------- #


def tilt_from_accelerometer(accel):
    """Roll and pitch from the direction of the measured specific force.

    At rest, the accelerometer reads "up" in body coordinates, so its direction
    reveals the tilt. Wrong while the drone accelerates sideways; see the docs.
    """
    ax, ay, az = accel
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.hypot(ay, az))
    return roll, pitch


def heading_from_magnetometer(field_body, roll, pitch):
    """Yaw from the magnetic field, after undoing roll and pitch ("tilt compensation")."""
    level = euler_to_rotation(roll, pitch, 0.0) @ field_body  # field as seen by a level drone
    # Field points north (+y) in the world; with yaw ψ it appears at angle (90° − ψ) from body x.
    return wrap_angle(np.pi / 2 - np.arctan2(level[1], level[0]))


def euler_rates(euler, body_rates):
    """Convert body rotation rates (p, q, r) into rates of change of roll, pitch and yaw."""
    roll, pitch, _ = euler
    p, q, r = body_rates
    sr, cr = np.sin(roll), np.cos(roll)
    cp, tp = np.cos(pitch), np.tan(pitch)
    return np.array([
        p + sr * tp * q + cr * tp * r,
        cr * q - sr * r,
        (sr * q + cr * r) / cp,
    ])


class AttitudeEstimator:
    """Complementary filter.

    Every step:   angle = angle + gyro-derived rate × dt                  (smooth, drifts)
    Then nudge:   angle += (dt / time_constant) × (accel/mag angle − angle)  (noisy, no drift)

    The time constant says how long the filter trusts the gyro before
    deferring to the absolute sensor. Short: quick to fix drift, but lets
    in noise and errors. Long: smooth, but drift lingers.

    The tilt time constant must be LONG on a multirotor. Its accelerometer
    reads mostly its own thrust, which always points along the body z axis,
    so the accelerometer "sees level" even while the drone is tilted and
    accelerating. Trust it too much and the estimate is dragged toward level
    every time the drone tilts. (Real autopilots solve this properly by
    fusing GPS velocity in an EKF.)

    A long time constant only works if the gyro doesn't drift, so the filter
    first CALIBRATES: for `calibration_time` seconds the drone must be still,
    and the average gyro reading is its bias, subtracted from then on. This
    is why real drones ask you not to move them while they boot.
    """

    def __init__(self, tilt_time_constant=20.0, heading_time_constant=1.0, magnetometer_rate=50.0,
                 calibration_time=1.0):
        self.tilt_time_constant = tilt_time_constant
        self.heading_time_constant = heading_time_constant
        self.magnetometer_interval = 1.0 / magnetometer_rate
        self.calibration_time = calibration_time
        self.euler = np.zeros(3)
        self.body_rates = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.ready = False  # True once calibration is finished
        self._samples = {"gyro": [], "accel": [], "magnetometer": []}
        self._elapsed = 0.0

    def _calibrate(self, gyro, accel, magnetometer, dt):
        """Average the sensors while the drone sits still; then set the bias and starting attitude."""
        self._samples["gyro"].append(gyro)
        self._samples["accel"].append(accel)
        if magnetometer is not None:
            self._samples["magnetometer"].append(magnetometer)
        self._elapsed += dt
        if self._elapsed >= self.calibration_time and self._samples["magnetometer"]:
            self.gyro_bias = np.mean(self._samples["gyro"], axis=0)
            roll, pitch = tilt_from_accelerometer(np.mean(self._samples["accel"], axis=0))
            yaw = heading_from_magnetometer(np.mean(self._samples["magnetometer"], axis=0), roll, pitch)
            self.euler = np.array([roll, pitch, yaw])
            self.ready = True

    def update(self, gyro, accel, magnetometer, dt):
        if not self.ready:
            self._calibrate(gyro, accel, magnetometer, dt)
            return self.euler

        self.body_rates = gyro - self.gyro_bias
        self.euler = self.euler + euler_rates(self.euler, self.body_rates) * dt

        roll_meas, pitch_meas = tilt_from_accelerometer(accel)
        tilt_blend = dt / self.tilt_time_constant
        self.euler[0] += tilt_blend * (roll_meas - self.euler[0])
        self.euler[1] += tilt_blend * (pitch_meas - self.euler[1])

        if magnetometer is not None:
            yaw_meas = heading_from_magnetometer(magnetometer, self.euler[0], self.euler[1])
            # The magnetometer reads less often, so each reading covers a longer interval.
            heading_blend = min(1.0, self.magnetometer_interval / self.heading_time_constant)
            self.euler[2] += heading_blend * wrap_angle(yaw_meas - self.euler[2])
        self.euler[2] = wrap_angle(self.euler[2])
        return self.euler


# ----------------------------------------------------------------------------- #
# Position: Kalman filter
# ----------------------------------------------------------------------------- #


class PositionEstimator:
    """A linear Kalman filter tracking position and velocity (6 numbers).

    It keeps two things:
        x  the estimate          [px, py, pz, vx, vy, vz]
        P  its uncertainty       6x6 covariance: how wrong x might be

    PREDICT (every IMU reading): move x forward using the measured
        acceleration, and grow P, because integrating noisy acceleration
        adds uncertainty.

    UPDATE (when GPS or barometer reads): compare the measurement with what
        x predicts. The Kalman gain K decides how far to move toward the
        measurement: a lot if P is large (we're unsure) or the sensor is
        precise; a little otherwise. Then P shrinks.
    """

    def __init__(self, accel_noise=0.5, gps_position_noise=0.6, gps_velocity_noise=0.1,
                 barometer_noise=0.15):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.accel_noise = accel_noise  # how much we distrust the accelerometer
        self.gps_noise = np.diag([gps_position_noise ** 2] * 3 + [gps_velocity_noise ** 2] * 3)
        self.barometer_noise = barometer_noise ** 2
        self.initialized = False  # becomes True at the first GPS fix

    def predict(self, accel_body, rotation, dt):
        """Advance the estimate using the accelerometer (rotated to the world frame)."""
        acceleration = rotation @ accel_body + GRAVITY_VECTOR  # add gravity back

        F = np.eye(6)  # how the state evolves: position += velocity × dt
        F[0:3, 3:6] = np.eye(3) * dt
        B = np.vstack([np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt])  # how acceleration enters

        self.x = F @ self.x + B @ acceleration
        Q = B @ B.T * self.accel_noise ** 2  # uncertainty added by noisy acceleration
        self.P = F @ self.P @ F.T + Q

    def _update(self, measurement, H, R):
        """The Kalman update for a measurement z = H x + noise with covariance R."""
        innovation = measurement - H @ self.x  # surprise: measured minus expected
        S = H @ self.P @ H.T + R  # uncertainty of that surprise
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_gps(self, position, velocity):
        if not self.initialized:
            # Start from the first fix rather than from (0, 0, 0) with a huge error.
            self.x = np.concatenate([position, velocity])
            self.P = self.gps_noise.copy()
            self.initialized = True
            return
        self._update(np.concatenate([position, velocity]), np.eye(6), self.gps_noise)

    def update_barometer(self, altitude):
        H = np.zeros((1, 6))
        H[0, 2] = 1.0  # the barometer sees only z
        self._update(np.array([altitude]), H, np.array([[self.barometer_noise]]))

    @property
    def position(self):
        return self.x[0:3]

    @property
    def velocity(self):
        return self.x[3:6]


class StateEstimator:
    """Runs both estimators on each set of sensor readings and returns an Estimate.

    `ready` is False during the start-up gyro calibration; don't fly until it's True.
    """

    def __init__(self):
        self.attitude = AttitudeEstimator()
        self.position = PositionEstimator()

    def update(self, readings, dt):
        euler = self.attitude.update(readings["gyro"], readings["accel"], readings["magnetometer"], dt)
        rotation = euler_to_rotation(*euler)
        self.position.predict(readings["accel"], rotation, dt)
        if readings["gps"] is not None:
            self.position.update_gps(readings["gps"]["position"], readings["gps"]["velocity"])
        if readings["barometer"] is not None:
            self.position.update_barometer(readings["barometer"])
        return Estimate(self.position.position.copy(), self.position.velocity.copy(),
                        euler.copy(), self.attitude.body_rates.copy())

    @property
    def ready(self):
        return self.attitude.ready and self.position.initialized
