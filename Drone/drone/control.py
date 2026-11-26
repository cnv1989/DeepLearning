"""The flight controller: from "go there" to motor speeds.

A quadcopter is controlled by a CASCADE of loops, each feeding the next.
Outer loops are slow and decide *where*; inner loops are fast and decide *how*:

   position setpoint
        │  PositionController       (P)    position error  -> desired velocity
        ▼
   velocity setpoint
        │  VelocityController       (PID)  velocity error  -> desired acceleration
        ▼
   acceleration setpoint
        │  acceleration_to_attitude        tilt the thrust toward the acceleration
        ▼
   attitude + thrust setpoint
        │  AttitudeController       (P)    attitude error  -> desired body rates
        ▼
   body rate setpoint
        │  RateController           (PID)  rate error      -> torques
        ▼
   thrust + torques ──► Mixer ──► four motor speeds

Why a cascade? Each loop only has to solve a simple problem, and inner loops
run much faster than the outer ones need, so to the outer loop the inner one
looks like it simply "does what it's told". PX4 and ArduPilot use this structure.
"""

from dataclasses import dataclass, field

import numpy as np

from drone.params import GRAVITY
from drone.rotations import (euler_to_rotation, quat_conjugate, quat_multiply, rotation_to_euler,
                             rotation_to_quat, wrap_angle)


class PID:
    """Proportional-Integral-Derivative controller, for one or more axes at once.

    output = kp · error  +  ki · ∫error dt  +  kd · d(error)/dt

    P  pushes toward the target, harder the further away it is.
    I  builds up over time to remove steady errors (wind, a heavy side).
    D  opposes fast changes, damping overshoot and oscillation.

    Practical details included:
      * Derivative on the measurement, not the error, so a sudden change of
        target doesn't cause a "derivative kick".
      * Integral limit ("anti-windup"): the integral can't grow without bound
        while the output is saturated or the drone is held on the ground.
      * Output limit.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, integral_limit=np.inf, output_limit=np.inf):
        self.kp, self.ki, self.kd = (np.asarray(g, dtype=float) for g in (kp, ki, kd))
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_measurement = None

    def update(self, setpoint, measurement, dt, feedforward=0.0):
        setpoint, measurement = np.asarray(setpoint, float), np.asarray(measurement, float)
        error = setpoint - measurement

        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)

        if self.previous_measurement is None:
            derivative = np.zeros_like(measurement)
        else:
            derivative = -(measurement - self.previous_measurement) / dt
        self.previous_measurement = measurement

        output = feedforward + self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.output_limit, self.output_limit)


@dataclass
class ControllerGains:
    position_p: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 1.5]))
    velocity_p: np.ndarray = field(default_factory=lambda: np.array([3.5, 3.5, 4.0]))
    velocity_i: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 2.0]))
    velocity_d: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    attitude_p: np.ndarray = field(default_factory=lambda: np.array([7.0, 7.0, 3.0]))
    rate_p: np.ndarray = field(default_factory=lambda: np.array([25.0, 25.0, 10.0]))
    rate_i: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 2.0]))
    rate_d: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.0]))
    max_speed_horizontal: float = 5.0  # m/s
    max_speed_vertical: float = 2.0  # m/s
    max_tilt: float = np.radians(35)
    max_body_rate: float = np.radians(220)  # rad/s


@dataclass
class Setpoint:
    """What the mission asks for at this moment."""
    position: np.ndarray
    yaw: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # feedforward, if known


class PositionController:
    """Position error -> velocity setpoint (P control, speed-limited)."""

    def __init__(self, gains):
        self.gains = gains

    def update(self, setpoint, estimate):
        velocity = setpoint.velocity + self.gains.position_p * (setpoint.position - estimate.position)
        horizontal_speed = np.linalg.norm(velocity[:2])
        if horizontal_speed > self.gains.max_speed_horizontal:
            velocity[:2] *= self.gains.max_speed_horizontal / horizontal_speed
        velocity[2] = np.clip(velocity[2], -self.gains.max_speed_vertical, self.gains.max_speed_vertical)
        return velocity


class VelocityController:
    """Velocity error -> acceleration setpoint (PID). The I term absorbs wind and trim errors."""

    def __init__(self, gains):
        self.pid = PID(gains.velocity_p, gains.velocity_i, gains.velocity_d, integral_limit=3.0)

    def update(self, velocity_setpoint, estimate, dt):
        return self.pid.update(velocity_setpoint, estimate.velocity, dt)


def acceleration_to_attitude(acceleration, yaw, mass, max_tilt):
    """Which way to tilt, and how hard to push, to get a desired acceleration.

    A quadcopter can only push along its own z axis. So to accelerate in some
    direction, it tilts that axis: the thrust vector must equal
        mass × (desired acceleration − gravity)
    Its direction becomes the desired body z axis; its length is the thrust.
    The yaw setpoint fixes the rotation about that axis.

    Returns (desired rotation matrix, thrust vector in world frame).
    """
    thrust_vector = mass * (acceleration + np.array([0.0, 0.0, GRAVITY]))
    thrust_vector[2] = max(thrust_vector[2], 0.1 * mass * GRAVITY)  # never ask to push downward

    # Limit the tilt: shrink the horizontal part if the vector leans too far.
    horizontal = np.linalg.norm(thrust_vector[:2])
    max_horizontal = thrust_vector[2] * np.tan(max_tilt)
    if horizontal > max_horizontal:
        thrust_vector[:2] *= max_horizontal / horizontal

    body_z = thrust_vector / np.linalg.norm(thrust_vector)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    body_y = np.cross(body_z, heading)
    body_y /= np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    return np.column_stack([body_x, body_y, body_z]), thrust_vector


class AttitudeController:
    """Attitude error -> body rate setpoint (P control on the quaternion error).

    The error is the rotation from the current attitude to the desired one,
    q_error = q_current⁻¹ ⊗ q_desired. For small errors, its vector part is
    half the rotation angle about each body axis, so 2 × vector part is the
    angle error, already in the body frame where the rate loop works.
    """

    def __init__(self, gains):
        self.gains = gains

    def update(self, desired_rotation, estimate):
        q_desired = rotation_to_quat(desired_rotation)
        q_error = quat_multiply(quat_conjugate(estimate.attitude), q_desired)
        if q_error[0] < 0:
            q_error = -q_error  # take the short way round
        angle_error = 2.0 * q_error[1:]
        rates = self.gains.attitude_p * angle_error
        return np.clip(rates, -self.gains.max_body_rate, self.gains.max_body_rate)


class RateController:
    """Body rate error -> torques (PID), scaled by inertia so gains mean "rad/s² per rad/s"."""

    def __init__(self, gains, inertia):
        self.pid = PID(gains.rate_p, gains.rate_i, gains.rate_d, integral_limit=0.5)
        self.inertia = inertia

    def update(self, rate_setpoint, estimate, dt):
        angular_acceleration = self.pid.update(rate_setpoint, estimate.body_rates, dt)
        rates = estimate.body_rates
        gyroscopic = np.cross(rates, self.inertia @ rates)  # cancel the gyroscopic effect
        return self.inertia @ angular_acceleration + gyroscopic


@dataclass
class ControlOutput:
    thrust: float
    torques: np.ndarray
    # Intermediate setpoints, kept for plotting and understanding.
    velocity_setpoint: np.ndarray
    acceleration_setpoint: np.ndarray
    euler_setpoint: np.ndarray
    rate_setpoint: np.ndarray


class FlightController:
    """The full cascade. Call update() once per control step."""

    def __init__(self, params, gains=None):
        self.params = params
        self.gains = gains or ControllerGains()
        self.position = PositionController(self.gains)
        self.velocity = VelocityController(self.gains)
        self.attitude = AttitudeController(self.gains)
        self.rate = RateController(self.gains, params.inertia)

    def reset_integrals(self):
        """Clear built-up integral terms, e.g. while sitting on the ground."""
        self.velocity.pid.reset()
        self.rate.pid.reset()

    def update(self, setpoint, estimate, dt):
        velocity_sp = self.position.update(setpoint, estimate)
        acceleration_sp = self.velocity.update(velocity_sp, estimate, dt)
        desired_rotation, thrust_vector = acceleration_to_attitude(
            acceleration_sp, setpoint.yaw, self.params.mass, self.gains.max_tilt)

        # Thrust: project the desired thrust vector onto where the body z axis points now,
        # so a drone that's still tilting doesn't shoot off sideways.
        current_body_z = estimate.rotation()[:, 2]
        thrust = max(float(thrust_vector @ current_body_z), 0.0)

        rate_sp = self.attitude.update(desired_rotation, estimate)
        torques = self.rate.update(rate_sp, estimate, dt)

        return ControlOutput(thrust, torques, velocity_sp, acceleration_sp,
                             rotation_to_euler(desired_rotation), rate_sp)


def euler_setpoint_to_rotation(roll, pitch, yaw):
    """Helper for examples that command an attitude directly."""
    return euler_to_rotation(roll, pitch, wrap_angle(yaw))
