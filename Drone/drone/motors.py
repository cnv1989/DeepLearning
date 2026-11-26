"""Motors, propellers and the mixer.

A quadcopter has no control surfaces like a plane's ailerons or rudder. Every
movement comes from changing the speeds of four motors:

    all four faster            -> more total thrust: climb
    left pair faster than right  -> roll right
    back pair faster than front -> pitch nose down: fly forward
    CCW pair faster than CW pair -> yaw clockwise (reaction torque)

Each spinning propeller produces
    thrust      F = k_f · ω²   (up, along the body z axis)
    drag torque Q = k_m · ω²   (resisting its spin; the body feels the opposite)

The MIXER turns what the flight controller wants (total thrust and three
torques) into four motor speeds. It's the inverse of the "allocation matrix",
which says what thrust and torques a given set of motor thrusts produces.
"""

import numpy as np


def allocation_matrix(params):
    """4x4 matrix A with  [T, τx, τy, τz] = A @ [F0, F1, F2, F3].

    T   total thrust                     = sum of motor thrusts
    τx  roll torque  (about body x)      = sum of  y_i · F_i
    τy  pitch torque (about body y)      = sum of -x_i · F_i
    τz  yaw torque   (about body z)      = sum of -direction_i · (k_m / k_f) · F_i
    """
    xs, ys = params.motor_positions[:, 0], params.motor_positions[:, 1]
    torque_per_thrust = params.torque_coefficient / params.thrust_coefficient
    return np.array([
        np.ones(4),
        ys,
        -xs,
        -params.motor_directions * torque_per_thrust,
    ])


class Mixer:
    """Converts a desired total thrust and body torques into four motor speeds."""

    def __init__(self, params):
        self.params = params
        self.inverse_allocation = np.linalg.inv(allocation_matrix(params))

    def motor_speeds(self, thrust, torques, max_motor_speed=None):
        """Desired (thrust in N, torques in N·m) -> motor speeds in rad/s.

        Motors can only push, not pull, and have a top speed. When the request
        is impossible, the thrusts are clipped to what the motors can do. A
        real flight controller is smarter: it gives up yaw first, to keep roll
        and pitch (which keep it upright) working.
        """
        max_speed = max_motor_speed or self.params.max_motor_speed
        max_thrust = self.params.thrust_coefficient * max_speed ** 2
        motor_thrusts = self.inverse_allocation @ np.array([thrust, *torques])
        motor_thrusts = np.clip(motor_thrusts, 0.0, max_thrust)
        return np.sqrt(motor_thrusts / self.params.thrust_coefficient)


class Motors:
    """Four motors whose speeds follow their commands with a short delay.

    Real motors can't change speed instantly: the rotor has inertia and the
    speed controller (ESC) takes time. A first-order lag models this: each
    time constant, the speed closes 63% of the gap to the command.
    """

    def __init__(self, params):
        self.params = params
        self.speeds = np.zeros(4)  # rad/s

    def update(self, commanded_speeds, dt):
        blend = 1.0 - np.exp(-dt / self.params.motor_time_constant)
        self.speeds = self.speeds + blend * (commanded_speeds - self.speeds)
        return self.speeds

    def thrusts(self):
        return self.params.thrust_coefficient * self.speeds ** 2

    def thrust_and_torques(self):
        """Total thrust (N) and body torques (N·m) the motors are producing now."""
        forces = allocation_matrix(self.params) @ self.thrusts()
        return forces[0], forces[1:]

    def mechanical_power(self):
        """Power the propellers take from the motors (W): torque × speed for each."""
        return float(np.sum(self.params.torque_coefficient * self.speeds ** 3))
