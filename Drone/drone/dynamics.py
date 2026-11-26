"""Rigid-body physics: how forces and torques move the drone.

This is the "real world" of the simulation. The rest of the software never
reads this state directly (a real drone can't); it only sees sensor readings.

The state of a rigid body in 3D:

    position     p  (world, m)
    velocity     v  (world, m/s)
    attitude     q  (quaternion, body -> world)
    body rates   ω  (body, rad/s): how fast it's spinning about its own axes

Newton's laws give how the state changes:

    dp/dt = v
    dv/dt = gravity + (R · [0, 0, thrust]) / mass + drag / mass
    dq/dt = 1/2 · q ⊗ (0, ω)
    dω/dt = I⁻¹ · (τ − ω × (I · ω))

The last line is Euler's rotation equation. The ω × Iω term is the
gyroscopic effect: a spinning body resists changes to its spin axis.

Equations like these are integrated numerically, here with the classic
4th-order Runge-Kutta method (RK4), which is far more accurate than simply
adding derivative × dt.
"""

from dataclasses import dataclass, field

import numpy as np

from drone.params import GRAVITY
from drone.rotations import quat_derivative, quat_normalize, quat_to_rotation

GRAVITY_VECTOR = np.array([0.0, 0.0, -GRAVITY])


@dataclass
class State:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    body_rates: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_vector(self):
        return np.concatenate([self.position, self.velocity, self.attitude, self.body_rates])

    @classmethod
    def from_vector(cls, x):
        return cls(x[0:3].copy(), x[3:6].copy(), x[6:10].copy(), x[10:13].copy())

    def rotation(self):
        return quat_to_rotation(self.attitude)


class QuadcopterDynamics:
    def __init__(self, params, state=None):
        self.params = params
        self.state = state or State()
        self.inertia = params.inertia
        self.inverse_inertia = np.linalg.inv(params.inertia)
        self.acceleration = np.zeros(3)  # last world-frame acceleration; the accelerometer reads it
        self.wind = np.zeros(3)  # world-frame wind velocity, m/s

    def derivative(self, x, thrust, torques):
        """d(state)/dt for state vector x, given motor thrust (N) and body torques (N·m)."""
        velocity, attitude, rates = x[3:6], x[6:10], x[10:13]
        rotation = quat_to_rotation(attitude)

        thrust_world = rotation @ np.array([0.0, 0.0, thrust])
        # Drag pushes against motion relative to the air (so wind pushes the drone).
        drag = -self.params.drag_coefficient * (velocity - self.wind)
        acceleration = GRAVITY_VECTOR + (thrust_world + drag) / self.params.mass

        gyroscopic = np.cross(rates, self.inertia @ rates)
        angular_acceleration = self.inverse_inertia @ (torques - gyroscopic)

        return np.concatenate([velocity, acceleration, quat_derivative(attitude, rates), angular_acceleration])

    def step(self, thrust, torques, dt):
        """Advance the physics by dt seconds with RK4. Thrust and torques are held constant."""
        x = self.state.as_vector()
        k1 = self.derivative(x, thrust, torques)
        k2 = self.derivative(x + 0.5 * dt * k1, thrust, torques)
        k3 = self.derivative(x + 0.5 * dt * k2, thrust, torques)
        k4 = self.derivative(x + dt * k3, thrust, torques)
        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[6:10] = quat_normalize(x_next[6:10])

        self.acceleration = (x_next[3:6] - x[3:6]) / dt
        self.state = State.from_vector(x_next)
        self._ground_contact()
        return self.state

    def _ground_contact(self):
        """The ground at z = 0 is solid: the drone can sit on it but not sink into it."""
        if self.state.position[2] < 0.0:
            self.state.position[2] = 0.0
            self.state.velocity[:] = 0.0
            self.state.body_rates[:] = 0.0
            self.acceleration = np.zeros(3)

    @property
    def on_ground(self):
        return self.state.position[2] <= 1e-6
