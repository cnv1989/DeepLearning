"""The physical description of the quadcopter.

The defaults describe a typical 1 kg, 450 mm-class quadcopter (motor-to-motor
diagonal 450 mm) on a 4-cell lithium-polymer battery, the size of many hobby
and research drones. Change them to see how a heavier, smaller or more
powerful drone behaves.

Motor layout, seen from above ("quad X", numbered as in PX4):

              front (+x)
         m2 (CW)    m0 (CCW)
              \\    /
               [  ]          +y is left
              /    \\
         m1 (CCW)   m3 (CW)
              back

Diagonal motors spin the same way. Two spin clockwise (CW) and two
counter-clockwise (CCW), so their reaction torques cancel in a hover.
"""

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81  # m/s²


@dataclass
class QuadParams:
    # ---- body ----
    mass: float = 1.0  # kg, including battery
    arm_length: float = 0.225  # m, centre to each motor
    # Moments of inertia (kg·m²): resistance to spinning about each body axis.
    # Yaw (z) is largest because all four motors are far from the z axis.
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.0082, 0.0082, 0.0149]))
    drag_coefficient: float = 0.25  # N per (m/s): air resistance on the whole body

    # ---- motors and propellers ----
    # thrust = thrust_coefficient * ω²   and   drag torque = torque_coefficient * ω²
    thrust_coefficient: float = 6.8e-6  # N / (rad/s)²
    torque_coefficient: float = 1.1e-7  # N·m / (rad/s)²
    max_motor_speed: float = 1000.0  # rad/s (about 9,500 rpm) at full battery voltage
    motor_time_constant: float = 0.02  # s: how quickly a motor reaches a new speed

    # ---- battery (4S LiPo) ----
    battery_cells: int = 4
    battery_capacity_mah: float = 3000.0
    battery_internal_resistance: float = 0.02  # ohms
    propulsion_efficiency: float = 0.6  # electrical power -> mechanical power at the props

    @property
    def weight(self):
        return self.mass * GRAVITY

    @property
    def max_thrust_per_motor(self):
        return self.thrust_coefficient * self.max_motor_speed ** 2

    @property
    def thrust_to_weight(self):
        """Total maximum thrust divided by weight. Above ~2 is needed for agile, safe flight."""
        return 4 * self.max_thrust_per_motor / self.weight

    @property
    def hover_motor_speed(self):
        """Motor speed at which the four motors exactly hold up the drone's weight."""
        return np.sqrt(self.weight / (4 * self.thrust_coefficient))

    @property
    def motor_positions(self):
        """(x, y) of each motor in the body frame, in motor-number order."""
        d = self.arm_length / np.sqrt(2)  # arms are at 45° to the body axes
        return np.array([[d, -d], [-d, d], [d, d], [-d, -d]])

    @property
    def motor_directions(self):
        """+1 for counter-clockwise (seen from above), -1 for clockwise."""
        return np.array([1, 1, -1, -1])
