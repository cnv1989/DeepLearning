"""Example 1: the physical drone. Motors, mixing, hovering and flight time.

    python examples/01_hover_and_motors.py

Guides: docs/01-drone-hardware.md, docs/04-motors-and-mixing.md, docs/05-battery.md
"""

import _setup  # noqa: F401
import numpy as np

from drone.battery import Battery
from drone.control import Setpoint
from drone.dynamics import State
from drone.motors import Mixer
from drone.params import QuadParams
from drone.plots import plot_flight
from drone.simulation import SimConfig, Simulator

params = QuadParams()
print("The drone")
print(f"  mass {params.mass} kg, weight {params.weight:.2f} N")
print(f"  max thrust {4 * params.max_thrust_per_motor:.1f} N -> thrust-to-weight {params.thrust_to_weight:.2f}")
print(f"  hover motor speed {params.hover_motor_speed:.0f} rad/s "
      f"({params.hover_motor_speed * 60 / (2 * np.pi):.0f} rpm, "
      f"{params.hover_motor_speed / params.max_motor_speed:.0%} of max)")

print("\nThe mixer: what each request does to the four motors (rad/s)")
print("  motors:                    m0 front-right  m1 back-left  m2 front-left  m3 back-right")
mixer = Mixer(params)
requests = {
    "hover": (params.weight, [0, 0, 0]),
    "climb (+20% thrust)": (params.weight * 1.2, [0, 0, 0]),
    "roll right (+x torque)": (params.weight, [0.1, 0, 0]),
    "pitch nose down (+y)": (params.weight, [0, 0.1, 0]),
    "yaw left (+z torque)": (params.weight, [0, 0, 0.02]),
}
for name, (thrust, torques) in requests.items():
    speeds = mixer.motor_speeds(thrust, np.array(torques, dtype=float))
    print(f"  {name:<26}" + "".join(f"{s:14.0f}" for s in speeds))

print("\nHovering for 10 s at 3 m with perfect state knowledge...")
simulator = Simulator(config=SimConfig(duration=10, use_estimator=False),
                      initial_state=State(position=np.array([0.0, 0.0, 3.0])))
log = simulator.run(setpoint_function=lambda t: Setpoint(np.array([0.0, 0.0, 3.0])))
print(f"  final position {np.round(log.position[-1], 3)}")
print(f"  motor speeds   {np.round(log.motor_speeds[-1])}")

battery = simulator.battery
print(f"  battery: {battery.voltage():.2f} V, drawing {battery.current:.1f} A, "
      f"{1 - battery.charge:.1%} used in 10 s")
print(f"  -> estimated hover time on a full battery: "
      f"{Battery(params).charge * params.battery_capacity_mah / 1000 * 3600 / battery.current / 60:.0f} minutes")
plot_flight(log, f"{_setup.OUTPUT}/01_hover.png", "Hover at 3 m")
