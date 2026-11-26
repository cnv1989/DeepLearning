"""A lithium-polymer (LiPo) battery.

Flight time is limited by the battery, and its voltage limits how fast the
motors can spin. This model captures the two effects that matter:

  * The voltage of a resting cell falls as it empties: about 4.2 V full,
    3.7 V in the middle, 3.3 V empty.
  * Drawing current makes the voltage sag further, by current × internal
    resistance. Under heavy load a battery can look emptier than it is.

Motor top speed is roughly proportional to voltage, so a sagging battery
leaves less spare thrust for manoeuvring. That's one reason drones feel
sluggish near the end of a flight.
"""

import numpy as np

# Resting voltage per cell at a given state of charge (0 = empty, 1 = full).
CHARGE_POINTS = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
CELL_VOLTAGE_POINTS = [3.3, 3.55, 3.65, 3.8, 3.95, 4.2]
FULL_CELL_VOLTAGE = 4.2


class Battery:
    def __init__(self, params, charge=1.0):
        self.params = params
        self.charge = charge  # state of charge, 0..1
        self.current = 0.0  # amps

    def resting_voltage(self):
        cell = np.interp(self.charge, CHARGE_POINTS, CELL_VOLTAGE_POINTS)
        return cell * self.params.battery_cells

    def voltage(self):
        """Voltage at the terminals, after sag from the current being drawn."""
        return self.resting_voltage() - self.current * self.params.battery_internal_resistance

    def voltage_fraction(self):
        """Voltage relative to a full battery; scales the motors' top speed."""
        return self.voltage() / (FULL_CELL_VOLTAGE * self.params.battery_cells)

    def draw(self, mechanical_power, dt):
        """Supply `mechanical_power` watts to the propellers for dt seconds."""
        electrical_power = mechanical_power / self.params.propulsion_efficiency
        self.current = electrical_power / max(self.resting_voltage(), 1e-3)
        capacity_amp_seconds = self.params.battery_capacity_mah / 1000 * 3600
        self.charge = max(0.0, self.charge - self.current * dt / capacity_amp_seconds)

    def estimated_flight_time(self):
        """Seconds of flight left at the current draw (rough)."""
        if self.current <= 0:
            return float("inf")
        remaining_amp_seconds = self.charge * self.params.battery_capacity_mah / 1000 * 3600
        return remaining_amp_seconds / self.current
