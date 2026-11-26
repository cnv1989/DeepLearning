"""The simulation loop: one pass per time step, in the same order as a real drone.

    ┌──────────────────────────────── one step (2 ms) ───────────────────────────────┐
    │                                                                                │
    │  true state ─► sensors ─► estimator ─► mission ─► flight controller ─► mixer   │
    │      ▲                                                                   │     │
    │      └──── physics ◄── motor thrusts & torques ◄── motors ◄── battery ◄──┘     │
    └────────────────────────────────────────────────────────────────────────────────┘

Everything left of the physics is "flight software" and only sees sensor
readings. The physics and the true state are the "real world". Keeping them
apart is what makes the simulation honest: with `use_estimator=False` the
controller is handed the true state instead, which is useful for learning
control on its own, but no real drone gets that luxury.

Real autopilots run each piece at its own rate (rate loop 1-8 kHz, position
loop 50-250 Hz, GPS 5-10 Hz). Here everything runs at 500 Hz except the
slow sensors, which keeps the code simple.
"""

from dataclasses import dataclass, field

import numpy as np

from drone.battery import Battery
from drone.control import FlightController
from drone.dynamics import QuadcopterDynamics, State
from drone.estimation import StateEstimator, estimate_from_true_state
from drone.mission import Mode
from drone.motors import Mixer, Motors
from drone.params import QuadParams
from drone.rotations import quat_to_euler
from drone.sensors import Sensors


@dataclass
class SimConfig:
    dt: float = 0.002  # s per step (500 Hz)
    duration: float = 60.0  # s, upper limit
    use_estimator: bool = True  # False: the controller sees the true state
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))  # steady wind, m/s (world)
    gust_strength: float = 0.0  # m/s, size of random gusts
    gust_time_constant: float = 2.0  # s, how quickly gusts change
    battery_charge: float = 1.0  # starting state of charge
    log_every: int = 5  # record every Nth step
    seed: int = 0


class FlightLog:
    """Everything recorded during a flight, as numpy arrays after `finish()`."""

    FIELDS = ["time", "position", "velocity", "euler", "body_rates",
              "est_position", "est_velocity", "est_euler",
              "setpoint_position", "setpoint_euler", "velocity_setpoint",
              "motor_speeds", "thrust", "battery_voltage", "battery_charge", "wind", "mode"]

    def __init__(self):
        self.data = {name: [] for name in self.FIELDS}
        self.gps = []  # (time, measured position)
        self.events = []

    def record(self, **values):
        for name, value in values.items():
            self.data[name].append(np.copy(value) if isinstance(value, np.ndarray) else value)

    def finish(self):
        for name, values in self.data.items():
            setattr(self, name, np.array(values))
        self.gps_time = np.array([t for t, _ in self.gps])
        self.gps_position = np.array([p for _, p in self.gps]).reshape(-1, 3)
        return self


class Simulator:
    def __init__(self, params=None, gains=None, config=None, sensor_noise=None, initial_state=None):
        self.params = params or QuadParams()
        self.config = config or SimConfig()
        self.dynamics = QuadcopterDynamics(self.params, initial_state or State())
        self.motors = Motors(self.params)
        self.mixer = Mixer(self.params)
        self.battery = Battery(self.params, self.config.battery_charge)
        self.sensors = Sensors(noise=sensor_noise, seed=self.config.seed)
        self.estimator = StateEstimator()
        self.controller = FlightController(self.params, gains)
        self.random = np.random.default_rng(self.config.seed + 1)
        self.gust = np.zeros(3)

        # Start the motors at hover speed if we begin in the air, so the drone doesn't drop.
        if self.dynamics.state.position[2] > 0:
            self.motors.speeds[:] = self.params.hover_motor_speed

    def _update_wind(self, dt):
        """Steady wind plus random gusts that wander smoothly (an Ornstein-Uhlenbeck process)."""
        c = self.config
        if c.gust_strength > 0:
            decay = -self.gust / c.gust_time_constant * dt
            kick = c.gust_strength * np.sqrt(2 * dt / c.gust_time_constant) * self.random.normal(size=3)
            kick[2] *= 0.3  # vertical gusts are weaker
            self.gust = self.gust + decay + kick
        self.dynamics.wind = c.wind + self.gust

    def _preflight_calibration(self):
        """Hold the drone still while the estimator calibrates its gyro, before the clock starts.

        On a real drone this happens on the ground after power-on. Here it also
        happens for flights that start in mid-air, as if the drone had been
        calibrated before being placed there.
        """
        still = State(self.dynamics.state.position.copy(), np.zeros(3),
                      self.dynamics.state.attitude.copy(), np.zeros(3))
        time = 0.0
        while not self.estimator.ready:
            self.estimator.update(self.sensors.read(still, np.zeros(3), time), self.config.dt)
            time += self.config.dt
        self.sensors.restart_clock()

    def run(self, mission=None, setpoint_function=None):
        """Fly a Mission, or follow `setpoint_function(time) -> Setpoint`, and return the log."""
        assert (mission is None) != (setpoint_function is None), "give a mission or a setpoint function"
        c, params = self.config, self.params
        log = FlightLog()
        self._preflight_calibration()

        steps = int(round(c.duration / c.dt))
        landed_at = None
        for step in range(steps):
            time = step * c.dt
            state = self.dynamics.state

            # --- flight software: sense, estimate, decide, control ---
            readings = self.sensors.read(state, self.dynamics.acceleration, time)
            if readings["gps"] is not None:
                log.gps.append((time, readings["gps"]["position"]))
            estimate = self.estimator.update(readings, c.dt)
            control_estimate = estimate if c.use_estimator else estimate_from_true_state(state)

            if mission is not None and mission.mode == Mode.DISARMED and self.estimator.ready:
                mission.arm(self.dynamics.state.position, time)  # sensors calibrated: take off
            if mission is not None:
                setpoint = mission.update(control_estimate, self.battery.charge,
                                          self.dynamics.on_ground, time, c.dt)
                motors_on = mission.motors_on
            else:
                setpoint, motors_on = setpoint_function(time), True

            if motors_on:
                output = self.controller.update(setpoint, control_estimate, c.dt)
                if self.dynamics.on_ground and output.thrust < params.weight:
                    self.controller.reset_integrals()  # don't wind up while sitting on the ground
                max_speed = params.max_motor_speed * min(1.0, self.battery.voltage_fraction())
                commanded = self.mixer.motor_speeds(output.thrust, output.torques, max_speed)
            else:
                output = None
                self.controller.reset_integrals()
                commanded = np.zeros(4)

            # --- the real world: motors, battery, wind, physics ---
            self.motors.update(commanded, c.dt)
            thrust, torques = self.motors.thrust_and_torques()
            self.battery.draw(self.motors.mechanical_power(), c.dt)
            self._update_wind(c.dt)
            self.dynamics.step(thrust, torques, c.dt)

            if step % c.log_every == 0:
                log.record(
                    time=time, position=state.position, velocity=state.velocity,
                    euler=quat_to_euler(state.attitude), body_rates=state.body_rates,
                    est_position=estimate.position, est_velocity=estimate.velocity, est_euler=estimate.euler,
                    setpoint_position=setpoint.position,
                    setpoint_euler=output.euler_setpoint if output else np.zeros(3),
                    velocity_setpoint=output.velocity_setpoint if output else np.zeros(3),
                    motor_speeds=self.motors.speeds, thrust=thrust,
                    battery_voltage=self.battery.voltage(), battery_charge=self.battery.charge,
                    wind=self.dynamics.wind,
                    mode=mission.mode.value if mission else "setpoint",
                )

            if mission is not None and mission.mode == Mode.LANDED:
                landed_at = landed_at if landed_at is not None else time
                if time - landed_at > 1.0:
                    break  # stop shortly after landing

        if mission is not None:
            log.events = mission.events
        return log.finish()
