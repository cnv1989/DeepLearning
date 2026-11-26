import numpy as np

from drone.battery import Battery
from drone.dynamics import QuadcopterDynamics, State
from drone.motors import Mixer, Motors, allocation_matrix
from drone.params import GRAVITY, QuadParams
from drone.rotations import (euler_to_quat, euler_to_rotation, quat_to_euler, quat_to_rotation,
                             rotation_to_euler, wrap_angle)

PARAMS = QuadParams()


def test_rotation_round_trips():
    for angles in [(0.1, -0.2, 0.7), (-0.4, 0.3, -2.9), (0.0, 0.0, 3.1)]:
        R = euler_to_rotation(*angles)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.allclose(rotation_to_euler(R), angles)
        assert np.allclose(quat_to_rotation(euler_to_quat(*angles)), R)
        assert np.allclose(quat_to_euler(euler_to_quat(*angles)), angles)


def test_sign_conventions():
    # Positive pitch tilts the thrust forward (+x); positive roll tilts it right (-y).
    assert euler_to_rotation(0, 0.3, 0)[0, 2] > 0
    assert euler_to_rotation(0.3, 0, 0)[1, 2] < 0
    assert np.isclose(wrap_angle(np.radians(350)), np.radians(-10))


def test_hover_thrust_balances_weight():
    dynamics = QuadcopterDynamics(PARAMS, State(position=np.array([0.0, 0.0, 5.0])))
    for _ in range(500):
        dynamics.step(PARAMS.weight, np.zeros(3), 0.002)
    assert np.allclose(dynamics.state.position, [0, 0, 5], atol=1e-9)


def test_free_fall_matches_physics():
    no_drag = QuadParams(drag_coefficient=0.0)
    dynamics = QuadcopterDynamics(no_drag, State(position=np.array([0.0, 0.0, 100.0])))
    for _ in range(500):  # 1 second
        dynamics.step(0.0, np.zeros(3), 0.002)
    assert np.isclose(dynamics.state.position[2], 100 - 0.5 * GRAVITY, atol=1e-6)
    assert np.isclose(dynamics.state.velocity[2], -GRAVITY, atol=1e-6)


def test_torque_spins_the_body():
    dynamics = QuadcopterDynamics(PARAMS, State(position=np.array([0.0, 0.0, 5.0])))
    dynamics.step(PARAMS.weight, np.array([0.01, 0.0, 0.0]), 0.002)
    expected_rate = 0.01 / PARAMS.inertia[0, 0] * 0.002  # α = τ / I
    assert np.isclose(dynamics.state.body_rates[0], expected_rate, rtol=1e-6)


def test_ground_stops_the_drone():
    dynamics = QuadcopterDynamics(PARAMS, State(position=np.array([0.0, 0.0, 0.01])))
    for _ in range(200):
        dynamics.step(0.0, np.zeros(3), 0.002)
    assert dynamics.state.position[2] == 0.0 and dynamics.on_ground


def test_mixer_inverts_allocation():
    mixer = Mixer(PARAMS)
    thrust, torques = 12.0, np.array([0.05, -0.03, 0.01])
    speeds = mixer.motor_speeds(thrust, torques)
    produced = allocation_matrix(PARAMS) @ (PARAMS.thrust_coefficient * speeds ** 2)
    assert np.allclose(produced, [thrust, *torques])


def test_mixer_directions():
    mixer = Mixer(PARAMS)
    hover = PARAMS.weight
    roll_right = mixer.motor_speeds(hover, np.array([0.1, 0, 0]))  # positive roll: left side up
    left_motors, right_motors = [1, 2], [0, 3]
    assert roll_right[left_motors].mean() > roll_right[right_motors].mean()
    yaw_left = mixer.motor_speeds(hover, np.array([0, 0, 0.02]))  # +z yaw comes from the CW motors
    assert yaw_left[[2, 3]].mean() > yaw_left[[0, 1]].mean()


def test_motor_lag():
    motors = Motors(PARAMS)
    motors.update(np.full(4, 500.0), PARAMS.motor_time_constant)  # one time constant
    assert np.allclose(motors.speeds, 500 * (1 - np.exp(-1)))


def test_battery_drains_and_sags():
    battery = Battery(PARAMS)
    full_voltage = battery.voltage()
    battery.draw(mechanical_power=200.0, dt=60.0)
    assert battery.charge < 1.0
    assert battery.voltage() < full_voltage
