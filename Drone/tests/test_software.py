import numpy as np

from drone.control import PID, Setpoint, acceleration_to_attitude
from drone.dynamics import GRAVITY_VECTOR, State
from drone.estimation import (AttitudeEstimator, PositionEstimator, heading_from_magnetometer,
                              tilt_from_accelerometer)
from drone.mission import Mission
from drone.params import GRAVITY
from drone.planning import OccupancyGrid, WaypointFollower, astar, line_is_free, shortcut_path
from drone.rotations import euler_to_rotation, rotation_to_euler
from drone.sensors import MAGNETIC_FIELD_WORLD
from drone.simulation import SimConfig, Simulator

# ---------------------------------------------------------------- estimation


def test_angles_from_accelerometer_and_magnetometer():
    roll, pitch, yaw = 0.2, -0.1, 1.2
    R = euler_to_rotation(roll, pitch, yaw)
    measured_roll, measured_pitch = tilt_from_accelerometer(R.T @ -GRAVITY_VECTOR)
    assert np.allclose([measured_roll, measured_pitch], [roll, pitch])
    assert np.isclose(heading_from_magnetometer(R.T @ MAGNETIC_FIELD_WORLD, roll, pitch), yaw)


def test_complementary_filter_removes_gyro_drift():
    estimator = AttitudeEstimator(tilt_time_constant=1.0)
    level_accel = np.array([0.0, 0.0, GRAVITY])
    biased_gyro = np.array([0.05, 0.0, 0.0])  # a gyro that thinks we're always rolling
    for _ in range(5000):
        estimator.update(biased_gyro, level_accel, None, 0.002)
    # Pure integration would say 0.5 rad; the filter settles at bias × time constant.
    assert abs(estimator.euler[0]) < 0.06


def test_kalman_filter_beats_raw_gps():
    rng = np.random.default_rng(0)
    kf = PositionEstimator()
    true_position = np.array([3.0, -2.0, 5.0])
    gps_errors, kf_errors = [], []
    for step in range(5000):
        kf.predict(np.array([0.0, 0.0, GRAVITY]) + rng.normal(0, 0.1, 3), np.eye(3), 0.002)
        if step % 50 == 0:
            gps = true_position + rng.normal(0, 0.6, 3)
            kf.update_gps(gps, rng.normal(0, 0.1, 3))
            if step > 1000:
                gps_errors.append(np.linalg.norm(gps - true_position))
                kf_errors.append(np.linalg.norm(kf.position - true_position))
    assert np.mean(kf_errors) < 0.5 * np.mean(gps_errors)


# ---------------------------------------------------------------- control


def test_pid_terms():
    pid = PID(kp=2.0, ki=1.0, kd=0.5)
    assert pid.update(1.0, 0.0, 0.1) == 2.0 * 1.0 + 1.0 * 0.1  # P + I (no D on the first call)
    out = pid.update(1.0, 0.2, 0.1)  # measurement moved up by 0.2 in 0.1 s -> D opposes it
    assert np.isclose(out, 2.0 * 0.8 + 1.0 * (0.1 + 0.08) + 0.5 * -2.0)


def test_pid_integral_limit():
    pid = PID(kp=0.0, ki=1.0, integral_limit=0.5)
    for _ in range(100):
        out = pid.update(1.0, 0.0, 0.1)
    assert out == 0.5


def test_acceleration_to_attitude():
    # Hover: level, thrust = weight.
    R, thrust = acceleration_to_attitude(np.zeros(3), 0.0, 1.0, np.radians(35))
    assert np.allclose(R, np.eye(3)) and np.isclose(thrust[2], GRAVITY)
    # Accelerate forward (+x) at g·tan(20°): pitch nose-down by 20°.
    R, _ = acceleration_to_attitude(np.array([GRAVITY * np.tan(np.radians(20)), 0, 0]), 0.0, 1.0, np.radians(35))
    assert np.isclose(np.degrees(rotation_to_euler(R)[1]), 20)
    # Asking for too much is limited to the maximum tilt.
    R, _ = acceleration_to_attitude(np.array([50.0, 0, 0]), 0.0, 1.0, np.radians(35))
    assert np.isclose(np.degrees(rotation_to_euler(R)[1]), 35)


# ---------------------------------------------------------------- planning


def test_astar_avoids_walls():
    grid = OccupancyGrid(10, 10, resolution=0.5)
    grid.add_box(4, 0, 5, 8)  # a wall with a gap at the top
    path = astar(grid, (1, 1), (9, 1))
    assert path is not None
    assert all(grid.is_free(grid.to_cell(p)) for p in path)
    assert max(p[1] for p in path) > 8  # had to go round the top
    smooth = shortcut_path(grid, path)
    assert len(smooth) < len(path)
    assert all(line_is_free(grid, a, b) for a, b in zip(smooth, smooth[1:]))


def test_astar_reports_no_path():
    grid = OccupancyGrid(10, 10)
    grid.add_box(4, 0, 5, 10)  # a complete wall
    assert astar(grid, (1, 1), (9, 1)) is None


def test_waypoint_follower_speed_profile():
    follower = WaypointFollower([[0, 0, 2], [20, 0, 2]], cruise_speed=2.0, max_acceleration=1.0)
    speeds = []
    for _ in range(2000):  # 20 s at 100 Hz
        setpoint = follower.update(np.array([0.0, 0.0, 2.0]), 0.01)
        speeds.append(setpoint.velocity[0])
    assert np.isclose(speeds[100], 1.0, atol=0.02)  # accelerating: 1 m/s after 1 s
    assert max(speeds) == 2.0  # never above cruise speed
    assert speeds[-1] == 0.0  # stopped at the waypoint
    assert np.isclose(setpoint.position[0], 20.0)


# ---------------------------------------------------------------- whole system


def test_full_mission_with_estimator():
    waypoints = [[4, 0, 2], [4, 4, 2]]
    log = Simulator(config=SimConfig(duration=40)).run(Mission(waypoints, takeoff_altitude=2))
    modes = [event[1] for event in log.events]
    assert any("mission complete" in m for m in modes)
    assert any("touched down" in m for m in modes)
    assert np.linalg.norm(log.position[-1][:2] - np.array([4, 4])) < 0.6  # landed near the last waypoint
    assert np.degrees(np.abs(log.euler[:, :2]).max()) < 40  # never tipped over


def test_hover_holds_position():
    simulator = Simulator(config=SimConfig(duration=5, use_estimator=False),
                          initial_state=State(position=np.array([0.0, 0.0, 3.0])))
    log = simulator.run(setpoint_function=lambda t: Setpoint(np.array([0.0, 0.0, 3.0])))
    assert np.allclose(log.position[-1], [0, 0, 3], atol=0.05)
