# Drone guides

One guide per component, in the order a drone's software uses them. Each
explains what the component is, why it matters, how it works, where it lives
in the code, the results from this simulator, and something to try.

| # | Guide | You'll learn |
|---|---|---|
| 1 | [The parts of a drone](01-drone-hardware.md) | frame, motors, ESCs, battery, flight controller, sensors; why quadcopters need computers to fly |
| 2 | [Frames and rotations](02-frames-and-rotations.md) | world vs body frame; rotation matrices, Euler angles, quaternions; gimbal lock |
| 3 | [Physics](03-physics.md) | Newton's and Euler's equations; why a drone must tilt to move; RK4 integration |
| 4 | [Motors and mixing](04-motors-and-mixing.md) | thrust and drag torque; the mixer; why yaw is weak; saturation; motor lag |
| 5 | [The battery](05-battery.md) | LiPo voltage, sag under load, flight time, low-battery failsafe |
| 6 | [Sensors](06-sensors.md) | gyro, accelerometer, magnetometer, barometer, GPS; the accelerometer trap |
| 7 | [State estimation](07-state-estimation.md) | complementary filter, Kalman filter, gyro calibration; the bug that flipped the drone |
| 8 | [Control](08-control.md) | PID, the control cascade, acceleration to attitude, tuning, integral action against wind |
| 9 | [Navigation and planning](09-navigation-and-planning.md) | occupancy grids, A*, shortcutting, the trapezoidal speed profile |
| 10 | [Flight modes and failsafes](10-flight-modes.md) | the flight as a state machine; arming; failsafes |
| 11 | [The simulation loop](11-the-simulation-loop.md) | how it all connects; time steps; truth vs estimate; flight logs and plots |
| 12 | [From simulation to real drones](12-from-simulation-to-real-drones.md) | how this maps to PX4 and ArduPilot; SITL and MAVLink; safety; extensions |

**Where to start:** 1, then 3, 4 and 8, which explain how a drone flies. Then 6
and 7, for how it knows where it is. Then 9–11, for flying a whole mission.
Guide 2 is a reference to come back to whenever rotations get confusing.

Every number quoted in the guides comes from running this code; rerun the
examples to reproduce them.
