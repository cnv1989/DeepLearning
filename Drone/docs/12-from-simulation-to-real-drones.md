# 12. From this simulator to real drones

This simulator implements the core ideas of real flight software in their
simplest form. Here's how each part maps to the real thing, and where to go next.

## How the pieces map

| This project | PX4 | ArduPilot |
|---|---|---|
| `estimation.py`: complementary filter + Kalman filter | EKF2 | EKF3 |
| `control.py`: `RateController` | `mc_rate_control` | rate controllers in `AC_AttitudeControl` |
| `control.py`: `AttitudeController` | `mc_att_control` | `AC_AttitudeControl` |
| `control.py`: position and velocity | `mc_pos_control` | `AC_PosControl` |
| `motors.py`: `Mixer` | control allocation | `AP_Motors` |
| `mission.py`: modes | flight modes, Commander | flight modes |
| `planning.py`: `WaypointFollower` | FlightTasks, trajectory generation | `AC_WPNav` |
| `simulation.py` | SITL with Gazebo / jMAVSim | SITL |
| `FlightLog` + `plots.py` | ULog + Flight Review | DataFlash logs |

The control structure (a P loop on position, PID on velocity, P on attitude,
PID on rates) is the same in both autopilots, and the names of their tuning
parameters will look familiar (for example, PX4's `MPC_XY_P`,
`MPC_XY_VEL_P_ACC`, `MC_ROLL_P` and `MC_ROLLRATE_P`).

## The next step: software-in-the-loop (SITL)

Real autopilot code can run on your computer, flying a simulated drone:

```bash
# PX4 with the Gazebo simulator (Linux/macOS; see the PX4 developer guide)
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot && make px4_sitl gz_x500
```

Then talk to it over **MAVLink**, the protocol drones use to talk to ground
stations and companion computers, with Python's MAVSDK or pymavlink:

```python
import asyncio
from mavsdk import System

async def fly():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(10)
    await drone.action.land()

asyncio.run(fly())
```

QGroundControl (a free ground station) shows the drone on a map, plans
missions, and replays logs.

## What real drones must also handle

| Topic | Why it matters |
|---|---|
| **Vibration** | motor vibration corrupts the accelerometer; IMUs are soft-mounted and readings filtered |
| **Sensor filtering** | low-pass and notch filters on the gyro remove motor noise before the rate loop |
| **Different loop rates** | the rate loop runs at kHz; GPS at 10 Hz; the estimator must handle the delays |
| **Saturation handling** | mixers prioritize staying upright over yaw ("airmode") |
| **System identification** | measuring a real drone's thrust curve, inertia and motor lag to tune well |
| **Safety** | pre-arm checks, geofences, failsafes, a pilot who can always take over |

## Safety and the law

Before flying a real drone outdoors, check your country's rules. Many require
registration, an online test, flying within visual line of sight, and staying
below about 120 m (400 ft) and away from airports and people. Test new code
in simulation first, then tethered or in a net, and always have a manual
kill switch.

## Where to learn more

* The PX4 and ArduPilot developer guides: architecture, flight modes and tuning.
* *Small Unmanned Aircraft: Theory and Practice*, Beard & McLain: estimation and control, in depth.
* *Probabilistic Robotics*, Thrun, Burgard & Fox: Kalman filters and beyond.
* *Planning Algorithms*, LaValle (free online): A*, RRT and much more.

## Ideas for extending this simulator

Roughly in order of difficulty:

1. Add a `HOLD` flight mode and a return-to-launch failsafe.
2. Run the rate loop at 1 kHz and the position loop at 50 Hz.
3. Mixer prioritization: sacrifice yaw before roll and pitch.
4. Add GPS delay, and see what it does to the Kalman filter; then compensate.
5. Make drag quadratic and add ground effect.
6. Replace the two estimators with one Extended Kalman Filter that also
   estimates gyro bias and wind.
7. Minimum-snap trajectories through the waypoints, so the drone needn't stop at corners.
8. Train a neural-network controller with reinforcement learning in this
   simulator, linking up with the [SmallLM](../../SmallLM/) project's RL ideas.
