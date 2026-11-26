# 10. Flight modes and failsafes

**Code:** [`drone/mission.py`](../drone/mission.py) · **Examples:** [`04_waypoint_mission.py`](../examples/04_waypoint_mission.py), [`06_wind_and_battery.py`](../examples/06_wind_and_battery.py)

## What it is

A whole flight is a sequence of phases, each with a different job. Flight
software organizes this as a **state machine**: the drone is always in exactly
one *mode*, each mode has a rule for choosing the setpoint, and clear events
switch between modes.

```
 DISARMED ──arm──► TAKEOFF ──at altitude──► MISSION ──route done──► LAND ──touched down──► LANDED
                       │                       │                    ▲
                       └──────battery < 20%────┴────────────────────┘     (failsafe)
```

| Mode | Setpoint | Leaves when |
|---|---|---|
| DISARMED | none; motors off | armed (after sensor calibration) |
| TAKEOFF | straight up at 1 m/s, over the start point | within 15 cm of the takeoff altitude |
| MISSION | the `WaypointFollower`'s carrot | the last waypoint is reached |
| LAND | straight down at 0.5 m/s, over the current point | the drone touches the ground |
| LANDED | none; motors off | – |

Each switch is recorded with its reason, as a real flight log does:

```
  0.0 s  disarmed -> takeoff: armed
  1.8 s  takeoff -> mission: reached takeoff altitude
 20.0 s  mission -> land: mission complete
 24.0 s  land -> landed: touched down
```

## Details that matter

* **Arming waits for calibration.** The simulator holds the drone still
  until the gyro calibration finishes (`Simulator._preflight_calibration`),
  then arms. Real drones refuse to arm until pre-arm checks pass: sensors
  calibrated, GPS fix, battery above minimum, and more.
* **Gradual setpoints.** Takeoff and landing move the altitude setpoint
  gradually, at a set climb or sink rate, rather than jumping. The same idea
  as the carrot in waypoint following.
* **Velocity feedforward.** Each mode also passes its intended velocity
  (climbing at 1 m/s, descending at 0.5 m/s) to the controller, so it doesn't
  lag behind.
* **Integrals reset on the ground.** While landed, or while the drone is
  still on the ground at takeoff, the controller's integrals are cleared to
  prevent windup.

## Failsafes

A **failsafe** is a mode switch triggered by trouble rather than by plan.
This simulator has one: **low battery**. Below 20% charge (`low_battery`),
the mission is abandoned and the drone lands where it is. Example 6 starts a
long mission at 23%; the failsafe fires about 29 seconds in.

Real autopilots have many failsafes, each with a configurable action (hold,
return to launch, land):

| Trigger | Typical action |
|---|---|
| battery low / critical | return to launch / land now |
| radio link lost | return to launch |
| GPS lost | hold altitude, land |
| geofence breached | return, or land |
| estimator unhealthy | land, or switch to manual |

Choosing these actions is a safety decision: landing immediately over water
or a road may be worse than flying home on a low battery.

## Real flight modes

PX4 and ArduPilot have many more modes, but each is the same pattern, a
rule for the setpoint:

| Mode | Setpoint comes from |
|---|---|
| Stabilize / Manual | the pilot's sticks set the *attitude* |
| Altitude hold | sticks set attitude; altitude held automatically |
| Position hold / Loiter | sticks set *velocity*; release them and it holds position |
| Mission / Auto | a list of waypoints (this simulator) |
| Return to launch (RTL) | climb, fly home, land |
| Offboard / Guided | a companion computer sends setpoints |

## Try it

Add a `HOLD` mode: when the mission finishes, hover at the last waypoint for
5 seconds before switching to LAND. You'll need a timer in `Mission` and one
more branch in `update`.
