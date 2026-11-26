# 2. Frames and rotations: describing where the drone points

**Code:** [`drone/rotations.py`](../drone/rotations.py) · **Tests:** `test_rotation_round_trips`, `test_sign_conventions` in [`tests/test_physics.py`](../tests/test_physics.py)

## Two points of view

Every vector in drone software belongs to a **frame**: a set of x, y, z axes.
Mixing them up is the most common source of bugs.

| Frame | Fixed to | Axes (this project) | Used for |
|---|---|---|---|
| **World** | the ground | x east, y north, z up | position, velocity, gravity, GPS, wind |
| **Body** | the drone | x forward, y left, z up | thrust, torques, gyro, accelerometer |

Thrust always points along the body's z axis. Gravity always points down the
world's z axis. To add them, one has to be converted into the other's frame.

> **Note:** PX4 and ArduPilot use NED (north, east, *down*) for the world and
> FRD (forward, right, *down*) for the body. The maths is identical, but
> signs flip. This project uses z-up because it's more intuitive for
> learning. Always check which convention code uses.

## The rotation matrix

A 3×3 matrix **R** whose columns are the body axes written in world
coordinates. It converts between frames:

```
v_world = R  · v_body        e.g. thrust:  R · [0, 0, T]  = thrust in the world
v_body  = Rᵀ · v_world       e.g. gravity as the accelerometer sees it
```

R's transpose is its inverse, because its columns are perpendicular unit vectors.

## Euler angles: roll, pitch, yaw

Three angles that are easy to picture:

| Angle | Rotation about | Positive means (here) |
|---|---|---|
| **Roll** φ | body x (forward) | left side up, so it moves right |
| **Pitch** θ | body y (left) | nose down, so it moves forward |
| **Yaw** ψ | world z (up) | turned to the left (counter-clockwise from above) |

They're applied in **ZYX order**: yaw first, then pitch, then roll
(`euler_to_rotation`). The order matters: rotating 90° about z then x is not
the same as x then z.

**Gimbal lock.** At pitch = ±90° (nose straight up or down), roll and yaw
rotate about the same axis and one degree of freedom disappears. The
equations divide by cos(pitch) and blow up (see `euler_rates` in
`estimation.py`). Fine for display and gentle flight; not for storing attitude
in aerobatic drones.

## Quaternions

Four numbers q = (w, x, y, z) with w² + x² + y² + z² = 1. A rotation by
angle α about unit axis **u** is

```
q = ( cos(α/2),  u · sin(α/2) )
```

Why the drone stores its attitude this way (`State.attitude`):

* **No gimbal lock**: they work in every orientation.
* **Cheap to integrate**: q̇ = ½ q ⊗ (0, ω) (`quat_derivative`), with no trigonometry.
* **Easy to fix**: integration slowly changes the length; dividing by the
  length (`quat_normalize`) restores it. A drifting rotation matrix is much
  harder to repair.
* **Easy to compare**: the error between two attitudes is
  q_err = q⁻¹ ⊗ q_desired. Its vector part is roughly half the angle error
  about each axis. The attitude controller uses exactly this.

`quat_multiply(a, b)` composes rotations (b first, then a), and
`quat_conjugate` reverses one.

## Angle wrapping

Yaw 179° and −179° are 2° apart, not 358°. Every angle difference must be
wrapped into [−180°, 180°) before use (`wrap_angle`). Otherwise a drone
facing south can spin almost a full turn the wrong way.

## Which representation, when?

| Use | Representation |
|---|---|
| Storing and integrating attitude | quaternion |
| Rotating vectors between frames | rotation matrix |
| Displaying, logging, human-set setpoints | Euler angles |
| Attitude error for control | quaternion error |

Converting between them is lossless (except at gimbal lock), and the tests
check every round trip.

## Try it

```python
from drone.rotations import euler_to_rotation
import numpy as np
R = euler_to_rotation(0, np.radians(20), 0)   # pitch nose-down 20°
print(R @ [0, 0, 1])                          # where the thrust points: forward and up
```
