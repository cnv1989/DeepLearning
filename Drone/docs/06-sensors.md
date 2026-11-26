# 6. Sensors

**Code:** [`drone/sensors.py`](../drone/sensors.py) · **Example:** [`03_sensors_and_estimation.py`](../examples/03_sensors_and_estimation.py)

## What it is

The flight software never knows the true state; it only gets sensor
readings, and every sensor is imperfect in its own way. `Sensors.read` turns
the true state into the readings a real drone would get, with realistic noise
and at realistic rates.

| Sensor | Measures | Frame | Rate here | Noise here | Main weakness |
|---|---|---|---|---|---|
| Gyroscope | rotation rates | body | 500 Hz | 0.005 rad/s + fixed bias | bias makes the integrated angle drift |
| Accelerometer | specific force | body | 500 Hz | 0.1 m/s² | can't tell tilt from acceleration |
| Magnetometer | Earth's magnetic field | body | 50 Hz | 2% | disturbed by motors, wiring, metal |
| Barometer | altitude (from pressure) | world | 50 Hz | 0.15 m | drifts with weather; disturbed by prop wash |
| GPS | position, velocity | world | 10 Hz | 0.6 m, 0.1 m/s | slow; metres of error; none indoors |

The gyroscope and accelerometer together form the **IMU** (inertial
measurement unit), a single chip on the flight controller.

## The gyroscope

A MEMS gyroscope measures rotation rate about each body axis. It's
excellent over short times: smooth and fast. But every gyroscope has a
**bias**, a small constant offset (typically around 0.01 rad/s here). Integrating the rate
to get an angle integrates the bias too: 0.01 rad/s is 34° of error per minute.

Real flight controllers measure the bias at start-up, while the drone sits
still, which is why they tell you not to move them while booting. See
[State estimation](07-state-estimation.md).

## The accelerometer: the subtle one

An accelerometer doesn't measure acceleration. It measures **specific
force**: every force except gravity, divided by mass.

```
specific force (body) = Rᵀ · (acceleration − gravity)
```

| Situation | Reading |
|---|---|
| sitting on a table | 9.81 m/s² **up**: the table pushing up |
| free fall | **zero**: nothing but gravity acts |
| hovering | 9.81 m/s² up the body z axis: the thrust |
| accelerating forward while tilted | still mostly along the body z axis: the thrust |

At rest, the reading points "up", so it shows which way the drone is tilted.
That's the basis of `tilt_from_accelerometer`.

**The multirotor trap.** On a quadcopter, the only large force is thrust, and
thrust *always* points along the body z axis. So the accelerometer reads
"straight up through the body" whether the drone is level or tilted 30° and
accelerating. It can't sense tilt in coordinated flight. This shaped the
estimator design, and caused the worst bug found while building this
simulator (see the next guide).

## The magnetometer

Measures Earth's magnetic field in body coordinates. Here the field points
north and dips downward (`MAGNETIC_FIELD_WORLD`), as it does in the northern
hemisphere. Its horizontal direction gives the heading, after correcting for
tilt. Real magnetometers need calibration for the drone's own magnetic
fields, which is why GPS/compass units sit on a mast, away from the power wiring.

## The barometer

Air pressure falls about 12 Pa per metre of height, so pressure gives
altitude. Good for relative height, but the absolute value drifts as the
weather changes, and propeller wash near the ground disturbs it.

## GPS

Gives position (and velocity) in the world frame, at only 5–10 Hz with about
a metre of error. Real GPS errors are also *correlated*: they wander slowly
rather than jumping independently each reading, which makes them harder to
average out than the simple noise modelled here. RTK GPS reaches
centimetres, with a base station.

## Sensor rates in the code

Slow sensors return `None` between readings. `Sensors._due` keeps a schedule
for each. The estimator must handle readings arriving at different times,
which is exactly what Kalman filters are designed for.

## Try it

In `SensorNoise` (drone/sensors.py), multiply `gps_position` by 5 and rerun
example 3. How much worse does the Kalman estimate get, compared with the raw
GPS? (It gets worse much less than GPS does, which is the whole point of
the estimator.)
