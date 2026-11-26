# 1. The parts of a drone

**Related code:** [`drone/params.py`](../drone/params.py) holds the numbers that describe the hardware.

## What a quadcopter is made of

```
                         GPS + compass (on a mast, away from magnetic noise)
                                     │
      propeller ─┐                   │                  ┌─ propeller
      motor ─────┤     ┌─────────────┴────────────┐     ├─ motor
                 └─────┤  flight controller (FC)  ├─────┘
           arm ────────┤  IMU · barometer · CPU   ├──────── arm
                 ┌─────┤  ESCs (often on a 4-in-1 │─────┐
      motor ─────┤     │  board under the FC)     │     ├─ motor
      propeller ─┘     └─────┬──────────┬─────────┘     └─ propeller
                             │          │
                       radio receiver   battery (LiPo) ──► power module
                       telemetry radio
```

| Part | What it does | In this simulator |
|---|---|---|
| **Frame** | holds everything together; arm length sets how much leverage the motors have | `arm_length`, `mass`, `inertia` |
| **Motors** | brushless DC motors spinning at thousands of rpm | `Motors` in `motors.py` |
| **Propellers** | turn rotation into thrust (and a drag torque) | `thrust_coefficient`, `torque_coefficient` |
| **ESCs** (electronic speed controllers) | turn the flight controller's commands into motor power; one per motor | the first-order lag in `Motors.update` |
| **Battery** | stores energy; a lithium-polymer (LiPo) pack, usually 3–6 cells | `Battery` in `battery.py` |
| **Flight controller** | a small computer running the flight software (PX4, ArduPilot, Betaflight) | everything in `estimation.py`, `control.py`, `mission.py` |
| **IMU** (gyroscope + accelerometer) | measures rotation and acceleration, hundreds to thousands of times a second | `Sensors` (gyro, accel) |
| **Barometer** | measures air pressure, giving altitude | `Sensors` (barometer) |
| **GPS + compass** | absolute position, and heading from Earth's magnetic field | `Sensors` (gps, magnetometer) |
| **Radio receiver** | the pilot's stick commands | not modelled; missions are autonomous |
| **Telemetry radio** | sends status to a ground station; receives missions | not modelled |

## Why four rotors?

A helicopter steers by tilting its rotor blades as they spin, which takes
complex mechanics. A quadcopter has fixed propellers and steers only by
changing motor speeds, controlled by software. Mechanically simple, but it
needs a computer correcting its balance hundreds of times a second, because
**a quadcopter is unstable**: without active control, any small tilt grows
until it flips. Everything in this project exists to keep that from happening
while getting where it's going.

## Key numbers for the simulated drone

| Quantity | Value | Why it matters |
|---|---|---|
| Mass | 1.0 kg | everything scales with weight |
| Arm length | 0.225 m (450 mm motor to motor, diagonally) | longer arms: more torque from the same thrust difference |
| Max thrust | 27.2 N (2.77 kg) | |
| **Thrust-to-weight ratio** | **2.77** | below ~1.5 it can barely manoeuvre; racing drones exceed 10 |
| Hover motor speed | 601 rad/s (5,735 rpm), 60% of max | the 40% left is the room the controller has to correct |
| Battery | 4S LiPo (4 cells, 14.8 V nominal), 3,000 mAh | |
| Hover power | about 95 W at the propellers | |
| Hover time | about 19 minutes | real drones of this class get 15–25 minutes |

## Motor layout ("quad X")

```
              front (+x)
         m2 (CW)    m0 (CCW)
              \    /
               [  ]          +y is left
              /    \
         m1 (CCW)   m3 (CW)
              back
```

Diagonally opposite motors spin the same way, and adjacent motors spin
opposite ways. Each propeller pushes air down and, by Newton's third law,
twists the drone the opposite way to its spin. With two spinning each way,
those twists cancel in a hover, and changing their balance is how the drone
yaws. The numbering follows PX4's convention.

## Try it

```bash
python examples/01_hover_and_motors.py
```

Then edit `QuadParams` in `drone/params.py`: double the `mass` and rerun.
What happens to the hover motor speed, the thrust-to-weight ratio and the
flight time?
