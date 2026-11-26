# 4. Motors, propellers and mixing

**Code:** [`drone/motors.py`](../drone/motors.py) · **Tests:** `test_mixer_inverts_allocation`, `test_mixer_directions`, `test_motor_lag` · **Example:** [`01_hover_and_motors.py`](../examples/01_hover_and_motors.py)

## What a propeller produces

A propeller spinning at ω rad/s produces, to a good approximation:

```
thrust       F = k_f · ω²       pushes the drone up its body z axis
drag torque  Q = k_m · ω²       resists the spin; the drone body is twisted the opposite way
```

Both grow with the **square** of speed: doubling speed gives 4× thrust. For
this drone, k_f = 6.8×10⁻⁶ N/(rad/s)², so at the hover speed of 601 rad/s,
each motor lifts 2.45 N (a quarter of the drone's weight).

Power is torque × speed = k_m·ω³. Since thrust grows with ω² and power with
ω³, **flying heavier costs disproportionately more power**, which is one reason
payload hurts flight time so much.

## From four thrusts to forces and torques: the allocation matrix

The four motor thrusts F₀…F₃ combine into one total thrust and three torques
(`allocation_matrix`):

```
 T  = F₀ + F₁ + F₂ + F₃                     total thrust
 τx = Σ yᵢ·Fᵢ                               roll:  left motors push the left side up
 τy = Σ −xᵢ·Fᵢ                              pitch: back motors push the nose down
 τz = Σ −directionᵢ · (k_m/k_f) · Fᵢ        yaw:   reaction to the propellers' spin
```

where (xᵢ, yᵢ) is motor i's position, and direction is +1 for
counter-clockwise, −1 for clockwise. That's a 4×4 matrix **A**:
[T, τx, τy, τz] = A · [F₀, F₁, F₂, F₃].

## Mixing: the reverse direction

The flight controller decides *what it wants*: a thrust and three torques.
The **mixer** works out *how*: four motor speeds.

```
[F₀, F₁, F₂, F₃] = A⁻¹ · [T, τx, τy, τz]          then  ωᵢ = √(Fᵢ / k_f)
```

From example 1, starting at hover (all 601 rad/s):

| Request | m0 front-right | m1 back-left | m2 front-left | m3 back-right |
|---|---|---|---|---|
| hover | 601 | 601 | 601 | 601 |
| climb (+20% thrust) | 658 | 658 | 658 | 658 |
| roll right | 581 | 619 | 619 | 581 |
| pitch nose down | 581 | 619 | 581 | 619 |
| yaw left | 561 | 561 | 637 | 637 |

Every manoeuvre is "speed some motors up, slow others down", keeping the
total thrust the same so the drone doesn't climb or sink while turning.

### Yaw is the weakest axis

Yaw torque comes only from the propellers' drag torque: each newton of
thrust difference makes 0.016 N·m of yaw torque (k_m/k_f). For roll and
pitch, the same newton acts on a 0.16 m lever arm, giving 0.16 N·m. So yaw
is about **10× weaker**: it needs much bigger speed differences for less effect. In the mission flight plot, each 90° yaw turn at a corner drives
two motors to **zero** and two to **maximum** for a moment: the motors
saturate.

### When the request is impossible: saturation

Motors can't spin backwards or faster than their maximum. When A⁻¹ asks for
the impossible, `Mixer.motor_speeds` simply clips each motor. Real flight
controllers are smarter about what to sacrifice: they keep roll and pitch
(which keep the drone upright) and give up yaw, and sometimes some thrust,
first. PX4 calls this "airmode" and "mixer prioritization". That's a good
extension to try.

## Motor lag

Real motors take time to change speed: the rotor has inertia, and the ESC
ramps the power. `Motors.update` models this as a **first-order lag**: in each
time constant (20 ms here), the speed closes 63% of the gap to its command.

```
speed ──────────────╭───────────────  command
                   ╱
                  ╱     63% of the way after one time constant
           ──────╯
```

This delay is why the fastest control loop (the rate loop) can't be made
arbitrarily aggressive: push it too hard and it reacts to its own delayed
corrections, and oscillates.

## Try it

* Change `torque_coefficient` in `params.py` to 3× its value and rerun example
  4. Yaw turns get snappier, and motor saturation at corners shrinks.
* Change `motor_time_constant` to 0.08 s (a sluggish motor) and rerun example
  2. Does the default tuning still work?
