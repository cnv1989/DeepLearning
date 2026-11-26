# 3. Physics: how forces move the drone

**Code:** [`drone/dynamics.py`](../drone/dynamics.py) · **Tests:** hover, free fall, torque and ground tests in [`tests/test_physics.py`](../tests/test_physics.py)

## What it is

The dynamics model is the simulator's "real world". Given the forces and
torques acting on the drone, it works out how the drone moves. Everything
else (sensors, estimator, controller) treats it as a black box, just as real
flight software can't see the true state of a real drone.

## The state: 13 numbers

| | Symbol | Frame | Units |
|---|---|---|---|
| position | p | world | m |
| velocity | v | world | m/s |
| attitude | q | quaternion, body → world | – |
| body rates | ω = (p, q, r) | body | rad/s |

That's everything needed to predict the future, given the inputs. (Motor
speeds and battery charge are tracked separately, in `Motors` and `Battery`.)

## Forces → linear motion (Newton's second law)

```
m · dv/dt = gravity + thrust + drag

  gravity  = (0, 0, −m·g)                    always straight down
  thrust   = R · (0, 0, T)                    along the body's z axis, rotated into the world
  drag     = −c · (v − wind)                  against motion through the air
```

The key consequence: **a quadcopter can only push along its own z axis.**
To move sideways it must tilt, so that part of the thrust points sideways,
and then it needs more total thrust to keep holding its height:

```
tilted by θ:   vertical thrust = T·cos θ      horizontal thrust = T·sin θ
```

At 35° of tilt (the controller's limit), holding altitude takes 1/cos 35° =
22% more thrust, and the drone accelerates sideways at g·tan 35° ≈ 6.9 m/s².

**Drag** here is a simple linear model (0.25 N per m/s). Real drag grows
roughly with speed squared, and propellers add their own "rotor drag", but
linear drag captures the important effects: a top speed, and wind pushing the
drone, since drag acts on velocity *relative to the air*.

## Torques → rotation (Euler's rotation equation)

```
I · dω/dt = τ − ω × (I · ω)
```

* **I**, the inertia matrix: resistance to spinning, like mass for rotation.
  For this drone, I = diag(0.0082, 0.0082, 0.0149) kg·m². Yaw has the most
  inertia because all four motors sit far from the z axis.
* **τ**: torques from the motors (see [Motors and mixing](04-motors-and-mixing.md)).
* **ω × Iω**: the gyroscopic term. A spinning body resists changes to its spin
  axis; that's why a spinning top doesn't fall over. It's small for slow
  flight, but the rate controller cancels it anyway.

Attitude then follows from the body rates: dq/dt = ½ q ⊗ (0, ω).

## Integrating: from rates of change to a trajectory

The equations above say how fast the state *changes*. To simulate, we step
forward in time. The simplest method, Euler's method, is
`x_next = x + dt · dx/dt`. It's inaccurate: it uses the slope at the start
of the step for the whole step, and errors pile up.

`QuadcopterDynamics.step` uses **4th-order Runge-Kutta (RK4)**:

```
k1 = slope at the start
k2 = slope at the middle, stepping halfway with k1
k3 = slope at the middle, stepping halfway with k2
k4 = slope at the end, stepping fully with k3
x_next = x + dt/6 · (k1 + 2·k2 + 2·k3 + k4)
```

Four slope evaluations per step, but the error per step shrinks with dt⁵
instead of dt². The free-fall test checks the result against the exact
answer (p = ½gt²) to a millionth of a metre.

The quaternion is renormalized after every step, and motor thrusts are held
constant during a step (they change slowly compared with 2 ms).

## The ground

`_ground_contact` is the simplest possible ground: the drone can't go below
z = 0, and touching it stops all motion. Real simulators (Gazebo, AirSim) model
landing gear, friction and bounce.

## What's left out

Real aircraft physics includes effects this model ignores. Each one is a
good extension:

* **Ground effect**: extra thrust within about one rotor diameter of the ground.
* **Blade flapping and rotor drag**: rotors tilt back in forward flight.
* **Motor/propeller interactions** and changes in thrust with airspeed.
* **Vortex ring state**: losing thrust when descending fast into your own downwash.

## Try it

```python
import numpy as np
from drone.dynamics import QuadcopterDynamics, State
from drone.params import QuadParams
p = QuadParams()
d = QuadcopterDynamics(p, State(position=np.array([0, 0, 10.0])))
for _ in range(500):
    d.step(thrust=p.weight * 1.1, torques=np.zeros(3), dt=0.002)   # 10% more than hover
print(d.state.position, d.state.velocity)   # climbing; drag limits the speed
```
