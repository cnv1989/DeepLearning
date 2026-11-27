"""Rotational dynamics of a 5-inch racing quadcopter, plus a Betaflight-style
PID rate controller to act as the "teacher".

Only attitude *rates* are simulated. That is all an inner-loop rate controller
(what Betaflight's PID loop does in acro mode) ever sees: gyro in, motors out.

Conventions
-----------
Body rates w = [roll, pitch, yaw] in deg/s (positive roll = right side down).
Motors 0..3 = front-left, front-right, rear-left, rear-right, command 0..1.
"""
import numpy as np

DT = 1.0 / 1000.0          # 1 kHz control loop
SUBSTEPS = 4               # physics integration steps per control step

# Mixer: motor = throttle + MIX @ [roll, pitch, yaw]
MIX = np.array([
    # roll  pitch  yaw
    [+1.0, -1.0, +1.0],    # front-left  (spins CW  -> yaw sign +)
    [-1.0, -1.0, -1.0],    # front-right (spins CCW)
    [+1.0, +1.0, -1.0],    # rear-left   (spins CCW)
    [-1.0, +1.0, +1.0],    # rear-right  (spins CW)
])


class Quad:
    """Rigid body with first-order motor lag, gyro noise and disturbances."""

    def __init__(self, rng, randomise=True):
        self.rng = rng
        r = (lambda lo, hi: rng.uniform(lo, hi)) if randomise else (lambda lo, hi: (lo + hi) / 2)
        self.J = np.array([r(1.3e-3, 1.8e-3), r(1.4e-3, 2.0e-3), r(2.3e-3, 3.0e-3)])  # kg m^2
        self.arm = r(0.055, 0.065)              # effective lever arm, m
        self.tmax = r(7.0, 9.0)                 # max thrust per motor, N
        self.kyaw = r(0.009, 0.012)             # yaw torque / thrust, m
        self.tau = r(0.015, 0.025)              # motor time constant, s
        # constant disturbance torque (CG offset, bent prop...) -> needs I-term
        self.bias = rng.normal(0, 0.004, 3) if randomise else np.zeros(3)
        self.gyro_noise = r(1.0, 3.0)           # deg/s RMS
        self.reset()

    def reset(self):
        self.w = np.zeros(3)                    # rad/s
        self.m = np.zeros(4)                    # actual motor output 0..1

    def torque(self, m):
        t = self.tmax * m
        return np.array([
            self.arm * (t[0] + t[2] - t[1] - t[3]),
            self.arm * (t[2] + t[3] - t[0] - t[1]),
            self.kyaw * (t[0] + t[3] - t[1] - t[2]),
        ])

    def step(self, motor_cmd):
        """Advance one control period. Returns the noisy gyro reading in deg/s."""
        cmd = np.clip(motor_cmd, 0.0, 1.0)
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.m += (cmd - self.m) * (h / self.tau)
            tq = self.torque(self.m) + self.bias + self.rng.normal(0, 0.003, 3)
            jw = self.J * self.w
            self.w += h * (tq - np.cross(self.w, jw)) / self.J
        return np.degrees(self.w) + self.rng.normal(0, self.gyro_noise, 3)


class PID:
    """Betaflight-flavoured rate PID: P on error, I with anti-windup and
    low-throttle reset, D on measurement (gyro), simple clipping mixer."""

    KP = np.array([0.0020, 0.0022, 0.0050])      # motor units per deg/s
    KI = np.array([0.006, 0.0066, 0.030])        # per deg/s * s
    KD = np.array([0.000020, 0.000022, 0.0])     # per deg/s^2
    I_LIMIT = np.array([0.15, 0.15, 0.15])
    D_ALPHA = 0.25                               # D-term low-pass (PT1, ~55 Hz at 1 kHz)

    def __init__(self):
        self.reset()

    def reset(self):
        self.i = np.zeros(3)
        self.prev_gyro = np.zeros(3)
        self.d = np.zeros(3)

    def __call__(self, setpoint, gyro, throttle):
        err = setpoint - gyro
        if throttle < 0.05:                        # disarmed / idle: no I build-up
            self.i[:] = 0
        else:
            self.i = np.clip(self.i + self.KI * err * DT, -self.I_LIMIT, self.I_LIMIT)
        self.d += self.D_ALPHA * (-(gyro - self.prev_gyro) / DT - self.d)
        self.prev_gyro = gyro.copy()
        axis = self.KP * err + self.i + self.KD * self.d
        return np.clip(throttle + MIX @ axis, 0.0, 1.0)


def stick_trajectory(rng, n, max_rate=(500, 500, 350)):
    """Pilot-like setpoints: holds, steps, smooth sweeps, and flips, plus throttle."""
    sp = np.zeros((n, 3))
    thr = np.zeros(n)
    t = 0
    while t < n:
        seg = int(rng.integers(150, 600))
        kind = rng.integers(0, 4)
        for a in range(3):
            target = rng.uniform(-1, 1) * max_rate[a] * rng.choice([0.0, 0.3, 1.0], p=[0.3, 0.4, 0.3])
            start = sp[t - 1, a] if t > 0 else 0.0
            if kind == 0:      # step
                sp[t:t + seg, a] = target
            elif kind == 1:    # ramp
                sp[t:t + seg, a] = np.linspace(start, target, seg)[: n - t]
            elif kind == 2:    # sine sweep
                f = rng.uniform(0.5, 6.0)
                sp[t:t + seg, a] = target * np.sin(2 * np.pi * f * np.arange(seg) * DT)[: n - t]
            else:              # centre stick
                sp[t:t + seg, a] = 0.0
        thr[t:t + seg] = np.clip(rng.uniform(0.15, 0.75) + rng.normal(0, 0.03), 0, 1)
        t += seg
    return sp[:n], thr[:n]


def fly(controller, quad, sp, thr):
    """Closed-loop flight. controller(setpoint, gyro, throttle) -> motors (0..1).
    Returns true rates, gyro readings and motor commands."""
    quad.reset()
    n = len(sp)
    gyro = np.zeros(3)
    rates = np.zeros((n, 3))
    gyros = np.zeros((n, 3))
    motors = np.zeros((n, 4))
    for t in range(n):
        m = controller(sp[t], gyro, thr[t])
        motors[t] = m
        gyros[t] = gyro
        gyro = quad.step(m)
        rates[t] = np.degrees(quad.w)
    return rates, gyros, motors


def tracking_rms(sp, rates, skip=100):
    return np.sqrt(np.mean((sp[skip:] - rates[skip:]) ** 2, axis=0))
