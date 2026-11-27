"""Bit-exact integer model of the TinyPilot chip, plus the PID "teacher".

Per control loop (1 kHz) the chip receives, over SPI:
    setpoint[3]  int16, deg/s        (from the pilot's sticks)
    gyro[3]      int16, deg/s        (from the IMU)
    throttle     uint8, 0..255
and returns 4 motor commands, uint8 0..255.

    ┌──────────── integer front end (state) ────────────┐     ┌──── neural net ────┐
    e  = sp - gyro                    -> f_e = sat8(e >> 1)                          │
    df4 += (prev - gyro) - (df4 >>> 2) -> f_d = sat8(df4 >>> 1)   10 x int8 ──► MLP ──► 4 x uint8 motors
    I  = clamp(I + e, ±I_LIM[axis])   -> f_i = sat8(I >>> 8)                         │
    throttle                          -> f_t = thr >> 1                              │
    └───────────────────────────────────────────────────┘     └────────────────────┘

The network replaces the PID gains, the mixer and the motor clipping.
Everything here is integer and maps 1:1 onto rtl/pilot_core.v.
"""
import json

import numpy as np

from quad_sim import MIX, PID

N_IN, N_OUT = 10, 4

CONFIG = dict(
    H=24,          # hidden neurons
    W_BITS=8,      # signed weight bits (both layers); 4 bits is too coarse for control
    B_BITS=16,     # signed bias bits
    S1=4,          # hidden requant shift
    S2=6,          # output requant shift
    ACC_W=24,      # accumulator width in hardware
)
# Per-axis integrator limits, in (deg/s * ms), matching the teacher's I_LIMIT
I_LIM = [int(round(PID.I_LIMIT[a] / PID.KI[a] * 1000)) for a in range(3)]
THR_IRESET = 13    # throttle below this (~5%) resets the integrators


def sat(x, lo, hi):
    return max(lo, min(hi, x))


def to_int16(v):
    return [sat(int(np.floor(x + 0.5)), -32768, 32767) for x in v]


class FrontEnd:
    """Integer state: previous gyro, filtered derivative, integrator."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = [0, 0, 0]
        self.df4 = [0, 0, 0]
        self.iacc = [0, 0, 0]

    def __call__(self, sp, gyro, thr):
        fe, fd, fi = [], [], []
        for a in range(3):
            e = sp[a] - gyro[a]
            if thr < THR_IRESET:
                self.iacc[a] = 0
            else:
                self.iacc[a] = sat(self.iacc[a] + e, -I_LIM[a], I_LIM[a])
            d = self.prev[a] - gyro[a]
            self.df4[a] = sat(self.df4[a] + d - (self.df4[a] >> 2), -32768, 32767)
            self.prev[a] = gyro[a]
            fe.append(sat(e >> 1, -127, 127))
            fd.append(sat(self.df4[a] >> 1, -127, 127))
            fi.append(sat(self.iacc[a] >> 8, -127, 127))
        return fe + fd + fi + [thr >> 1]


def teacher(f):
    """The PID law expressed on the (quantised) features -> motors 0..1 (float).
    This is what the network learns to imitate."""
    f = np.asarray(f, dtype=float)
    e = f[..., 0:3] * 2.0                      # deg/s
    d = f[..., 3:6] * 2.0 / 4.0 * 1000.0       # deg/s^2  (df4/4 per ms)
    i = f[..., 6:9] * 256.0 / 1000.0           # deg/s * s
    thr = f[..., 9:10] * 2.0 / 255.0
    axis = PID.KP * e + PID.KI * i + PID.KD * d
    return np.clip(thr + axis @ MIX.T, 0.0, 1.0)


def requant(acc, shift, relu):
    if relu and acc <= 0:
        return 0
    return sat((acc + (1 << (shift - 1))) >> shift, 0, 255)


class IntNet:
    """Bit-exact integer MLP: 10 x int8 -> H x uint8 (ReLU) -> 4 x uint8."""

    def __init__(self, w):
        self.c = w["config"]
        self.W1, self.b1 = w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __call__(self, f):
        c = self.c
        h = [requant(self.b1[j] + sum(wi * fi for wi, fi in zip(self.W1[j], f)), c["S1"], True)
             for j in range(c["H"])]
        return [requant(self.b2[k] + sum(wi * hi for wi, hi in zip(self.W2[k], h)), c["S2"], False)
                for k in range(N_OUT)]


class Chip:
    """Front end + network: exactly what the hardware does each loop."""

    def __init__(self, net):
        self.fe = FrontEnd()
        self.net = net

    def reset(self):
        self.fe.reset()

    def step(self, sp_i, gyro_i, thr_u8):
        return self.net(self.fe(sp_i, gyro_i, thr_u8))

    def controller(self):
        """Adapter for quad_sim.fly: float in/out."""
        def ctl(sp, gyro, thr):
            m = self.step(to_int16(sp), to_int16(gyro), sat(int(thr * 255 + 0.5), 0, 255))
            return np.array(m) / 255.0
        return ctl


def quantised_pid():
    """The teacher running on the integer front end (no network)."""
    fe = FrontEnd()

    def ctl(sp, gyro, thr):
        return teacher(fe(to_int16(sp), to_int16(gyro), sat(int(thr * 255 + 0.5), 0, 255)))
    ctl.reset = fe.reset
    return ctl
