"""Turn the PID flight controller into a neural network.

1. Behaviour cloning: fly many simulated quads with the (integer front end +)
   PID teacher, record (features, motor commands), fit the network.
2. DAgger: fly with the *network* in control, record the states IT visits,
   label them with what the teacher would have done, retrain. Repeat.
   (Pure cloning only sees states the teacher visits; small network errors
   drift the quad into states it was never trained on.)
3. Quantisation-aware training: the forward pass is the exact integer
   hardware arithmetic, gradients use the straight-through estimator.

Usage: python3 model/train.py
Writes model/weights.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402
import quad_sim as qs  # noqa: E402


def rnd(x):
    return np.floor(x + 0.5)


class QATNet:
    def __init__(self, cfg, rng):
        self.c = cfg
        H = cfg["H"]
        self.wmax = (1 << (cfg["W_BITS"] - 1)) - 1
        self.bmax = (1 << (cfg["B_BITS"] - 1)) - 1
        self.p = {
            "W1": rng.normal(0, 2.0, (H, pilot.N_IN)),
            "b1": np.zeros(H),
            "W2": rng.normal(0, 2.0, (pilot.N_OUT, H)),
            "b2": np.zeros(pilot.N_OUT),
        }
        self.lr = {"W1": 0.02, "b1": 0.5, "W2": 0.02, "b2": 4.0}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def q(self):
        p = self.p
        return {
            "W1": np.clip(rnd(p["W1"]), -self.wmax, self.wmax),
            "b1": np.clip(rnd(p["b1"]), -self.bmax, self.bmax),
            "W2": np.clip(rnd(p["W2"]), -self.wmax, self.wmax),
            "b2": np.clip(rnd(p["b2"]), -self.bmax, self.bmax),
        }

    def forward(self, F):
        c, q = self.c, self.q()
        a1 = F @ q["W1"].T + q["b1"]
        r1 = np.floor((a1 + (1 << (c["S1"] - 1))) / (1 << c["S1"]))
        h = np.where(a1 > 0, np.clip(r1, 0, 255), 0)
        z = h @ q["W2"].T + q["b2"]
        ylin = np.floor((z + (1 << (c["S2"] - 1))) / (1 << c["S2"]))
        return np.clip(ylin, 0, 255), (F, q, a1, r1, h, ylin)

    def step(self, F, T):
        """One Adam step on MSE against targets T (0..255)."""
        c = self.c
        y, (F, q, a1, r1, h, ylin) = self.forward(F)
        err = ylin - T
        # clipping-aware: no penalty for overshooting a saturated target
        err[(T <= 0) & (ylin < 0)] = 0
        err[(T >= 255) & (ylin > 255)] = 0
        B = F.shape[0]
        dz = 2 * err / B / (1 << c["S2"]) / 255.0
        g = {"W2": dz.T @ h, "b2": dz.sum(0)}
        dh = dz @ q["W2"]
        da1 = dh * (a1 > 0) * (r1 < 255.5) / (1 << c["S1"])
        g["W1"] = da1.T @ F
        g["b1"] = da1.sum(0)
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= self.lr[k] * mh / (np.sqrt(vh) + 1e-12)
        return np.mean(err ** 2)

    def export(self):
        q = self.q()
        return {"config": self.c, "I_LIM": pilot.I_LIM, "THR_IRESET": pilot.THR_IRESET,
                **{k: v.astype(int).tolist() for k, v in q.items()}}


class FastChip:
    """Vectorised version of pilot.Chip for quick simulation (same arithmetic)."""

    def __init__(self, w):
        self.c = w["config"]
        self.W1, self.b1 = np.array(w["W1"]), np.array(w["b1"])
        self.W2, self.b2 = np.array(w["W2"]), np.array(w["b2"])
        self.fe = pilot.FrontEnd()
        self.log = []

    def net(self, f):
        c = self.c
        a1 = self.W1 @ f + self.b1
        h = np.where(a1 > 0, np.clip((a1 + (1 << (c["S1"] - 1))) >> c["S1"], 0, 255), 0)
        z = self.W2 @ h + self.b2
        return np.clip((z + (1 << (c["S2"] - 1))) >> c["S2"], 0, 255)

    def controller(self):
        self.fe.reset()

        def ctl(sp, gyro, thr):
            f = self.fe(pilot.to_int16(sp), pilot.to_int16(gyro), pilot.sat(int(thr * 255 + 0.5), 0, 255))
            self.log.append(f)
            return self.net(np.array(f, dtype=np.int64)) / 255.0
        return ctl


def collect_teacher(rng, flights, steps):
    feats = []
    for _ in range(flights):
        quad = qs.Quad(rng)
        sp, thr = qs.stick_trajectory(rng, steps)
        fe = pilot.FrontEnd()

        def ctl(s, g, t):
            f = fe(pilot.to_int16(s), pilot.to_int16(g), pilot.sat(int(t * 255 + 0.5), 0, 255))
            feats.append(f)
            return pilot.teacher(f)
        qs.fly(ctl, quad, sp, thr)
    return feats


def collect_student(rng, w, flights, steps):
    chip = FastChip(w)
    for _ in range(flights):
        quad = qs.Quad(rng)
        sp, thr = qs.stick_trajectory(rng, steps)
        qs.fly(chip.controller(), quad, sp, thr)
    return chip.log


def evaluate(rng, w, flights=8, steps=6000):
    res = {"float PID": [], "quantised PID": [], "TinyPilot NN": []}
    for _ in range(flights):
        quad = qs.Quad(rng)
        sp, thr = qs.stick_trajectory(rng, steps)
        for name, ctl in (("float PID", qs.PID()), ("quantised PID", pilot.quantised_pid()),
                          ("TinyPilot NN", FastChip(w).controller())):
            rates, _, _ = qs.fly(ctl, quad, sp, thr)
            res[name].append(qs.tracking_rms(sp, rates))
    return {k: np.mean(v, 0) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dagger", type=int, default=4, help="DAgger rounds")
    ap.add_argument("--steps", type=int, default=6000, help="SGD steps per round")
    ap.add_argument("--out", default=os.path.join(HERE, "weights.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = dict(pilot.CONFIG)
    t0 = time.time()

    F = collect_teacher(rng, flights=40, steps=4000)
    print(f"behaviour cloning data: {len(F)} samples from 40 teacher flights")
    net = QATNet(cfg, rng)
    for rnd_i in range(args.dagger + 1):
        X = np.array(F, dtype=float)
        T = pilot.teacher(X) * 255.0
        for s in range(args.steps):
            idx = rng.integers(0, len(X), 512)
            net.step(X[idx], T[idx])
        y, _ = net.forward(X)
        rmse = np.sqrt(np.mean((y - T) ** 2))
        w = net.export()
        ev = evaluate(np.random.default_rng(1000 + rnd_i), w, flights=3, steps=4000)
        print(f"round {rnd_i}: {len(X):6d} samples, motor RMSE {rmse:5.2f}/255, "
              f"closed-loop rate error NN {ev['TinyPilot NN'].round(1)} vs PID {ev['quantised PID'].round(1)} deg/s "
              f"[{time.time() - t0:.0f}s]")
        if rnd_i < args.dagger:
            new = collect_student(rng, w, flights=20, steps=4000)
            F += new
    with open(args.out, "w") as f:
        json.dump(w, f)
    print(f"saved {args.out}")

    # verify the vectorised simulator net == bit-exact reference
    ref, fast = pilot.IntNet(w), FastChip(w)
    for f in F[:: max(1, len(F) // 300)]:
        assert ref(f) == fast.net(np.array(f, dtype=np.int64)).tolist()

    ev = evaluate(np.random.default_rng(12345), w)
    print("\nheld-out evaluation (8 random quads x 6 s of aggressive flying), RMS rate error deg/s:")
    print(f"  {'controller':<16}{'roll':>8}{'pitch':>8}{'yaw':>8}")
    for k, v in ev.items():
        print(f"  {k:<16}{v[0]:8.1f}{v[1]:8.1f}{v[2]:8.1f}")


if __name__ == "__main__":
    main()
