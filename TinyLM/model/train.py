"""Train TinyLM with quantisation-aware training (QAT), in plain NumPy.

The forward pass computes EXACTLY the integer arithmetic of the hardware
(integer embeddings, ternary weights, integer biases, shift-and-round
requantisation). Gradients flow through the rounding/ternarising steps with
the straight-through estimator (STE): we pretend round(x) has derivative 1.

Usage:  python3 model/train.py --config tt
Writes: model/weights/<config>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charset  # noqa: E402
import config as configs  # noqa: E402
from ref_model import TinyLM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def rnd(x):
    """Round half up -- the same rounding the hardware does."""
    return np.floor(x + 0.5)


def ternary(w):
    """Ternary Weight Network quantiser: values become -1, 0 or +1."""
    delta = 0.7 * np.mean(np.abs(w))
    return np.sign(w) * (np.abs(w) > delta)


def make_dataset(tokens, n):
    padded = [0] * n + tokens
    X = np.array([padded[i:i + n] for i in range(len(tokens))], dtype=np.int64)
    Y = np.array(tokens, dtype=np.int64)
    return X, Y


class QATModel:
    def __init__(self, cfg, rng):
        self.c = cfg
        V, D, H, ND = cfg["V"], cfg["D"], cfg["H"], cfg["ND"]
        self.emax = (1 << (cfg["E_BITS"] - 1)) - 1        # e.g. 7 for 4 bits
        self.hmax = (1 << cfg["H_BITS"]) - 1              # e.g. 15
        self.bmax = (1 << (cfg["B_BITS"] - 1)) - 1
        # "latent" float parameters; the forward pass only ever sees quantised versions
        self.p = {
            "E": rng.normal(0, 3.0, (V, D)),
            "W1": rng.normal(0, 1.0, (H, ND)),
            "b1": np.zeros(H),
            "W2": rng.normal(0, 1.0, (V, H)),
            "b2": np.zeros(V),
            "tau": np.array([0.05]),  # softmax temperature, training only
        }
        self.lr = {"E": 0.05, "W1": 0.01, "b1": 0.1, "W2": 0.01, "b2": 0.1, "tau": 0.002}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def quantised(self):
        p = self.p
        return {
            "E": np.clip(rnd(p["E"]), -self.emax, self.emax),
            "W1": ternary(p["W1"]),
            "b1": np.clip(rnd(p["b1"]), -self.bmax, self.bmax),
            "W2": ternary(p["W2"]),
            "b2": np.clip(rnd(p["b2"]), -self.bmax, self.bmax),
        }

    def forward(self, X, Y=None):
        c, q = self.c, self.quantised()
        B = X.shape[0]
        x = q["E"][X].reshape(B, c["ND"])                 # (B, ND)   integers
        a1 = x @ q["W1"].T + q["b1"]                      # (B, H)    integers
        r = np.maximum(a1, 0) / (1 << c["S1"])
        h = np.clip(rnd(r), 0, self.hmax)                 # (B, H)    integers
        z = h @ q["W2"].T + q["b2"]                       # (B, V)    integer logits
        cache = (X, q, x, a1, r, h, z)
        if Y is None:
            return z, cache
        logits = z * self.p["tau"][0]
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(B), Y] + 1e-12))
        return loss, probs, cache

    def backward(self, probs, Y, cache):
        X, q, x, a1, r, h, z = cache
        c, B = self.c, X.shape[0]
        tau = self.p["tau"][0]
        dlog = probs.copy()
        dlog[np.arange(B), Y] -= 1
        dlog /= B
        g = {}
        g["tau"] = np.array([np.sum(dlog * z)])
        dz = dlog * tau
        g["W2"] = dz.T @ h                                       # STE through ternary
        g["b2"] = dz.sum(0) * (np.abs(self.p["b2"]) < self.bmax + 0.5)
        dh = dz @ q["W2"]
        # STE through round + clip + relu + shift
        da1 = dh * (a1 > 0) * (r < self.hmax + 0.5) / (1 << c["S1"])
        g["W1"] = da1.T @ x
        g["b1"] = da1.sum(0) * (np.abs(self.p["b1"]) < self.bmax + 0.5)
        dx = (da1 @ q["W1"]).reshape(B, c["N"], c["D"])
        dE = np.zeros_like(self.p["E"])
        np.add.at(dE, X, dx)
        g["E"] = dE * (np.abs(self.p["E"]) < self.emax + 0.5)
        return g

    def adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr[k] * mh / (np.sqrt(vh) + eps)
        self.p["tau"] = np.maximum(self.p["tau"], 1e-3)

    def export(self):
        q = self.quantised()
        return {
            "config": self.c,
            "E": q["E"].astype(int).tolist(),
            "W1": q["W1"].astype(int).tolist(),
            "b1": q["b1"].astype(int).tolist(),
            "W2": q["W2"].astype(int).tolist(),
            "b2": q["b2"].astype(int).tolist(),
            "tau": float(self.p["tau"][0]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tt", choices=list(configs.CONFIGS))
    ap.add_argument("--corpus", default=os.path.join(HERE, "data", "corpus.txt"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = configs.get(args.config)
    rng = np.random.default_rng(args.seed)
    with open(args.corpus) as f:
        tokens = charset.encode(f.read())
    X, Y = make_dataset(tokens, cfg["N"])
    print(f"config={cfg['name']} corpus={len(tokens)} tokens")

    m = QATModel(cfg, rng)
    best = (1e9, None)
    for step in range(args.steps + 1):
        idx = rng.integers(0, len(X), args.batch)
        loss, probs, cache = m.forward(X[idx], Y[idx])
        m.adam(m.backward(probs, Y[idx], cache))
        if step % 500 == 0:
            full_loss, _, _ = m.forward(X, Y)
            z, _ = m.forward(X)
            acc = np.mean(z.argmax(1) == Y)
            print(f"step {step:5d}  loss {full_loss:.3f}  next-char accuracy {acc*100:5.1f}%")
            if full_loss < best[0]:
                best = (full_loss, m.export())

    w = best[1]
    # Sanity check: the bit-exact integer model must agree with the training forward pass.
    ref = TinyLM(w)
    for i in range(200):
        assert ref.logits(list(X[i])) == [int(v) for v in
               (np.clip(rnd(np.maximum(np.array(w["E"])[X[i]].reshape(-1) @ np.array(w["W1"]).T + w["b1"], 0)
                              / (1 << cfg["S1"])), 0, (1 << cfg["H_BITS"]) - 1) @ np.array(w["W2"]).T
                + w["b2"])], "integer reference model disagrees with training forward pass"

    out = args.out or os.path.join(HERE, "weights", f"{cfg['name']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(w, f)
    nz1 = sum(v != 0 for row in w["W1"] for v in row)
    nz2 = sum(v != 0 for row in w["W2"] for v in row)
    print(f"saved {out}")
    print(f"params: E {cfg['V']}x{cfg['D']}, W1 {cfg['H']}x{cfg['ND']} ({nz1} non-zero), "
          f"W2 {cfg['V']}x{cfg['H']} ({nz2} non-zero)")

    for noise in (False, True):
        ref.reset()
        out_toks = ref.generate(charset.encode("the "), 120, noise_en=noise)
        print(f"sample (noise={'on ' if noise else 'off'}): the {charset.decode(out_toks)!r}")


if __name__ == "__main__":
    main()
