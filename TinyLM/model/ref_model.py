"""Bit-exact integer reference model of the TinyLM hardware.

Every number here is an integer and every operation has an exact hardware
equivalent. If the Verilog and this file ever disagree, one of them has a bug.
The testbenches compare the RTL against vectors produced by this file.
"""
import json

LFSR_SEED = 0xACE1
LFSR_TAPS = 0xB400  # 16-bit Galois LFSR, maximal length


def lfsr_step(s: int) -> int:
    return (s >> 1) ^ (LFSR_TAPS if (s & 1) else 0)


def requant(acc: int, s1: int, hmax: int) -> int:
    """h = clamp(round(relu(acc) / 2**s1), 0, hmax), rounding half up.
    Hardware: (acc + 2**(s1-1)) >>> s1, then saturate."""
    if acc <= 0:
        return 0
    r = (acc + (1 << (s1 - 1))) >> s1
    return min(r, hmax)


class TinyLM:
    def __init__(self, weights: dict):
        self.w = weights
        self.cfg = weights["config"]
        self.reset()

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    # ---- hardware state -------------------------------------------------
    def reset(self):
        self.ctx = [0] * self.cfg["N"]   # ctx[0] = oldest, ctx[N-1] = newest
        self.lfsr = LFSR_SEED

    def push(self, tok: int):
        self.ctx = self.ctx[1:] + [tok]

    # ---- the network -----------------------------------------------------
    def hidden(self, ctx):
        c, w = self.cfg, self.w
        x = [w["E"][t][d] for t in ctx for d in range(c["D"])]
        hmax = (1 << c["H_BITS"]) - 1
        h = []
        for j in range(c["H"]):
            acc = w["b1"][j] + sum(wi * xi for wi, xi in zip(w["W1"][j], x))
            h.append(requant(acc, c["S1"], hmax))
        return h

    def logits(self, ctx):
        """Raw integer logits z[k] (no noise)."""
        h = self.hidden(ctx)
        return [self.w["b2"][k] + sum(wi * hi for wi, hi in zip(self.w["W2"][k], h))
                for k in range(self.cfg["V"])]

    def start(self, noise_en: bool) -> int:
        """One inference, exactly as the hardware does it:
        logits are produced in order k = 0..V-1, the LFSR advances once per
        logit, optional noise = low NOISE_BITS of the LFSR (before advancing),
        argmax keeps the FIRST maximum, result is shifted into the context."""
        z = self.logits(self.ctx)
        mask = (1 << self.cfg["NOISE_BITS"]) - 1
        best_val, best_idx = None, 0
        for k in range(self.cfg["V"]):
            v = z[k] + ((self.lfsr & mask) if noise_en else 0)
            self.lfsr = lfsr_step(self.lfsr)
            if best_val is None or v > best_val:
                best_val, best_idx = v, k
        self.push(best_idx)
        return best_idx

    def generate(self, prompt_tokens, n, noise_en=False):
        for t in prompt_tokens:
            self.push(t)
        return [self.start(noise_en) for _ in range(n)]
