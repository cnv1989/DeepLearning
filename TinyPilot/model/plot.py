"""Plot PID vs TinyPilot network on the same quad and the same stick inputs.
Usage: python3 model/plot.py  -> docs/flight.png  (needs matplotlib)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot  # noqa: E402
import quad_sim as qs  # noqa: E402

rng = np.random.default_rng(7)
quad = qs.Quad(rng)
n = 2500
sp, thr = qs.stick_trajectory(rng, n)
sp[:300] = 0
sp[300:900, 0] = 400                 # a hard roll step
sp[900:1500, 0] = 0
chip = pilot.Chip(pilot.IntNet.load(os.path.join(HERE, "weights.json")))
r_pid, _, m_pid = qs.fly(pilot.quantised_pid(), quad, sp, thr)
r_nn, _, m_nn = qs.fly(chip.controller(), quad, sp, thr)
t = np.arange(n) * qs.DT * 1000

fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for a, name in enumerate(["roll", "pitch", "yaw"]):
    ax[a].plot(t, sp[:, a], "k--", lw=1, label="setpoint")
    ax[a].plot(t, r_pid[:, a], lw=1.2, label="PID")
    ax[a].plot(t, r_nn[:, a], lw=1.2, label="TinyPilot NN", alpha=0.8)
    ax[a].set_ylabel(f"{name} deg/s")
ax[0].legend(loc="upper right")
ax[3].plot(t, m_pid[:, 0] * 255, lw=1, label="PID motor 0")
ax[3].plot(t, m_nn[:, 0] * 255, lw=1, label="NN motor 0", alpha=0.8)
ax[3].set_ylabel("motor (0-255)")
ax[3].set_xlabel("time (ms)")
ax[3].legend(loc="upper right")
fig.suptitle("Same quad, same sticks: PID teacher vs integer neural network")
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "docs", "flight.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}")
