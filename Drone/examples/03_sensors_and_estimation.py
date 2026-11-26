"""Example 3: noisy sensors, and the estimators that clean them up.

    python examples/03_sensors_and_estimation.py

Flies a short mission using only sensor readings, then compares the
estimates with the truth. Guides: docs/06-sensors.md, docs/07-state-estimation.md
"""

import _setup  # noqa: F401
import numpy as np

from drone.mission import Mission
from drone.plots import plot_estimation
from drone.simulation import SimConfig, Simulator

log = Simulator(config=SimConfig(duration=40)).run(Mission([[5, 0, 2], [5, 5, 3], [0, 5, 2]]))

gps_truth = np.array([log.position[np.argmin(np.abs(log.time - t))] for t in log.gps_time])
gps_error = np.linalg.norm(log.gps_position - gps_truth, axis=1)
kf_error = np.linalg.norm(log.est_position - log.position, axis=1)
tilt_error = np.degrees(np.abs(log.est_euler[:, :2] - log.euler[:, :2]))
yaw_error = np.degrees(np.abs((log.est_euler[:, 2] - log.euler[:, 2] + np.pi) % (2 * np.pi) - np.pi))

print("Position error (how far the belief is from the truth)")
print(f"  raw GPS readings:     mean {gps_error.mean():.2f} m   worst {gps_error.max():.2f} m")
print(f"  Kalman filter:        mean {kf_error.mean():.2f} m   worst {kf_error.max():.2f} m")
print("Attitude error (complementary filter)")
print(f"  roll/pitch:           mean {tilt_error.mean():.2f}°   worst {tilt_error.max():.2f}°")
print(f"  yaw:                  mean {yaw_error.mean():.2f}°   worst {yaw_error.max():.2f}°")
plot_estimation(log, f"{_setup.OUTPUT}/03_estimation.png")
