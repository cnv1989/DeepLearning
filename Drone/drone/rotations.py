"""Describing orientation in 3D: quaternions, rotation matrices and Euler angles.

Frames used everywhere in this project (both right-handed):

    World frame  x = east, y = north, z = up          (fixed to the ground)
    Body frame   x = forward, y = left, z = up        (fixed to the drone)

A rotation R turns body-frame vectors into world-frame vectors:

    v_world = R @ v_body          v_body = R.T @ v_world

Three ways to store the same orientation:

    Euler angles (roll, pitch, yaw)  easy to read; used for display and setpoints.
                                     Breaks down ("gimbal lock") at pitch = ±90°.
    Rotation matrix R (3x3)          easy to apply to vectors.
    Quaternion q = (w, x, y, z)      4 numbers, no gimbal lock, cheap to integrate.
                                     Used to store the drone's attitude.

Euler angles here use the common aerospace "ZYX" order: rotate by yaw about z,
then pitch about the new y, then roll about the newest x.
Positive roll tips the left side up; positive pitch tips the nose down.
"""

import numpy as np


def wrap_angle(angle):
    """Wrap an angle to the range [-pi, pi). 350 degrees becomes -10 degrees."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------- #
# Quaternions
# ----------------------------------------------------------------------------- #


def quat_multiply(a, b):
    """Compose rotations: rotating by b, then by a, equals rotating by quat_multiply(a, b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conjugate(q):
    """The opposite rotation (for unit quaternions, the inverse)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """Rescale to length 1. Integration slowly drifts the length; this fixes it."""
    return q / np.linalg.norm(q)


def quat_derivative(q, body_rates):
    """How fast the quaternion changes when the body spins at `body_rates` (rad/s).

    q_dot = 1/2 * q ⊗ (0, ω)
    """
    return 0.5 * quat_multiply(q, np.array([0.0, *body_rates]))


def quat_to_rotation(q):
    """Unit quaternion -> 3x3 rotation matrix (body to world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_quat(R):
    """3x3 rotation matrix -> unit quaternion (with w >= 0)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


# ----------------------------------------------------------------------------- #
# Euler angles
# ----------------------------------------------------------------------------- #


def euler_to_rotation(roll, pitch, yaw):
    """Roll, pitch, yaw (radians, ZYX order) -> rotation matrix (body to world)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    yaw_matrix = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pitch_matrix = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    roll_matrix = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return yaw_matrix @ pitch_matrix @ roll_matrix


def rotation_to_euler(R):
    """Rotation matrix -> (roll, pitch, yaw) in radians, ZYX order."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll, pitch, yaw):
    return rotation_to_quat(euler_to_rotation(roll, pitch, yaw))


def quat_to_euler(q):
    return rotation_to_euler(quat_to_rotation(q))
