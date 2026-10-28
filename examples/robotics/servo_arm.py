"""The soft-servoed yaw-pitch-pitch arm the robotics examples share.

Three affine links on revolute servos, plus a 1 kg payload welded to the
forearm's tip. `add_servo_arm` builds it into a world.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import make_bar_trimesh  # noqa: E402

Z_BASE     = 0.5                  # shoulder height (m)
L1, L2     = 0.40, 0.35           # upper arm, forearm (m)
HALF       = 0.035                # link half-thickness (m)
PAYLOAD    = 0.04                 # payload cube half-size (m)
PAYLOAD_KG = 1.0
REACH2     = L2 + PAYLOAD         # elbow to payload centre (m)
GAINS      = np.array([150.0, 200.0, 100.0])   # yaw, shoulder, elbow servo k (N m/rad)


def make_box(center, half, spacing=None):
    """A closed box mesh; with `spacing` (m), its faces are subdivided about that finely."""
    h = np.asarray(half, dtype=np.float64)
    res = (1, 1, 1) if spacing is None else tuple(max(1, round(2 * x / spacing)) for x in h)
    return make_bar_trimesh(tuple(2 * h), res, origin=tuple(np.asarray(center) - h))


@dataclass
class ServoArm:
    """Handles into the world `add_servo_arm` built."""

    bodies: list                                # base, upper arm, forearm, payload
    acts: list                                  # yaw, shoulder, elbow servos
    payload: int
    rest_vertices: dict = field(default_factory=dict)   # body -> (n, 3) rest surface (m)
    faces: dict = field(default_factory=dict)           # body -> (m, 3) surface triangles


def add_servo_arm(w, spacing=None):
    """Add the arm to world `w`; returns its `ServoArm` handles.

    `spacing` (m) subdivides each link's box mesh about that finely; by default each link is a
    plain 8-vertex box."""
    aff = trusty.affine
    rest, faces = {}, {}

    def body(center, half, density):
        V, F = make_box(center, half, spacing)
        b = aff.add_affine_body(w, V, F, density=density, stiffness=1e8)
        rest[b], faces[b] = V, F
        return b

    base = body((0, 0, Z_BASE), (0.05, 0.05, 0.05), 300.0)
    upper = body((L1 / 2, 0, Z_BASE), (L1 / 2, HALF, HALF), 300.0)
    fore = body((L1 + L2 / 2, 0, Z_BASE), (L2 / 2, HALF, HALF), 300.0)
    payload = body((L1 + REACH2, 0, Z_BASE), (PAYLOAD,) * 3,
                   PAYLOAD_KG / (2 * PAYLOAD) ** 3)

    yaw = aff.add_revolute_joint(w, base, (0, 0, Z_BASE - 0.05), (0, 0, Z_BASE + 0.05))
    shoulder = aff.add_revolute_joint(w, upper, (0, -HALF, Z_BASE), (0, HALF, Z_BASE), base)
    elbow = aff.add_revolute_joint(w, fore, (L1, -HALF, Z_BASE), (L1, HALF, Z_BASE), upper)
    aff.add_fixed_joint(w, payload, (L1 + L2, 0, Z_BASE), (0, 0.03, 0), (0, 0, 0.03), fore, 1e8)
    acts = [aff.add_actuator(w, j, stiffness=k) for j, k in zip((yaw, shoulder, elbow), GAINS)]
    return ServoArm([base, upper, fore, payload], acts, payload, rest, faces)
