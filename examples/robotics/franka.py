"""Franka FR3 arm as a chain of affine bodies, built from its URDF.

Shared by the Franka examples. Each link becomes one near-rigid affine body
whose shape is the URDF collision mesh; links joined by a `fixed` joint (the
last arm link and the hand) become one body. Every arm joint is a revolute
joint driven by an actuator, and each finger slides on a prismatic joint. The
arm is built in the pose `q0`, so an actuator's target is the joint's offset
from `q0`. `tcp` is the tool centre point between the fingertips.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_mesh, signed_volume  # noqa: E402

ROBOT_DIR = Path(__file__).resolve().parent.parent / "data" / "franka_emika_panda"
URDF = ROBOT_DIR / "urdf" / "fr3_franka_hand.urdf"

ARM_JOINTS    = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_LINKS  = ["fr3_leftfinger", "fr3_rightfinger"]
HAND_LINK     = "fr3_hand"
Q_READY       = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # rad

ARM_GAINS       = [5e4, 5e4, 5e4, 5e4, 1.25e4, 1.25e4, 5e3]  # N*m/rad, shoulder to wrist
FINGER_GAIN     = 2e3   # N/m
JOINT_STIFFNESS = 1e8   # how hard each joint holds its two links together
BODY_STIFFNESS  = 1e9   # Pa; large keeps the links near-rigid
LIMIT_STIFFNESS = 1e3   # joint-limit barrier weight
ARM_LIMIT_MARGIN    = 0.05  # rad; distance from a joint limit where its barrier starts
FINGER_LIMIT_MARGIN = 2e-3  # m
ANGLE_RANGE    = np.pi - 0.05  # rad; a joint limit must lie inside one turn of the build pose
ARM_FRICTION    = 0.3
FINGER_FRICTION = 1.0   # pads grip by friction alone

TCP_OFFSET      = 0.1034   # m, from the hand frame to the fingertip centre, along its z
GRIPPER_OPEN    = 0.075    # m between the pads; clears the finger limit barrier

GRASP_SAFETY      = 5.0    # grip force over the force needed to hold the object's weight
GRASP_DHAT        = 1e-4   # m, contact barrier width; objects rest this far above their support
GRASP_KAPPA       = 1e5    # contact barrier stiffness
GRASP_EPSV        = 1e-3   # m/s, sliding speed below which friction holds
GRASP_CLOSE_TIME  = 0.6    # s, open to squeeze
GRASP_SETTLE_TIME = 0.3    # s, hold closed before lifting


@dataclass
class Franka:
    bodies: dict            # body name -> affine body
    arm_actuators: list     # one per arm joint, shoulder to wrist
    finger_actuators: list  # empty without fingers
    hand: object            # the body carrying the hand
    hand_pose: np.ndarray   # (4, 4) hand frame at rest, z along the fingers
    tcp: np.ndarray         # (4, 4) tool centre point at rest, the hand frame moved along its z
    arm_limits: np.ndarray  # (7, 2) URDF joint limits (rad), lower and upper
    arm_speeds: np.ndarray  # (7,) URDF joint speed limits (rad/s)
    q0: np.ndarray          # (7,) the build pose (rad)
    finger0: float          # finger slide at rest (m)

    def set_arm(self, world, q):
        """Drive the arm toward joint angles `q` (7,) (rad)."""
        for actuator, qi, q0i in zip(self.arm_actuators, q, self.q0):
            trusty.affine.set_actuator_target(world, actuator, float(qi - q0i))

    def set_gripper(self, world, width):
        """Drive the fingers toward an opening of `width` (m) between them."""
        for actuator in self.finger_actuators:
            trusty.affine.set_actuator_target(world, actuator, 0.5 * width - self.finger0)


def _transform(elem):
    """(4, 4) from a URDF `<origin xyz rpy>`: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    T = np.eye(4)
    o = elem.find("origin")
    if o is None:
        return T
    r, p, y = (float(v) for v in o.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                 @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                 @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def _rotation(axis, angle):
    """Rotation by `angle` (rad) about the unit `axis`."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _box(size):
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    V = (corners - 0.5) * size
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)
    return V, F


def _collision_mesh(link):
    """The link's collision shapes as one triangle mesh in the link frame."""
    Vs, Fs, n = [], [], 0
    for c in link.findall("collision"):
        shape = c.find("geometry")[0]
        if shape.tag == "mesh":
            rel = shape.get("filename").removeprefix("package://franka_emika_panda/")
            V, F = load_mesh(ROBOT_DIR / rel)
        else:  # the fingers are boxes
            V, F = _box(np.array([float(v) for v in shape.get("size").split()]))
        T = _transform(c)
        Vs.append(V @ T[:3, :3].T + T[:3, 3])
        Fs.append(F + n)
        n += len(V)
    return (np.vstack(Vs), np.vstack(Fs)) if Vs else None


def build_franka(world, q0=Q_READY, fingers=True, finger0=0.03,
                 finger_margin=FINGER_LIMIT_MARGIN) -> Franka:
    """Add an FR3 arm to `world`, its base at the origin, posed at `q0` (rad).

    With `fingers`, the two hand fingers are added too, each slid `finger0` (m)
    open from closed. Each joint stops short of its URDF limit; a finger's limit barrier
    starts `finger_margin` (m) from its closed limit, so pinching a sheet thinner than
    that margin needs a smaller one.
    """
    robot = ET.parse(URDF).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = robot.findall("joint")
    if not fingers:
        joints = [j for j in joints if j.get("name") not in FINGER_JOINTS]
    q = dict(zip(ARM_JOINTS, q0)) | {name: finger0 for name in FINGER_JOINTS}

    # Pose every link at q0 (joints are listed parent first) and merge links
    # joined by fixed joints into the body of their parent.
    pose = {"base": np.eye(4)}
    owner = {"base": "base"}
    for j in joints:
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        motion = np.eye(4)
        if j.get("type") != "fixed":
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            if j.get("type") == "revolute":
                motion[:3, :3] = _rotation(axis, q[j.get("name")])
            else:
                motion[:3, 3] = q[j.get("name")] * axis
        pose[child] = pose[parent] @ _transform(j) @ motion
        owner[child] = owner[parent] if j.get("type") == "fixed" else child

    limits = {j.get("name"): [float(v) for v in (j.find("limit").get("lower"),
                                                 j.find("limit").get("upper"))] for j in joints
              if j.find("limit") is not None}
    speeds = {j.get("name"): float(j.find("limit").get("velocity")) for j in joints
              if j.find("limit") is not None}
    parts = {}
    for name, T in pose.items():
        mesh = _collision_mesh(links[name])
        if mesh is not None:
            V, F = mesh
            parts.setdefault(owner[name], []).append((V @ T[:3, :3].T + T[:3, 3], F))

    bodies = {}
    for name, meshes in parts.items():
        offsets = np.cumsum([0] + [len(V) for V, _ in meshes])
        V = np.vstack([V for V, _ in meshes])
        F = np.vstack([F + o for (_, F), o in zip(meshes, offsets)]).astype(np.int32)
        mass = sum(float(links[n].find("inertial/mass").get("value"))
                   for n, o in owner.items() if o == name and links[n].find("inertial") is not None)
        volume = signed_volume(V, F)
        pad = name in FINGER_LINKS
        bodies[name] = trusty.affine.add_affine_body(
            world, V, F, density=mass / volume, stiffness=BODY_STIFFNESS,
            friction_mu=FINGER_FRICTION if pad else ARM_FRICTION)

    # The base is bolted to the world.
    trusty.affine.add_fixed_joint(world, bodies[owner["fr3_link0"]], (0.0, 0.0, 0.0),
                                  (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), stiffness=JOINT_STIFFNESS)

    arm, grip = [], []
    for j in joints:
        name = j.get("name")
        if j.get("type") == "fixed":
            continue
        T = pose[j.find("parent").get("link")] @ _transform(j)
        axis = T[:3, :3] @ np.array([float(v) for v in j.find("axis").get("xyz").split()])
        child = bodies[owner[j.find("child").get("link")]]
        parent = bodies[owner[j.find("parent").get("link")]]
        if name in ARM_JOINTS:
            joint = trusty.affine.add_revolute_joint(
                world, child, tuple(T[:3, 3] - 0.05 * axis), tuple(T[:3, 3] + 0.05 * axis),
                body_j=parent, stiffness=JOINT_STIFFNESS)
            lower, upper = limits[name]
            trusty.affine.add_joint_limit(
                world, joint, max(lower - q[name], -ANGLE_RANGE), min(upper - q[name], ANGLE_RANGE),
                margin=ARM_LIMIT_MARGIN, stiffness=LIMIT_STIFFNESS)
            arm.append(trusty.affine.add_actuator(
                world, joint, stiffness=ARM_GAINS[ARM_JOINTS.index(name)]))
        else:
            joint = trusty.affine.add_prismatic_joint(
                world, child, tuple(pose[j.find("child").get("link")][:3, 3]), tuple(axis),
                body_j=parent, stiffness=JOINT_STIFFNESS)
            lower, upper = limits[name]
            trusty.affine.add_joint_limit(
                world, joint, max(lower, 5e-4) - finger0, upper - finger0, margin=finger_margin,
                stiffness=LIMIT_STIFFNESS)
            grip.append(trusty.affine.add_actuator(world, joint, stiffness=FINGER_GAIN))

    tcp = pose[HAND_LINK].copy()
    tcp[:3, 3] += TCP_OFFSET * tcp[:3, 2]
    return Franka(bodies=bodies, arm_actuators=arm, finger_actuators=grip,
                  hand=bodies[owner[HAND_LINK]], hand_pose=pose[HAND_LINK], tcp=tcp,
                  arm_limits=np.array([limits[n] for n in ARM_JOINTS]),
                  arm_speeds=np.array([speeds[n] for n in ARM_JOINTS]),
                  q0=np.asarray(q0, float), finger0=finger0 if fingers else 0.0)


def grasp_contact_config(dhat=GRASP_DHAT, kappa=GRASP_KAPPA, epsv=GRASP_EPSV,
                         broadphase=trusty.contact.BroadphaseKind.AffineRelative):
    """Contact settings for pads gripping an object by friction; a deformable object needs `TwoLevel`."""
    return trusty.contact.Config(dhat=dhat, kappa=kappa, epsv=epsv, broadphase=broadphase)


def grasp_command_width(width, mass, mu, *, safety=GRASP_SAFETY, gravity=9.81):
    """Gripper command that squeezes an object of `width` (m) with `safety` times the
    force that holds it against gravity by friction `mu`.

    @return (command width (m), grip force per pad (N))
    """
    force = safety * mass * gravity / (2.0 * 0.5 * (mu + FINGER_FRICTION))
    return width - 2.0 * force / FINGER_GAIN, force


def meshes(world, franka):
    """[(name, V, F, rgb)] of every body of the arm, for rendering."""
    out = []
    for name, body in franka.bodies.items():
        V, F = trusty.affine.surface(world, body)
        out.append((name, np.asarray(V), np.asarray(F),
                    (0.25, 0.25, 0.28) if name in FINGER_LINKS else (0.92, 0.92, 0.94)))
    return out


def register_meshes(ps, world, franka):
    """Show every body of the arm in polyscope; returns {body name: mesh}."""
    out = {}
    for name, body in franka.bodies.items():
        V, F = trusty.affine.surface(world, body)
        out[name] = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F),
                                             color=(0.92, 0.92, 0.94), smooth_shade=True)
    return out


def update_meshes(world, franka, meshes):
    for name, body in franka.bodies.items():
        meshes[name].update_vertex_positions(np.asarray(trusty.affine.surface(world, body)[0]))
