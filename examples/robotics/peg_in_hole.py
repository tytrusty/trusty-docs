"""Franka FR3 inserts a chamfered rigid peg into a tight, chamfered hole.

A near-rigid 20 mm x 60 mm peg is picked off the table, carried over a fixture's blind hole
with a small lateral offset, and driven straight down. The controller knows nothing about the
hole: the two chamfers meet, contact pushes the peg toward the hole axis, and the arm's servos
give way until the peg seats. The hole leaves `--clearance` of room around the peg at its
tightest. Headless, the run prints how deep the peg went, how far it sat from the hole axis,
and how much it tilted.

Usage:
    python examples/robotics/peg_in_hole.py                  # polyscope
    python examples/robotics/peg_in_hole.py --no-viewer      # headless
    python examples/robotics/peg_in_hole.py --no-viewer --clearance 2e-5
    python examples/robotics/peg_in_hole.py --record peg_in_hole.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.spatial.transform import Rotation

import trusty

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, frame_points, frame_pose, make_T
from servo_arm import make_box
from utils import make_lathe, signed_volume

TABLE_Z = -0.005                  # table top (m): just below the robot base
PEG_XY = (0.42, -0.15)
HOLE_XY = (0.50, 0.12)
PEG_R, PEG_L = 0.010, 0.060       # circumscribed radius and length (m)
PEG_CHAMFER = 0.001               # 45 degree tip chamfer (m)
PEG_MASS = 0.10                   # kg
BLOCK_SIDE, BLOCK_H = 0.070, 0.045   # fixture footprint and height (m)
HOLE_DEPTH = 0.030                # blind hole (m)
HOLE_CHAMFER = 0.0015             # 45 degree entrance chamfer (m)
SIDES = 64                        # polygon sides of the peg and the hole
MU = 0.3                          # peg and fixture friction (the finger pads are 1.0)
GRASP_BELOW_TOP = 0.010           # TCP below the peg top (m)
PRESS = 0.001                     # commanded travel past the hole bottom (m)


def clearance_gap(hole_r, n):
    """Radial room (m) of a centred peg in the hole at the hole's tightest side."""
    return hole_r * math.cos(math.pi / n) - PEG_R


class Peg:
    """The peg; `pose()` is the (4, 4) frame at its bottom centre, z along its axis."""

    def __init__(self, world, bottom):
        V, F = make_lathe([(0, 0), (PEG_R - PEG_CHAMFER, 0), (PEG_R, PEG_CHAMFER),
                           (PEG_R, PEG_L - 5e-4), (PEG_R - 5e-4, PEG_L), (0, PEG_L)], SIDES)
        volume = signed_volume(V, F)
        self.world = world
        self.body = trusty.affine.add_affine_body(
            world, np.ascontiguousarray(V + bottom), np.ascontiguousarray(F),
            density=PEG_MASS / volume, stiffness=fr.BODY_STIFFNESS, friction_mu=MU)
        trusty.control.track(world, self.body, frame_points(make_T(t=bottom)))

    def pose(self):
        return frame_pose(trusty.control.tracked_points(self.world, self.body))

    def surface(self):
        V, F = trusty.affine.surface(self.world, self.body)
        return np.asarray(V), np.asarray(F)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--clearance", type=float, default=1e-4,
                   help="radial room between the peg and the hole at the tightest side (m)")
    p.add_argument("--misalign", type=float, default=7.5e-4,
                   help="lateral offset of the approach from the hole axis (m)")
    p.add_argument("--insert-time", type=float, default=3.0, help="seconds for the insertion stroke")
    args = p.parse_args()

    hole_r = (PEG_R + args.clearance) / math.cos(math.pi / SIDES)
    dhat = 0.25 * args.clearance     # the contact barrier must stay narrower than the clearance

    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, fr.grasp_contact_config(dhat=dhat))
    # Two contact widths below the fixture's base, so the fixture is not in contact with the plane.
    trusty.contact.add_plane(world, np.array([0.0, 0.0, TABLE_Z - 2.0 * dhat]),
                             np.array([0.0, 0.0, 1.0]), friction_mu=MU)

    top_z = TABLE_Z + BLOCK_H
    bottom_z = top_z - HOLE_DEPTH
    square = lambda th: 0.5 * BLOCK_SIDE / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))  # noqa: E731
    Vf, Ff = make_lathe([(0, 0), (square, 0), (square, BLOCK_H), (hole_r + HOLE_CHAMFER, BLOCK_H),
                         (hole_r, BLOCK_H - HOLE_CHAMFER), (hole_r, BLOCK_H - HOLE_DEPTH),
                         (0, BLOCK_H - HOLE_DEPTH)], SIDES)
    Vf = Vf + [HOLE_XY[0], HOLE_XY[1], TABLE_Z]
    trusty.contact.add_wall(world, "fixture", np.asfortranarray(Vf), np.asfortranarray(Ff),
                            friction_mu=MU)

    franka = fr.build_franka(world)
    peg = Peg(world, (PEG_XY[0], PEG_XY[1], TABLE_Z + dhat))
    grip_width, grip_force = fr.grasp_command_width(2 * PEG_R, PEG_MASS, MU)
    print(f"peg {2e3 * PEG_R:.0f} mm x {1e3 * PEG_L:.0f} mm, hole {1e3 * HOLE_DEPTH:.0f} mm deep, "
          f"{1e6 * clearance_gap(hole_r, SIDES):.0f} um clearance, approach offset "
          f"{1e3 * args.misalign:.2f} mm; grip command {1e3 * grip_width:.2f} mm gives "
          f"{grip_force:.2f} N per pad")

    R0 = franka.tcp[:3, :3]
    peg_top = peg.pose()[:3, 3] + [0, 0, PEG_L]
    grasp = make_T(R0, peg_top - [0, 0, GRASP_BELOW_TOP])
    hover = make_T(R0, grasp[:3, 3] + [0, 0, 0.08])
    lifted = make_T(R0, grasp[:3, 3] + [0, 0, top_z + 0.03 - TABLE_Z])   # peg bottom 3 cm over the fixture

    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add([
        Waypoint(0.5, name="settle"),
        Waypoint(2.0, hover, fr.GRIPPER_OPEN, "approach"),
        Waypoint(1.0, grasp, name="descend"),
        Waypoint(fr.GRASP_CLOSE_TIME, gripper=grip_width, name="close"),
        Waypoint(fr.GRASP_SETTLE_TIME, name="settle grip"),
        Waypoint(1.0, lifted, name="lift")])
    driver = Driver(world, franka, trajectory, args.dt)
    hand = driver.controller
    state = {"insert": False, "depth": 0.0, "tilt": 0.0, "grip_tilt": 0.0}

    def insertion():
        """Hand poses that bring the peg, plumb, over the hole and down to its bottom."""
        T_peg = peg.pose()
        in_hand = np.linalg.inv(hand.pose()) @ T_peg
        tilt = Rotation.from_rotvec(np.cross(T_peg[:3, 2], [0, 0, 1]))   # swing the axis to vertical
        state["grip_tilt"] = math.degrees(math.acos(T_peg[2, 2]))
        R_plumb = tilt.as_matrix() @ T_peg[:3, :3]
        offset = args.misalign * R0[:, 1]                                 # along the finger axis

        def hand_at(z):
            bottom = [HOLE_XY[0] + offset[0], HOLE_XY[1] + offset[1], z]
            return make_T(R_plumb, bottom) @ np.linalg.inv(in_hand)

        above, down = hand_at(top_z + 0.005), hand_at(bottom_z - PRESS)
        retreat = make_T(down[:3, :3], down[:3, 3] + [0, 0, 0.08])
        return [Waypoint(2.5, hand_at(top_z + 0.035), name="carry"),
                Waypoint(0.8, above, name="align"),
                Waypoint(args.insert_time, down, name="insert"),
                Waypoint(0.6, name="seated"),
                Waypoint(0.5, gripper=fr.GRIPPER_OPEN, name="release"),
                Waypoint(0.8, retreat, name="retreat")]

    if args.steps is None:
        insert_time = 2.5 + 0.8 + args.insert_time + 0.6 + 0.5 + 0.8
        args.steps = int(np.ceil((sum(w.duration for w in trajectory.queue) + insert_time)
                                 / args.dt)) + 10

    def step():
        if trajectory.done and not state["insert"]:
            state["insert"] = True
            trajectory.add(insertion())
        phase = trajectory.name
        driver.step()
        T = peg.pose()
        if phase in ("insert", "seated"):
            state["depth"] = top_z - T[2, 3]
            state["tilt"] = max(state["tilt"], math.degrees(math.acos(T[2, 2])))
            state["offset"] = 1e6 * np.linalg.norm(T[:2, 3] - HOLE_XY)

    def meshes():
        V, F = peg.surface()
        return fr.meshes(world, franka) + [("peg", V, F, (0.85, 0.55, 0.20))]

    def setup_scene(ps):
        V, F = make_box((0.4, 0.0, TABLE_Z - 0.01), (0.6, 0.6, 0.01))
        ps.register_surface_mesh("table", V, F, color=(0.55, 0.45, 0.35))
        ps.register_surface_mesh("fixture", Vf, Ff, color=(0.45, 0.55, 0.65), smooth_shade=False)

    common.run(args, step=step, meshes=meshes,
               gui_text=lambda: [f"{trajectory.name}: depth {1e3 * state['depth']:.2f} mm"],
               camera=((0.88, -0.12, 0.30), (0.46, -0.01, 0.12)), setup_scene=setup_scene)

    T = peg.pose()
    print(f"peg tilted {state['grip_tilt']:.2f} deg in the grip before the insertion")
    print(f"deepest insertion {1e3 * state['depth']:.2f} mm of {1e3 * HOLE_DEPTH:.0f} mm; "
          f"after release {1e3 * (top_z - T[2, 3]):.2f} mm, {1e6 * np.linalg.norm(T[:2, 3] - HOLE_XY):.0f} um "
          f"from the hole axis, tilt at most {state['tilt']:.3f} deg while inserted")
    print(driver.summary())


if __name__ == "__main__":
    main()
