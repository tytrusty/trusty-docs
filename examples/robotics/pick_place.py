"""Franka FR3 picks three cubes off a table and stacks them into a tower, held by friction alone.

Each cube starts at a different yaw; the hand turns to put its fingers across a pair
of faces, squeezes with a few times the force that holds the cube's weight, lifts, and
sets it on the tower. Each place uses the cube-in-hand pose measured after the lift. The
arm follows the planned hand path through `trusty.control`, steered by the sensitivity
of its own steps. Headless, the run prints how much each cube slipped in the grip and
where the tower ended up.

Usage:
    python examples/robotics/pick_place.py                  # polyscope
    python examples/robotics/pick_place.py --no-viewer      # headless
    python examples/robotics/pick_place.py --record pick_place.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.spatial.transform import Rotation

import trusty

import common
import franka as fr
from pose_control import (Driver, Trajectory, Waypoint, frame_points, frame_pose, make_T,
                          rot_z)
from servo_arm import make_box

CUBE_STARTS = [(0.42, -0.22, 20.0), (0.58, -0.06, -30.0), (0.36, 0.04, 40.0)]   # x, y (m), yaw (deg)
TOWER_XY = (0.50, 0.22)
TABLE_Z = -0.005    # table top (m): just below the robot base so the first link never touches it
CUBE_COLORS = [(0.85, 0.30, 0.25), (0.25, 0.55, 0.85), (0.95, 0.75, 0.20)]
HOVER, CARRY = 0.10, 0.12    # approach height above a grasp, and carry height above a place (m)
PLACE_DROP = 1e-3            # release height above the resting place pose (m)


class Cube:
    """A near-rigid cube; `pose()` is its (4, 4) frame."""

    def __init__(self, world, center, yaw, size, mass, mu):
        V, F = make_box(np.zeros(3), (0.5 * size,) * 3)
        V = V @ rot_z(yaw).T + np.asarray(center, float)
        self.world = world
        self.body = trusty.affine.add_affine_body(
            world, np.ascontiguousarray(V), np.ascontiguousarray(F),
            density=mass / size**3, stiffness=fr.BODY_STIFFNESS, friction_mu=mu)
        self.rest = make_T(rot_z(yaw), center)
        trusty.control.track(world, self.body, frame_points(self.rest))

    def pose(self):
        return frame_pose(trusty.control.tracked_points(self.world, self.body))

    def surface(self):
        V, F = trusty.affine.surface(self.world, self.body)
        return np.asarray(V), np.asarray(F)


def hand_yaw_for(R_hand, cube_R):
    """Smallest hand yaw (rad) putting the finger axis normal to a pair of cube faces."""
    finger = R_hand[:, 1]
    face = cube_R[:, 0]
    a = math.atan2(face[1], face[0]) - math.atan2(finger[1], finger[0])
    return (a + 0.25 * math.pi) % (0.5 * math.pi) - 0.25 * math.pi


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--size", type=float, default=0.045, help="cube edge (m)")
    p.add_argument("--mass", type=float, default=0.2, help="cube mass (kg)")
    p.add_argument("--mu", type=float, default=0.8, help="cube friction (the finger pads are 1.0)")
    p.add_argument("--safety", type=float, default=fr.GRASP_SAFETY,
                   help="grip force over the force that holds a cube's weight; raising it squeezes harder")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, fr.grasp_contact_config())
    trusty.contact.add_plane(world, np.array([0.0, 0.0, TABLE_Z]), np.array([0.0, 0.0, 1.0]),
                             friction_mu=args.mu)
    franka = fr.build_franka(world)
    s = args.size
    cubes = [Cube(world, (x, y, TABLE_Z + 0.5 * s + fr.GRASP_DHAT), math.radians(yaw), s,
                  args.mass, args.mu) for x, y, yaw in CUBE_STARTS]
    grip_width, grip_force = fr.grasp_command_width(s, args.mass, args.mu, safety=args.safety)
    print(f"{len(cubes)} cubes of {1e3 * s:.0f} mm, {args.mass:g} kg; grip command "
          f"{1e3 * grip_width:.2f} mm gives {grip_force:.2f} N per pad")

    R0 = franka.tcp[:3, :3]                      # tool pointing down
    open_width = fr.GRIPPER_OPEN
    targets = [make_T(np.eye(3), (TOWER_XY[0], TOWER_XY[1], TABLE_Z + 0.5 * s + k * s))
               for k in range(len(cubes))]

    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add([Waypoint(0.5, name="settle")])
    driver = Driver(world, franka, trajectory, args.dt)
    hand = driver.controller

    def pick(cube):
        T = cube.pose()
        R = rot_z(hand_yaw_for(R0, T[:3, :3])) @ R0
        return [Waypoint(2.0, make_T(R, T[:3, 3] + [0, 0, HOVER]), open_width, "approach"),
                Waypoint(1.0, make_T(R, T[:3, 3]), name="descend"),
                Waypoint(fr.GRASP_CLOSE_TIME, gripper=grip_width, name="close"),
                Waypoint(fr.GRASP_SETTLE_TIME, name="settle"),
                Waypoint(1.0, make_T(R, T[:3, 3] + [0, 0, CARRY]), name="lift")]

    def place(cube, k):
        in_hand = np.linalg.inv(hand.pose()) @ cube.pose()
        best = None            # the cube's yaw on the tower is free modulo 90 degrees
        for quarter in range(4):
            T_hand = targets[k] @ make_T(rot_z(0.5 * math.pi * quarter)) @ np.linalg.inv(in_hand)
            cost = np.linalg.norm(Rotation.from_matrix(T_hand[:3, :3] @ R0.T).as_rotvec())
            if best is None or cost < best[0]:
                best = (cost, T_hand)
        T_hand = best[1].copy()
        T_hand[:3, 3] += [0, 0, PLACE_DROP]
        above = T_hand.copy()
        above[:3, 3] += [0, 0, CARRY]
        retreat = make_T(T_hand[:3, :3], T_hand[:3, 3] + [0, 0, HOVER])
        return [Waypoint(2.5, above, name="carry"), Waypoint(1.2, T_hand, name="lower"),
                Waypoint(0.3, name="rest"), Waypoint(0.5, gripper=open_width, name="release"),
                Waypoint(0.8, retreat, name="retreat")]

    plan = {"k": 0, "stage": "pick", "hold": None}
    trajectory.add(pick(cubes[0]))
    segment = sum(w.duration for w in pick(cubes[0]) + place(cubes[0], 0))
    if args.steps is None:
        args.steps = int(np.ceil((0.5 + len(cubes) * segment + 1.0) / args.dt)) + 10
    slip = {"mm": 0.0, "deg": 0.0}

    def next_segment():
        k = plan["k"]
        if plan["stage"] == "pick":
            plan["stage"] = "place"
            trajectory.add(place(cubes[k], k))
        elif k + 1 < len(cubes):
            plan["k"], plan["stage"] = k + 1, "pick"
            trajectory.add(pick(cubes[k + 1]))
        elif plan["stage"] == "place":
            plan["stage"] = "home"
            trajectory.add([Waypoint(1.0, franka.tcp, name="home")])

    def step():
        if trajectory.done:
            next_segment()
        phase, cube = trajectory.name, cubes[plan["k"]]
        driver.step()
        if phase in ("lift", "carry"):
            in_hand = np.linalg.inv(hand.pose()) @ cube.pose()
            if phase == "lift" and plan["hold"] is None:
                plan["hold"] = in_hand
            H = plan["hold"]
            slip["mm"] = max(slip["mm"], 1e3 * np.linalg.norm(in_hand[:3, 3] - H[:3, 3]))
            slip["deg"] = max(slip["deg"], math.degrees(np.linalg.norm(
                Rotation.from_matrix(H[:3, :3].T @ in_hand[:3, :3]).as_rotvec())))
        else:
            plan["hold"] = None

    def meshes():
        out = fr.meshes(world, franka)
        for i, c in enumerate(cubes):
            V, F = c.surface()
            out.append((f"cube{i}", V, F, CUBE_COLORS[i % len(CUBE_COLORS)]))
        return out

    def setup_scene(ps):
        V, F = make_box((0.4, 0.0, TABLE_Z - 0.01), (0.6, 0.6, 0.01))
        ps.register_surface_mesh("table", V, F, color=(0.55, 0.45, 0.35))
        V, F = make_box((TOWER_XY[0], TOWER_XY[1], TABLE_Z + 5e-4), (0.5 * s, 0.5 * s, 5e-4))
        ps.register_surface_mesh("target", V, F, color=(0.3, 0.8, 0.3), transparency=0.5)

    common.run(args, step=step, meshes=meshes,
               gui_text=lambda: [f"cube {plan['k']} {plan['stage']}: {trajectory.name}"],
               camera=((1.15, -0.68, 0.66), (0.36, 0.02, 0.30)), setup_scene=setup_scene)

    for i, (cube, target) in enumerate(zip(cubes, targets)):
        err = 1e3 * np.linalg.norm(cube.pose()[:3, 3] - target[:3, 3])
        print(f"cube {i}: {err:.2f} mm from its place in the tower")
    print(f"largest slip in the grip while carrying: {slip['mm']:.2f} mm, {slip['deg']:.2f} deg")
    print(driver.summary())


if __name__ == "__main__":
    main()
