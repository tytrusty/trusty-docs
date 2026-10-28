"""Franka FR3 tracing a square with its tool tip, steered through the sensitivity of its own steps.

The hand points straight down and traces a 20 cm square 25 cm above a table
plane. Every step the controller reads where the tool tip is, where the next
step would take it, and how it moves with the seven joint servos, and solves
for the servo targets that put it on the path. Headless, the run prints how far
the tip stayed from the path.

Usage:
    python examples/robotics/reach.py                  # polyscope
    python examples/robotics/reach.py --no-viewer      # headless
    python examples/robotics/reach.py --record reach.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, make_T, pose_error


def square_waypoints(tcp, center, side, edge_time):
    corners = [np.asarray(center, float) + 0.5 * side * np.array(s)
               for s in ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))]
    T = [make_T(tcp[:3, :3], p) for p in corners]
    wps = [Waypoint(0.5, name="settle"), Waypoint(2.0, T[0], name="approach")]
    wps += [Waypoint(edge_time, T[(k + 1) % 4], name=f"edge{k}") for k in range(4)]
    return wps + [Waypoint(0.5, name="hold")], T


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--side", type=float, default=0.2, help="square side (m)")
    p.add_argument("--edge-time", type=float, default=1.5, help="seconds per edge")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, trusty.contact.Config(
        dhat=1e-3, broadphase=trusty.contact.BroadphaseKind.AffineRelative))
    trusty.contact.add_plane(world, np.array([0.0, 0.0, -0.005]), np.array([0.0, 0.0, 1.0]))
    franka = fr.build_franka(world)

    waypoints, corners = square_waypoints(
        franka.tcp, (0.45, 0.0, 0.25), args.side, args.edge_time)
    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add(waypoints)
    driver = Driver(world, franka, trajectory, args.dt)
    if args.steps is None:
        args.steps = int(np.ceil(sum(w.duration for w in waypoints) / args.dt)) + 1
    errors = []

    def step():
        phase = trajectory.name
        T = driver.step()
        if phase.startswith("edge"):
            errors.append(pose_error(driver.controller.pose(), T))

    common.run(args, step=step, meshes=lambda: fr.meshes(world, franka),
               gui_text=lambda: [f"phase {trajectory.name}"],
               camera=((0.95, -0.80, 0.65), (0.22, 0.0, 0.45)),
               setup_scene=lambda ps: ps.register_curve_network(
                   "path", np.array([T[:3, 3] for T in corners]), "loop",
                   radius=0.002, color=(0.9, 0.3, 0.2)))

    if errors:
        e = np.array(errors)
        print(f"tip distance from the commanded path: max {1e3 * np.linalg.norm(e[:, :3], axis=1).max():.2f} mm, "
              f"rotation max {1e3 * np.linalg.norm(e[:, 3:], axis=1).max():.2f} mrad")
    print(driver.summary())


if __name__ == "__main__":
    main()
