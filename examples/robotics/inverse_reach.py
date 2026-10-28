"""Inverse reach: click a point and the Franka's tool tip goes there, steered through the
sensitivity of its own steps.

Every step the arm reads where its tool tip is, where the next step would take it, and how it
moves with the seven joint servos (`trusty.control`), and solves for the servo targets that carry
it to the target. A column stands in the middle of the table, and every link of the arm is kept
clear of it as a constraint of the same solve. In the viewer, a click on the table (or on anything
else drawn) sets the target, drawn as a red sphere 10 cm above the clicked point; the tip moves
there with the hand pointing down. Headless, the tip circles the column and the run prints how far
it stayed from the path and how close the arm came to the column.

Usage:
    python examples/robotics/inverse_reach.py                  # polyscope: click to aim
    python examples/robotics/inverse_reach.py --no-viewer      # headless: circle the column
    python examples/robotics/inverse_reach.py --no-column      # an empty table
    python examples/robotics/inverse_reach.py --record inverse_reach.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import trusty

import common
import franka as fr
from pose_control import Driver, HandController, Trajectory, Waypoint, make_T, pose_error
from servo_arm import make_box
from utils import make_lathe

SHOULDER = np.array([0.0, 0.0, 0.333])    # m; the arm's second joint
REACH = 0.6                               # m; targets are kept this close to the shoulder
MIN_RADIUS = 0.25                         # m; and this far from the base's vertical axis
SPEED = 0.5                               # m/s; how fast the tip moves to a new target
CLICK_SLOP = 4.0                          # px; a press that moves farther is a camera drag
HOVER = 0.1                               # m; the target sits this far above a clicked point
COLUMN_CENTER, COLUMN_RADIUS, COLUMN_HEIGHT = np.array([0.45, 0.0]), 0.06, 0.3
BARRIER_STIFFNESS = 1e8                   # Pa; the servos are stiff, so a soft barrier lets them squeeze a gap to nothing
MARGIN = 0.06                             # m; links stay this far from the column
CIRCLE_RADIUS, CIRCLE_HEIGHT, CIRCLE_PERIOD = 0.2, 0.25, 6.0


def clamp_to_reach(p):
    """The point nearest `p` that the hand can reach pointing down: inside a ball about the
    shoulder, in front of the base and off its axis, above the floor."""
    d = p - SHOULDER
    n = np.linalg.norm(d)
    p = SHOULDER + d * min(1.0, REACH / n) if n > 0 else p
    xy = np.array([max(p[0], 0.1), p[1]])     # in front of the base
    xy = xy * max(1.0, MIN_RADIUS / np.hypot(*xy))
    return np.array([xy[0], xy[1], max(p[2], 0.02)])


def column_distance(points):
    """Signed distance (m) from `points` (n, 3) to the column, negative inside it."""
    q = np.stack([np.hypot(*(points[:, :2] - COLUMN_CENTER).T) - COLUMN_RADIUS,
                  np.abs(points[:, 2] - 0.5 * COLUMN_HEIGHT) - 0.5 * COLUMN_HEIGHT], axis=1)
    return np.minimum(q.max(axis=1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)


def clear_of_column(p):
    """The point nearest `p` that is `MARGIN` or more outside the column."""
    h = 1e-4
    for _ in range(4):
        phi = column_distance(p[None])[0]
        if phi >= MARGIN:
            break
        grad = np.array([(column_distance((p + h * e)[None])[0] - column_distance((p - h * e)[None])[0])
                         for e in np.eye(3)])
        p = p + grad / np.linalg.norm(grad) * (MARGIN - phi)
    return p


def circle(t):
    """The tip's path in the headless run: a horizontal circle around the column (m)."""
    a = 2.0 * math.pi * t / CIRCLE_PERIOD
    return np.array([*(COLUMN_CENTER + CIRCLE_RADIUS * np.array([math.cos(a), math.sin(a)])),
                     CIRCLE_HEIGHT])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--laps", type=int, default=1, help="laps of the circle in the headless run")
    p.add_argument("--no-column", action="store_true", help="leave the column out")
    args = p.parse_args()

    trusty.check_capabilities("affine")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    column, column_mesh = not args.no_column, []
    if column:
        trusty.contact.enable(world, trusty.contact.Config(
            dhat=1e-3, kappa=BARRIER_STIFFNESS,
            broadphase=trusty.contact.BroadphaseKind.AffineRelative))
        V, F = make_lathe([(0, 0), (COLUMN_RADIUS, 0), (COLUMN_RADIUS, COLUMN_HEIGHT),
                           (0, COLUMN_HEIGHT)], 48)
        V = V + [*COLUMN_CENTER, 0.0]
        trusty.contact.add_wall(world, "column", V, F)
        column_mesh = [("column", V, F, (0.45, 0.5, 0.6))]
    franka = fr.build_franka(world)
    R_down = franka.tcp[:3, :3]
    scripted = args.no_viewer or args.record is not None
    trajectory = Trajectory(franka.tcp, 2 * franka.finger0)
    controller = HandController(world, franka, keep_out=column_distance if column else None,
                                margin=MARGIN)
    driver = Driver(world, franka, trajectory, args.dt, controller)
    tip = controller.pose
    errors, closest = [], [math.inf]

    if scripted:
        steps_per_lap = round(CIRCLE_PERIOD / args.dt)
        trajectory.add([Waypoint(1.0, make_T(R_down, circle(0.0)), name="approach")])
        trajectory.add([Waypoint(args.dt, make_T(R_down, circle((k + 1) * args.dt)), name="circle")
                        for k in range(args.laps * steps_per_lap)])
        if args.steps is None:
            args.steps = int(1.0 / args.dt) + args.laps * steps_per_lap + 1

    state = {"target": franka.tcp[:3, 3].copy(), "down": None}

    def aim(point):
        point = clamp_to_reach(point)
        if column:
            point = clear_of_column(point)
        state["target"] = point
        distance = np.linalg.norm(point - tip()[:3, 3])
        trajectory.go([Waypoint(max(0.4, distance / SPEED), make_T(R_down, point), name="aim")])

    def step():
        phase = trajectory.name
        T = driver.step()
        if scripted and phase == "circle":
            errors.append(np.linalg.norm(pose_error(tip(), T)[:3]))
        closest[0] = min(closest[0], controller.clearance())

    def interact(ps, psim):
        io = psim.GetIO()
        if psim.IsMouseClicked(0) and not io.WantCaptureMouse:
            state["down"] = np.array(psim.GetMousePos())
        if psim.IsMouseReleased(0) and state["down"] is not None:
            pos = np.array(psim.GetMousePos())
            if np.linalg.norm(pos - state["down"]) < CLICK_SLOP:
                hit = np.array(ps.screen_coords_to_world_position(pos))
                if np.all(np.isfinite(hit)):      # a click on the background hits nothing
                    aim(hit + [0.0, 0.0, HOVER])
            state["down"] = None
        ps.get_point_cloud("target").update_point_positions(state["target"][None, :])

    def setup_scene(ps):
        V, F = make_box((0.4, 0.0, -0.01), (0.6, 0.6, 0.01))
        ps.register_surface_mesh("table", V, F, color=(0.55, 0.45, 0.35))
        if scripted:
            path = np.array([circle(k * CIRCLE_PERIOD / 64) for k in range(64)])
            ps.register_curve_network("path", path, "loop", radius=0.002, color=(0.9, 0.3, 0.2))
        else:
            marker = ps.register_point_cloud("target", state["target"][None, :])
            marker.set_color((0.9, 0.1, 0.1))
            marker.set_radius(0.02, relative=False)

    common.run(args, step=step, meshes=lambda: fr.meshes(world, franka) + column_mesh,
               gui_text=lambda: [f"target {np.round(state['target'], 3)} m",
                                 f"tip {np.round(tip()[:3, 3], 3)} m"],
               camera=((1.5, -1.1, 0.9), (0.4, 0.0, 0.4)), setup_scene=setup_scene,
               interact=None if scripted else interact, play=True)

    if errors:
        e = np.array(errors)
        print(f"tip distance from the circle: RMS {1e3 * np.sqrt(np.mean(e**2)):.2f} mm, "
              f"max {1e3 * e.max():.2f} mm")
    if column and scripted:
        print(f"closest the arm came to the column: {1e3 * closest[0]:.1f} mm outside it "
              f"(kept at least {1e3 * MARGIN:.0f} mm)")
    print(driver.summary())


if __name__ == "__main__":
    main()
