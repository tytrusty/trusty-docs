"""Franka FR3 folds a towel in half along its diagonal, pinching one corner between its fingers.

A square towel lies on a table with one corner hanging a few centimetres over the edge. The
hand slides its fingers over that corner, pinches the single layer, swings it in an arc over
the fold line and lets go onto the opposite corner. The arm follows the planned hand path
through `trusty.control`, steered by the sensitivity of its own steps. Headless, the run
prints how far the folded corner landed from its target, how much of the flap lies over the
fold line, and how much the corner slipped in the pinch.

Usage:
    python examples/robotics/cloth_fold.py                  # polyscope
    python examples/robotics/cloth_fold.py --no-viewer      # headless
    python examples/robotics/cloth_fold.py --record cloth_fold.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import trusty

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, make_T
from servo_arm import make_box

SIDE, RES = 0.30, 24             # towel side (m) and quads per side
CENTER = np.array([0.45, 0.0])   # towel centre (x, y), on the fold line
OVERHANG = 0.03                  # corner A hangs this far past the table edge (m)
GRIP_DEPTH = 0.012               # pad centre this far inside the corner tip (m)
GRIP_OPEN, GRIP_CLOSE = 0.040, -0.004   # finger command while approaching, and pinching (m)
ARC_RADIUS = 0.19                # corner path about the fold line (m); the flap is 0.21 long
RELEASE_TILT = 0.6               # wrist roll at release (x pi): the tip leans past C and drops on it
RELEASE_DZ = 0.04                # corner tip height over C at release (m)
RELEASE_OPEN = 0.070             # finger opening after release (m)


def towel_mesh(res):
    """Unit-square sheet: UV (n, 2), triangles (m, 3), corner ids A, B, C, D counter-clockwise.

    Quads split along B-D, so the fold line is a row of mesh edges."""
    s = np.linspace(0.0, 1.0, res + 1)
    UV = np.stack(np.meshgrid(s, s, indexing="ij"), -1).reshape(-1, 2)
    vid = lambda i, j: i * (res + 1) + j     # noqa: E731
    F = [t for i in range(res) for j in range(res)
         for t in ((vid(i, j), vid(i + 1, j), vid(i, j + 1)),
                   (vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))]
    return UV, np.array(F, np.int32), (vid(0, 0), vid(res, 0), vid(res, res), vid(0, res))


def hand_R(roll):
    """Hand orientation rolled by `roll` about world x; the fingers close along z at roll 0."""
    z = np.array([0.0, math.cos(roll), -math.sin(roll)])
    x = np.array([1.0, 0.0, 0.0])
    return np.c_[x, np.cross(z, x), z]


def tcp_for_tip(tip, roll):
    """Hand pose that holds a corner tip at `tip`, GRIP_DEPTH in from the pad centre."""
    R = hand_R(roll)
    return make_T(R, np.asarray(tip, float) + GRIP_DEPTH * R[:, 2])


def fold_waypoints(tip_A, tip_C):
    """Pinch corner A, arc it over the fold line, release over C, retreat."""
    out = np.array([0.0, -0.10, 0.0])
    lift = tip_A + [0, 0, 0.02]
    wps = [Waypoint(0.5, name="settle"),
           Waypoint(2.5, tcp_for_tip(tip_A + out + [0, 0, 0.08], 0.0), GRIP_OPEN, "approach"),
           Waypoint(1.0, tcp_for_tip(tip_A + out, 0.0), name="descend"),
           Waypoint(1.5, tcp_for_tip(tip_A, 0.0), name="insert"),
           Waypoint(1.0, gripper=GRIP_CLOSE, name="pinch"),
           Waypoint(0.4, name="settle_grip"),
           Waypoint(1.0, tcp_for_tip(lift, 0.0), name="lift")]
    mid = np.array([CENTER[0], CENTER[1], tip_A[2]])
    for k, phi in enumerate(np.linspace(0.0, math.pi, 7)[1:-1]):
        tip = mid + [0.0, -ARC_RADIUS * math.cos(phi), 0.02 + ARC_RADIUS * math.sin(phi)]
        wps.append(Waypoint(0.9, tcp_for_tip(tip, min(phi, RELEASE_TILT * math.pi)), name=f"arc{k}"))
    above = tip_C + [0, 0, RELEASE_DZ]
    R = hand_R(RELEASE_TILT * math.pi)
    return wps + [Waypoint(1.2, tcp_for_tip(above, RELEASE_TILT * math.pi), name="lay"),
                  Waypoint(0.3, name="settle_lay"),
                  Waypoint(0.6, gripper=RELEASE_OPEN, name="release"),
                  Waypoint(1.0, make_T(R, tcp_for_tip(above, RELEASE_TILT * math.pi)[:3, 3]
                                       - 0.10 * R[:, 2]), name="retreat")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--dhat", type=float, default=2.5e-4,
                   help="contact distance (m); it sets how close the pads can pinch the cloth")
    p.add_argument("--youngs", type=float, default=1e6, help="cloth Young's modulus (Pa)")
    p.add_argument("--thickness", type=float, default=1e-3, help="cloth thickness (m)")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact", "shells")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(
        dhat=args.dhat, epsv=0.1, broadphase=trusty.contact.BroadphaseKind.TwoLevel))
    franka = fr.build_franka(world, finger_margin=0.1 * args.thickness)

    UV, F, (iA, iB, iC, iD) = towel_mesh(RES)
    half_diag = SIDE / math.sqrt(2.0)
    edge_y = CENTER[1] - half_diag + OVERHANG
    table_lo, table_hi = (0.15, edge_y, -0.04), (0.80, CENTER[1] + half_diag + 0.10, 0.0)
    table_V, table_F = make_box(0.5 * (np.array(table_lo) + table_hi),
                                0.5 * (np.array(table_hi) - table_lo))
    trusty.contact.add_wall(world, "table", np.asfortranarray(table_V),
                            np.asfortranarray(table_F.astype(np.int32)), friction_mu=0.4)

    cfg = trusty.shells.ShellConfig()
    cfg.youngs_modulus, cfg.poisson_ratio = args.youngs, 0.3
    cfg.thickness, cfg.density = args.thickness, 400.0
    cfg.bending_model = trusty.shells.BendingModel.QuadraticLagrange
    e_u = np.array([1.0, 1.0]) / math.sqrt(2.0)
    e_v = np.array([-1.0, 1.0]) / math.sqrt(2.0)
    xy = CENTER + np.outer(SIDE * (UV[:, 0] - 0.5), e_u) + np.outer(SIDE * (UV[:, 1] - 0.5), e_v)
    z0 = 0.5 * args.thickness + args.dhat
    X0 = np.c_[xy, np.full(len(xy), z0)]
    towel = trusty.shells.add_shell(world, X0, F, cfg, friction_mu=0.5)

    flap = np.flatnonzero(X0[:, 1] < 0.5 * (X0[iB, 1] + X0[iD, 1]) - 1e-9)
    waypoints = fold_waypoints(X0[iA], X0[iC] + [0, 0, -z0])
    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add(waypoints)
    driver = Driver(world, franka, trajectory, args.dt)
    if args.steps is None:
        args.steps = int(np.ceil(sum(w.duration for w in waypoints) / args.dt)) + 1
    slip = {"hold": None, "mm": 0.0}

    def cloth():
        return np.asarray(trusty.shells.read_positions(world, towel))

    def step():
        phase = trajectory.name
        driver.step()
        if phase in ("settle_grip", "lift", "lay") or phase.startswith("arc"):
            X = cloth()
            in_hand = np.linalg.solve(driver.controller.pose(), np.r_[X[iA], 1.0])[:3]
            if slip["hold"] is None:
                slip["hold"] = in_hand
            slip["mm"] = max(slip["mm"], 1e3 * np.linalg.norm(in_hand - slip["hold"]))

    common.run(args, step=step,
               meshes=lambda: fr.meshes(world, franka) + [("towel", cloth(), F, (0.30, 0.55, 0.85))],
               gui_text=lambda: [f"phase {trajectory.name}"],
               camera=((1.0, -0.35, 0.55), (0.45, 0.0, 0.10)),
               setup_scene=lambda ps: ps.register_surface_mesh(
                   "table", table_V, table_F, color=(0.55, 0.45, 0.35)))

    X = cloth()
    on_top = 100.0 * np.mean(X[flap, 1] > 0.5 * (X[iB, 1] + X[iD, 1]) - 5e-3)
    print(f"corner A landed {1e3 * np.linalg.norm(X[iA] - X[iC]):.1f} mm from corner C; "
          f"{on_top:.0f}% of the flap lies over the fold line")
    print(f"largest slip of the corner in the pinch: {slip['mm']:.1f} mm")
    print(driver.summary())


if __name__ == "__main__":
    main()
