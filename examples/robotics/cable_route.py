"""Franka FR3 presses a cable into three snap-in clips, one after another.

A 50 cm cable is held at one end and lies across a table beside three spring-jawed clips. The
hand grips the cable beside each clip, carries it over the clip and presses it down; the cable
forces the clip's two jaws apart, drops into the slot, and the jaws close behind it. The arm
follows the planned hand path through `trusty.control`, steered by the sensitivity of its own
steps. Headless, the run prints for each clip whether the cable ended up in its slot and how
far the jaws opened to let it in.

Usage:
    python examples/robotics/cable_route.py                  # polyscope
    python examples/robotics/cable_route.py --no-viewer      # headless
    python examples/robotics/cable_route.py --record cable_route.mp4
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

import trusty

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, make_T, rot_z
from servo_arm import make_box

# Cable.
CABLE_LEN, CABLE_R, CABLE_EDGE = 0.50, 0.0025, 0.004   # length, radius, segment length (m)
CABLE_E, CABLE_RHO, CABLE_MU = 1.0e7, 1400.0, 0.6      # jacketed cable: bending stiffness ~ 3e-4 N m^2

# Clip cross-section in the clip frame: x along the slot, y across, z up (m).
SLOT_W = 0.0065        # slot width: 0.75 mm clearance either side of the cable
LIP_GAP = 0.0036       # opening between the lips, narrower than the cable so it snaps in
JAW_T, CLIP_HALF_L = 0.0020, 0.004
SEAT_Z = 0.008         # slot floor, above the table
JAW_GAP = 0.0008       # clearance between a jaw's bottom and the base
LIP_UNDER = SEAT_Z + 2 * CABLE_R + 0.0006
LIP_TIP_T = 0.0008
JAW_K = 0.015          # jaw torsion spring (N m/rad)
JAW_RHO, CLIP_MU = 1200.0, 0.2

TABLE_Z, TABLE_MU = -0.005, 0.4
DHAT = 5e-4

GRASP_DEPTH = 0.0085   # cable centre below the hand's tool point (the fingertips reach 9.5 mm)
OPEN_W = 0.03          # finger opening while approaching (m)
PRESS_OFF = 0.022      # grip point past the clip centre along the slot (m)
OVERPRESS = 0.005      # push the held cable this far below its seated height (m)
CARRY_RISE = 0.025     # held cable this far above its seated height while carrying (m)

CONNECTOR = np.array([0.28, -0.24])      # where the cable is held
ROUTE_HEADINGS = np.radians([20.0, 40.0, 60.0])
SEG_LEN = 0.13
INIT_HEADING = math.radians(-40.0)


def prism(poly_yz, x0, x1):
    """Closed, outward-wound extrusion along x of a convex counter-clockwise (y, z) polygon."""
    P = np.asarray(poly_yz, float)
    n = len(P)
    V = np.vstack([np.c_[np.full(n, x0), P], np.c_[np.full(n, x1), P]])
    F = [[0, i + 1, i] for i in range(1, n - 1)] + [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    return V, np.array(F, np.int32)


def jaw_mesh(side):
    """One jaw (side +1: the +y wall, -1: its mirror) with a chamfered lip, and its hinge (y, z)."""
    zj = SEAT_Z + JAW_GAP
    yi, yo, yt = 0.5 * SLOT_W, 0.5 * SLOT_W + JAW_T, 0.5 * LIP_GAP
    z_tip = LIP_UNDER + LIP_TIP_T
    zG, zB = z_tip + (yi - yt), z_tip + (yo - yt)
    wall = prism([(yi, zj), (yo, zj), (yo, zB), (yi, zG)], -CLIP_HALF_L, CLIP_HALF_L)
    lip = prism([(yt, LIP_UNDER), (yi, LIP_UNDER), (yi, zG), (yt, z_tip)], -CLIP_HALF_L, CLIP_HALF_L)
    V = np.vstack([wall[0], lip[0]])
    F = np.vstack([wall[1], lip[1] + len(wall[0])])
    if side < 0:
        V[:, 1] *= -1
        F = F[:, ::-1]
    return V, F, (side * yo, zj)


@dataclass
class Clip:
    """A snap-in clip: a fixed base and two spring-hinged jaws, slot axis along `heading`."""

    center: np.ndarray     # (3,) world
    heading: float         # rad
    jaws: list             # affine bodies
    base: tuple            # (V, F) of the fixed base

    def to_local(self, X):
        return (np.asarray(X) - self.center) @ rot_z(self.heading)

    def seat(self, X):
        """(y, z) of the cable centreline where it crosses the slot's mid-plane, in the clip frame."""
        L = self.to_local(X)
        for a, b in zip(L[:-1], L[1:]):
            if a[0] * b[0] <= 0.0 and a[0] != b[0]:
                p = a + (b - a) * (-a[0] / (b[0] - a[0]))
                if abs(p[1]) < 0.02 and p[2] < 0.03:      # this clip, not a far crossing
                    return p[1], p[2]
        return None

    def lip_gap(self, world):
        """Opening between the two lips (m)."""
        y = [self.to_local(np.asarray(trusty.affine.surface(world, b)[0]))[:, 1] for b in self.jaws]
        return float(y[0].min() - y[1].max())

    def seated(self, X):
        s = self.seat(X)
        return s is not None and abs(s[0]) < 0.5 * SLOT_W and SEAT_Z < s[1] < LIP_UNDER


def add_clip(world, center_xy, heading, index):
    c = np.r_[center_xy, TABLE_Z]
    R = rot_z(heading)
    half_w = 0.5 * SLOT_W + JAW_T
    Vb, Fb = make_box((0.0, 0.0, 0.5 * SEAT_Z), (CLIP_HALF_L, half_w, 0.5 * SEAT_Z))
    Vb = Vb @ R.T + c
    trusty.contact.add_wall(world, f"clip{index}_base", np.asfortranarray(Vb),
                            np.asfortranarray(Fb.astype(np.int32)), friction_mu=CLIP_MU)
    jaws = []
    for side in (+1, -1):
        V, F, (hy, hz) = jaw_mesh(side)
        body = trusty.affine.add_affine_body(
            world, np.ascontiguousarray(V @ R.T + c), np.ascontiguousarray(F), density=JAW_RHO,
            stiffness=fr.BODY_STIFFNESS, friction_mu=CLIP_MU)
        hinge = trusty.affine.add_revolute_joint(
            world, body, tuple(R @ [-CLIP_HALF_L, hy, hz] + c), tuple(R @ [CLIP_HALF_L, hy, hz] + c),
            stiffness=fr.JOINT_STIFFNESS)
        trusty.affine.add_actuator(world, hinge, stiffness=JAW_K)
        jaws.append(body)
    return Clip(center=c, heading=heading, jaws=jaws, base=(Vb, Fb))


def cable_rest():
    """Straight lie along INIT_HEADING, rising to the connector's exit height over the first 4 cm."""
    n = int(round(CABLE_LEN / CABLE_EDGE)) + 1
    s = np.linspace(0.0, CABLE_LEN, n)
    X = np.zeros((n, 3))
    X[:, :2] = CONNECTOR + s[:, None] * [math.cos(INIT_HEADING), math.sin(INIT_HEADING)]
    z_lie, z_exit = TABLE_Z + CABLE_R + 0.6 * DHAT, TABLE_Z + SEAT_Z + CABLE_R
    u = np.clip(1.0 - s / 0.04, 0.0, 1.0)
    X[:, 2] = z_lie + (z_exit - z_lie) * u * u * (3 - 2 * u)
    return X


def tube(X, r, nseg=8):
    """Surface mesh of a tube of radius `r` around the polyline X."""
    n = len(X)
    T = np.gradient(X, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.cross(T, [0.0, 0.0, 1.0])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    a = np.linspace(0.0, 2 * math.pi, nseg, endpoint=False)
    V = (X[:, None] + r * (np.cos(a)[None, :, None] * N[:, None]
                           + np.sin(a)[None, :, None] * B[:, None])).reshape(-1, 3)
    i, k = np.meshgrid(np.arange(n - 1), np.arange(nseg), indexing="ij")
    a0, a1 = i * nseg + k, i * nseg + (k + 1) % nseg
    F = np.stack([np.stack([a0, a1, a1 + nseg], -1), np.stack([a0, a1 + nseg, a0 + nseg], -1)], 2)
    return V, F.reshape(-1, 3).astype(np.int32)


def hand_R(yaw):
    """Hand orientation pointing down, fingers closing across the direction `yaw` (hand x)."""
    return rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def grasp_cycle(X, index, anchor, clip, arc, k):
    """Waypoints that grip cable vertex `index`, press it into `clip` and let go.

    `anchor` (2,) is where the cable is already held and `arc` (m) the cable length from there
    to the grip point."""
    p = X[index]
    t = X[min(index + 1, len(X) - 1)] - X[max(index - 1, 0)]
    yaw_t = math.atan2(t[1], t[0])
    u = clip.center[:2] - anchor
    u /= np.linalg.norm(u)
    press = clip.center[:2] + PRESS_OFF * u
    # The chord from the anchor equals the arc, so carrying does not tug the seated clips out.
    carry = anchor + math.sqrt(max(arc**2 - CARRY_RISE**2, 0.0)) * u
    yaw_u = math.atan2(u[1], u[0])
    # The cable tangent is +-hand x; take the sign that turns the hand least.
    _, yaw_grip, yaw_press = min(
        (max(abs(wrap(yaw_t + s)), abs(wrap(yaw_u + s)))
         + (10.0 if abs(wrap(yaw_u - yaw_t)) > 0.9 * math.pi else 0.0), wrap(yaw_t + s),
         wrap(yaw_u + s)) for s in (0.0, math.pi))
    z_grip = max(p[2], TABLE_Z + CABLE_R) + GRASP_DEPTH
    z_seat = TABLE_Z + SEAT_Z + CABLE_R + GRASP_DEPTH
    pose = lambda yaw, xy, z: make_T(hand_R(yaw), [xy[0], xy[1], z])   # noqa: E731
    return [Waypoint(1.5, pose(yaw_grip, p, z_grip + 0.05), OPEN_W, f"c{k}_approach"),
            Waypoint(1.0, pose(yaw_grip, p, z_grip), name=f"c{k}_descend"),
            Waypoint(0.6, gripper=0.0, name=f"c{k}_close"),
            Waypoint(0.8, pose(yaw_grip, p, z_grip + 0.03), name=f"c{k}_lift"),
            Waypoint(2.0, pose(yaw_press, carry, z_seat + CARRY_RISE), name=f"c{k}_carry"),
            Waypoint(1.5, pose(yaw_press, press, z_seat - OVERPRESS), name=f"c{k}_press"),
            Waypoint(0.3, name=f"c{k}_hold"),
            Waypoint(0.5, gripper=OPEN_W, name=f"c{k}_release"),
            Waypoint(0.8, pose(yaw_press, press, z_seat - OVERPRESS + 0.06), name=f"c{k}_up")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--slack", type=float, default=-0.002,
                   help="cable length (m) beyond the route between the last clip and the grip point; "
                        "negative pulls the cable taut, which presses it onto the lips")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact", "rods")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=80))
    trusty.contact.enable(world, trusty.contact.Config(dhat=DHAT))
    # Two contact widths below the clips' bases, so they are not in contact with the plane.
    trusty.contact.add_plane(world, np.array([0.0, 0.0, TABLE_Z - 2.0 * DHAT]),
                             np.array([0.0, 0.0, 1.0]), friction_mu=TABLE_MU)
    franka = fr.build_franka(world)

    route = [CONNECTOR]
    for a in ROUTE_HEADINGS:
        route.append(route[-1] + SEG_LEN * np.array([math.cos(a), math.sin(a)]))
    headings = 0.5 * (ROUTE_HEADINGS + np.r_[ROUTE_HEADINGS[1:], ROUTE_HEADINGS[-1]])
    clips = [add_clip(world, route[i + 1], headings[i], i) for i in range(3)]

    material = trusty.rods.RodMaterial()
    material.youngs_modulus, material.poisson_ratio = CABLE_E, 0.4
    material.radius, material.density = CABLE_R, CABLE_RHO
    X0 = cable_rest()
    cable = trusty.rods.add_rod(world, X0, material, friction_mu=CABLE_MU)
    trusty.rods.pin_vertices(world, cable, [0])
    trusty.rods.pin_twist(world, cable, [0])

    reach = np.cumsum([np.linalg.norm(route[i + 1] - route[i]) for i in range(3)])
    arcs = [np.linalg.norm(route[k + 1] - route[k]) + PRESS_OFF + args.slack for k in range(3)]
    grip_vertex = [int(round((reach[k] + PRESS_OFF + args.slack) / CABLE_EDGE)) for k in range(3)]

    def cable_points():
        return np.asarray(trusty.rods.read_positions(world, cable))

    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add([Waypoint(0.5, name="settle")])
    driver = Driver(world, franka, trajectory, args.dt)
    cycle = grasp_cycle(X0, grip_vertex[0], route[0], clips[0], arcs[0], 0)
    if args.steps is None:
        args.steps = int(np.ceil((0.5 + 3 * sum(w.duration for w in cycle) + 3.0) / args.dt))
    state = {"k": 0}
    spread = [0.0] * 3

    def step():
        if trajectory.done:
            k = state["k"]
            if k < 3:
                trajectory.add(grasp_cycle(cable_points(), grip_vertex[k], route[k], clips[k],
                                           arcs[k], k))
                state["k"] += 1
            else:
                trajectory.add([Waypoint(2.0, franka.tcp, name="home")])
        driver.step()
        for i, clip in enumerate(clips):
            spread[i] = max(spread[i], clip.lip_gap(world))

    def meshes():
        out = fr.meshes(world, franka)
        for i, clip in enumerate(clips):
            for side, body in enumerate(clip.jaws):
                V, F = trusty.affine.surface(world, body)
                out.append((f"clip{i}_jaw{side}", np.asarray(V), np.asarray(F), (0.2, 0.45, 0.85)))
        return out + [("cable", *tube(cable_points(), CABLE_R), (0.95, 0.55, 0.15))]

    def setup_scene(ps):
        g = 0.9
        ps.register_surface_mesh(
            "table", np.array([[-0.1, -g, TABLE_Z], [g, -g, TABLE_Z], [g, g, TABLE_Z], [-0.1, g, TABLE_Z]]),
            np.array([[0, 1, 2], [0, 2, 3]]), color=(0.75, 0.7, 0.62))
        for i, clip in enumerate(clips):
            ps.register_surface_mesh(f"clip{i}_base", *clip.base, color=(0.15, 0.3, 0.6))

    common.run(args, step=step, meshes=meshes, setup_scene=setup_scene,
               gui_text=lambda: [f"phase {trajectory.name}"],
               camera=((0.50, -0.55, 0.38), (0.38, -0.07, 0.04)),
               follow=common.follow_tcp(driver.controller, (0.12, -0.48, 0.34), (0.43, -0.13, 0.0),
                                        z_max=0.08))

    X = cable_points()
    for i, clip in enumerate(clips):
        s = clip.seat(X)
        where = f"{1e3 * s[0]:+.2f} mm off the slot centre" if s else "not at the clip"
        print(f"clip {i}: cable {'in' if clip.seated(X) else 'not in'} the slot ({where}); "
              f"lips opened to {1e3 * spread[i]:.2f} mm for a {2e3 * CABLE_R:.1f} mm cable, "
              f"now {1e3 * clip.lip_gap(world):.2f} mm")
    print(driver.summary())


if __name__ == "__main__":
    main()
