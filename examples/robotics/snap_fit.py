"""Franka FR3 picks up a compliant snap-fit clip and presses it into a slot until its barbs latch.

The clip is one FEM solid: a stem the fingers grip, a base the fingertips press on, and
two barbed cantilever arms. The hand carries it over a housing, pushes the arms down
through the slot, where they bend inward, and the barbs spring back out under the lip. A
short tug then pulls up on the latched clip. The arm follows the planned hand path through
`trusty.control`, steered by the sensitivity of its own steps. Headless, the run prints
how far the arms bent, where the barbs ended up under the lip and how the clip answered
the tug.

Usage:
    python examples/robotics/snap_fit.py                  # polyscope
    python examples/robotics/snap_fit.py --no-viewer      # headless
    python examples/robotics/snap_fit.py --no-viewer --res 2   # coarser clip, faster
    python examples/robotics/snap_fit.py --record snap_fit.mp4
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
from utils import signed_volume

TABLE_Z = -0.005             # table top (m): just below the robot base
CLIP_XY = (0.45, -0.14)
SLOT_XY = (0.45, 0.12)

# Clip profile (m), in the x-z plane with the arm tips at z = 0.
ARM_T = 2.4e-3               # arm thickness (bending direction x)
ARM_OUT = 7.0e-3             # arm outer face |x|
ARM_LEN = 22e-3              # tip to handle
BARB_H = 5e-3                # barb height; its flat latch face is at z = BARB_H
BARB_P = 2.4e-3              # barb reach past the arm's outer face
RAMP_TIP = 0.3e-3            # barb width at the tip (ramp starts here)
BARB_LAND = 1e-3             # land between the ramp and the latch face ...
LAND_BACK = 0.3e-3           # ... tapering inward: a square latch corner snags and the arms buckle
BASE_H = 4e-3                # handle base: wider along y, the fingertips press on it
STEM_H = 14e-3               # gripped stem above the base
THICK = 10e-3                # stem + arms extent along y (the gripper's squeeze axis)
BASE_Y = 20e-3               # base extent along y
CLIP_E, CLIP_NU, CLIP_RHO, CLIP_MU = 2e8, 0.38, 900.0, 0.3

# Housing. Barb span 18.8 mm vs slot 16 mm: 1.4 mm arm deflection, 1.1 mm latch overlap.
SLOT_W = 16e-3
LIP_Z0 = 6e-3                # underside of the lip above the table
LIP_T = 3e-3
LIP_W = 22e-3                # each lip box along x
LIP_D = 40e-3                # along y
WALL_MU = 0.1
LIP_CHAMFER = 0.6e-3         # 45 deg lead-in: a sharp lip corner jams on the barb ramp
LEG_W = 5e-3

SHOULDER_GAP = 1.5e-3        # fingertips above the base top when grasping
INSERT_LOAD = 5.0            # N, pull the grip is sized for (the press itself goes through the base)
PRESS_TIP = 0.5e-3           # arm-tip height above the table at full press: the latch face clears the lip
TUG = 3e-3                   # TCP lift while still holding the latched clip (m)
DURATION = dict(settle=0.5, approach=2.0, hover=0.3, descend=1.0, close=fr.GRASP_CLOSE_TIME,
                settle_grip=fr.GRASP_SETTLE_TIME, lift=1.0, carry=2.0, align=1.0, steady=0.3,
                press=2.5, dwell=0.3, tug=0.8, release=0.6, retreat=1.0, observe=0.5)


def _block(c00, c10, c11, c01, nx, nz):
    """Bilinear (nx, nz) grid over the x-z quad c00-c10-c11-c01: points and CCW cells."""
    c = [np.asarray(p, float) for p in (c00, c10, c11, c01)]
    u, v = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, nz + 1), indexing="ij")
    u, v = u.ravel()[:, None], v.ravel()[:, None]
    P = (1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3]
    idx = lambda i, j: i * (nz + 1) + j    # noqa: E731
    i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    i, j = i.ravel(), j.ravel()
    Q = np.stack([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)], axis=1)
    return P, Q


def clip_profile(res):
    """2D x-z quad mesh of the clip profile, `res` cells across an arm.

    @return (P (n,2), Q (m,4), wide (m,) bool quad-in-base, left / right latch-corner ids)
    """
    h = ARM_T / res
    n = lambda L, scale=1.0: max(1, int(math.ceil(L / (scale * h) - 1e-9)))   # noqa: E731
    zr, zb, zt = BARB_H - BARB_LAND, ARM_LEN + BASE_H, ARM_LEN + BASE_H + STEM_H
    nr, nl, na, nb, ns, nm, npb = (n(zr), n(BARB_LAND), n(ARM_LEN - BARB_H), n(BASE_H), n(STEM_H, 2),
                                   n(2 * (ARM_OUT - ARM_T), 2), n(BARB_P))
    xi, xo = ARM_OUT - ARM_T, ARM_OUT
    blocks = []                    # (c00, c10, c11, c01, nx, nz, wide)
    for s in (1.0, -1.0):
        m = lambda x, z: (s * x, z)    # noqa: E731
        blocks += [(m(xi, 0), m(xo, 0), m(xo, zr), m(xi, zr), res, nr, False),
                   (m(xi, zr), m(xo, zr), m(xo, BARB_H), m(xi, BARB_H), res, nl, False),
                   (m(xi, BARB_H), m(xo, BARB_H), m(xo, ARM_LEN), m(xi, ARM_LEN), res, na, False),
                   (m(xo, 0), m(xo + RAMP_TIP, 0), m(xo + BARB_P, zr), m(xo, zr), npb, nr, False),
                   (m(xo, zr), m(xo + BARB_P, zr), m(xo + BARB_P - LAND_BACK, BARB_H), m(xo, BARB_H),
                    npb, nl, False),
                   (m(xi, ARM_LEN), m(xo, ARM_LEN), m(xo, zb), m(xi, zb), res, nb, True),
                   (m(xi, zb), m(xo, zb), m(xo, zt), m(xi, zt), res, ns, False)]
    blocks += [((-xi, ARM_LEN), (xi, ARM_LEN), (xi, zb), (-xi, zb), nm, nb, True),
               ((-xi, zb), (xi, zb), (xi, zt), (-xi, zt), nm, ns, False)]
    Ps, Qs, W, off = [], [], [], 0
    for c00, c10, c11, c01, nx, nz, wide in blocks:
        P, Q = _block(c00, c10, c11, c01, nx, nz)
        Ps.append(P)
        Qs.append(Q + off)
        W.append(np.full(len(Q), wide))
        off += len(P)
    P, Q = np.vstack(Ps), np.vstack(Qs)
    _, first, inv = np.unique(np.round(P * 1e9).astype(np.int64), axis=0, return_index=True,
                              return_inverse=True)
    order = np.argsort(first)                 # weld coincident points, first-appearance order
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    P, Q = P[first[order]], remap[inv.ravel()][Q]
    tip = lambda sx: np.flatnonzero(np.all(np.isclose(  # noqa: E731
        P, [sx * (xo + BARB_P - LAND_BACK), BARB_H], atol=1e-9), axis=1))
    return P, Q, np.concatenate(W), tip(-1.0), tip(1.0)


def clip_mesh(res, origin):
    """Clip hex mesh extruded along y, arm-tip centre at `origin`.

    @return (V (N,3), H (M,8), left / right latch-corner node ids)
    """
    P, Q, wide, tl, tr = clip_profile(res)
    h = ARM_T / res
    nyc = max(2, int(round(THICK / (2 * h))))                      # stem / arm layers
    nyo = max(1, int(round(0.5 * (BASE_Y - THICK) / (2 * h))))     # extra base layers per side
    ys = np.r_[np.linspace(-0.5 * BASE_Y, -0.5 * THICK, nyo + 1)[:-1],
               np.linspace(-0.5 * THICK, 0.5 * THICK, nyc + 1),
               np.linspace(0.5 * THICK, 0.5 * BASE_Y, nyo + 1)[1:]]
    n2 = len(P)
    V = np.concatenate([np.c_[P[:, 0], np.full(n2, y), P[:, 1]] for y in ys]) + np.asarray(origin, float)
    H = np.concatenate([np.c_[Q[wide | (nyo <= k < nyo + nyc)] + k * n2,
                              Q[wide | (nyo <= k < nyo + nyc)] + (k + 1) * n2]
                        for k in range(len(ys) - 1)]).astype(np.int32)
    e1, e2, e3 = (V[H[:, 1]] - V[H[:, 0]], V[H[:, 3]] - V[H[:, 0]], V[H[:, 4]] - V[H[:, 0]])
    neg = np.einsum("ij,ij->i", np.cross(e1, e2), e3) < 0         # mirrored blocks wind the other way
    H[neg] = H[neg][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    used = np.zeros(len(V), bool)                                  # drop points only narrow quads used
    used[H.ravel()] = True
    new_id = np.cumsum(used) - 1
    tl = np.concatenate([tl + k * n2 for k in range(nyo, nyo + nyc + 1)])
    tr = np.concatenate([tr + k * n2 for k in range(nyo, nyo + nyc + 1)])
    return (np.ascontiguousarray(V[used]), np.ascontiguousarray(new_id[H].astype(np.int32)),
            new_id[tl], new_id[tr])


def prism_mesh(poly, y0, y1):
    """Closed, outward-wound extrusion of the convex x-z polygon `poly` (k,2) over y in [y0, y1]."""
    P = np.asarray(poly, float)
    k = len(P)
    V = np.r_[np.c_[P[:, 0], np.full(k, y0), P[:, 1]], np.c_[P[:, 0], np.full(k, y1), P[:, 1]]]
    F = [[0, i, i + 1] for i in range(1, k - 1)] + [[k, k + i + 1, k + i] for i in range(1, k - 1)]
    for i in range(k):
        j = (i + 1) % k
        F += [[i, k + i, k + j], [i, k + j, j]]
    F = np.array(F, np.int32)
    return (V, F) if signed_volume(V, F) > 0 else (V, F[:, [0, 2, 1]].copy())


def housing_parts():
    """The housing as static meshes [(V, F)]: per side a chamfered lip on a leg."""
    out = []
    y0, y1 = SLOT_XY[1] - 0.5 * LIP_D, SLOT_XY[1] + 0.5 * LIP_D
    z0, z1, ch = TABLE_Z + LIP_Z0, TABLE_Z + LIP_Z0 + LIP_T, LIP_CHAMFER
    for s in (-1.0, 1.0):
        e, f = SLOT_XY[0] + s * 0.5 * SLOT_W, SLOT_XY[0] + s * (0.5 * SLOT_W + LIP_W)    # slot edge, outer end
        lip = [(e, z0), (f, z0), (f, z1), (e + s * ch, z1), (e, z1 - ch)]
        leg = [(f - s * LEG_W, TABLE_Z), (f, TABLE_Z), (f, z0), (f - s * LEG_W, z0)]
        for poly in (lip, leg):
            out.append(prism_mesh(poly if s > 0 else poly[::-1], y0, y1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--res", type=int, default=3, help="hex cells across an arm's thickness")
    p.add_argument("--youngs", type=float, default=CLIP_E,
                   help="clip Young's modulus (Pa); lowering it lets the arms bend further")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact", "fem")
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, fr.grasp_contact_config(
        broadphase=trusty.contact.BroadphaseKind.TwoLevel))
    # Two contact widths below the housing's base, so the housing is not in contact with the plane.
    trusty.contact.add_plane(world, np.array([0.0, 0.0, TABLE_Z - 2.0 * fr.GRASP_DHAT]),
                             np.array([0.0, 0.0, 1.0]), friction_mu=0.5)
    housing = housing_parts()
    for k, (V, F) in enumerate(housing):
        trusty.contact.add_wall(world, f"housing{k}", np.asfortranarray(V),
                                np.asfortranarray(F, dtype=np.int32), friction_mu=WALL_MU)
    franka = fr.build_franka(world)

    V, H, tip_l, tip_r = clip_mesh(args.res, (CLIP_XY[0], CLIP_XY[1], TABLE_Z + 0.5 * fr.GRASP_DHAT))
    clip = trusty.fem.add_hex_solid(
        world, trusty.make_hex_mesh(V, H),
        trusty.StableNeoHookean(youngs_modulus=args.youngs, poisson_ratio=CLIP_NU),
        density=CLIP_RHO, friction_mu=CLIP_MU)
    clip_F = trusty.fem.read_surface_triangles(world, clip)
    mass = CLIP_RHO * signed_volume(V, clip_F)
    bottom = np.flatnonzero(V[:, 2] < V[:, 2].min() + 1e-9)          # arm-tip nodes
    rest_spread = float(np.mean(V[tip_r, 0]) - np.mean(V[tip_l, 0]))

    grip_width, grip_force = fr.grasp_command_width(THICK, mass + INSERT_LOAD / 9.81, CLIP_MU, safety=2.0)
    print(f"clip: {len(V)} nodes, {len(H)} hexes (res {args.res}), {1e3 * mass:.1f} g, "
          f"E {args.youngs / 1e6:g} MPa; grip command {1e3 * grip_width:.2f} mm gives "
          f"{grip_force:.2f} N per pad")

    def positions():
        return np.asarray(trusty.fem.read_positions(world, clip))

    def tip_center(X):
        return X[bottom].mean(axis=0)

    def spread(X):
        return float(np.linalg.norm((X[tip_r].mean(axis=0) - X[tip_l].mean(axis=0))[:2]))

    R = franka.tcp[:3, :3]
    finger_below = franka.tcp[2, 3] - min(
        float(np.asarray(trusty.affine.surface(world, franka.bodies[n])[0])[:, 2].min())
        for n in fr.FINGER_LINKS)
    shoulder = ARM_LEN + BASE_H + finger_below   # arm tips to TCP, fingertips just above the base
    clip0 = np.array([CLIP_XY[0], CLIP_XY[1], TABLE_Z])
    D = DURATION

    trajectory = Trajectory(franka.tcp, 2 * franka.finger0)
    driver = Driver(world, franka, trajectory, args.dt)
    hand = driver.controller

    def tip_in_hand():
        """Arm-tip centre relative to the TCP (m), world axes: the grasp may be off-centre."""
        return tip_center(positions()) - hand.pose()[:3, 3]

    def grasp():
        at = lambda dz: make_T(R, clip0 + [0.0, 0.0, shoulder + SHOULDER_GAP + dz])   # noqa: E731
        return [Waypoint(D["descend"], at(0.0), name="descend"),
                Waypoint(D["close"], gripper=grip_width, name="close"),
                Waypoint(D["settle_grip"], name="settle_grip"),
                Waypoint(D["lift"], at(0.04), name="lift")]

    def carry():
        goal = np.array([*SLOT_XY, TABLE_Z + LIP_Z0 + LIP_T + 3e-3]) - tip_in_hand()
        return [Waypoint(D["carry"], make_T(R, goal + [0, 0, 0.03]), name="carry"),
                Waypoint(D["align"], make_T(R, goal), name="align"),
                Waypoint(D["steady"], name="steady")]

    def insert():
        press = np.array([*SLOT_XY, TABLE_Z + PRESS_TIP]) - tip_in_hand()
        at = lambda dz: make_T(R, press + [0.0, 0.0, dz])    # noqa: E731
        return [Waypoint(D["press"], at(0.0), name="press"),
                Waypoint(D["dwell"], name="dwell"),
                Waypoint(D["tug"], at(TUG), name="tug"),
                Waypoint(D["release"], gripper=fr.GRIPPER_OPEN, name="release"),
                Waypoint(D["retreat"], at(0.06), name="retreat"),
                Waypoint(D["observe"], name="observe")]

    above = clip0 + [0.0, 0.0, shoulder + SHOULDER_GAP + 0.06]
    trajectory.add([Waypoint(D["settle"], name="settle"),
                    Waypoint(D["approach"], make_T(R, above), fr.GRIPPER_OPEN, "approach"),
                    Waypoint(D["hover"], name="hover")])
    stages = [grasp, carry, insert]
    if args.steps is None:
        args.steps = int(np.ceil(sum(D.values()) / args.dt)) + 5
    seen = {"stage": "approach", "peak_deflection": 0.0, "held": None, "slip": 0.0,
            "tug_z0": None, "tug_clip": 0.0, "tug_hand": 0.0}

    def step():
        if trajectory.done and stages:
            builder = stages.pop(0)
            seen["stage"] = builder.__name__
            trajectory.add(builder())
        phase = trajectory.name
        driver.step()
        X = positions()
        seen["peak_deflection"] = max(seen["peak_deflection"], 0.5 * (rest_spread - spread(X)))
        if phase in ("lift", "carry", "align", "steady", "press"):
            T = hand.pose()
            c = T[:3, :3].T @ (tip_center(X) - T[:3, 3])
            seen["held"] = c if seen["held"] is None else seen["held"]
            seen["slip"] = max(seen["slip"], float(np.linalg.norm(c - seen["held"])))
        if phase == "tug":
            z, z_hand = tip_center(X)[2], hand.pose()[2, 3]
            seen["tug_z0"] = seen["tug_z0"] or (z, z_hand)
            seen["tug_clip"] = max(seen["tug_clip"], z - seen["tug_z0"][0])
            seen["tug_hand"] = max(seen["tug_hand"], z_hand - seen["tug_z0"][1])

    def meshes():
        return fr.meshes(world, franka) + [("clip", positions(), clip_F, (0.95, 0.55, 0.15))]

    def setup_scene(ps):
        V, F = make_box((0.4, 0.0, TABLE_Z - 0.01), (0.4, 0.4, 0.01))
        ps.register_surface_mesh("table", V, F, color=(0.55, 0.45, 0.35))
        for k, (Vb, Fb) in enumerate(housing):
            ps.register_surface_mesh(f"housing{k}", Vb, Fb, color=(0.45, 0.5, 0.6))

    common.run(args, step=step, meshes=meshes,
               gui_text=lambda: [f"{seen['stage']}: {trajectory.name}",
                                 f"arm spread {1e3 * spread(positions()):.2f} mm (rest {1e3 * rest_spread:.2f})"],
               camera=((0.88, -0.12, 0.30), (0.46, -0.01, 0.12)), setup_scene=setup_scene)

    X = positions()
    edge, lip = 0.5 * SLOT_W, TABLE_Z + LIP_Z0
    overlap = min(float(np.abs(X[ids, 0] - SLOT_XY[0]).min()) - edge for ids in (tip_l, tip_r))
    below = lip - max(float(X[ids, 2].max()) for ids in (tip_l, tip_r))
    print(f"arms bent in by {1e3 * seen['peak_deflection']:.2f} mm each (the latch overlaps the slot by "
          f"{1e3 * (ARM_OUT + BARB_P - LAND_BACK - edge):.2f} mm)")
    print(f"barbs end {1e3 * overlap:.2f} mm past the slot edges and {1e3 * below:.2f} mm below the lip; "
          f"the arms ended {100 * abs(spread(X) - rest_spread) / rest_spread:.2f}% from their rest spread")
    print(f"clip moved {1e3 * seen['slip']:.2f} mm in the grip on the way; during the tug the hand rose "
          f"{1e3 * seen['tug_hand']:.2f} mm and the clip {1e3 * seen['tug_clip']:.2f} mm")
    print(driver.summary())


if __name__ == "__main__":
    main()
