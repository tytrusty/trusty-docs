"""Fin Ray finger wrapping around a sphere pushed into it.

A Fin Ray finger is a soft triangular frame: a front and a back flank lean in
to meet at the tip, joined by thin ribs that slope up toward the back. Pressed
at its middle, the front flank pushes the ribs back, they swing about their
ends, and the frame shears so the tip curls toward the load: the finger wraps
around what presses it, where a solid finger would bend away. Here the finger
stands with its base clamped, and a near-rigid sphere slides sideways into the
front flank and back out.

Quasi-static stepping: each step is the finger in equilibrium with the
sphere's current push. The sphere rides a rail, driven toward its target by an
actuator that gives way as the finger pushes back.

Usage:
    python examples/robotics/finray_finger.py               # polyscope
    python examples/robotics/finray_finger.py --no-viewer   # headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

LENGTH = 0.070             # m, base to tip
WIDTH = 0.030              # m, front flank to back flank at the base
DEPTH = 0.018              # m
WALL = 0.002               # m, flank thickness
RIB = 0.0012               # m, rib thickness
BASE = 0.003               # m, solid plate at the clamped base
NUM_RIBS = 5
RIB_ANGLE = 25.0           # deg above horizontal, rising from the front flank
CELLS_PER_RIB = 2          # elements across a rib
LAYERS = 3                 # element layers through the depth

SPHERE_RADIUS = 0.010      # m
GAP = 0.0005               # m, sphere to front flank at the start
PUSH = 0.010               # m, the actuator's target travel
STEPS_PER_CYCLE = 40


def _segment_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _inward_normal(a, b, c):
    """Unit normal of line a-b pointing to the side of c."""
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    return n if (c - a) @ n > 0 else -n


def _clip(poly, p, n):
    """The part of convex polygon `poly` (k, 2) with (x - p) . n >= 0."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = (a - p) @ n, (b - p) @ n
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + (b - a) * (da / (da - db)))
    return np.array(out).reshape(-1, 2)


def _signed_area(poly):
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(x @ np.roll(z, -1) - z @ np.roll(x, -1))


def _corners():
    """Front base corner, back base corner and tip of the profile, in (x, z)."""
    return np.array([0.0, 0.0]), np.array([WIDTH, 0.0]), np.array([0.5 * WIDTH, LENGTH])


def front_flank_x(z):
    A, _, T = _corners()
    return A[0] + (T[0] - A[0]) * z / LENGTH


def profile_polygon():
    """Outer triangle (3, 2) CCW and the cells between ribs as CW convex holes."""
    A, B, T = _corners()
    outer = np.array([A, B, T])
    nf, nb = _inward_normal(A, T, B), _inward_normal(B, T, A)
    inner = _clip(_clip(outer, A + WALL * nf, nf), B + WALL * nb, nb)
    inner = _clip(inner, np.array([0.0, BASE]), np.array([0.0, 1.0]))
    a = math.radians(RIB_ANGLE)
    d = np.array([math.cos(a), math.sin(a)])
    up = np.array([-d[1], d[0]])
    anchors = [A + zf * (T - A) for zf in np.linspace(BASE / LENGTH + 0.12, 0.80, NUM_RIBS)]
    holes = []
    for k in range(NUM_RIBS + 1):
        c = inner
        if k > 0:
            c = _clip(c, anchors[k - 1] + 0.5 * RIB * up, up)
        if k < NUM_RIBS:
            c = _clip(c, anchors[k] - 0.5 * RIB * up, -up)
        if len(c) >= 3 and _signed_area(c) > 0.0:
            holes.append(c[::-1].copy())
    return outer, holes


def _in_convex(P, poly):
    """P (n, 2) strictly inside the convex polygon `poly` (either winding)."""
    s = np.sign(_signed_area(poly))
    ok = np.ones(len(P), bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        ok &= s * (e[0] * (P[:, 1] - a[1]) - e[1] * (P[:, 0] - a[0])) > 0.0
    return ok


def _in_profile(P, outer, holes):
    m = _in_convex(P, outer)
    for H in holes:
        m &= ~_in_convex(P, H)
    return m


def _loop_distance(P, loops):
    d = np.full(len(P), np.inf)
    for L in loops:
        for a, b in zip(L, np.roll(L, -1, axis=0)):
            d = np.minimum(d, _segment_dist(P, a, b))
    return d


def _sample_loop(loop, h):
    """Points on closed `loop` spaced <= h, consecutive segment pairs, and a corner mask."""
    pts, corner = [], []
    for a, b in zip(loop, np.roll(loop, -1, axis=0)):
        n = max(1, math.ceil(np.linalg.norm(b - a) / h - 1e-9))
        pts.append(a + (np.arange(n)[:, None] / n) * (b - a))
        corner.append(np.arange(n) == 0)
    P = np.concatenate(pts)
    i = np.arange(len(P))
    return P, np.stack([i, np.roll(i, -1)], axis=1), np.concatenate(corner)


def _edges(T):
    return np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)


def triangulate_profile(smooth: int = 10):
    """Delaunay triangulation of the profile that keeps its boundary edges.

    Corners are numbered first so `extrude_tets` splits the prisms at the tip
    through the tip corner itself.
    @return (V (n, 2) in (x, z), T (m, 3) int, CCW)
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise SystemExit("meshing the finger requires scipy: `pip install scipy`") from e

    h = min(WALL, RIB) / CELLS_PER_RIB
    outer, holes = profile_polygon()
    loops = [outer] + holes
    Pb, Sb, corner = [], [], []
    for L in loops:
        P, S, c = _sample_loop(L, h)
        Sb.append(S + sum(len(p) for p in Pb))
        Pb.append(P)
        corner.append(c)
    Pb, Sb = np.concatenate(Pb), np.sort(np.concatenate(Sb), axis=1)
    lo, hi = outer.min(0), outer.max(0)
    rows = []
    for j, z in enumerate(np.arange(lo[1], hi[1] + h, 0.5 * math.sqrt(3.0) * h)):
        x = np.arange(lo[0] + 0.5 * h * (j % 2), hi[0] + h, h)
        rows.append(np.stack([x, np.full_like(x, z)], axis=1))
    Pi = np.concatenate(rows)
    Pi = Pi[_in_profile(Pi, outer, holes)]
    Pi = Pi[_loop_distance(Pi, loops) > 0.6 * h]

    # Smooth the interior points a few times, keeping them clear of the boundary.
    nb = len(Pb)
    for it in range(smooth + 1):
        P = np.concatenate([Pb, Pi])
        T = Delaunay(P).simplices
        T = T[_in_profile(P[T].mean(1), outer, holes)]
        if it == smooth:
            break
        E = np.unique(_edges(T), axis=0)
        acc, cnt = np.zeros_like(P), np.zeros(len(P))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(acc, E[:, a], P[E[:, b]])
            np.add.at(cnt, E[:, a], 1.0)
        moved = acc[nb:] / np.maximum(cnt[nb:], 1.0)[:, None]
        ok = ((cnt[nb:] > 0) & _in_profile(moved, outer, holes)
              & (_loop_distance(moved, loops) > 0.5 * h))
        Pi = np.where(ok[:, None], moved, Pi)

    E, cnt = np.unique(_edges(T), axis=0, return_counts=True)
    if cnt.max() != 2 or {tuple(e) for e in E[cnt == 1]} != {tuple(s) for s in Sb}:
        raise RuntimeError("the triangulation does not follow the finger's outline")
    a = P[T]
    cross = ((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
             - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0]))
    T = np.where((cross < 0.0)[:, None], T[:, [0, 2, 1]], T)
    corner = np.r_[np.concatenate(corner), np.zeros(len(Pi), bool)]
    order = np.r_[np.nonzero(corner)[0], np.nonzero(~corner)[0]]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return P[order], rank[T]


def extrude_tets(V2, T2):
    """Extrude the profile through the depth (y) into prisms of 3 tets each, split by
    vertex index so neighbouring prisms share their side faces."""
    n = len(V2)
    ys = np.linspace(0.0, DEPTH, LAYERS + 1)
    V = np.concatenate([np.c_[V2[:, 0], np.full(n, y), V2[:, 1]] for y in ys])
    a, b, c = np.sort(T2, axis=1).T
    tets = []
    for k in range(LAYERS):
        lo, up = k * n, (k + 1) * n
        tets += [np.stack([a + lo, b + lo, c + lo, c + up], 1),
                 np.stack([a + lo, b + lo, b + up, c + up], 1),
                 np.stack([a + lo, a + up, b + up, c + up], 1)]
    tets = np.concatenate(tets)
    p = [V[tets[:, i]] for i in range(4)]
    volume = np.einsum("ij,ij->i", p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0]))
    tets = np.where((volume < 0.0)[:, None], tets[:, [1, 0, 2, 3]], tets)
    return V, tets.astype(np.int32)


def icosphere(radius: float, center, subdiv: int = 2):
    """A closed icosphere surface (outward-wound)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = np.array([
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ], dtype=np.float64)
    F = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int32)
    for _ in range(subdiv):
        mid: dict[tuple[int, int], int] = {}
        verts = list(V)
        new_F = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((V[a] + V[b]) / 2.0)
            return mid[key]

        for a, b, c in F:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_F += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V = np.array(verts, dtype=np.float64)
        F = np.array(new_F, dtype=np.int32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    return V + np.asarray(center, dtype=np.float64), F


def build_world(backend: str = "auto"):
    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=backend, time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=100))
    # A soft TPU.
    material = trusty.StableNeoHookean(youngs_modulus=30e6, poisson_ratio=0.45)
    V, tets = extrude_tets(*triangulate_profile())
    finger = trusty.fem.add_tet_solid(world, trusty.make_tet_mesh(V, tets), material,
                                      density=1200.0, friction_mu=0.5)
    trusty.fem.pin_face(world, finger, axis=2, coord=0.0)

    # The sphere slides along x on a rail, driven by an actuator.
    height = 0.5 * LENGTH
    center = (front_flank_x(height) - GAP - SPHERE_RADIUS, 0.5 * DEPTH, height)
    Vs, Fs = icosphere(SPHERE_RADIUS, center)
    sphere = trusty.affine.add_affine_body(world, Vs, Fs, density=1200.0, stiffness=1e8,
                                           friction_mu=0.5)
    rail = trusty.affine.add_prismatic_joint(world, sphere, center, (1.0, 0.0, 0.0),
                                             stiffness=1e8)
    actuator = trusty.affine.add_actuator(world, rail, stiffness=1e5)

    trusty.contact.set_self_contact(world, finger, False)
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-4))
    return world, finger, sphere, actuator


def push(step: int) -> float:
    return PUSH * 0.5 * (1.0 - np.cos(2.0 * np.pi * step / STEPS_PER_CYCLE))


def advance(world, actuator, step):
    trusty.affine.set_actuator_target(world, actuator, push(step + 1))
    world.step()


def tip_toward_sphere(world, finger, rest):
    """How far the tip has moved toward the sphere, against the push (m)."""
    x = np.asarray(trusty.fem.read_positions(world, finger))
    tip = rest[:, 2] > rest[:, 2].max() - 1e-9
    return float(rest[tip, 0].mean() - x[tip, 0].mean())


def run_headless(world, finger, actuator, steps):
    rest = np.asarray(trusty.fem.read_positions(world, finger)).copy()
    for i in range(steps):
        advance(world, actuator, i)
        print(f"step {i:3d}  newton {world.last_report().iterations:3d}  "
              f"sphere {1e3 * trusty.affine.actuator_value(world, actuator):5.2f} mm  "
              f"tip toward sphere {1e3 * tip_toward_sphere(world, finger, rest):+5.2f} mm")


def run_polyscope(world, finger, sphere, actuator, steps):
    ps = init_polyscope(headless=False)
    ps_finger = ps.register_surface_mesh(
        "finger", np.asarray(trusty.fem.read_positions(world, finger)).copy(),
        np.asarray(trusty.fem.read_surface_triangles(world, finger)), edge_width=0.5)
    V, F = trusty.affine.surface(world, sphere)
    ps_sphere = ps.register_surface_mesh("sphere", np.asarray(V), np.asarray(F),
                                         color=(0.85, 0.35, 0.2))
    ps.look_at((0.0, -0.13, 0.035), (0.005, 0.0, 0.035))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            advance(world, actuator, state["i"])
            state["i"] += 1
            ps_finger.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, finger)))
            ps_sphere.update_vertex_positions(np.asarray(trusty.affine.surface(world, sphere)[0]))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=STEPS_PER_CYCLE)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    world, finger, sphere, actuator = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, finger, actuator, args.steps)
    else:
        run_polyscope(world, finger, sphere, actuator, args.steps)


if __name__ == "__main__":
    main()
