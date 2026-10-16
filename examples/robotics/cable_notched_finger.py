"""Notched cable-driven finger: a tendon closes trapezoidal notches.

A soft finger is a solid spine with a row of blocks on one side, separated by
trapezoidal notches that widen toward the open face. A tendon runs through
the blocks near that face, crossing each notch in the open, from a winch at
the clamped root to the tip. The finger hangs from its root. Winding the tendon in pulls the blocks
together: the notches close and the finger curls toward them, while the
spine, which the tendon does not cross, bends without stretching. The
finger collides with itself, so once a notch's walls meet they push back:
past that point the tension climbs steeply for little more curl.

The winch winds in and lets out on a cycle. Static time stepping: each step
is the equilibrium for the current pull-in.

Usage:
    python examples/robotics/cable_notched_finger.py               # polyscope
    python examples/robotics/cable_notched_finger.py --no-viewer   # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

bc = trusty.boundary_conditions

NUM_BLOCKS = 7
BLOCK = 0.012              # m, block pitch along the finger
WIDTH = 0.015              # m
SPINE = 0.005              # m, spine thickness
HEIGHT = 0.016             # m, spine + blocks
NOTCH_INNER = 0.001        # m, notch width at its bottom, by the spine
NOTCH_OUTER = 0.005        # m, notch width at the open face
CABLE_DEPTH = 0.002        # m, tendon below the open face
CELLS = (4, 2, 3, 6)       # cells per block, per notch, across, and through the height
SPINE_CELLS = 2
PULL_MAX = 0.032           # m, the full pull-in
STEPS_PER_CYCLE = 40


def notch_width(h):
    """Notch width at height `h` (m) above the spine's open-side face."""
    return NOTCH_INNER + (NOTCH_OUTER - NOTCH_INNER) * np.clip(h / (HEIGHT - SPINE), 0.0, 1.0)


def hanging(p):
    """Finger frame (x along it, y across, z through it) to the world, hanging down: z up."""
    return np.stack([p[:, 2], p[:, 1], -p[:, 0]], axis=1)


def finger_mesh():
    """Hex mesh in the finger frame; the notches open at z = 0."""
    n_block, n_notch, ny, nz = CELLS
    # Each node column belongs to a block or to a notch. A notch column's x
    # depends on height, so the notch walls slant.
    columns = []  # (kind, index, fraction across it)
    for b in range(NUM_BLOCKS):
        for i in range(n_block + 1):
            columns.append(("block", b, i / n_block))
        if b + 1 < NUM_BLOCKS:
            for i in range(1, n_notch):
                columns.append(("notch", b, i / n_notch))
    length = NUM_BLOCKS * BLOCK

    def x_at(kind, b, s, h):
        g = notch_width(h)
        if kind == "notch":
            return (b + 1) * BLOCK - 0.5 * g + s * g
        lo = 0.0 if b == 0 else b * BLOCK + 0.5 * g
        hi = length if b == NUM_BLOCKS - 1 else (b + 1) * BLOCK - 0.5 * g
        return lo + s * (hi - lo)

    zs = np.concatenate([np.linspace(0.0, HEIGHT - SPINE, nz + 1),
                         np.linspace(HEIGHT - SPINE, HEIGHT, SPINE_CELLS + 1)[1:]])
    ys = np.linspace(0.0, WIDTH, ny + 1)
    nx = len(columns)
    verts = np.array([[x_at(*columns[i], HEIGHT - SPINE - z), y, z]
                      for i in range(nx) for y in ys for z in zs])
    node = np.arange(len(verts)).reshape(nx, ny + 1, len(zs))

    in_notch = [left[:2] != right[:2] or left[0] == "notch"
                for left, right in zip(columns[:-1], columns[1:])]
    solid = np.ones((nx - 1, ny, len(zs) - 1), dtype=bool)
    solid[in_notch, :, :nz] = False
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    i, j, k = np.nonzero(solid)
    hexes = np.stack([node[i + a, j + b, k + c] for a, b, c in corners], axis=1)
    used, hexes = np.unique(hexes, return_inverse=True)
    return verts[used], hexes.reshape(-1, 8).astype(np.int32)


def tendon_points():
    """Root, then each block's two notch-side walls, then the tip; just inside the blocks."""
    h = HEIGHT - SPINE - CABLE_DEPTH
    inset = 1e-4
    xs = [0.0]
    for b in range(NUM_BLOCKS - 1):
        wall = (b + 1) * BLOCK - 0.5 * notch_width(h)
        xs += [wall - inset, wall + notch_width(h) + inset]
    xs.append(NUM_BLOCKS * BLOCK - inset)
    return np.array([[x, 0.5 * WIDTH, CABLE_DEPTH] for x in xs])


def build_world(backend: str = "auto"):
    trusty.check_capabilities("contact")
    world = trusty.World(backend=backend, time_stepping="static",
                         newton=trusty.NewtonConfig(max_iters=100))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.45)
    verts, hexes = finger_mesh()
    finger = trusty.fem.add_hex_solid(world, trusty.make_hex_mesh(hanging(verts), hexes),
                                      material, density=1100.0)
    trusty.fem.pin_face(world, finger, axis=2, coord=0.0)
    cable = bc.add_cable(world, finger, hanging(tendon_points()), bc.CableMode.Length)
    # Self-contact: the closing notch walls press on each other instead of passing through.
    trusty.contact.set_self_contact(world, finger, True)
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-4))
    return world, finger, cable


def pull_in(step: int) -> float:
    return PULL_MAX * 0.5 * (1.0 - np.cos(2.0 * np.pi * step / STEPS_PER_CYCLE))


def advance(world, cable, rest_length, step):
    bc.set_cable_length(world, cable, rest_length - pull_in(step + 1))
    world.step()


def run_headless(world, finger, cable, steps):
    rest_length = bc.cable_length(world, cable)
    rest = np.asarray(trusty.fem.read_positions(world, finger)).copy()
    tip = rest[:, 2] < rest[:, 2].min() + 1e-9
    for i in range(steps):
        advance(world, cable, rest_length, i)
        x = np.asarray(trusty.fem.read_positions(world, finger))
        print(f"step {i:3d}  newton {world.last_report().iterations:3d}  "
              f"pull-in {1e3 * pull_in(i + 1):5.2f} mm  "
              f"tension {bc.cable_tension(world, cable):6.2f} N  "
              f"tip moved {1e3 * np.linalg.norm((x[tip] - rest[tip]).mean(axis=0)):6.2f} mm")


def run_polyscope(world, finger, cable, steps):
    ps = init_polyscope(headless=False)
    rest_length = bc.cable_length(world, cable)
    mesh = trusty.fem.read_mesh(world, finger)
    ps_mesh = ps.register_volume_mesh("finger", np.asarray(mesh.vertices).copy(),
                                      hexes=np.asarray(mesh.hexes))
    ps_mesh.set_edge_width(1.0)
    pts = np.asarray(bc.cable_points(world, cable))
    edges = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
    ps_cable = ps.register_curve_network("tendon", pts, edges)
    ps_cable.set_radius(0.004, relative=True)
    ps_cable.set_color((0.85, 0.15, 0.1))
    ps.look_at((0.0, -0.22, -0.04), (0.0, 0.0, -0.04))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            advance(world, cable, rest_length, state["i"])
            state["i"] += 1
            ps_mesh.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, finger)))
            ps_cable.update_node_positions(np.asarray(bc.cable_points(world, cable)))
        psim.Text(f"step {state['i']} / {steps}   pull-in {1e3 * pull_in(state['i']):.2f} mm   "
                  f"tension {bc.cable_tension(world, cable):.2f} N")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=2 * STEPS_PER_CYCLE)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    world, finger, cable = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, finger, cable, args.steps)
    else:
        run_polyscope(world, finger, cable, args.steps)


if __name__ == "__main__":
    main()
