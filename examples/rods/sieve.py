"""Rods of different thickness sorted by a sieve.

A grate of thick fixed bars leaves square holes between them, and a row of
upright rods, each thicker than the last, drops onto it. Contact treats every
rod as a tube of its own radius, so a rod slips through its hole when its
diameter is smaller than the hole and is caught on the bars when it is not.
The rods that pass land on the floor below.

The headless run prints each rod's diameter against the hole and where it
came to rest.

Usage:
    python examples/rods/sieve.py
    python examples/rods/sieve.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402

BAR_RADIUS  = 0.02                        # grate bars (m)
PITCH       = 0.1                         # bar spacing; holes are PITCH - 2 * BAR_RADIUS wide
ROD_RADII   = [0.006, 0.012, 0.02, 0.026, 0.036, 0.045]
ROD_LENGTH  = 0.3
DROP_HEIGHT = 0.1                         # rod bottoms start this far above the grate
TILT        = np.radians(4.0)             # off vertical, so caught rods topple onto the grate
FLOOR_Z     = -0.6
N_ROD, N_BAR = 16, 25
TOP_Z       = 2 * BAR_RADIUS + 0.005      # upper bar layer clears the lower one

BAR_COLOR = (0.45, 0.45, 0.50)


def hole_width() -> float:
    return PITCH - 2 * BAR_RADIUS


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods", "contact")
    world = trusty.World(backend=backend, timestep=1.0 / 120.0,
                         time_stepping="backward_euler")
    trusty.contact.enable(world, trusty.contact.Config(dhat=1e-3))
    trusty.contact.add_plane(world, np.array([0.0, 0.0, FLOOR_Z]),
                             np.array([0.0, 0.0, 1.0]), friction_mu=0.5)

    bar = trusty.rods.RodMaterial()
    bar.youngs_modulus = 1e9
    bar.radius = BAR_RADIUS
    bar.density = 1000.0

    # Two crossed layers of bars, the upper at TOP_Z; the rods drop into the
    # row of holes centered at x = (k + 1/2) PITCH, y = 0.
    n = len(ROD_RADII)
    xs = np.linspace(-PITCH, (n + 1) * PITCH, N_BAR)
    ys = np.linspace(-3 * PITCH, 3 * PITCH, N_BAR)
    bars = []
    for j in range(-3, 3):
        bars.append(np.stack([xs, np.full(N_BAR, (j + 0.5) * PITCH), np.zeros(N_BAR)], axis=1))
    for k in range(n + 1):
        bars.append(np.stack([np.full(N_BAR, k * PITCH), ys,
                              np.full(N_BAR, TOP_Z)], axis=1))
    bar_ids = []
    for X in bars:
        rod = trusty.rods.add_rod(world, X, bar)
        trusty.rods.pin_vertices(world, rod, list(range(N_BAR)))
        bar_ids.append(rod)

    rod_ids = []
    z0 = TOP_Z + BAR_RADIUS + DROP_HEIGHT
    for k, r in enumerate(ROD_RADII):
        mat = trusty.rods.RodMaterial()
        mat.youngs_modulus = 1e8
        mat.radius = r              # also the rod's thickness in contact
        mat.density = 1000.0
        X = np.zeros((N_ROD, 3))
        s = np.linspace(0.0, ROD_LENGTH, N_ROD)
        X[:, 0] = (k + 0.5) * PITCH
        X[:, 1] = np.tan(TILT) * s
        X[:, 2] = z0 + s
        rod_ids.append(trusty.rods.add_rod(world, X, mat, friction_mu=0.3))
    return world, bar_ids, rod_ids


def _rod_color(k: int):
    t = k / max(1, len(ROD_RADII) - 1)
    return (0.95 - 0.35 * t, 0.65 - 0.35 * t, 0.25 + 0.5 * t)


def run_viewer(world, bar_ids, rod_ids, steps: int):
    ps = init_polyscope(headless=False)
    nets = []
    for k, rod in enumerate(bar_ids + rod_ids):
        x = np.asarray(trusty.rods.read_positions(world, rod))
        is_bar = k < len(bar_ids)
        net = ps.register_curve_network(
            f"bar{k}" if is_bar else f"rod{k - len(bar_ids)}", x, "line",
            color=BAR_COLOR if is_bar else _rod_color(k - len(bar_ids)))
        net.set_radius(BAR_RADIUS if is_bar else ROD_RADII[k - len(bar_ids)],
                       relative=False)
        nets.append((net, rod))
    floor = np.array([[-0.3, -0.5], [0.9, -0.5], [0.9, 0.5], [-0.3, 0.5]])
    ps.register_surface_mesh("floor", np.column_stack([floor, np.full(4, FLOOR_Z)]),
                             np.array([[0, 1, 2], [0, 2, 3]]), color=(0.85, 0.85, 0.85))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for net, rod in nets[len(bar_ids):]:
                net.update_node_positions(np.asarray(trusty.rods.read_positions(world, rod)))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(world, rod_ids, steps: int):
    for _ in range(steps):
        world.step()
    print(f"after {steps} steps (holes {hole_width() * 1e3:.0f} mm wide):")
    for r, rod in zip(ROD_RADII, rod_ids):
        low = np.asarray(trusty.rods.read_positions(world, rod))[:, 2].min()
        where = "fell through" if low < FLOOR_Z + 0.1 else "caught"
        print(f"  {2 * r * 1e3:4.0f} mm rod: {where}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print which rods passed the sieve")
    args = parser.parse_args()

    world, bar_ids, rod_ids = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, rod_ids, args.steps)
    else:
        run_viewer(world, bar_ids, rod_ids, args.steps)


if __name__ == "__main__":
    main()
