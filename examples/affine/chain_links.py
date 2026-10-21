"""A chain of interlocked steel links swinging from a fixed end.

Each link is an oval loop of round wire, threaded through its neighbours, and
the first link is welded in place. Released straight out to the side, the
chain falls, swings under the weld and hangs. Contact treats every link as a
tube of the wire's thickness, so the links pull on one another through
contact alone and never pass through each other. The headless run prints the
height of the free end and the closest any two neighbouring wires come.

Usage:
    python examples/affine/chain_links.py
    python examples/affine/chain_links.py --no-viewer --steps 480
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402

WIRE_RADIUS = 0.002      # m
HALF_LENGTH = 0.018      # link centerline, center to tip (m)
HALF_WIDTH = 0.008       # link centerline, center to side (m)
PITCH = 2 * HALF_LENGTH - HALF_WIDTH   # each tip sits at the centre of the next link's bend
STEEL = 7850.0
STEEL_FRICTION = 0.5     # dry steel on steel


def link_centerline(n_arc: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Closed oval centerline in the xz-plane, centred at the origin, and its edges."""
    points = []
    for cx, start in ((HALF_LENGTH - HALF_WIDTH, -np.pi / 2),
                      (HALF_WIDTH - HALF_LENGTH, np.pi / 2)):
        for t in np.linspace(start, start + np.pi, n_arc + 1):
            points.append([cx + HALF_WIDTH * np.cos(t), 0.0, HALF_WIDTH * np.sin(t)])
    n = len(points)
    return np.array(points), np.array([[i, (i + 1) % n] for i in range(n)])


def build_world(num_links: int, backend: str):
    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=backend, timestep=1.0 / 240.0)
    # Links slower than 1 mm/s against each other count as stuck.
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-4, epsv=1e-3))

    X, E = link_centerline()
    links = []
    for k in range(num_links):
        # Alternate links turn a quarter turn about the chain's axis to interlock.
        Xk = X[:, [0, 2, 1]] if k % 2 else X
        links.append(trusty.affine.add_affine_curve(
            world, Xk + [k * PITCH, 0.0, 0.0], E, WIRE_RADIUS,
            density=STEEL, stiffness=1e9, friction_mu=STEEL_FRICTION))
    trusty.affine.add_fixed_joint(world, links[0], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                  (0.0, 1.0, 0.0))
    return world, links, E


def link_points(world, link) -> np.ndarray:
    return np.asarray(trusty.affine.surface(world, link)[0])


def closest_neighbour_gap(world, links) -> float:
    """Smallest distance between the wires of two consecutive links (m)."""
    gap = np.inf
    for a, b in zip(links, links[1:]):
        Pa, Pb = link_points(world, a), link_points(world, b)
        gap = min(gap, np.linalg.norm(Pa[:, None] - Pb[None], axis=2).min() - 2 * WIRE_RADIUS)
    return gap


def run_headless(world, links, steps: int):
    print(f"{len(links)} links, pitch {PITCH * 1e3:.0f} mm; the weld is at z = 0")
    print(f"{'step':>5} {'free end z (m)':>15} {'closest wires (mm)':>19} {'iters':>6}")
    closest = np.inf
    for i in range(1, steps + 1):
        world.step()
        gap = closest_neighbour_gap(world, links)
        closest = min(closest, gap)
        if i % 40 == 0:
            z = link_points(world, links[-1])[:, 2].mean()
            print(f"{i:5d} {z:15.3f} {gap * 1e3:19.2f} {world.last_report().iterations:6d}")
    print(f"closest the wires of any two neighbours came: {closest * 1e3:.3f} mm")


def run_viewer(world, links, E, steps: int):
    ps = init_polyscope(headless=False)
    nets = []
    for k, link in enumerate(links):
        net = ps.register_curve_network(f"link{k}", link_points(world, link), E,
                                        color=(0.55, 0.58, 0.64) if k % 2 else (0.72, 0.62, 0.45))
        net.set_radius(WIRE_RADIUS, relative=False)
        nets.append((net, link))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for net, link in nets:
                net.update_node_positions(link_points(world, link))
        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"last solve: iters={r.iterations}  residual={r.final_residual:.2e}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--links", type=int, default=12, help="number of links in the chain")
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--no-viewer", action="store_true",
                   help="headless: print the free end's height and the closest wire gap")
    args = p.parse_args()

    world, links, E = build_world(args.links, args.backend)
    if args.no_viewer:
        run_headless(world, links, args.steps)
    else:
        run_viewer(world, links, E, args.steps)


if __name__ == "__main__":
    main()
