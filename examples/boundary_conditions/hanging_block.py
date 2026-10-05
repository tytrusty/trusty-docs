"""A block hung from a soft pin: pin stiffness and the pin's reaction force.

A 16 kg block hangs from its top face, which is held by a spring pin
(`pin_to_target`) at its rest position. A spring pin gives a little under load:
the pinned face settles below its targets by about weight / (k * V), where k is
the pin stiffness (Pa) and V the volume the pinned nodes carry. Each step
solves for the hanging equilibrium, so the example raises the stiffness with
`set_pin_stiffness` and re-solves: the gap shrinks tenfold each time, while the
pin's reaction force (`pin_total_force`) stays equal to the block's weight.

Usage:
    python examples/boundary_conditions/hanging_block.py              # live polyscope
    python examples/boundary_conditions/hanging_block.py --no-viewer  # print the table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

SIZE = (0.2, 0.2, 0.4)         # m; the block hangs from its top face at z = 0.4
RES = (4, 4, 8)
DENSITY = 1000.0               # kg/m^3, so the block weighs 16 kg
STIFFNESSES = (1e6, 1e7, 1e8)  # Pa; the pin is stiffened through these


def build_world(backend: str = "auto"):
    world = trusty.World(backend=backend,
                         time_stepping="static")  # every step solves for the equilibrium

    mesh = trusty.make_beam_hex_mesh(size=SIZE, res=RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.4)
    body = trusty.fem.add_hex_solid(world, mesh, material, density=DENSITY)

    rest = np.asarray(mesh.vertices)
    top = np.flatnonzero(rest[:, 2] > SIZE[2] - 1e-9)  # the top face's nodes
    pin = trusty.boundary_conditions.pin_to_target(
        world, body, top.tolist(), rest[top], stiffness=STIFFNESSES[0])
    return world, body, pin, mesh, top


def hang(world, pin, stiffness: float):
    """Set the pin's stiffness and solve for the new equilibrium."""
    trusty.boundary_conditions.set_pin_stiffness(world, pin, stiffness)
    world.step()


def measure(world, body, pin, top):
    """The pin's gap below its targets (m), and its reaction force (N)."""
    x = np.asarray(trusty.fem.read_positions(world, body))
    targets = trusty.boundary_conditions.pin_targets(world, pin)
    gap = float(np.mean(targets[:, 2] - x[top, 2]))
    forces = trusty.boundary_conditions.pin_forces(world, pin)  # one row per pinned node
    total = trusty.boundary_conditions.pin_total_force(world, pin)  # their sum (fx, fy, fz)
    return gap, forces, np.asarray(total)


def weight():
    return DENSITY * float(np.prod(SIZE)) * 9.81


def run_headless(world, body, pin, top):
    print(f"block weight: {weight():.2f} N")
    print(f"{'stiffness (Pa)':>15}  {'gap (m)':>9}  {'reaction fz (N)':>15}")
    for k in STIFFNESSES:
        hang(world, pin, k)
        if not world.last_report().converged:
            raise SystemExit(f"k={k:.0e}: the solve did not converge")
        gap, _, total = measure(world, body, pin, top)
        print(f"{k:15.0e}  {gap:9.2e}  {total[2]:15.2f}")
        if abs(-total[2] - weight()) > 1e-6 * weight():
            raise SystemExit("the reaction force does not balance the weight")
    print("OK: the reaction balances the weight; a stiffer pin sags less.")


def run_polyscope(world, body, pin, mesh, top):
    ps = init_polyscope(headless=False)
    m = ps.register_volume_mesh("block", trusty.fem.read_positions(world, body),
                                hexes=np.asarray(mesh.hexes))
    m.set_edge_width(1.0)
    ps.register_point_cloud("targets", trusty.boundary_conditions.pin_targets(world, pin),
                            radius=0.006)
    state = {"log_k": np.log10(STIFFNESSES[0])}

    def refresh():
        hang(world, pin, 10.0 ** state["log_k"])
        x = np.asarray(trusty.fem.read_positions(world, body))
        m.update_vertex_positions(x)
        _, forces, _ = measure(world, body, pin, top)
        nodes = ps.register_point_cloud("pinned nodes", x[top], radius=0.004)
        nodes.add_vector_quantity("pin force", forces, enabled=True)

    refresh()

    def callback():
        import polyscope.imgui as psim
        changed, state["log_k"] = psim.SliderFloat("log10 stiffness (Pa)", state["log_k"],
                                                   5.5, 9.0)
        if changed:
            refresh()
        gap, _, total = measure(world, body, pin, top)
        psim.Text(f"gap below targets {gap:.2e} m")
        psim.Text(f"reaction fz {total[2]:.2f} N   (weight {weight():.2f} N)")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: step through the stiffnesses and print the table")
    args = ap.parse_args()

    world, body, pin, mesh, top = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, body, pin, top)
    else:
        run_polyscope(world, body, pin, mesh, top)


if __name__ == "__main__":
    main()
