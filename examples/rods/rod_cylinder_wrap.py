"""Rods wrapping a cylinder.

A row of parallel elastic rods drops onto a fixed cylinder, drapes over it, and
wraps down its sides. Contact treats each rod as a tube of its own radius:
rods stay one radius off the cylinder and two radii off each other. Friction
lets the rods grip and stay wrapped instead of sliding off.

`--num-rods` sets how many rods drop; the rod radius shrinks with the count so
neighbours never overlap along the cylinder. `--cylinder-radius` sizes the
cylinder: a smaller cylinder lets the fixed-length rods wrap further around it.
The headless run prints how close the rods come to the cylinder.

Usage:
    uv run examples/rods/rod_cylinder_wrap.py
    uv run examples/rods/rod_cylinder_wrap.py --num-rods 12
    uv run examples/rods/rod_cylinder_wrap.py --cylinder-radius 0.15
    uv run examples/rods/rod_cylinder_wrap.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


N, LENGTH  = 30, 1.0        # rod resolution; length > cylinder diameter -> wraps
CYL_R      = 0.3           # default cylinder radius (see --cylinder-radius)
Y_SPAN     = 0.6            # rods spread over y in [-Y_SPAN/2, +Y_SPAN/2]
MAX_RADIUS = 0.02
DROP_ABOVE = 0.15          # rod centerline starts this far above the cylinder top


def make_cylinder(R: float, y0: float, y1: float, na: int, nl: int):
    """Lateral surface of a cylinder of radius R along the y-axis."""
    ys = np.linspace(y0, y1, nl + 1)
    th = 2.0 * np.pi * np.arange(na) / na
    V = np.zeros(((nl + 1) * na, 3))
    for l, y in enumerate(ys):
        V[l * na:(l + 1) * na] = np.stack(
            [R * np.cos(th), np.full(na, y), R * np.sin(th)], axis=1)
    F = []
    for l in range(nl):
        for a in range(na):
            a1 = (a + 1) % na
            v00, v01 = l * na + a, l * na + a1
            v10, v11 = (l + 1) * na + a, (l + 1) * na + a1
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    return V, np.asarray(F, dtype=np.int32)


def rod_radius(num_rods: int) -> float:
    """Radius that keeps neighbouring rods clear: 2*radius stays below the
    y-spacing, capped at MAX_RADIUS so few-rod worlds keep the tuned size."""
    spacing = Y_SPAN / (num_rods - 1) if num_rods > 1 else Y_SPAN
    return min(MAX_RADIUS, 0.3 * spacing)


def build_world(backend: str = "auto", *, num_rods: int = 5, cyl_r: float = CYL_R):
    trusty.check_capabilities("rods", "contact")
    radius = rod_radius(num_rods)

    world = trusty.World(backend=backend, timestep=1.0 / 120.0, time_stepping="bdf2")
    trusty.contact.enable(world, trusty.contact.Config(
        dhat=min(0.01, 0.5 * radius),   # contact band scales with the rod
    ))
    mu = 0.4    # friction so the rods grip and stay wrapped: the cylinder's and each rod's
    Vc, Fc = make_cylinder(cyl_r, -0.6, 0.6, 48, 24)
    trusty.contact.add_wall(world, "cylinder", Vc, Fc, friction_mu=mu)

    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e6        # floppy enough to drape around the cylinder
    mat.radius = radius             # also the rod's thickness in contact
    mat.density = 1000.0

    ids = []
    for ri in range(num_rods):
        y = -0.5 * Y_SPAN + Y_SPAN * ri / (num_rods - 1) if num_rods > 1 else 0.0
        X = np.zeros((N, 3))
        X[:, 0] = np.linspace(0.0, LENGTH, N) - LENGTH / 2
        X[:, 1] = y
        X[:, 2] = cyl_r + DROP_ABOVE
        ids.append(trusty.rods.add_rod(world, X, mat, friction_mu=mu))

    return world, ids, radius, (Vc, Fc)


def _rod_edges(n: int) -> np.ndarray:
    return np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)


def _register_visuals(ps, world, ids, radius, cylinder):
    Vc, Fc = cylinder
    ps.register_surface_mesh("cylinder", Vc, Fc, color=(0.6, 0.6, 0.6))
    nets = []
    for k, rid in enumerate(ids):
        X   = np.asarray(trusty.rods.read_positions(world, rid))
        net = ps.register_curve_network(f"rod{k}", X, _rod_edges(len(X)),
                                        color=(0.85, 0.55, 0.25))
        net.set_radius(radius, relative=False)
        nets.append(net)
    return nets


def _update(nets, world, ids):
    for net, rid in zip(nets, ids):
        net.update_node_positions(
            np.asarray(trusty.rods.read_positions(world, rid)))


def run_polyscope(world, ids, radius, cylinder, steps: int):
    ps   = init_polyscope(headless=False)
    nets = _register_visuals(ps, world, ids, radius, cylinder)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update(nets, world, ids)

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance_one()
        elif state["playing"]:
            advance_one()
        psim.Text(f"step {state['i']} / {steps}")
        report = world.last_report()
        psim.Text(
            f"last solve: iters={report.iterations}  "
            f"residual={report.final_residual:.3e}  "
            f"{'converged' if report.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(world, ids, radius, cyl_r: float, steps: int):
    for i in range(steps):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: Newton did not converge")
    xs = [np.asarray(trusty.rods.read_positions(world, rid)) for rid in ids]
    gap = min(np.hypot(x[:, 0], x[:, 2]).min() for x in xs) - cyl_r - radius
    low = min(x[:, 2].min() for x in xs)
    print(f"after {steps} steps: closest rod surface {gap * 1e3:.1f} mm off the "
          f"cylinder; rod ends down to z = {low:.3f} m (cylinder axis at z = 0)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print how the rods came to rest")
    parser.add_argument("--num-rods", type=int, default=5)
    parser.add_argument("--cylinder-radius", type=float, default=CYL_R,
                        help="cylinder radius; smaller -> rods wrap further")
    args = parser.parse_args()

    world, ids, radius, cylinder = build_world(
        backend=args.backend, num_rods=args.num_rods, cyl_r=args.cylinder_radius)
    print(f"{args.num_rods} rods on a cylinder (R = {args.cylinder_radius}, "
          f"rod radius = {radius:.4f})")

    if args.no_viewer:
        run_headless(world, ids, radius, args.cylinder_radius, args.steps)
    else:
        run_polyscope(world, ids, radius, cylinder, args.steps)


if __name__ == "__main__":
    main()
