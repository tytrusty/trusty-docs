"""A closed spherical shell, pinned around its equator, inflated by internal pressure.

A rubber ball's skin is held along its equator and pressurized from inside.
The two hemispheres bulge out like a pair of balloons, and the pinned ring
stays put as a sharp waist between them. The sphere is curved at rest, so
before any pressure is applied it carries no bending stress and stays exactly
where it is.

``--bending`` picks the bending model: ``discrete_shell`` (the default) is
cheaper, with only vertex positions as unknowns; ``bac`` is more accurate but
adds one unknown per edge. ``--no-viewer`` runs headless and prints the step
time, Newton iterations and enclosed volume as the pressure ramps up.

Usage:
    python examples/shells/inflated_sphere.py
    python examples/shells/inflated_sphere.py --bending bac
    python examples/shells/inflated_sphere.py --no-viewer --steps 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


RADIUS = 0.1          # m
SUBDIV = 4            # icosphere subdivisions: 2562 vertices
PRESSURE = 1.2e4      # Pa, reached at the end of the ramp
HOLD_STEPS = 5        # steps at zero pressure before the ramp starts

BENDING = {"discrete_shell": trusty.shells.BendingModel.DiscreteShell,
           "bac": trusty.shells.BendingModel.Bac}


def icosphere(radius: float, subdiv: int):
    """A closed icosphere surface, outward-wound, centred at the origin."""
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
    return V / np.linalg.norm(V, axis=1, keepdims=True) * radius, F


def equator_ring(V, F):
    """Vertices separating the two hemispheres: every vertex on z = 0, plus the
    nearer end of each edge that crosses it, so no free edge spans the equator."""
    z = V[:, 2]
    tol = 1e-9 * np.abs(z).max()
    ring = set(np.flatnonzero(np.abs(z) < tol).tolist())
    edges = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                              axis=1), axis=0)
    for a, b in edges:
        if z[a] * z[b] < 0 and a not in ring and b not in ring:
            ring.add(int(a) if abs(z[a]) < abs(z[b]) else int(b))
    return sorted(ring)


def build_world(backend: str = "auto", bending: str = "discrete_shell", subdiv: int = SUBDIV):
    trusty.check_capabilities("shells")
    V, F = icosphere(RADIUS, subdiv)
    # Quasi-static: each step relaxes to equilibrium at that step's pressure. No
    # gravity, so with no pressure the ball has nothing to respond to.
    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         gravity=(0.0, 0.0, 0.0),
                         time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=100, tolerance=1e-4))
    config = trusty.shells.ShellConfig()
    config.youngs_modulus = 1.0e6       # Pa, a soft rubber
    config.poisson_ratio = 0.4
    config.thickness = 2.0e-3           # m
    config.density = 1.1e3              # kg/m^3
    config.bending_model = BENDING[bending]
    body = trusty.shells.add_shell(world, V, F, config)

    ring = equator_ring(V, F)
    trusty.shells.pin_vertices(world, body, ring)

    # Every face of the closed shell bounds the pressurized volume.
    cavity = trusty.boundary_conditions.add_pressure_cavity(world, body, list(range(len(F))))
    return world, body, cavity, ring


def pressure_at(step: int, steps: int, p_max: float = PRESSURE) -> float:
    """Gauge pressure for step `step`: zero for the hold, then a linear ramp.

    The ramp is gentle on purpose: a large jump in pressure in one step can
    stall the solve.
    """
    ramp = max(1, steps - HOLD_STEPS)
    return p_max * min(1.0, max(0, step - HOLD_STEPS + 1) / ramp)


def advance(world, cavity, step: int, steps: int):
    trusty.boundary_conditions.set_cavity_pressure(world, cavity, pressure_at(step, steps))
    world.step()


def run_headless(world, body, cavity, ring, steps: int, bending: str):
    V0 = np.asarray(trusty.shells.read_positions(world, body)).copy()
    num_edges = 3 * len(np.asarray(trusty.shells.read_triangles(world, body))) // 2
    unknowns = 3 * len(V0) + (num_edges if bending == "bac" else 0)
    print(f"{bending}: {len(V0)} vertices, {unknowns} unknowns, "
          f"{len(ring)} vertices pinned around the equator")
    print(" step  pressure (kPa)  newton  step time (ms)  volume (L)")
    times, iters = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        advance(world, cavity, i, steps)
        times.append(time.perf_counter() - t0)
        report = world.last_report()
        iters.append(report.iterations)
        if i < HOLD_STEPS or (i + 1) % 5 == 0 or i + 1 == steps:
            volume = trusty.boundary_conditions.cavity_volume(world, cavity)
            print(f"{i:5d}  {pressure_at(i, steps) / 1e3:14.2f}  {report.iterations:6d}  "
                  f"{times[-1] * 1e3:14.1f}  {volume * 1e3:10.4f}")
    x = np.asarray(trusty.shells.read_positions(world, body))
    r = np.linalg.norm(x, axis=1)
    print(f"mean step {np.mean(times) * 1e3:.1f} ms, {np.mean(iters):.1f} Newton iterations "
          f"per step ({sum(iters)} total)")
    print(f"the hemispheres bulge to {r.max() / RADIUS:.2f}x the rest radius; the pinned ring "
          f"moved {np.abs(x[ring] - V0[ring]).max():.1e} m")


def run_polyscope(world, body, cavity, steps: int):
    ps = init_polyscope(headless=False)
    F = np.asarray(trusty.shells.read_triangles(world, body))
    mesh = ps.register_surface_mesh(
        "shell", np.asarray(trusty.shells.read_positions(world, body)), F, smooth_shade=True)
    mesh.set_edge_width(0.5)
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            advance(world, cavity, state["i"], steps)
            state["i"] += 1
            mesh.update_vertex_positions(np.asarray(trusty.shells.read_positions(world, body)))
        if state["i"]:
            volume = trusty.boundary_conditions.cavity_volume(world, cavity)
            psim.Text(f"step {state['i']} / {steps}   "
                      f"p = {pressure_at(state['i'] - 1, steps) / 1e3:.2f} kPa   "
                      f"V = {volume * 1e3:.3f} L")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--bending", choices=sorted(BENDING), default="discrete_shell",
                        help="bending model of the shell")
    parser.add_argument("--subdiv", type=int, default=SUBDIV,
                        help="icosphere subdivisions; each one quadruples the triangles")
    parser.add_argument("--steps", type=int, default=45,
                        help="steps, including the first few at zero pressure")
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print step time, iterations and volume")
    args = parser.parse_args()

    world, body, cavity, ring = build_world(args.backend, args.bending, args.subdiv)
    if args.no_viewer:
        run_headless(world, body, cavity, ring, args.steps, args.bending)
    else:
        run_polyscope(world, body, cavity, args.steps)


if __name__ == "__main__":
    main()
