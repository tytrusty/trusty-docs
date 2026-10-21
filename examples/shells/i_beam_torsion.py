"""A thin-walled steel I-beam, clamped at one end and twisted at the other.

The beam is one shell mesh: two flanges and a web, welded along two lines
where three plates meet. The welds carry bending moment as well as force, so
web and flanges stay close to square while the beam twists, and the flanges
bend sideways in opposite directions as the clamp stops them from warping.
Each step solves for static equilibrium, so one step gives the final shape.

``--no-viewer`` prints the tip twist next to the thin-walled beam estimate
with the warping held at the clamp (Vlasov torsion), and the angle between
web and flanges at the tip.

Usage:
    python examples/shells/i_beam_torsion.py              # live polyscope
    python examples/shells/i_beam_torsion.py --no-viewer  # print the twist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


LENGTH = 2.0              # m, along x
DEPTH = 0.2               # m, between the flange mid-planes
FLANGE_WIDTH = 0.2        # m
THICKNESS = 0.004         # m, every plate
YOUNGS, POISSON, DENSITY = 2.0e11, 0.3, 7850.0
TORQUE = 800.0            # N m about the beam axis, at the free end
CELL = 0.025              # m, quad size


def make_i_beam():
    """Flanges at z = 0 and z = DEPTH and a web at y = 0, sharing the vertex
    rows along the two weld lines. Returns (V, F, flange tip faces)."""
    nx = round(LENGTH / CELL)
    V, F = [], []

    def row(y, z):
        ids = list(range(len(V), len(V) + nx + 1))
        V.extend((LENGTH * i / nx, y, z) for i in range(nx + 1))
        return ids

    def strip(root, tip, rows):
        """Quads from row `root` to row `tip`; returns the faces at x = LENGTH."""
        p0, p1 = np.array(V[root[0]]), np.array(V[tip[0]])
        rs = [root] + [row(*(p0 + k / rows * (p1 - p0))[1:]) for k in range(1, rows)] + [tip]
        end_faces = []
        for a, b in zip(rs[:-1], rs[1:]):
            for i in range(nx):
                if i == nx - 1:
                    end_faces += [len(F), len(F) + 1]
                F.append((a[i], a[i + 1], b[i + 1]))
                F.append((a[i], b[i + 1], b[i]))
        return end_faces

    half = round(FLANGE_WIDTH / 2 / CELL)
    bottom, top = row(0, 0), row(0, DEPTH)
    tips = {}
    for name, root, z in (("bottom", bottom, 0.0), ("top", top, DEPTH)):
        tips[name] = (strip(root, row(-FLANGE_WIDTH / 2, z), half)
                      + strip(root, row(FLANGE_WIDTH / 2, z), half))
    strip(bottom, top, round(DEPTH / CELL))     # the web
    return np.array(V), np.array(F, dtype=np.int32), tips


def build(backend: str = "auto"):
    trusty.check_capabilities("shells")
    world = trusty.World(backend=backend, gravity=(0.0, 0.0, 0.0),
                         time_stepping="static",   # each step solves equilibrium
                         newton=trusty.NewtonConfig(max_iters=100))
    V, F, tips = make_i_beam()

    config = trusty.shells.ShellConfig()
    config.youngs_modulus = YOUNGS     # Pa
    config.poisson_ratio = POISSON
    config.thickness = THICKNESS       # m
    config.density = DENSITY           # kg/m^3
    # One mesh: an edge shared by a flange and the web is a weld.
    beam = trusty.shells.add_shell(world, V, F, config)

    # Clamp the x = 0 end: two rows of vertices hold rotation as well as position.
    trusty.shells.pin_vertices(world, beam, np.flatnonzero(V[:, 0] < 1.5 * CELL).tolist())

    # A couple at the free end: the top flange pushed toward -y, the bottom one
    # toward +y, each over its last row of faces (area CELL * FLANGE_WIDTH).
    push = TORQUE / DEPTH / (CELL * FLANGE_WIDTH)      # Pa
    for name, sign in (("top", -1.0), ("bottom", 1.0)):
        faces = tips[name]
        trusty.boundary_conditions.attach_surface_load(
            world, beam, faces, np.tile([0.0, sign * push, 0.0], (len(faces), 1)))
    return world, beam, V, F


def tip_section(V, x):
    """Positions of the free end's web and flange vertices, sorted along each plate."""
    end = np.isclose(V[:, 0], LENGTH)
    web = np.flatnonzero(end & np.isclose(V[:, 1], 0.0))
    top = np.flatnonzero(end & np.isclose(V[:, 2], DEPTH))
    return x[web[np.argsort(V[web, 2])]], x[top[np.argsort(V[top, 1])]]


def tip_twist(V, x) -> float:
    """Rotation of the free end's web about the beam axis (degrees)."""
    web, _ = tip_section(V, x)
    d = web[-1] - web[0]
    return float(np.degrees(np.arctan2(-d[1], d[2])))


def weld_angle(V, x) -> float:
    """Angle between the web and the top flange at the free end (degrees)."""
    web, top = tip_section(V, x)
    d_web = web[-1] - web[0]
    d_flange = top[-1] - top[0]
    c = d_web @ d_flange / np.linalg.norm(d_web) / np.linalg.norm(d_flange)
    return float(np.degrees(np.arccos(c)))


def vlasov_twist() -> float:
    """Small-twist tip rotation of a cantilever I-beam under an end torque with
    the clamp holding warping: T / GJ (L - tanh(kL) / k), k^2 = GJ / (E Cw)."""
    G = YOUNGS / (2.0 * (1.0 + POISSON))
    J = (2.0 * FLANGE_WIDTH + DEPTH) * THICKNESS**3 / 3.0
    Cw = THICKNESS * FLANGE_WIDTH**3 / 12.0 * DEPTH**2 / 2.0
    k = np.sqrt(G * J / (YOUNGS * Cw))
    return float(np.degrees(TORQUE / (G * J) * (LENGTH - np.tanh(k * LENGTH) / k)))


def run_viewer(world, beam, V, F, steps: int):
    ps = init_polyscope(headless=False)
    mesh = ps.register_surface_mesh("I-beam", V, F, color=(0.62, 0.66, 0.72))
    mesh.set_edge_width(1.0)
    ps.reset_camera_to_home_view()
    state = {"i": 0}

    def callback():
        import polyscope.imgui as psim
        if psim.Button("step") and state["i"] < steps:
            state["i"] += 1
            world.step()
            mesh.update_vertex_positions(trusty.shells.read_positions(world, beam))
        x = trusty.shells.read_positions(world, beam)
        psim.Text(f"step {state['i']} / {steps}")
        psim.Text(f"tip twist {tip_twist(V, x):.2f} deg, web-flange angle {weld_angle(V, x):.2f} deg")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the twist")
    args = parser.parse_args()

    world, beam, V, F = build(args.backend)
    if args.no_viewer:
        for _ in range(args.steps):
            world.step()
        x = trusty.shells.read_positions(world, beam)
        print(f"tip twist {tip_twist(V, x):.2f} deg   (Vlasov estimate {vlasov_twist():.2f} deg)")
        print(f"web-flange angle at the tip {weld_angle(V, x):.2f} deg   (90 at rest)")
        return
    run_viewer(world, beam, V, F, args.steps)


if __name__ == "__main__":
    main()
