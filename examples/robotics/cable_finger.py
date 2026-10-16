"""Cable-driven soft finger.

A soft silicone finger is clamped at its base and actuated by a tendon running
through it near its underside, root to tip. Pulling the tendon shortens the
bottom fibres, so the finger curls down. The tendon is a polyline of rest-space
points the tet body interpolates (each moves with its containing tet); with
``--anchor surface`` the same points lie on the underside instead, so the tendon
is glued to the skin. Its energy acts on the path length.

Two drive modes:

- ``tension`` ramps a prescribed tension T (N): energy T * L.
- ``length`` ramps a prescribed pull-in (m): a winch shortens the cable's rest
  length to L_t, and the elastic cable (axial stiffness k, the default) pulls
  with energy 0.5 * k * max(L - L_t, 0)^2.

Static time stepping under gravity: each step is the equilibrium for the
current command.

Usage:
    python examples/robotics/cable_finger.py               # polyscope
    python examples/robotics/cable_finger.py --mode length
    python examples/robotics/cable_finger.py --anchor surface
    python examples/robotics/cable_finger.py --no-viewer   # headless PNGs
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

bc = trusty.boundary_conditions

FINGER_SIZE = (0.08, 0.02, 0.015)  # length x width x thickness (m)
FINGER_RES = (24, 4, 6)
TENDON_DEPTH = 0.25                # interior tendon height, fraction of thickness
# N, the fully-ramped tension; the surface tendon has twice the lever arm.
TENSION_MAX = {"interior": 6.0, "surface": 3.0}
PULL_MAX = 0.01                    # m, the fully-ramped pull-in (length mode)
NUM_CABLE_POINTS = 17


def build_world(backend: str = "auto", anchor: str = "interior", mode: str = "tension"):
    world = trusty.World(backend=backend, time_stepping="static")
    material = trusty.StableNeoHookean(youngs_modulus=1.8e5, poisson_ratio=0.45)
    mesh = trusty.make_beam_tet_mesh(size=FINGER_SIZE, res=FINGER_RES)
    finger = trusty.fem.add_tet_solid(world, mesh, material, density=1000.0)

    verts = np.asarray(mesh.vertices)
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    trusty.fem.pin_face(world, finger, axis=0, coord=x_min)

    # Tendon along the finger's centreline, base to tip.
    xs = np.linspace(x_min, x_max, NUM_CABLE_POINTS)
    y_mid = 0.5 * (verts[:, 1].min() + verts[:, 1].max())
    z = z_min + TENDON_DEPTH * (z_max - z_min) if anchor == "interior" else z_min
    points = np.stack([xs, np.full_like(xs, y_mid), np.full_like(xs, z)], axis=1)
    cable = bc.add_cable(world, finger, points,
                         bc.CableMode.Length if mode == "length" else bc.CableMode.Tension)
    return world, finger, cable


def _command(world, cable, mode: str, tension_max: float, rest_length: float,
             step: int, steps: int) -> str:
    s = min(1.0, (step + 1) / max(1, steps))
    if mode == "tension":
        bc.set_cable_tension(world, cable, s * tension_max)
        return f"T = {s * tension_max:.2f} N"
    bc.set_cable_length(world, cable, rest_length - s * PULL_MAX)
    return f"pull-in = {1e3 * s * PULL_MAX:.1f} mm"


def _register(ps, world, finger, cable):
    mesh = trusty.fem.read_tet_mesh(world, finger)
    ps_mesh = ps.register_volume_mesh("finger", np.asarray(mesh.vertices).copy(),
                                      tets=np.asarray(mesh.tets))
    ps.set_transparency_mode("pretty")  # the tendon runs inside
    ps_mesh.set_transparency(0.3)
    pts = np.asarray(bc.cable_points(world, cable))
    edges = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
    ps_cable = ps.register_curve_network("cable", pts, edges)
    ps_cable.set_radius(0.008, relative=True)
    ps_cable.set_color((0.85, 0.15, 0.1))
    ps.look_at((0.04, -0.2, -0.01), (0.04, 0.0, -0.01))  # side-on: the curl is in xz
    return ps_mesh, ps_cable


def _update(world, finger, cable, ps_mesh, ps_cable):
    ps_mesh.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, finger)))
    ps_cable.update_node_positions(np.asarray(bc.cable_points(world, cable)))


def run_polyscope(world, finger, cable, command, steps):
    ps = init_polyscope(headless=False)
    rest_length = bc.cable_length(world, cable)
    ps_mesh, ps_cable = _register(ps, world, finger, cable)
    state = {"i": 0, "playing": False, "label": "slack"}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["label"] = command(rest_length, state["i"])
            world.step()
            state["i"] += 1
            _update(world, finger, cable, ps_mesh, ps_cable)
        psim.Text(f"step {state['i']} / {steps}   {state['label']}   "
                  f"tension = {bc.cable_tension(world, cable):.2f} N")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, finger, cable, command, steps, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rest_length = bc.cable_length(world, cable)
    ps_mesh, ps_cable = _register(ps, world, finger, cable)
    ps.screenshot(str(out_dir / "finger_0000.png"), transparent_bg=False)
    for i in range(steps):
        command(rest_length, i)
        world.step()
        _update(world, finger, cable, ps_mesh, ps_cable)
        ps.screenshot(str(out_dir / f"finger_{i + 1:04d}.png"), transparent_bg=False)
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--mode", choices=["tension", "length"], default="tension")
    parser.add_argument("--anchor", choices=["interior", "surface"], default="interior",
                        help="tendon through the finger, or along its underside")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    world, finger, cable = build_world(args.backend, args.anchor, args.mode)
    command = functools.partial(_command, world, cable, args.mode, TENSION_MAX[args.anchor],
                                steps=args.steps)

    if args.no_viewer:
        run_screenshots(world, finger, cable, command, args.steps, args.out)
    else:
        run_polyscope(world, finger, cable, command, args.steps)


if __name__ == "__main__":
    main()
