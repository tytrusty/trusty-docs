"""Cable-steered soft tentacle (four tendons, length-driven).

A soft square tentacle stands on a clamped base, with four tendons threaded
through its interior, one near each side, root to tip. A winch on each tendon
shortens its rest length in turn: the side it pulls on shortens, so the tentacle
bends toward it, and pulling the four in sequence sweeps the tip around a
circle. Each tendon is an elastic cable of the default axial stiffness, so it
stretches a little under load.

The tentacle is a quadratic (Q2) hex body; the tendon points sit wherever the
tendon runs, not on mesh nodes, and the body interpolates them. No gravity: the
tentacle is neutrally buoyant, as in water.

Usage:
    python examples/robotics/cable_tentacle.py               # polyscope
    python examples/robotics/cable_tentacle.py --mode tension
    python examples/robotics/cable_tentacle.py --no-viewer   # headless PNGs
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

WIDTH = 0.03                # square cross-section side (m)
LENGTH = 0.2                # tentacle length along z (m)
RES = (2, 2, 10)            # Q2 hex resolution
TENDON_OFFSET = 0.01        # tendon distance from the axis (m)
NUM_CABLE_POINTS = 11
PULL_MAX = 0.012            # m, the fully-wound pull-in (length mode)
TENSION_MAX = 4.0           # N, the full tension (tension mode)
STEPS_PER_PHASE = 12        # ramp one tendon up, then hand over to the next


def build_world(backend: str = "auto", mode: str = "length"):
    world = trusty.World(backend=backend, time_stepping="static", gravity=(0.0, 0.0, 0.0))
    material = trusty.StableNeoHookean(youngs_modulus=2e5, poisson_ratio=0.45)
    mesh = trusty.transform_mesh(trusty.make_beam_hex_mesh(size=(WIDTH, WIDTH, LENGTH), res=RES),
                                 t=(-WIDTH / 2, -WIDTH / 2, 0.0))
    tentacle = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0,
                                        order=trusty.fem.ElementOrder.Quadratic)
    trusty.fem.pin_face(world, tentacle, axis=2, coord=0.0)

    zs = np.linspace(0.0, LENGTH, NUM_CABLE_POINTS)
    cables = []
    for angle in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):  # +x, +y, -x, -y sides
        x, y = TENDON_OFFSET * np.cos(angle), TENDON_OFFSET * np.sin(angle)
        points = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], axis=1)
        cables.append(bc.add_cable(
            world, tentacle, points,
            bc.CableMode.Length if mode == "length" else bc.CableMode.Tension))
    return world, tentacle, cables


def _command(world, cables, rest_length, mode: str, step: int) -> str:
    """Tendon `step // STEPS_PER_PHASE` ramps in while the previous one lets go."""
    phase, k = divmod(step, STEPS_PER_PHASE)
    s = (k + 1) / STEPS_PER_PHASE
    active, previous = phase % len(cables), (phase - 1) % len(cables)
    for i, cable in enumerate(cables):
        w = s if i == active else (1.0 - s if i == previous and phase > 0 else 0.0)
        if mode == "tension":
            bc.set_cable_tension(world, cable, w * TENSION_MAX)
        else:
            bc.set_cable_length(world, cable, rest_length - w * PULL_MAX)
    return f"tendon {active} ({'+x +y -x -y'.split()[active]})"


def _register(ps, world, tentacle, cables):
    ps_mesh = ps.register_surface_mesh(
        "tentacle", np.asarray(trusty.fem.read_positions(world, tentacle)).copy(),
        np.asarray(trusty.fem.read_surface_triangles(world, tentacle)))
    ps.set_transparency_mode("pretty")  # the tendons run inside
    ps_mesh.set_transparency(0.35)
    colors = [(0.85, 0.15, 0.1), (0.1, 0.55, 0.85), (0.95, 0.65, 0.1), (0.3, 0.7, 0.3)]
    ps_cables = []
    for i, cable in enumerate(cables):
        pts = np.asarray(bc.cable_points(world, cable))
        edges = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
        c = ps.register_curve_network(f"tendon_{i}", pts, edges)
        c.set_radius(0.006, relative=True)
        c.set_color(colors[i])
        ps_cables.append(c)
    ps.look_at((0.4, -0.15, 0.22), (0.0, 0.0, 0.1))
    return ps_mesh, ps_cables


def _update(world, tentacle, cables, ps_mesh, ps_cables):
    ps_mesh.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, tentacle)))
    for cable, c in zip(cables, ps_cables):
        c.update_node_positions(np.asarray(bc.cable_points(world, cable)))


def run_polyscope(world, tentacle, cables, mode: str, steps: int):
    ps = init_polyscope(headless=False)
    rest_length = bc.cable_length(world, cables[0])
    ps_mesh, ps_cables = _register(ps, world, tentacle, cables)
    state = {"i": 0, "playing": False, "label": "slack"}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["label"] = _command(world, cables, rest_length, mode, state["i"])
            world.step()
            state["i"] += 1
            _update(world, tentacle, cables, ps_mesh, ps_cables)
        tensions = ", ".join(f"{bc.cable_tension(world, c):.2f}" for c in cables)
        psim.Text(f"step {state['i']} / {steps}   {state['label']}   tensions = {tensions} N")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, tentacle, cables, mode: str, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rest_length = bc.cable_length(world, cables[0])
    ps_mesh, ps_cables = _register(ps, world, tentacle, cables)
    ps.screenshot(str(out_dir / "tentacle_0000.png"), transparent_bg=False)
    for i in range(steps):
        label = _command(world, cables, rest_length, mode, i)
        world.step()
        _update(world, tentacle, cables, ps_mesh, ps_cables)
        ps.screenshot(str(out_dir / f"tentacle_{i + 1:04d}.png"), transparent_bg=False)
        if (i + 1) % STEPS_PER_PHASE == 0:
            tip = np.asarray(trusty.fem.read_positions(world, tentacle))
            tip = tip[tip[:, 2] > tip[:, 2].max() - 0.02].mean(axis=0)
            print(f"step {i + 1:3d}  {label}: tip at ({1e3 * tip[0]:+.1f}, {1e3 * tip[1]:+.1f}) mm")
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--mode", choices=["length", "tension"], default="length")
    parser.add_argument("--steps", type=int, default=5 * STEPS_PER_PHASE)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    world, tentacle, cables = build_world(args.backend, args.mode)
    if args.no_viewer:
        run_screenshots(world, tentacle, cables, args.mode, args.steps, args.out)
    else:
        run_polyscope(world, tentacle, cables, args.mode, args.steps)


if __name__ == "__main__":
    main()
