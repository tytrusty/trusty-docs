"""Drawstring curtain (a cable on a shell's midsurface, tension-driven).

A thin shell curtain hangs from a pinned top edge, with a drawstring threaded
along its bottom hem, corner to corner. Pulling the drawstring with a rising
tension shortens the hem, which gathers into pleats while the curtain's weight
keeps the rest hanging. The cable points sit on the shell's rest midsurface,
which is where a shell interpolates them; a slight ripple in the rest shape
picks which way each pleat folds.

Self-contact is on: the pleats fold against each other. Without it nothing
stops a pleat from folding shut, and once the drawstring outpulls the curtain's
bending the run stops.

Usage:
    python examples/boundary_conditions/cable_curtain.py               # polyscope
    python examples/boundary_conditions/cable_curtain.py --no-viewer   # headless PNGs
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

WIDTH = 0.6                 # curtain width along x (m)
HEIGHT = 0.5                # curtain drop along z (m)
RES = (30, 24)
RIPPLE = 0.01               # m, rest-shape pleat amplitude
NUM_PLEATS = 3
TENSION_MAX = 1.5           # N, the fully-ramped drawstring tension


def make_curtain():
    xs = np.linspace(0.0, WIDTH, RES[0] + 1)
    zs = np.linspace(0.0, HEIGHT, RES[1] + 1)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    Y = RIPPLE * np.sin(2 * np.pi * NUM_PLEATS * X / WIDTH)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def vid(i, j):
        return j * (RES[0] + 1) + i

    tris = []
    for j in range(RES[1]):
        for i in range(RES[0]):
            tris.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            tris.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("shells", "contact")
    world = trusty.World(backend=backend, timestep=0.01)
    trusty.contact.enable(world, trusty.contact.Config(kappa=1e4, dhat=2e-3))
    V, F = make_curtain()
    cfg = trusty.shells.ShellConfig()
    cfg.youngs_modulus = 5e5
    cfg.poisson_ratio = 0.3
    cfg.thickness = 1e-3
    cfg.density = 300.0
    curtain = trusty.shells.add_shell(world, V, F, cfg)
    trusty.shells.pin_vertices(world, curtain, np.flatnonzero(V[:, 2] > HEIGHT - 1e-9).tolist())

    # The drawstring: the bottom-hem vertices, corner to corner. Any points on
    # the midsurface would do; these happen to be vertices.
    hem = np.flatnonzero(V[:, 2] < 1e-9)
    hem = hem[np.argsort(V[hem, 0])]
    cable = bc.add_cable(world, curtain, V[hem])
    return world, curtain, cable


def _command(world, cable, step: int, steps: int) -> str:
    T = TENSION_MAX * min(1.0, (step + 1) / (0.6 * steps))  # ramp, then hold
    bc.set_cable_tension(world, cable, T)
    return f"T = {T:.2f} N"


def _register(ps, world, curtain, cable):
    ps_curtain = ps.register_surface_mesh(
        "curtain", np.asarray(trusty.shells.read_positions(world, curtain)).copy(),
        np.asarray(trusty.shells.read_triangles(world, curtain)), color=(0.35, 0.45, 0.75))
    ps_curtain.set_edge_width(0.5)
    pts = np.asarray(bc.cable_points(world, cable))
    edges = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
    ps_cable = ps.register_curve_network("drawstring", pts, edges)
    ps_cable.set_radius(0.004, relative=True)
    ps_cable.set_color((0.85, 0.15, 0.1))
    ps.look_at((0.3, -1.1, 0.45), (0.3, 0.0, 0.2))
    return ps_curtain, ps_cable


def _update(world, curtain, cable, ps_curtain, ps_cable):
    ps_curtain.update_vertex_positions(np.asarray(trusty.shells.read_positions(world, curtain)))
    ps_cable.update_node_positions(np.asarray(bc.cable_points(world, cable)))


def run_polyscope(world, curtain, cable, steps: int):
    ps = init_polyscope(headless=False)
    ps_curtain, ps_cable = _register(ps, world, curtain, cable)
    state = {"i": 0, "playing": False, "label": "slack"}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["label"] = _command(world, cable, state["i"], steps)
            world.step()
            state["i"] += 1
            _update(world, curtain, cable, ps_curtain, ps_cable)
        psim.Text(f"step {state['i']} / {steps}   {state['label']}   "
                  f"hem length = {bc.cable_length(world, cable):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, curtain, cable, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_curtain, ps_cable = _register(ps, world, curtain, cable)
    ps.screenshot(str(out_dir / "curtain_0000.png"), transparent_bg=False)
    for i in range(steps):
        label = _command(world, cable, i, steps)
        world.step()
        _update(world, curtain, cable, ps_curtain, ps_cable)
        ps.screenshot(str(out_dir / f"curtain_{i + 1:04d}.png"), transparent_bg=False)
        if (i + 1) % 25 == 0:
            print(f"step {i + 1:4d}  {label}: hem length {bc.cable_length(world, cable):.3f} m")
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    world, curtain, cable = build_world(args.backend)
    if args.no_viewer:
        run_screenshots(world, curtain, cable, args.steps, args.out)
    else:
        run_polyscope(world, curtain, cable, args.steps)


if __name__ == "__main__":
    main()
