"""Shell flag stitched into a flexible FEM pole.

A soft quadratic (Q2) hex pole stands on a clamped base. A thin shell flag hangs
off its upper half; the flag's root edge runs up the pole's axis, inside the
pole, where there are no mesh nodes. Those root vertices are *stitched* onto the
pole: they carry no degrees of freedom of their own and follow the pole's
motion. Under
gravity the flag droops and its weight sways the pole.

Any FEM body can anchor a stitch (tet or hex, linear or quadratic), as can an
embedded, affine or shell body; see fem_embedded_flag.py and
shell_affine_flag.py for the other anchors.

Usage:
    python examples/boundary_conditions/shell_fem_flag.py               # polyscope
    python examples/boundary_conditions/shell_fem_flag.py --no-viewer   # headless PNGs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

POLE_WIDTH = 0.06           # square cross-section side (m)
POLE_HEIGHT = 1.0           # m
FLAG_LEN = 0.6              # flag length off the pole axis (x, m)
FLAG_HEIGHT = 0.4           # flag height, hung from the pole's top (z, m)
FLAG_RES = (12, 8)


def make_sheet(x0, x1, z0, z1, resx, resz):
    """Two-triangle-per-quad flat sheet in the y = 0 plane."""
    xs = np.linspace(x0, x1, resx + 1)
    zs = np.linspace(z0, z1, resz + 1)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)

    def vid(i, j):
        return j * (resx + 1) + i

    tris = []
    for j in range(resz):
        for i in range(resx):
            tris.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            tris.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("shells")
    world = trusty.World(backend=backend, timestep=0.01)

    # -- Anchor: a soft Q2 hex pole, clamped at z = 0, centred on the z axis. --
    mesh = trusty.transform_mesh(
        trusty.make_beam_hex_mesh(size=(POLE_WIDTH, POLE_WIDTH, POLE_HEIGHT), res=(1, 1, 8)),
        t=(-POLE_WIDTH / 2, -POLE_WIDTH / 2, 0.0))
    material = trusty.StableNeoHookean(youngs_modulus=2e7, poisson_ratio=0.4)
    pole = trusty.fem.add_hex_solid(world, mesh, material, density=500.0,
                                    order=trusty.fem.ElementOrder.Quadratic)
    trusty.fem.pin_face(world, pole, axis=2, coord=0.0)

    # -- Follower: a shell flag whose x = 0 edge lies on the pole's axis. --
    V, F = make_sheet(0.0, FLAG_LEN, POLE_HEIGHT - FLAG_HEIGHT, POLE_HEIGHT, *FLAG_RES)
    cfg = trusty.shells.ShellConfig()
    cfg.youngs_modulus = 5e5
    cfg.poisson_ratio = 0.3
    cfg.thickness = 2e-3
    cfg.density = 400.0
    flag = trusty.shells.add_shell(world, V, F, cfg)

    root = np.flatnonzero(np.abs(V[:, 0]) < 1e-9).tolist()
    trusty.boundary_conditions.stitch(world, flag, root, pole)
    print(f"Q2 hex pole + shell flag, {len(root)} root verts stitched along the pole's axis.")
    return world, pole, flag


def _register(ps, world, pole, flag):
    ps_pole = ps.register_surface_mesh(
        "pole", np.asarray(trusty.fem.read_positions(world, pole)).copy(),
        np.asarray(trusty.fem.read_surface_triangles(world, pole)), color=(0.55, 0.55, 0.6))
    ps_flag = ps.register_surface_mesh(
        "flag", np.asarray(trusty.shells.read_positions(world, flag)).copy(),
        np.asarray(trusty.shells.read_triangles(world, flag)), color=(0.9, 0.5, 0.2))
    ps_flag.set_edge_width(1.0)
    ps.look_at((1.3, -2.2, 1.0), (0.25, 0.0, 0.6))
    return ps_pole, ps_flag


def _update(world, pole, flag, ps_pole, ps_flag):
    ps_pole.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, pole)))
    ps_flag.update_vertex_positions(np.asarray(trusty.shells.read_positions(world, flag)))


def run_polyscope(world, pole, flag, steps: int):
    ps = init_polyscope(headless=False)
    ps_pole, ps_flag = _register(ps, world, pole, flag)
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            _update(world, pole, flag, ps_pole, ps_flag)
        r = world.last_report()
        psim.Text(f"step {state['i']} / {steps}   newton iters = {r.iterations}")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, pole, flag, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_pole, ps_flag = _register(ps, world, pole, flag)
    ps.screenshot(str(out_dir / "flag_0000.png"), transparent_bg=False)
    rest = np.asarray(trusty.fem.read_positions(world, pole)).copy()
    top = np.flatnonzero(rest[:, 2] > POLE_HEIGHT - 1e-6)
    for i in range(steps):
        world.step()
        _update(world, pole, flag, ps_pole, ps_flag)
        ps.screenshot(str(out_dir / f"flag_{i + 1:04d}.png"), transparent_bg=False)
        if (i + 1) % 50 == 0:
            tip = np.asarray(trusty.fem.read_positions(world, pole))[top].mean(axis=0)
            print(f"step {i + 1:4d}: pole top at x = {1e3 * tip[0]:+.1f} mm, "
                  f"flag low point z = {np.asarray(trusty.shells.read_positions(world, flag))[:, 2].min():.3f} m")
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    world, pole, flag = build_world(args.backend)
    if args.no_viewer:
        run_screenshots(world, pole, flag, args.steps, args.out)
    else:
        run_polyscope(world, pole, flag, args.steps)


if __name__ == "__main__":
    main()
