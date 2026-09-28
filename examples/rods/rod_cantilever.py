"""Rod cantilever.

An elastic rod clamped at one end droops under gravity, using the discrete
elastic rods (DER) discretization.

Usage:
    python examples/rods/rod_cantilever.py
    python examples/rods/rod_cantilever.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


N, LENGTH, RADIUS = 24, 1.0, 0.02


def build_rod(n: int, length: float, arc: float) -> np.ndarray:
    """A rod of `n` vertices clamped at the origin, curving down by `arc` rad."""
    X = np.zeros((n, 3))
    seg = length / (n - 1)
    pos = np.zeros(3)
    X[0] = pos
    for i in range(1, n):
        ang = arc * (i - 0.5) / (n - 1)
        pos = pos + seg * np.array([np.cos(ang), 0.0, -np.sin(ang)])
        X[i] = pos
    return X


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("rods")

    X = build_rod(N, LENGTH, 0.0)

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0

    world = trusty.World(cfg)
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e8      # ~10% static tip droop for this geometry
    mat.radius = RADIUS
    mat.density = 1000.0

    rod = trusty.rods.add_rod(world, X, mat)
    trusty.rods.pin_vertices(world, rod, [0, 1])   # clamp the base


    return world, rod


def _rod_edges(n: int) -> np.ndarray:
    return np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)


def _register_visuals(ps, world, rod):
    X   = np.asarray(trusty.rods.read_positions(world, rod))
    net = ps.register_curve_network("rod", X, _rod_edges(len(X)),
                                    color=(0.85, 0.55, 0.25))
    net.set_radius(RADIUS, relative=False)
    return net


def run_polyscope(world, rod, steps: int):
    ps  = init_polyscope(headless=False)
    net = _register_visuals(ps, world, rod)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            net.update_node_positions(
                np.asarray(trusty.rods.read_positions(world, rod)))

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


def run_screenshots(world, rod, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = _register_visuals(ps, world, rod)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"rod_cantilever_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        net.update_node_positions(
            np.asarray(trusty.rods.read_positions(world, rod)))
        snapshot(i)

    report = world.last_report()
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")
    print(f"Last solve: iters={report.iterations}  "
          f"residual={report.final_residual:.3e}  "
          f"{'converged' if report.converged else 'DIVERGED'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["cpu", "accelerate"], default="cpu")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_rod_cantilever"))
    args = parser.parse_args()

    world, rod = build_world(backend=args.backend)
    if args.no_viewer:
        run_screenshots(world, rod, args.steps, args.out)
    else:
        run_polyscope(world, rod, args.steps)


if __name__ == "__main__":
    main()
