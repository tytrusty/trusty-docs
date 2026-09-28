"""Spring pins with moving targets + IPC contact.

A cantilever bar:
  - Left face pinned at its rest position with high stiffness (acts as a
    fixed clamp).
  - Right face pinned to a target that follows a sinusoid in z.
  - A floor below (IPC contact) prevents the bar from punching through
    even when the prescribed motion would push it there.

By default opens a polyscope viewer that animates the bar following the
prescribed sinusoid; ``--no-viewer`` writes PNG frames via the EGL
backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402


BAR_SIZE = (1.0, 0.2, 0.2)
BAR_RES  = (10, 2, 2)
FLOOR_Z  = -0.2
DRIVE_AMP        = 0.27
DRIVE_PERIOD_STEPS = 60


def build_world(backend: str):
    trusty.check_capabilities("contact")

    mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)

    cfg = trusty.SimulatorConfig()
    cfg.backend = backend
    cfg.timestep = 1.0 / 60.0
    cfg.contact.enabled = True
    cfg.contact.dhat = 5e-3
    cfg.contact.kappa = 1e4

    cfg.newton.max_iters = 60

    world = trusty.World(cfg)
    body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)

    V = np.asarray(mesh.vertices)
    left_nodes = np.where(V[:, 0] < 1e-9)[0].tolist()
    right_nodes = np.where(V[:, 0] > BAR_SIZE[0] - 1e-9)[0].tolist()
    left_targets = V[left_nodes].copy()
    right_targets_init = V[right_nodes].copy()

    trusty.boundary_conditions.pin_to_target(
        world, body, left_nodes, left_targets, stiffness=1e8)
    right_pin = trusty.boundary_conditions.pin_to_target(
        world, body, right_nodes, right_targets_init, stiffness=5e6)

    trusty.add_floor_plane(world, z=FLOOR_Z)

    return world, body, right_pin, right_nodes, right_targets_init


def prescribed_targets(step: int, right_targets_init: np.ndarray):
    phase = (2 * np.pi * step) / DRIVE_PERIOD_STEPS
    dz = DRIVE_AMP * np.sin(phase)
    targets = right_targets_init.copy()
    targets[:, 2] += dz
    return targets, dz


def _register_visuals(ps, world, body):
    mesh = trusty.fem.read_mesh(world, body)
    verts = np.asarray(mesh.vertices).copy()
    hexes = np.asarray(mesh.hexes)
    ps_mesh = ps.register_volume_mesh("bar", verts, hexes=hexes)
    ps_mesh.set_edge_width(1.0)

    Lx = BAR_SIZE[0]
    floor = np.array(
        [[-0.5, -0.5, FLOOR_Z], [Lx + 0.5, -0.5, FLOOR_Z],
         [Lx + 0.5, 0.5, FLOOR_Z], [-0.5, 0.5, FLOOR_Z]])
    ps.register_surface_mesh(
        "floor", floor, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        color=(0.6, 0.6, 0.6))
    return ps_mesh


def run_polyscope(backend: str, n_steps: int):
    ps = init_polyscope(headless=False)
    world, body, right_pin, _, right_targets_init = build_world(backend)
    ps_mesh = _register_visuals(ps, world, body)

    state = {"i": 0, "playing": False}   # play starts OFF

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step") or state["playing"]:
            if state["i"] < n_steps:
                targets, _ = prescribed_targets(state["i"], right_targets_init)
                trusty.boundary_conditions.set_pin_targets(world, right_pin, targets)
                world.step()
                state["i"] += 1
                ps_mesh.update_vertex_positions(
                    np.asarray(trusty.fem.read_mesh(world, body).vertices))
        psim.Text(f"step {state['i']} / {n_steps}")
        rep = world.last_report()
        psim.Text(
            f"last solve: iters={rep.iterations}  "
            f"residual={rep.final_residual:.3e}  "
            f"{'converged' if rep.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(backend: str, n_steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    world, body, right_pin, _, right_targets_init = build_world(backend)
    ps_mesh = _register_visuals(ps, world, body)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"moving_pin_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    last_residual = float("nan")
    for i in range(n_steps):
        targets, _ = prescribed_targets(i, right_targets_init)
        trusty.boundary_conditions.set_pin_targets(world, right_pin, targets)
        world.step()
        last_residual = world.last_report().final_residual
        ps_mesh.update_vertex_positions(
            np.asarray(trusty.fem.read_mesh(world, body).vertices))
        snapshot(i + 1)

    print(f"Wrote {n_steps + 1} screenshots to {out_dir}/")
    print(f"Last residual: {last_residual:.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend",
                    choices=["cpu", "cuda", "accelerate"], default="cpu")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out_moving_pin"))
    args = ap.parse_args()

    if args.no_viewer:
        run_screenshots(args.backend, args.steps, args.out)
    else:
        run_polyscope(args.backend, args.steps)


if __name__ == "__main__":
    main()
