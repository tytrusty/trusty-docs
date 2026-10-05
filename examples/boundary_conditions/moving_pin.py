"""Spring pins with moving targets, and a floor in the way.

A soft cantilever bar:
  - its left face is pinned at its rest position with a high stiffness, so it
    acts as a clamp;
  - its right face is pinned to targets that move up and down on a sine wave;
  - a floor below stops the bar. The targets dip below the floor, but a spring
    pin gives, so the bar rests on the floor instead of passing through it.

Usage:
    python examples/boundary_conditions/moving_pin.py              # polyscope
    python examples/boundary_conditions/moving_pin.py --no-viewer  # print the drive
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

    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(dhat=5e-3, kappa=1e4))
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


def drive(world, right_pin, right_targets_init, step: int):
    """Move the right face's targets for this step, then advance."""
    targets, dz = prescribed_targets(step, right_targets_init)
    trusty.boundary_conditions.set_pin_targets(world, right_pin, targets)
    world.step()
    return dz


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
                drive(world, right_pin, right_targets_init, state["i"])
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


def run_headless(backend: str, n_steps: int):
    world, body, right_pin, right_nodes, right_targets_init = build_world(backend)
    print(f"{'step':>4}  {'target dz':>9}  {'face dz':>8}  {'lowest z':>8}  pin force z (N)")
    lowest = np.inf
    for i in range(n_steps):
        dz = drive(world, right_pin, right_targets_init, i)
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: the solve did not converge")
        x = np.asarray(trusty.fem.read_positions(world, body))
        lowest = min(lowest, x[:, 2].min())
        if i % 10 == 0:
            face_dz = x[right_nodes, 2].mean() - right_targets_init[:, 2].mean()
            fz = trusty.boundary_conditions.pin_total_force(world, right_pin)[2]
            print(f"{i:4d}  {dz:+9.3f}  {face_dz:+8.3f}  {x[:, 2].min():+8.3f}  {fz:+.1f}")
    print(f"Lowest point reached: z = {lowest:.3f} (floor at z = {FLOOR_Z})")
    if lowest < FLOOR_Z - 1e-3:
        raise SystemExit("the bar passed through the floor")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend",
                    choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: print the drive and the pin force every 10 steps")
    args = ap.parse_args()

    if args.no_viewer:
        run_headless(args.backend, args.steps)
    else:
        run_polyscope(args.backend, args.steps)


if __name__ == "__main__":
    main()
