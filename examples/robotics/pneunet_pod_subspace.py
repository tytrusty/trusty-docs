"""PneuNet finger reduced to a few displacement modes learned from a full simulation.

The finger of `pneunet_finger.py`, root clamped, is first inflated as a full
solid, and every step's displacement field is kept as a snapshot. The leading
singular vectors of the snapshots (proper orthogonal decomposition) become the
displacement modes of a reduced finger with only `--modes` coordinates, and an
elastic cubature rule evaluates its elasticity on a small set of weighted
elements. The full finger (blue) and the reduced one (translucent orange) are
then inflated side by side, and the panel compares their tip drop and time per
step. The modes are fixed in the rest frame, so they reproduce this clamped
finger well but would not follow a finger that is carried through a large
rotation.

Usage:
    python examples/robotics/pneunet_pod_subspace.py                 # polyscope
    python examples/robotics/pneunet_pod_subspace.py --reduced-only  # reduced finger alone
    python examples/robotics/pneunet_pod_subspace.py --no-viewer     # headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402
from pneunet_finger import cavity_faces, finger_voxels, voxel_hex_mesh  # noqa: E402

bc = trusty.boundary_conditions
subspace = trusty.subspace

MATERIAL = dict(youngs_modulus=6e5, poisson_ratio=0.45)
DENSITY = 1080.0


def make_world(backend: str):
    return trusty.World(backend=backend, time_stepping="quasi_static",
                        newton=trusty.NewtonConfig(max_iters=50))


def full_finger(world, mesh):
    body = trusty.fem.add_hex_solid(world, mesh, trusty.StableNeoHookean(**MATERIAL),
                                    density=DENSITY)
    trusty.fem.pin_face(world, body, axis=0, coord=0.0)
    return body, bc.add_pressure_cavity(world, body, cavity_faces(world, body))


def pod_modes(mesh, backend: str, p_max: float, snapshots: int, modes: int) -> np.ndarray:
    """Leading `modes` displacement modes (3N, modes) of the full finger inflated to `p_max`."""
    world = make_world(backend)
    body, cavity = full_finger(world, mesh)
    rest = np.asarray(trusty.fem.read_positions(world, body)).copy()
    u = []
    for i in range(snapshots):
        bc.set_cavity_pressure(world, cavity, p_max * (i + 1) / snapshots)
        world.step()
        u.append((np.asarray(trusty.fem.read_positions(world, body)) - rest).reshape(-1))
    U, s, _ = np.linalg.svd(np.array(u).T, full_matrices=False)
    print("relative singular values:", " ".join(f"{v / s[0]:.1e}" for v in s[:modes + 2]))
    return np.asfortranarray(U[:, :modes])


def reduced_finger(world, mesh, modes: np.ndarray, cubature: bool):
    body = subspace.add_hex_body(world, mesh, trusty.StableNeoHookean(**MATERIAL),
                                 density=DENSITY, basis=subspace.Basis.modal(modes))
    if cubature:
        rule = subspace.fit_cubature(world, body)
        subspace.set_cubature(world, body, rule)
        print(f"cubature: {len(rule.elements)} of {len(np.asarray(mesh.hexes))} elements")
    return body, bc.add_pressure_cavity(world, body, cavity_faces(world, body))


class Comparison:
    """The reduced finger and, unless `reduced_only`, the full one, inflated together."""

    def __init__(self, args):
        self.mesh = voxel_hex_mesh(finger_voxels())
        modes = pod_modes(self.mesh, args.backend, args.pressure, args.snapshots, args.modes)
        self.full = None
        if not args.reduced_only:
            self.full_world = make_world(args.backend)
            self.full, self.full_cavity = full_finger(self.full_world, self.mesh)
        self.world = make_world(args.backend)
        self.reduced, self.cavity = reduced_finger(self.world, self.mesh, modes, args.cubature)
        self.rest = np.asarray(self.mesh.vertices).copy()
        self.tip = self.rest[:, 0] > self.rest[:, 0].max() - 1e-9
        self.p_max, self.steps, self.i = args.pressure, args.steps, 0
        self.t_full = self.t_reduced = 0.0

    def step(self):
        p = self.p_max * (self.i + 1) / self.steps
        if self.full is not None:
            bc.set_cavity_pressure(self.full_world, self.full_cavity, p)
            t = time.perf_counter()
            self.full_world.step()
            self.t_full += time.perf_counter() - t
        bc.set_cavity_pressure(self.world, self.cavity, p)
        t = time.perf_counter()
        self.world.step()
        self.t_reduced += time.perf_counter() - t
        self.i += 1

    def full_positions(self):
        return np.asarray(trusty.fem.read_positions(self.full_world, self.full))

    def reduced_positions(self):
        return np.asarray(subspace.deformed_positions(self.world, self.reduced))

    def summary(self) -> str:
        n = max(self.i, 1)
        def drop(x):
            return 1e3 * (self.rest[self.tip, 2] - x[self.tip, 2]).mean()
        text = (f"reduced {drop(self.reduced_positions()):5.1f} mm, "
                f"{1e3 * self.t_reduced / n:5.1f} ms/step")
        if self.full is not None:
            text = (f"full {drop(self.full_positions()):5.1f} mm, "
                    f"{1e3 * self.t_full / n:6.1f} ms/step   " + text)
        return "tip drop: " + text


def run_headless(sim):
    for _ in range(sim.steps):
        sim.step()
        if sim.i % 10 == 0:
            print(f"step {sim.i:3d}  p = {sim.p_max * sim.i / sim.steps / 1e3:4.1f} kPa  "
                  + sim.summary())


def run_polyscope(sim, modes: int):
    ps = init_polyscope(headless=False)
    ps.set_up_dir("z_up")
    hexes = np.asarray(sim.mesh.hexes)
    reduced = ps.register_volume_mesh(f"reduced, {modes} modes", sim.rest, hexes=hexes,
                                      color=(0.95, 0.55, 0.25))
    if sim.full is None:
        reduced.set_edge_width(0.5)
    else:
        full = ps.register_volume_mesh("full", sim.rest, hexes=hexes, color=(0.3, 0.55, 0.85))
        full.set_edge_width(0.5)
        ps.set_transparency_mode("pretty")
        reduced.set_transparency(0.45)
    L = float(sim.rest[:, 0].max())
    ps.look_at((0.5 * L, -2.4 * L, -0.2 * L), (0.5 * L, 0.01, -0.3 * L))
    state = {"playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and sim.i < sim.steps:
            sim.step()
            reduced.update_vertex_positions(sim.reduced_positions())
            if sim.full is not None:
                full.update_vertex_positions(sim.full_positions())
        psim.Text(f"step {sim.i} / {sim.steps}   "
                  f"p = {sim.p_max * sim.i / sim.steps / 1e3:.1f} kPa")
        psim.Text(sim.summary())

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--pressure", type=float, default=40e3,
                        help="peak gauge pressure in the finger (Pa); higher curls further")
    parser.add_argument("--modes", type=int, default=5,
                        help="displacement modes, the reduced finger's coordinates")
    parser.add_argument("--snapshots", type=int, default=40,
                        help="full-solid inflation steps recorded as snapshots; the "
                             "pressure may rise by about 1 kPa per step")
    parser.add_argument("--no-cubature", dest="cubature", action="store_false",
                        help="evaluate elasticity on every element instead of a fitted rule")
    parser.add_argument("--reduced-only", action="store_true",
                        help="simulate only the reduced finger, so its time per step is alone")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    trusty.check_capabilities("subspace")
    sim = Comparison(args)
    if args.no_viewer:
        run_headless(sim)
    else:
        run_polyscope(sim, args.modes)


if __name__ == "__main__":
    main()
