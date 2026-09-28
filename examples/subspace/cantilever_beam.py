"""Pinned subspace cantilever beam under gravity.

A hex beam is pinned at one face and gravity-loaded. The state is the
length-``r`` reduced vector ``z`` (``x_full = x0 + B z``); the basis
defaults to ``handles`` (affine-expanded skinning eigenmodes,
``12 * modes`` columns) which handles the cantilever's tip rotation
well with a small handle count.

Usage:
    python examples/subspace/cantilever_beam.py
    python examples/subspace/cantilever_beam.py --basis modal
    python examples/subspace/cantilever_beam.py --backend cuda
    python examples/subspace/cantilever_beam.py --no-viewer --steps 240

Install polyscope with ``pip install trusty-sim[viewer]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tempfile

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis, pack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


DROP_HEIGHT = 0.3
FLOOR_Z     = 0.0


def build_world_and_basis(tmp_dir: Path, basis_kind: str, modes: int,
                          backend: str = "cpu"):
    """Pinned cantilever + chosen basis. The mesh is lifted by
    ``DROP_HEIGHT`` so the free tip sags toward the floor at
    ``FLOOR_Z``.
    """
    base_mesh = trusty.make_beam_hex_mesh(size=(0.5, 0.1, 0.1), res=(10, 2, 2))
    material  = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)

    rest_verts = np.asarray(base_mesh.vertices, dtype=np.float64).copy()
    rest_verts[:, 2] += DROP_HEIGHT
    hexes = np.asarray(base_mesh.hexes)
    mesh  = trusty.make_hex_mesh(np.ascontiguousarray(rest_verts),
                                 np.ascontiguousarray(hexes))
    mh    = build_basis.mesh_hash_sha256(rest_verts, hexes)

    # Precompute world: fem body, pinned face. `fem_rest_K_M` extracts
    # K + M with pin rows zeroed so the eigenproblem is well-conditioned.
    pc_world = trusty.World()
    pc_beam  = trusty.fem.add_hex_solid(pc_world, mesh, material, density=1000.0)
    trusty.fem.pin_face(pc_world, pc_beam, axis=0, coord=rest_verts[:, 0].min())

    if basis_kind == "handles":
        basis_obj = build_basis.build_skinning_eigenmodes(
            world=pc_world, num_nodes=rest_verts.shape[0],
            rest_positions=rest_verts, num_handles=modes, mesh_hash=mh)
    elif basis_kind == "modal":
        basis_obj = build_basis.build_eigenmodes(
            world=pc_world, num_nodes=rest_verts.shape[0],
            rest_positions=rest_verts, r_modal=modes, mesh_hash=mh)
    else:
        raise ValueError(f"unknown basis kind {basis_kind!r}")

    basis_path = tmp_dir / "beam.basis"
    pack.save(basis_path, basis_obj)
    basis = trusty.subspace.load(str(basis_path))

    sim_cfg = trusty.SimulatorConfig()
    sim_cfg.timestep = 1.0 / 60.0
    sim_cfg.backend  = backend
    sim_cfg.contact.enabled = True

    world = trusty.World(sim_cfg)
    beam  = trusty.subspace.add_hex_body(
        world, mesh, material, density=1000.0, basis=basis)
    trusty.fem.pin_face(world, beam, axis=0, coord=rest_verts[:, 0].min())
    trusty.add_floor_plane(world, FLOOR_Z)
    return world, beam, mesh


def _register_visuals(ps, mesh):
    rest_verts = np.asarray(mesh.vertices).copy()
    hexes      = np.asarray(mesh.hexes)
    ps_mesh = ps.register_volume_mesh("beam", rest_verts, hexes=hexes)
    ps_mesh.set_edge_width(1.0)

    pad = 0.2
    x_lo, x_hi = float(rest_verts[:, 0].min()) - pad, float(rest_verts[:, 0].max()) + pad
    y_lo, y_hi = float(rest_verts[:, 1].min()) - pad, float(rest_verts[:, 1].max()) + pad
    floor_verts = np.array([
        [x_lo, y_lo, FLOOR_Z], [x_hi, y_lo, FLOOR_Z],
        [x_hi, y_hi, FLOOR_Z], [x_lo, y_hi, FLOOR_Z]])
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
    ps.register_surface_mesh("floor", floor_verts, floor_faces)
    return ps_mesh


def run_polyscope(body, mesh, world, steps: int):
    ps = init_polyscope(headless=False)
    ps_mesh = _register_visuals(ps, mesh)
    state = {"i": 0, "simulate": False}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        psim.SameLine()
        step_now = psim.Button("step")
        if (step_now or state["simulate"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(
                np.asarray(trusty.subspace.deformed_positions(world, body=body)))
        psim.Text(f"step {state['i']} / {steps}")
        report = world.last_report()
        psim.Text(
            f"last solve: iters={report.iterations}  "
            f"residual={report.final_residual:.3e}  "
            f"{'converged' if report.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(body, mesh, world, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register_visuals(ps, mesh)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"subspace_cantilever_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        ps_mesh.update_vertex_positions(
            np.asarray(trusty.subspace.deformed_positions(world, body=body)))
        snapshot(i)

    report = world.last_report()
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")
    print(f"Last solve: iters={report.iterations}  "
          f"residual={report.final_residual:.3e}  "
          f"{'converged' if report.converged else 'DIVERGED'}")


def main():
    trusty.check_capabilities("subspace", "contact")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "cuda", "accelerate"],
                   default="cpu", help="solver backend (default: cpu)")
    p.add_argument("--basis", choices=["modal", "handles"], default="handles",
                   help="subspace basis kind (default: handles)")
    p.add_argument("--modes", type=int, default=6,
                   help="basis size: handle count for handles, "
                        "modal column count for modal (default: 6)")
    p.add_argument("--steps", type=int, default=180,
                   help="total simulation steps (default: 180 = 3 s at 1/60)")
    p.add_argument("--no-viewer", action="store_true",
                   help="headless: write PNG screenshots instead of polyscope")
    p.add_argument("--out", type=Path, default=Path("out_cantilever"),
                   help="output directory for --no-viewer mode")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        world, beam, mesh = build_world_and_basis(
            tmp_dir, args.basis, args.modes, args.backend)

        if args.no_viewer:
            run_screenshots(beam, mesh, world, args.steps, args.out)
        else:
            run_polyscope(beam, mesh, world, args.steps)


if __name__ == "__main__":
    main()
