"""Coupled subspace + FEM drop: two beams in one simulator.

One hex beam is driven by a reduced-coordinate (subspace) basis;
the second is driven by full-FEM. Both fall under gravity onto a
floor; subspace<->FEM IPC contact is set up automatically when
`sim_cfg.contact.enabled` is set, and both bodies are solved
together in one implicit step (CPU or CUDA).

Pipeline:
  1. Build two translated hex beams in one world.
  2. Precompute a basis on a single beam-rest configuration and
     save it as a `.basis` file.
  3. Build a coupled simulator: body 0 -> subspace (loads the
     basis), body 1 -> FEM. Contact (FEM-floor, subspace-floor,
     and cross-block subspace<->FEM) is governed entirely by
     `sim_cfg.contact`.

Usage:
    python examples/subspace/coupled_drop.py
    python examples/subspace/coupled_drop.py --backend cuda
    python examples/subspace/coupled_drop.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


BEAM_SIZE        = (0.4, 0.1, 0.1)
BEAM_RES         = (8, 2, 2)
DROP_HEIGHT      = 0.15
FLOOR_Z          = 0.0
BEAM_SEPARATION  = 0.0  # subspace stacked above fem in z, separated by 2*dhat


def build_basis(tmp_dir: Path, num_handles: int) -> Path:
    """Precompute one skinning-eigenmodes basis on a one-body world of
    the source rest mesh."""
    from trusty.subspace.precompute import build_basis as bb, pack
    mesh     = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    world    = trusty.World()
    trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)

    rv    = np.asarray(mesh.vertices, dtype=np.float64)
    hexes = np.asarray(mesh.hexes)
    basis = bb.build_skinning_eigenmodes(
        world=world,
        num_nodes=rv.shape[0],
        rest_positions=rv,
        num_handles=num_handles,
        mesh_hash=bb.mesh_hash_sha256(rv, hexes))
    path = tmp_dir / "beam.basis"
    pack.save(path, basis)
    return path


def add_beam(world, z_offset: float, basis=None):
    """Add one hex beam translated to (0, 0, z_offset). Returns (body, mesh).
    Pass `basis` to attach a subspace body; omit it for a fem body."""
    src   = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    verts = np.asarray(src.vertices, dtype=np.float64).copy()
    verts[:, 2] += z_offset
    mesh = trusty.make_hex_mesh(verts, np.asarray(src.hexes, dtype=np.int32))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    if basis is not None:
        body = trusty.subspace.add_hex_body(
            world, mesh, material, density=1000.0, basis=basis)
    else:
        body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
    return body, mesh


def main():
    trusty.check_capabilities("subspace", "contact", "fem")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "cuda", "accelerate"],
                   default="cpu", help="solver backend (default: cpu)")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--modes", type=int, default=6,
                   help="skinning-handle count for the subspace basis "
                        "(default: 6)")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--out", type=Path, default=Path("out_coupled"))
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        basis_path = build_basis(td, args.modes)
        basis = trusty.subspace.load(str(basis_path))

        sim_cfg = trusty.SimulatorConfig()
        sim_cfg.timestep = 1.0 / 60.0
        sim_cfg.backend  = args.backend
        sim_cfg.contact.enabled = True

        world = trusty.World(sim_cfg)
        # body 0 -> subspace, sits above body 1 so they collide as they
        # both drop under gravity (if --coupled-contact is on).
        body_sub, mesh_sub = add_beam(
            world, z_offset=DROP_HEIGHT * 2 + BEAM_SEPARATION, basis=basis)
        # body 1 -> fem, the floor catches it.
        body_fem, mesh_fem = add_beam(world, z_offset=DROP_HEIGHT)

        # One contact knob covers every body in the simulator: the
        # subspace bodies' floor, the FEM bodies' floor, and the
        # cross-block subspace<->FEM IPC all live on `sim_cfg.contact`
        # and are wired automatically by the contact module's
        # surface-driven dispatch.
        trusty.add_floor_plane(world, FLOOR_Z)

        n_sub = trusty.subspace.num_bodies(world)
        print(f"Coupled simulator: {n_sub} subspace body, 1 fem body")

        if args.no_viewer:
            run_screenshots(world, body_sub, body_fem,
                            mesh_sub, mesh_fem,
                            args.steps, args.out)
        else:
            run_polyscope(world, body_sub, body_fem,
                          mesh_sub, mesh_fem, args.steps)


def _floor_quad(extent: float = 1.0):
    return (
        np.array([
            [-extent, -extent, FLOOR_Z], [ extent, -extent, FLOOR_Z],
            [ extent,  extent, FLOOR_Z], [-extent,  extent, FLOOR_Z]],
            dtype=np.float64),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))


def _register_visuals(ps, world, body_fem, mesh_sub, mesh_fem):
    fv, ff = _floor_quad()
    ps.register_surface_mesh("floor", fv, ff, color=(0.6, 0.6, 0.6))

    rest_sub = np.asarray(mesh_sub.vertices)
    ps_sub = ps.register_volume_mesh(
        "subspace_beam", rest_sub, hexes=np.asarray(mesh_sub.hexes))
    ps_sub.set_edge_width(1.0)

    fem_verts = np.asarray(trusty.fem.read_mesh(world, body_fem).vertices).copy()
    ps_fem = ps.register_volume_mesh(
        "fem_beam", fem_verts, hexes=np.asarray(mesh_fem.hexes))
    ps_fem.set_edge_width(1.0)
    return ps_sub, ps_fem


def _update_visuals(world, body_sub, body_fem, ps_sub, ps_fem):
    ps_sub.update_vertex_positions(
        np.asarray(trusty.subspace.deformed_positions(world, body=body_sub)))
    ps_fem.update_vertex_positions(
        np.asarray(trusty.fem.read_mesh(world, body_fem).vertices))


def run_screenshots(world, body_sub, body_fem, mesh_sub, mesh_fem,
                    steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_sub, ps_fem = _register_visuals(ps, world, body_fem, mesh_sub, mesh_fem)
    ps.reset_camera_to_home_view()

    def snap(idx):
        ps.screenshot(str(out_dir / f"coupled_{idx:04d}.png"),
                      transparent_bg=False)

    snap(0)
    for i in range(1, steps + 1):
        world.step()
        _update_visuals(world, body_sub, body_fem, ps_sub, ps_fem)
        snap(i)

    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def run_polyscope(world, body_sub, body_fem, mesh_sub, mesh_fem,
                  steps: int):
    ps = init_polyscope(headless=False)
    ps_sub, ps_fem = _register_visuals(ps, world, body_fem, mesh_sub, mesh_fem)
    state = {"i": 0, "simulate": False}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        psim.SameLine()
        step_now = psim.Button("step")
        if (step_now or state["simulate"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(world, body_sub, body_fem, ps_sub, ps_fem)
        psim.Text(f"step {state['i']} / {steps}")
        rep = world.last_report()
        psim.Text(
            f"last solve: iters={rep.iterations}  "
            f"residual={rep.final_residual:.3e}  "
            f"{'converged' if rep.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
