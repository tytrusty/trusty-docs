"""Reduced-coordinate (subspace) beam dropped onto a floor plane.

Demonstrates the subspace simulator with the CPU and CUDA backends:
the beam is unpinned (free body), gravity pulls it down, and an IPC
floor plane catches it. The state is a length-``r`` reduced vector
``z`` (``x_full = x0 + B z``); per Newton iter the subspace term
projects gravity / inertia / elastic Hessians and a CUDA dense LDLT
(or CPU dense LDLT) solves the small dense system.

Pick the basis kind with ``--basis``:

  - ``handles`` (default): affine-expanded skinning eigenmodes
    (``12 * modes`` columns). Rotation-rich, captures tumbling
    motion with few handles.
  - ``modal``: plain modal eigenmodes (``modes`` columns). Cheap,
    good for small bending.

Pick the element kind with ``--element hex|tet``. Hex uses
``trusty.subspace.add_hex_body``; tet uses ``add_tet_body``. The
subspace term, contact, and basis machinery all work identically
across both.

Pick the backend with ``--backend cpu|cuda``. CUDA needs a CUDA-enabled
build; the current wheels are CPU / Accelerate only.

Usage:

    python examples/subspace/beam_drop.py
    python examples/subspace/beam_drop.py --element tet
    python examples/subspace/beam_drop.py --backend cuda --modes 8
    python examples/subspace/beam_drop.py --no-viewer --steps 240

Install polyscope with ``pip install trusty-sim[viewer]``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis, pack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


def _register_basis_columns(ps_mesh, basis):
    """Register every basis column on the mesh as scalar (magnitude) and
    vector (per-vertex 3D) quantities. Polyscope's UI lets the user pick
    which one to display from the structure panel."""
    N = basis.num_nodes
    r_total = basis.B.shape[1]
    for k in range(r_total):
        col = basis.B[:, k].reshape(N, 3)
        mag = np.linalg.norm(col, axis=1)
        ps_mesh.add_scalar_quantity(f"basis[{k}] |.|", mag)
        ps_mesh.add_vector_quantity(f"basis[{k}] vec", col)


DROP_HEIGHT = 0.3
FLOOR_Z     = 0.0


def build_world_and_basis(tmp_dir: Path, basis_kind: str, element: str,
                          modes: int, backend: str):
    """Build the unpinned beam world + chosen subspace basis.

    The base beam from ``make_beam_*_mesh`` is centred near the origin;
    we translate every node up by ``DROP_HEIGHT`` so the runtime beam
    starts above the floor and falls under gravity. The basis is
    precomputed against the translated rest (K + M are translation-
    invariant so the modes' shapes are unaffected). ``modes`` is the
    column count for ``modal`` and the handle count for ``handles`` /
    ``modal+handles``.
    """
    if element == "hex":
        base_mesh    = trusty.make_beam_hex_mesh(size=(0.5, 0.5, 0.5), res=(6, 6, 6))
        fem_add      = trusty.fem.add_hex_solid
        subspace_add = trusty.subspace.add_hex_body
        cells        = np.asarray(base_mesh.hexes)
        wrap_mesh    = trusty.make_hex_mesh
    elif element == "tet":
        base_mesh    = trusty.make_beam_tet_mesh(size=(0.5, 0.5, 0.5), res=(6, 6, 6))
        fem_add      = trusty.fem.add_tet_solid
        subspace_add = trusty.subspace.add_tet_body
        cells        = np.asarray(base_mesh.tets)
        wrap_mesh    = trusty.make_tet_mesh
    else:
        raise ValueError(f"unknown element kind {element!r}")
    material = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)

    # Translate every node up by DROP_HEIGHT. Both the precompute and
    # the runtime world use this lifted mesh, so x0 is consistently
    # above the floor.
    rest_verts = np.asarray(base_mesh.vertices, dtype=np.float64).copy()
    rest_verts[:, 2] += DROP_HEIGHT
    mesh = wrap_mesh(np.ascontiguousarray(rest_verts),
                     np.ascontiguousarray(cells))
    mh = build_basis.mesh_hash_sha256(rest_verts, cells)

    # Precompute world: a fem body so `fem_rest_K_M` can extract K + M.
    pc_world = trusty.World()
    fem_add(pc_world, mesh, material, density=1000.0)

    if basis_kind == "modal":
        basis_obj = build_basis.build_eigenmodes(
            world=pc_world, num_nodes=rest_verts.shape[0],
            rest_positions=rest_verts, r_modal=modes, mesh_hash=mh)
    elif basis_kind == "handles":
        basis_obj = build_basis.build_skinning_eigenmodes(
            world=pc_world, num_nodes=rest_verts.shape[0],
            rest_positions=rest_verts, num_handles=modes, mesh_hash=mh)
    else:
        raise ValueError(f"unknown basis kind {basis_kind!r}")

    basis_path = tmp_dir / "beam.basis"
    pack.save(basis_path, basis_obj)
    basis = trusty.subspace.load(str(basis_path))

    sim_cfg = trusty.SimulatorConfig()
    sim_cfg.timestep = 1.0 / 60.0
    sim_cfg.backend  = backend
    sim_cfg.newton.tolerance = 1e-5
    sim_cfg.integrator = trusty.IntegratorType.BDF2
    sim_cfg.contact.enabled = True

    # Runtime world: subspace-tagged body of the lifted geometry,
    # carrying the basis as a `SubspaceBody` component.
    world = trusty.World(sim_cfg)
    beam  = subspace_add(world, mesh, material, density=1000.0, basis=basis)
    trusty.add_floor_plane(world, FLOOR_Z)
    return world, beam, mesh, rest_verts, cells, basis_path


def _register_visuals(ps, rest_verts, cells, element):
    kw = {"hexes": cells} if element == "hex" else {"tets": cells}
    ps_mesh = ps.register_volume_mesh("beam", rest_verts, **kw)
    ps_mesh.set_edge_width(1.0)
    pad = 0.3
    x_lo, x_hi = float(rest_verts[:, 0].min()) - pad, float(rest_verts[:, 0].max()) + pad
    y_lo, y_hi = float(rest_verts[:, 1].min()) - pad, float(rest_verts[:, 1].max()) + pad
    floor_verts = np.array([
        [x_lo, y_lo, FLOOR_Z], [x_hi, y_lo, FLOOR_Z],
        [x_hi, y_hi, FLOOR_Z], [x_lo, y_hi, FLOOR_Z]])
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
    ps.register_surface_mesh("floor", floor_verts, floor_faces)
    return ps_mesh


def run_polyscope(rest_verts, cells, element, body, world, basis,
                  steps: int, label: str):
    ps = init_polyscope(headless=False)
    ps_mesh = _register_visuals(ps, rest_verts, cells, element)
    _register_basis_columns(ps_mesh, basis)
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
        psim.Text(f"step {state['i']} / {steps}    backend={label}")
        rep = world.last_report()
        psim.Text(
            f"last solve: iters={rep.iterations}  "
            f"residual={rep.final_residual:.3e}  "
            f"{'converged' if rep.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(rest_verts, cells, element, body, world, basis,
                 steps: int, label: str, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register_visuals(ps, rest_verts, cells, element)
    _register_basis_columns(ps_mesh, basis)
    ps.reset_camera_to_home_view()

    def snap(i):
        ps.screenshot(str(out_dir / f"subspace_beam_{label}_{i:04d}.png"),
                      transparent_bg=False)
    snap(0)
    for i in range(1, steps + 1):
        world.step()
        ps_mesh.update_vertex_positions(
            np.asarray(trusty.subspace.deformed_positions(world, body=body)))
        snap(i)
    rep = world.last_report()
    print(f"[{label}] wrote {steps + 1} frames; last iters={rep.iterations}, "
          f"residual={rep.final_residual:.3e}, "
          f"{'converged' if rep.converged else 'DIVERGED'}")


def main():
    trusty.check_capabilities("subspace", "contact")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "cuda", "accelerate"],
                   default="cpu",
                   help="simulator backend (default: cpu). 'accelerate' uses "
                        "Apple's sparse solver.")
    p.add_argument("--basis", choices=["modal", "handles"],
                   default="handles",
                   help="subspace basis kind (default: handles). "
                        "'handles' uses affine-expanded skinning eigenmodes; "
                        "'modal' uses plain modal eigenmodes.")
    p.add_argument("--element", choices=["hex", "tet"], default="hex",
                   help="element kind (default: hex). 'tet' exercises "
                        "subspace.add_tet_body + the tet-aware subspace "
                        "elastic / contact path.")
    p.add_argument("--modes", type=int, default=6,
                   help="basis size: modal column count for --basis modal, "
                        "handle count for --basis handles (default: 6)")
    p.add_argument("--steps", type=int, default=240,
                   help="total simulation steps (default: 240 = 4 s at 1/60)")
    p.add_argument("--no-viewer", action="store_true",
                   help="headless: write PNG screenshots instead of polyscope")
    p.add_argument("--out", type=Path, default=Path("out"),
                   help="output directory for --no-viewer mode")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit(
            "This build does not include CUDA support; "
            "run with --backend cpu.")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        world, beam, mesh, rest_verts, cells, basis_path = build_world_and_basis(
            tmp, args.basis, args.element, args.modes, args.backend)
        basis = pack.load(basis_path)
        label = f"{args.backend}-{args.element}-{args.basis}"
        if args.no_viewer:
            run_headless(rest_verts, cells, args.element, beam, world, basis,
                         args.steps, label, args.out)
        else:
            run_polyscope(rest_verts, cells, args.element, beam, world, basis,
                          args.steps, label)


if __name__ == "__main__":
    main()
