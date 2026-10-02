"""Hex and tet meshes: built-in beams, and a tet mesh loaded from a file.

Builds one world with a structured hex beam, a structured tet beam and a tet
ball read from ``data/ball.msh`` (Gmsh format 2.2), the way you would load your
own tetrahedralized geometry. Nothing is pinned, so stepping
lets all three fall under gravity.

Usage:
    uv run examples/fem/mesh_types.py                   # live polyscope
    uv run examples/fem/mesh_types.py --no-viewer       # print mesh sizes
    uv run examples/fem/mesh_types.py --mesh my.msh     # your own tet mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
COLORS = {"hex beam": (0.94, 0.33, 0.31), "tet beam": (0.78, 0.79, 0.84),
          "loaded mesh": (0.35, 0.75, 0.70)}


def build_world(mesh_path: Path = DATA / "ball.msh", backend: str = "auto"):
    # Structured beams, built in: (Lx, Ly, Lz) and elements per axis.
    hex_mesh = trusty.make_beam_hex_mesh(size=(1.0, 0.1, 0.1), res=(20, 2, 2))
    tet_mesh = trusty.make_beam_tet_mesh(size=(1.0, 0.1, 0.1), res=(20, 2, 2))

    # Your own tet mesh, from a Gmsh .msh file (format 2.2) ...
    loaded = trusty.read_tet_msh(str(mesh_path))
    # ... or from NumPy arrays, e.g. produced by TetGen, fTetWild or meshio.
    loaded = trusty.make_tet_mesh(loaded.vertices, loaded.tets)

    # Place each mesh before adding it: rotate with A, translate with t.
    tet_mesh = trusty.transform_mesh(tet_mesh, t=(0.0, 0.0, -0.3))
    loaded = trusty.transform_mesh(loaded, t=(1.3, 0.05, -0.1))

    world = trusty.World(backend=backend)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    bodies = {
        "hex beam":    trusty.fem.add_hex_solid(world, hex_mesh, material, density=1000.0),
        "tet beam":    trusty.fem.add_tet_solid(world, tet_mesh, material, density=1000.0),
        "loaded mesh": trusty.fem.add_tet_solid(world, loaded, material, density=1000.0),
    }
    return world, bodies, {"hex beam": hex_mesh, "tet beam": tet_mesh,
                           "loaded mesh": loaded}


def run_viewer(world, bodies, meshes, steps: int):
    ps = init_polyscope(headless=False)
    shown = {}
    for name, body in bodies.items():
        mesh = meshes[name]
        cells = {"hexes": np.asarray(mesh.hexes)} if hasattr(mesh, "hexes") \
            else {"tets": np.asarray(mesh.tets)}
        m = ps.register_volume_mesh(name, trusty.fem.read_positions(world, body), **cells)
        m.set_color(COLORS[name])
        m.set_edge_width(1.0)
        shown[name] = (m, body)
    ps.reset_camera_to_home_view()

    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play (fall under gravity)", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for m, body in shown.values():
                m.update_vertex_positions(trusty.fem.read_positions(world, body))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mesh", type=Path, default=DATA / "ball.msh",
                    help="tet mesh to load (Gmsh .msh, format 2.2)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: print each mesh's size")
    args = ap.parse_args()

    world, bodies, meshes = build_world(args.mesh, args.backend)
    if args.no_viewer:
        for name, mesh in meshes.items():
            print(f"{name:<12} {mesh.num_nodes:5d} nodes  {mesh.num_elems:5d} elements")
        return
    run_viewer(world, bodies, meshes, args.steps)


if __name__ == "__main__":
    main()
