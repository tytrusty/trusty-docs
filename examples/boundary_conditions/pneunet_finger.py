"""PneuNet bending finger: a pressurized cavity curls a soft actuator.

A voxelized silicone finger (hex FEM) carries a row of chambers on a solid base;
a channel through the base joins every chamber into one closed void. The void
is picked as a connected component of the boundary surface
(`trusty.triangle_components`) and driven by a gauge pressure through
`boundary_conditions.add_pressure_cavity` -- the conservative potential
`-p V(x)`, a follower pressure on the chamber walls. The chambers balloon
against each other above the thicker base, so the finger curls toward it.

The root end is pinned and the pressure ramps from zero to `--pressure`
over `--steps` steps via `set_cavity_pressure`; `cavity_volume` reads back
the enclosed volume each step. Quasi-static stepping by default, not
`static`: a `static` solve of this finger fails by about 1 kPa.

Usage:
    python examples/boundary_conditions/pneunet_finger.py             # polyscope
    python examples/boundary_conditions/pneunet_finger.py --no-viewer # headless PNGs
    python examples/boundary_conditions/pneunet_finger.py --check     # headless self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

CELL = 2.5e-3          # voxel edge (m)
NUM_CHAMBERS = 8
CHAMBER_CELLS = 3      # interior length of a chamber along the finger
WIDTH_CELLS = 8        # side walls + interior across the finger
HEIGHT_CELLS = 10      # base + chambers + roof
BASE_CELLS = 2         # solid strain-limiting base below the channel


def finger_voxels() -> np.ndarray:
    """Solid-cell mask (nx, ny, nz): chambers on a base, joined by a channel."""
    period = CHAMBER_CELLS + 3  # front wall, interior, back wall, gap
    nx = NUM_CHAMBERS * period - 1
    ny, nz = WIDTH_CELLS, HEIGHT_CELLS
    solid = np.ones((nx, ny, nz), dtype=bool)
    channel_z = BASE_CELLS
    for c in range(NUM_CHAMBERS):
        x0 = c * period
        # Chamber interior: inside its own walls, from the channel to the roof.
        solid[x0 + 1:x0 + 1 + CHAMBER_CELLS, 1:ny - 1, channel_z:nz - 1] = False
        # Gap to the next chamber, above the channel's roof.
        if c + 1 < NUM_CHAMBERS:
            solid[x0 + period - 1, :, channel_z + 2:] = False
    # The channel runs through every wall, sealed at both ends.
    solid[1:nx - 1, ny // 2 - 1:ny // 2 + 1, channel_z] = False
    return solid


def voxel_hex_mesh(solid: np.ndarray):
    """Hex mesh of the solid cells, in the canonical trilinear node order."""
    nx, ny, nz = solid.shape
    node = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    i, j, k = np.nonzero(solid)
    hexes = np.stack([node[i + a, j + b, k + c] for a, b, c in corners], axis=1)
    used, hexes = np.unique(hexes, return_inverse=True)
    grid = np.stack(np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    verts = CELL * grid[used].astype(float)
    return trusty.make_hex_mesh(verts, hexes.reshape(-1, 8).astype(np.int32))


def cavity_faces(world, body) -> list[int]:
    """Boundary triangles of the internal void: the component off the outer skin."""
    tris = np.asarray(trusty.fem.read_surface_triangles(world, body))
    label = np.asarray(trusty.triangle_components(tris))
    if label.max() != 1:
        raise RuntimeError(f"expected the outer skin plus one void, got {label.max() + 1} "
                           "surface components")
    verts = np.asarray(trusty.fem.read_positions(world, body))
    skin_corner = int(np.argmin(verts.sum(axis=1)))  # the (0, 0, 0) corner
    outer = label[np.flatnonzero((tris == skin_corner).any(axis=1))[0]]
    return [int(f) for f in np.flatnonzero(label != outer)]


def build_world(backend: str = "auto", time_stepping: str = "quasi_static"):
    world = trusty.World(backend=backend,
                         time_stepping=time_stepping,
                         timestep=1.0 / 60.0,
                         newton=trusty.NewtonConfig(max_iters=50))
    # A Dragon-Skin-like silicone.
    material = trusty.StableNeoHookean(youngs_modulus=6e5, poisson_ratio=0.45)
    body = trusty.fem.add_hex_solid(world, voxel_hex_mesh(finger_voxels()), material,
                                    density=1080.0)
    trusty.fem.pin_face(world, body, axis=0, coord=0.0)
    cavity = trusty.boundary_conditions.add_pressure_cavity(
        world, body, cavity_faces(world, body))
    return world, body, cavity


def pressure_at(step: int, steps: int, p_max: float) -> float:
    return p_max * min(1.0, (step + 1) / max(1, steps))


def tip_deflection(world, body, rest) -> float:
    """Mean -z displacement of the free end's nodes (m); > 0 curls toward the base."""
    x = np.asarray(trusty.fem.read_positions(world, body))
    tip = rest[:, 0] > rest[:, 0].max() - 1e-9
    return float(rest[tip, 2].mean() - x[tip, 2].mean())


def advance(world, cavity, step, steps, p_max):
    trusty.boundary_conditions.set_cavity_pressure(world, cavity, pressure_at(step, steps, p_max))
    world.step()


def _register(ps, world, body):
    mesh = trusty.fem.read_mesh(world, body)
    verts = np.asarray(mesh.vertices)
    ps_mesh = ps.register_volume_mesh("finger", verts.copy(), hexes=np.asarray(mesh.hexes))
    ps_mesh.set_edge_width(1.0)
    # Side view (z up), framed for the curl toward -z.
    ps.set_up_dir("z_up")
    length = float(verts[:, 0].max())
    ps.look_at((0.5 * length, -2.2 * length, -0.3 * length), (0.5 * length, 0.0, -0.3 * length))
    return ps_mesh


def run_polyscope(world, body, cavity, steps, p_max):
    ps = init_polyscope(headless=False)
    ps_mesh = _register(ps, world, body)
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            advance(world, cavity, state["i"], steps, p_max)
            state["i"] += 1
            ps_mesh.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))
        if state["i"]:
            p = pressure_at(state["i"] - 1, steps, p_max)
            v = trusty.boundary_conditions.cavity_volume(world, cavity)
            psim.Text(f"step {state['i']} / {steps}   p = {p / 1e3:.1f} kPa   "
                      f"V = {v * 1e6:.2f} cm^3")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, cavity, steps, p_max, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register(ps, world, body)
    ps.screenshot(str(out_dir / "finger_0000.png"), transparent_bg=False)
    for i in range(steps):
        advance(world, cavity, i, steps, p_max)
        ps_mesh.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))
        ps.screenshot(str(out_dir / f"finger_{i + 1:04d}.png"), transparent_bg=False)
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def run_check(world, body, cavity, steps, p_max):
    rest = np.asarray(trusty.fem.read_positions(world, body)).copy()
    volumes = []
    for i in range(steps):
        advance(world, cavity, i, steps, p_max)
        report = world.last_report()
        assert report.converged, f"step {i} did not converge"
        volumes.append(trusty.boundary_conditions.cavity_volume(world, cavity))
        print(f"step {i:3d}  p = {pressure_at(i, steps, p_max) / 1e3:6.2f} kPa  "
              f"newton {report.iterations:2d}  V = {volumes[-1] * 1e6:7.3f} cm^3  "
              f"tip -dz = {tip_deflection(world, body, rest) * 1e3:7.2f} mm")
    assert all(b > a for a, b in zip(volumes, volumes[1:])), "cavity did not keep inflating"
    assert tip_deflection(world, body, rest) > 5e-3, "finger did not curl toward its base"
    print("OK: the cavity inflated monotonically and the finger curled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--time-stepping", choices=["quasi_static", "backward_euler"],
                        default="quasi_static")
    parser.add_argument("--pressure", type=float, default=40e3,
                        help="final gauge pressure (Pa)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--check", action="store_true",
                        help="headless self-test: assert the finger curls, no viewer")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    world, body, cavity = build_world(args.backend, args.time_stepping)
    if args.check:
        run_check(world, body, cavity, args.steps, args.pressure)
    elif args.no_viewer:
        run_screenshots(world, body, cavity, args.steps, args.pressure, args.out)
    else:
        run_polyscope(world, body, cavity, args.steps, args.pressure)


if __name__ == "__main__":
    main()
