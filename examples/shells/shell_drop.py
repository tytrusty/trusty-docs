"""Square shell sheet dropped onto a static sphere via IPC contact.

A horizontal sheet falls under gravity and lands on a kinematic sphere
(a `ContactWall`). IPC barriers between the sheet's triangles and the
sphere's triangles keep them apart.

Backend: CPU, CUDA, or Accelerate (``--backend cuda``). A shells body's
mixed K=3 (positions) + K=1 (edge alphas) layout routes to the mixed
scalar assembler + sparse Cholesky on every backend (Eigen / cuDSS /
Apple Accelerate), and the IPC contact term runs on the chosen backend too.

Usage:
    python examples/shells/shell_drop.py
    python examples/shells/shell_drop.py --backend cuda
    python examples/shells/shell_drop.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


SIDE = 1.0
RES  = 16
DROP_Z = 0.6
SPHERE_CENTER = (0.5, 0.5, 0.0)
SPHERE_RADIUS = 0.25


def make_square_sheet(side: float, res: int, z0: float):
    xs = np.linspace(0.0, side, res + 1)
    ys = np.linspace(0.0, side, res + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z    = np.full_like(X, z0)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    def vid(i, j):
        return j * (res + 1) + i

    tris = []
    for j in range(res):
        for i in range(res):
            tris.append((vid(i, j),     vid(i + 1, j),     vid(i + 1, j + 1)))
            tris.append((vid(i, j),     vid(i + 1, j + 1), vid(i,     j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def make_uv_sphere(center, radius: float, n_lat: int = 12, n_lon: int = 24):
    """Closed UV sphere triangle mesh. Outward winding."""
    cx, cy, cz = center
    verts = [(cx, cy, cz + radius)]                                     # north
    for j in range(1, n_lat):
        theta = np.pi * j / n_lat
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for i in range(n_lon):
            phi = 2.0 * np.pi * i / n_lon
            verts.append((cx + radius * sin_t * np.cos(phi),
                          cy + radius * sin_t * np.sin(phi),
                          cz + radius * cos_t))
    verts.append((cx, cy, cz - radius))                                 # south
    V = np.asarray(verts, dtype=np.float64)

    south = len(V) - 1
    tris = []
    # North cap.
    for i in range(n_lon):
        a = 1 + i
        b = 1 + (i + 1) % n_lon
        tris.append((0, a, b))
    # Middle rings.
    for j in range(n_lat - 2):
        base0 = 1 + j * n_lon
        base1 = 1 + (j + 1) * n_lon
        for i in range(n_lon):
            a = base0 + i
            b = base0 + (i + 1) % n_lon
            c = base1 + (i + 1) % n_lon
            d = base1 + i
            tris.append((a, b, c))
            tris.append((a, c, d))
    # South cap.
    base = 1 + (n_lat - 2) * n_lon
    for i in range(n_lon):
        a = base + i
        b = base + (i + 1) % n_lon
        tris.append((a, south, b))
    return V, np.asarray(tris, dtype=np.int32)


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("shells", "contact")

    V_sh, F_sh = make_square_sheet(SIDE, RES, DROP_Z)
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 120.0
    cfg.newton.max_iters = 50
    cfg.integrator = trusty.IntegratorType.BDF2
    # cfg.newton.linear_solver = trusty.LinearSolverType.Pcg

    cfg.contact.enabled = True
    cfg.contact.kappa   = 1.0e4

    world  = trusty.World(cfg)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 1.0e5
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 1.0e-3
    sh_cfg.density        = 1.0e3
    body = trusty.shells.add_shell(world, V_sh, F_sh, sh_cfg)

    V_sp, F_sp = make_uv_sphere(SPHERE_CENTER, SPHERE_RADIUS)
    trusty.contact.add_wall(world, "sphere", V_sp, F_sp)


    return world, body, (V_sp, F_sp)


def _register_visuals(ps, world, body, sphere):
    V = np.asarray(trusty.shells.read_positions(world, body))
    F = np.asarray(trusty.shells.read_triangles(world, body))
    ps_sheet = ps.register_surface_mesh("shell", V, F, smooth_shade=False)
    ps_sheet.set_edge_width(1.0)
    V_sp, F_sp = sphere
    ps.register_surface_mesh("sphere", V_sp, F_sp, color=(0.6, 0.6, 0.6))
    return ps_sheet


def run_polyscope(world, body, sphere, steps: int):
    ps = init_polyscope(headless=False)
    ps_mesh = _register_visuals(ps, world, body, sphere)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(
                np.asarray(trusty.shells.read_positions(world, body)))

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
            f"{'converged' if report.converged else 'DIVERGED'}"
        )

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, sphere, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register_visuals(ps, world, body, sphere)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"shell_{idx:04d}.png"), transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        ps_mesh.update_vertex_positions(
            np.asarray(trusty.shells.read_positions(world, body)))
        snapshot(i)

    report = world.last_report()
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")
    print(f"Last solve: iters={report.iterations}  "
          f"residual={report.final_residual:.3e}  "
          f"{'converged' if report.converged else 'DIVERGED'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["cpu", "cuda", "accelerate"], default="cpu")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_shell_drop"))
    args = parser.parse_args()

    world, body, sphere = build_world(backend=args.backend)
    if args.no_viewer:
        run_screenshots(world, body, sphere, args.steps, args.out)
    else:
        run_polyscope(world, body, sphere, args.steps)


if __name__ == "__main__":
    main()
