"""Cantilever beam with a moving Nitsche pin at the free end.

A high-res rectangular bar is voxelized, the left face Nitsche-pinned
at rest, and the right face Nitsche-pinned to a target that follows a
vertical sinusoid. The beam bends elastically as the right end is
driven up and down.

Defaults open a polyscope viewer; ``--no-viewer`` writes PNG frames via
the EGL backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope, make_bar_trimesh, mean_edge_length  # noqa: E402


# --- Geometry & drive --------------------------------------------------
BAR_SIZE     = (1.0, 0.2, 0.2)
BAR_FACE_RES = (16, 4, 4)        # surface triangulation per dimension
VOXEL_FACTOR = 1.5               # hex edge ≈ VOXEL_FACTOR * mean tri edge

GAMMA            = 50.0
DRIVE_AMP        = 0.30
DRIVE_PERIOD_STEPS = 60

# Centroid x-band (fraction of x-extent) within which a triangle is
# considered to lie on the corresponding end face.
PIN_BAND_FRAC = 5e-2


def _end_face_indices(V: np.ndarray, F: np.ndarray):
    """Return (left_face_tris, right_face_tris) — triangle indices into
    F whose centroid sits in the leftmost/rightmost band of x-extent."""
    x_min = float(V[:, 0].min())
    x_max = float(V[:, 0].max())
    band  = PIN_BAND_FRAC * (x_max - x_min)
    cx    = V[F, 0].mean(axis=1)
    left  = np.flatnonzero(cx - x_min < band).astype(int).tolist()
    right = np.flatnonzero(x_max - cx < band).astype(int).tolist()
    return left, right


def _right_face_verts(V: np.ndarray, F: np.ndarray, right_faces):
    """Indices into V that participate in any right-face triangle."""
    return sorted({int(v) for f in right_faces for v in F[f]})


def build_world(backend: str):
    trusty.check_capabilities("embedded")

    V, F = make_bar_trimesh(BAR_SIZE, BAR_FACE_RES)
    left_faces, right_faces = _end_face_indices(V, F)
    right_verts = _right_face_verts(V, F, right_faces)
    if not left_faces or not right_faces:
        raise SystemExit(
            "moving_nitsche: failed to identify both end-face triangle "
            "groups — bump PIN_BAND_FRAC or increase BAR_FACE_RES.")

    avg_edge   = mean_edge_length(V, F)
    voxel_size = VOXEL_FACTOR * avg_edge

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0
    cfg.gravity  = (0.0, 0.0, 0.0)
    cfg.newton.max_iters = 80
    cfg.newton.tolerance = 1e-6

    world = trusty.World(cfg)
    mat   = trusty.StableNeoHookean(youngs_modulus=2e6, poisson_ratio=0.4)
    body  = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=voxel_size, material=mat, density=1000.0)

    trusty.embedded.attach_weak_pin(
        world, body,
        pin_faces=left_faces + right_faces,
        initial_targets=V.copy(),
        gamma=GAMMA)


    print(f"[moving_nitsche] bar={BAR_SIZE}  surface_verts={len(V)}  "
          f"surface_tris={len(F)}  voxel_size={voxel_size:.4f}  "
          f"left_pinned={len(left_faces)} tris  "
          f"right_pinned={len(right_faces)} tris")

    return world, body, V, F, right_verts


def prescribed_targets(step: int, V_rest: np.ndarray, right_verts):
    """Hold all surface verts at rest, except the right-face verts —
    those get pulled up/down along z by a sinusoid."""
    phase = (2 * np.pi * step) / DRIVE_PERIOD_STEPS
    dz    = DRIVE_AMP * np.sin(phase)
    targets = V_rest.copy()
    targets[right_verts, 2] += dz
    return targets, dz


def _register_visuals(ps, world, body, F, V, right_verts):
    surf = np.asarray(trusty.embedded.read_embedded_surface(world, body)).copy()
    ps_mesh = ps.register_surface_mesh("beam", surf, F, smooth_shade=False)
    ps_mesh.set_color((0.85, 0.55, 0.25))
    ps_mesh.set_edge_width(1.0)
    ps_target = ps.register_point_cloud(
        "right_targets", V[right_verts].copy(), radius=0.018,
        color=(1.0, 0.4, 0.2))
    return ps_mesh, ps_target


def run_polyscope(backend: str, n_steps: int):
    ps = init_polyscope(headless=False)
    world, body, V, F, right_verts = build_world(backend)
    ps_mesh, ps_target = _register_visuals(ps, world, body, F, V, right_verts)

    state = {"i": 0, "playing": True}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step") or state["playing"]:
            if state["i"] < n_steps:
                targets, _dz = prescribed_targets(state["i"], V, right_verts)
                trusty.embedded.set_pin_targets(world, body, targets)
                world.step()
                state["i"] += 1
                ps_mesh.update_vertex_positions(np.asarray(
                    trusty.embedded.read_embedded_surface(world, body)))
                ps_target.update_point_positions(targets[right_verts])
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

    world, body, V, F, right_verts = build_world(backend)
    ps_mesh, ps_target = _register_visuals(ps, world, body, F, V, right_verts)

    Lx = BAR_SIZE[0]
    ps.reset_camera_to_home_view()
    ps.look_at(camera_location=(0.5 * Lx, -2.0, 0.5),
               target=(0.5 * Lx, 0.0, 0.0))

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"moving_nitsche_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    last_residual = float("nan")
    for i in range(n_steps):
        targets, _dz = prescribed_targets(i, V, right_verts)
        trusty.embedded.set_pin_targets(world, body, targets)
        world.step()
        last_residual = world.last_report().final_residual
        ps_mesh.update_vertex_positions(np.asarray(
            trusty.embedded.read_embedded_surface(world, body)))
        ps_target.update_point_positions(targets[right_verts])
        snapshot(i + 1)

    print(f"Wrote {n_steps + 1} screenshots to {out_dir}/")
    print(f"Last residual: {last_residual:.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend",
                    choices=["cpu", "cuda", "accelerate"], default="cpu")
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out_moving_nitsche"),
                    help="Directory to write PNG frames in --no-viewer mode.")
    args = ap.parse_args()

    if args.no_viewer:
        run_screenshots(args.backend, args.steps, args.out)
    else:
        run_polyscope(args.backend, args.steps)


if __name__ == "__main__":
    main()
