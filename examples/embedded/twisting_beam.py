"""Twisting beam: an embedded beam pinned at both ends, with one end's
Nitsche targets rotating about the beam axis. The bulk twists like a
torsion bar.

Geometry: a 4 m × 0.4 m × 0.4 m bar centered on the x-axis, voxelized
into hex elements. Triangles whose centroids sit in the leftmost
fraction of x are "left-pinned" at rest; the rightmost fraction are
"right-pinned" with targets that rotate about the x-axis through the
right face's centroid by `theta(t)`. theta ramps from 0 to `--turns`
revolutions over `--steps` steps.

Targets are rewritten each step with `embedded.set_pin_targets(world,
body, ...)`. The embedded `sync_in` stage copies them into the module's
own storage, and the term recomputes per-QP `x_bar` from the bary table
at `on_step_begin` -- re-uploading to the device on CUDA.

Run:
    python examples/embedded/twisting_beam.py
    python examples/embedded/twisting_beam.py --steps 240 --turns 1.0
    python examples/embedded/twisting_beam.py --no-viewer  # PNG frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope, make_bar_trimesh  # noqa: E402


# --- Geometry ---------------------------------------------------------

BAR_SIZE  = (4.0, 0.4, 0.4)   # length along x; cross-section y, z
BAR_RES   = (40, 6, 6)        # surface triangulation per dimension
# Default voxel edge = min(Ly, Lz) / DEFAULT_VOXEL_FACTOR. Smaller
# factors -> coarser hex grid (fewer DOFs, faster, blockier twist);
# larger factors -> finer grid (smoother, slower).
DEFAULT_VOXEL_FACTOR = 4.0

# Pin all triangles whose centroid x lies in this fraction of the
# extent at each end of the beam.
END_BAND_FRAC = 0.04


def select_end_faces(
    V: np.ndarray,
    F: np.ndarray,
    band_frac: float = END_BAND_FRAC,
) -> tuple[list[int], list[int]]:
    """Return (left_face_ids, right_face_ids) -- triangles whose centroid
    x sits within `band_frac` of (x_max - x_min) at each end."""
    centroid_x = V[F].mean(axis=1)[:, 0]
    x_min, x_max = float(centroid_x.min()), float(centroid_x.max())
    band = band_frac * (x_max - x_min)
    left  = [i for i in range(F.shape[0]) if centroid_x[i] <= x_min + band]
    right = [i for i in range(F.shape[0]) if centroid_x[i] >= x_max - band]
    return left, right


def vertices_of(F: np.ndarray, face_ids: list[int]) -> list[int]:
    return sorted({int(v) for fid in face_ids for v in F[fid]})


# --- Twist target generator ------------------------------------------

def rotate_about_x(points: np.ndarray, theta: float, axis_yz: np.ndarray) -> np.ndarray:
    """Rotate `points` about the x-axis through `axis_yz = (y0, z0)`.

    Points rotate in the (y, z) plane: their x is unchanged; y and z
    rotate around the pivot `axis_yz` by angle `theta`.
    """
    c, s = np.cos(theta), np.sin(theta)
    out = points.copy()
    dy = points[:, 1] - axis_yz[0]
    dz = points[:, 2] - axis_yz[1]
    out[:, 1] = axis_yz[0] + c * dy - s * dz
    out[:, 2] = axis_yz[1] + s * dy + c * dz
    return out


# --- Main ------------------------------------------------------------

def run(
    *,
    backend: str,
    n_steps: int,
    turns: float,
    timestep: float,
    gamma: float,
    voxel_factor: float,
    viewer: bool = True,
    screenshot_dir: Path | None = None,
    show_hexes: bool = True,
    quiet: bool = False,
) -> int:
    trusty.check_capabilities("embedded")
    V, F = make_bar_trimesh(BAR_SIZE, BAR_RES,
                            origin=(0.0, -BAR_SIZE[1] / 2, -BAR_SIZE[2] / 2))
    Ly, Lz = BAR_SIZE[1], BAR_SIZE[2]
    voxel_size = min(Ly, Lz) / max(voxel_factor, 1e-9)
    left_faces, right_faces = select_end_faces(V, F)
    left_verts  = vertices_of(F, left_faces)
    right_verts = vertices_of(F, right_faces)
    if not left_verts or not right_verts:
        raise SystemExit("twisting_beam: end-face selection produced empty sets")

    if not quiet:
        print(f"[{backend}] surface: V={V.shape[0]} verts, F={F.shape[0]} tris")
        print(f"[{backend}] pinned ends: {len(left_verts)} left + "
              f"{len(right_verts)} right vertices, "
              f"{len(left_faces)} left + {len(right_faces)} right tris")

    # Material parameters
    E_modulus = 5e5
    nu        = 0.3

    world = trusty.World(backend=backend,
                         timestep=timestep,
                         gravity=(0.0, 0.0, 0.0),  # isolate the twist driver
                         newton=trusty.NewtonConfig(max_iters=80, tolerance=1e-4))
    # NeoHookeanBW: log(J) singularity stresses the inversion-free
    # initial-step-size filter under large rotations.
    mat = trusty.NeohookeanBW(youngs_modulus=E_modulus, poisson_ratio=nu)
    body  = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=voxel_size, material=mat, density=1000.0)
    if not quiet:
        print(f"[{backend}] voxel_size={voxel_size:.4f}  "
              f"(factor={voxel_factor:g}, ~{Ly/voxel_size:.1f} voxels across "
              f"the {Ly:.2f} m cross-section)")

    # Initial Nitsche targets = rest positions for *all* surface verts.
    # Only the rows referenced by pinned faces actually drive the term;
    # the others are irrelevant and we leave them at rest.
    pin_targets = V.copy()

    pin_faces = sorted(set(left_faces + right_faces))
    trusty.embedded.attach_weak_pin(
        world, body, pin_faces=pin_faces,
        initial_targets=pin_targets, gamma=gamma)

    # Pivot for the right-end rotation: the rest centroid of the right
    # pinned vertices, projected onto the (y, z) plane.
    right_pivot_yz = V[right_verts][:, 1:3].mean(axis=0)

    # Two structures: the smooth embedded surface (the user-facing
    # mesh) and the underlying hex volume mesh (the actual simulator
    # DOFs). Both deform in lockstep with the simulation, so you can
    # see voxel-staircase boundaries and how the twist propagates
    # through the bulk.
    headless = screenshot_dir is not None
    ps = init_polyscope(headless=headless) if (viewer or headless) else None
    ps_mesh = None
    ps_hex = None
    if ps is not None:
        ps_mesh = ps.register_surface_mesh("beam (embedded surface)", V, F)
        ps_mesh.set_smooth_shade(False)
        ps_mesh.set_edge_width(0.0)
        ps_mesh.set_color((0.45, 0.6, 0.85))

        if show_hexes:
            hex_mesh = trusty.fem.read_mesh(world, body)
            ps_hex = ps.register_volume_mesh(
                "voxel hexes",
                np.asarray(hex_mesh.vertices),
                hexes=np.asarray(hex_mesh.hexes))
            ps_hex.set_color((0.45, 0.6, 0.85))
            ps_hex.set_edge_width(1.0)
            ps_hex.set_transparency(0.55)

    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        # 3/4 side view: the 4 m beam runs roughly horizontally across
        # the frame with the twist visible. Camera is offset along both
        # -y and -x so the right-end rotation is clearly visible.
        cx = 0.5 * BAR_SIZE[0]
        ps.look_at(
            camera_location=(cx - 0.3 * BAR_SIZE[0],
                             -0.7 * BAR_SIZE[0],
                             0.5 * BAR_SIZE[0]),
            target=(cx, 0.0, 0.0))

    target_theta = 2 * np.pi * float(turns)
    history: list[float] = []  # observed twist angle at the right face
    state = {"i": 0, "playing": False, "failed": False}   # play starts OFF

    def advance() -> bool:
        """Ramp theta one step, drive the pin, measure the observed twist.

        False if Newton failed to converge, which stops the run.
        """
        i = state["i"]
        theta = ((i + 1) / n_steps) * target_theta

        # Left pinned verts stay at rest (already set); right pinned verts
        # rotate by theta about the x-axis through `right_pivot_yz`.
        new_targets = pin_targets.copy()
        new_targets[right_verts] = rotate_about_x(
            V[right_verts], theta, right_pivot_yz)

        trusty.embedded.set_pin_targets(world, body, new_targets)
        world.step()
        report = world.last_report()
        if not report.converged:
            print(f"[{backend}] step {i+1}/{n_steps} did NOT converge "
                  f"(residual={report.final_residual:.2e})")
            state["failed"] = True
            return False

        # Observed twist on the right face: mean angle change in (y, z)
        # relative to the pivot, unwrapped against the previous observation
        # so it keeps accumulating past +-pi.
        x_surf = np.asarray(trusty.embedded.read_embedded_surface(world, body))
        state["x_surf"] = x_surf
        rest = V[right_verts][:, 1:3] - right_pivot_yz
        cur  = x_surf[right_verts][:, 1:3] - right_pivot_yz
        rest_ang = np.arctan2(rest[:, 1], rest[:, 0])
        cur_ang  = np.arctan2(cur[:, 1],  cur[:, 0])
        d_ang = np.angle(np.exp(1j * (cur_ang - rest_ang)))
        observed = float(np.mean(d_ang))
        if history:
            jump = observed - history[-1]
            if jump > np.pi:
                observed -= 2 * np.pi
            elif jump < -np.pi:
                observed += 2 * np.pi
        history.append(observed)
        state["i"] = i + 1

        if not quiet and (i % max(1, n_steps // 20) == 0 or i == n_steps - 1):
            print(f"  [{backend}] step {i+1:4d}/{n_steps}  "
                  f"theta_target={theta:+.3f}  theta_obs={observed:+.3f}  "
                  f"newton={report.iterations:3d}  "
                  f"res={report.final_residual:.2e}")
        return True

    def refresh() -> None:
        if ps_mesh is None:
            return
        ps_mesh.update_vertex_positions(state["x_surf"])
        if ps_hex is not None:
            ps_hex.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, body).vertices))

    every = max(1, n_steps // 60)

    if screenshot_dir is not None:
        while state["i"] < n_steps:
            i = state["i"]
            if not advance():
                return 1
            if i % every == 0:
                refresh()
                ps.screenshot(
                    str(screenshot_dir / f"twisting_beam_{i:04d}.png"),
                    transparent_bg=False)
    elif ps is not None:
        state["x_surf"] = np.asarray(
            trusty.embedded.read_embedded_surface(world, body))
        refresh()

        def cb():
            import polyscope.imgui as psim
            _, state["playing"] = psim.Checkbox("play", state["playing"])
            psim.SameLine()
            stepped = psim.Button("step")
            if (stepped or state["playing"]) and state["i"] < n_steps \
                    and not state["failed"]:
                if advance():
                    refresh()
                else:
                    state["playing"] = False
            psim.Text(f"step {state['i']} / {n_steps}")
            psim.Text(f"twist target {(state['i'] / n_steps) * target_theta:+.3f} rad"
                      f"   observed {(history[-1] if history else 0.0):+.3f} rad")
            rep = world.last_report()
            psim.Text(f"iters={rep.iterations}  res={rep.final_residual:.2e}")
            if state["failed"]:
                psim.Text("Newton did not converge -- run stopped")

        ps.set_user_callback(cb)
        ps.show()
    else:
        while state["i"] < n_steps:
            if not advance():
                return 1

    # Final summary. In the viewer the window can be closed early, so this
    # reports whatever was actually simulated.
    if not history:
        print(f"[{backend}] no steps simulated")
        return 1 if state["failed"] else 0
    err = abs(history[-1] - target_theta) if state["i"] == n_steps else float("nan")
    print(f"[{backend}] target twist = {target_theta:+.3f} rad "
          f"({turns:.3f} turns)")
    print(f"[{backend}] observed twist at right face = {history[-1]:+.3f} rad "
          f"(error {err:+.3e} rad) after {state['i']}/{n_steps} steps")

    if state["failed"]:
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend",
                    choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    ap.add_argument("--steps", type=int, default=240,
                    help="Number of timesteps over which the twist ramps "
                         "from 0 to `--turns` revolutions.")
    ap.add_argument("--turns", type=float, default=1.0,
                    help="Total revolutions of the right end about the "
                         "beam axis.")
    ap.add_argument("--timestep", type=float, default=1.0 / 60.0)
    ap.add_argument("--gamma", type=float, default=80.0,
                    help="Nitsche stabilization parameter; higher = "
                         "stiffer pin (less tracking offset).")
    ap.add_argument("--voxel-factor", type=float,
                    default=DEFAULT_VOXEL_FACTOR,
                    help="Number of voxels across the bar's smallest "
                         "cross-section dimension. Larger -> finer hex "
                         "grid, slower but smoother. Default 4.")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Headless: write PNG screenshots via the EGL "
                         "polyscope backend instead of opening a window.")
    ap.add_argument("--out", type=Path, default=Path("out_twisting_beam"),
                    help="Output directory for --no-viewer screenshots.")
    ap.add_argument("--no-hexes", action="store_true",
                    help="Disable the voxel-hex overlay; show only the "
                         "embedded surface.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rc = run(
        backend=args.backend,
        n_steps=args.steps,
        turns=args.turns,
        timestep=args.timestep,
        gamma=args.gamma,
        voxel_factor=args.voxel_factor,
        viewer=not args.no_viewer,
        screenshot_dir=args.out if args.no_viewer else None,
        show_hexes=not args.no_hexes,
        quiet=args.quiet,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
