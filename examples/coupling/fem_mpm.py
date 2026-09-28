"""fem x mpm contact -- an MPM particle blob dropped onto a FEM slab.

A deformable FEM slab rests on a floor plane, and an MPM particle body is
dropped onto it. Contact covers:

  - FEM slab vs floor
  - MPM particles vs floor
  - MPM particles vs the FEM slab's surface

MPM particles do not collide with each other through contact; the MPM
material model handles that. Runs on the CPU and CUDA backends.


Usage:
    python examples/coupling/fem_mpm.py                     # polyscope viewer
    python examples/coupling/fem_mpm.py --backend cuda      # on CUDA
    python examples/coupling/fem_mpm.py --no-viewer         # headless self-check
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def shifted_hex_mesh(size, res, translate):
    """A structured slab mesh translated to a target base corner."""
    m = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(m.vertices).copy()
    V = V - V.min(axis=0)                      # base corner at origin
    V = V + np.asarray(translate, dtype=np.float64)
    return trusty.make_hex_mesh(V, np.asarray(m.hexes))


def build_world(backend: str):
    trusty.check_capabilities("contact", "mpm")

    floor_z = 0.0
    cell    = 0.05

    cfg = trusty.SimulatorConfig()
    cfg.backend          = backend
    cfg.timestep         = 0.005
    cfg.newton.max_iters = 80
    cfg.contact.enabled  = True
    cfg.contact.dhat     = 2e-3

    mpm_cfg = trusty.mpm.MpmConfig()
    mpm_cfg.cell_size         = cell
    mpm_cfg.cdpi_domain_scale = 0.25
    cfg.mpm = mpm_cfg

    world = trusty.World(cfg)

    # FEM slab resting just above the floor (non-penetrating start).
    slab_mesh = shifted_hex_mesh((0.8, 0.8, 0.15), (4, 4, 1),
                                 (-0.4, -0.4, floor_z + 0.005))
    mat  = trusty.StableNeoHookean(youngs_modulus=3e5, poisson_ratio=0.40)
    slab = trusty.fem.add_hex_solid(world, slab_mesh, mat, density=1000.0)

    # One MPM blob well above the slab (gap >> dhat). A soft, near-fluid
    # material so it splats rather than bounces.
    pts = trusty.generate_particle_grid(
        lo=[-0.15, -0.15, floor_z + 0.45],
        hi=[0.15, 0.15, floor_z + 0.70],
        spacing=cell * 0.5, mode="uniform")
    mmat = trusty.mpm.MpmMaterial()
    mmat.density = 1000.0
    mmat.mu      = 1e3
    mmat.lam     = 1e4
    particle_volume = (cell * 0.5) ** 3
    trusty.mpm.add_mpm_particles(world, pts, particle_volume, mmat)

    trusty.add_floor_plane(world, floor_z)


    return world, slab, floor_z


def slab_min_z(world, slab) -> float:
    return float(np.asarray(trusty.fem.read_mesh(world, slab).vertices)[:, 2].min())


def particle_min_z(world) -> float:
    P = np.asarray(trusty.mpm.read_particles(world))
    return float(P[:, 2].min()) if P.size else float("inf")


def run_headless(world, slab, floor_z, steps: int):
    print(f"fem x mpm contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}  "
                  f"slab_z={slab_min_z(world, slab):+.4f}  "
                  f"part_z={particle_min_z(world):+.4f}")
    slab_z = slab_min_z(world, slab)
    part_z = particle_min_z(world)
    print(f"Done. slab min_z={slab_z:+.4f}, particle min_z={part_z:+.4f}, "
          f"{diverged} non-converged steps.")
    # Nothing tunnels through the floor (allow a small dhat-scale margin).
    assert slab_z > floor_z - 0.02, "FEM slab fell through the floor"
    assert part_z > floor_z - 0.02, "MPM particles fell through the floor"
    assert diverged == 0, f"{diverged} steps did not converge"
    print("OK: stable fem<->mpm contact.")


def run_polyscope(world, slab, floor_z, steps: int):
    try:
        import polyscope as ps
    except ImportError:
        print("polyscope not installed; running headless.")
        run_headless(world, slab, floor_z, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Floor at floor_z (matches trusty.add_floor_plane).
    fx = fy = 0.7
    floor_v = np.array(
        [[-fx, -fy, floor_z], [fx, -fy, floor_z],
         [fx, fy, floor_z], [-fx, fy, floor_z]], dtype=np.float64)
    floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ps.register_surface_mesh("floor", floor_v, floor_f, color=(0.85, 0.85, 0.85))

    slab_mesh = trusty.fem.read_mesh(world, slab)
    ps_slab = ps.register_volume_mesh(
        "fem slab", np.asarray(slab_mesh.vertices).copy(),
        hexes=np.asarray(slab_mesh.hexes), color=(0.30, 0.55, 0.85))

    X0 = np.asarray(trusty.mpm.read_particles(world))
    pcloud = ps.register_point_cloud("mpm particles", X0,
                                     radius=0.008,
                                     point_render_mode="sphere")
    pcloud.add_scalar_quantity("z", X0[:, 2], enabled=True)

    state = {"i": 0, "playing": False}  # opens paused -- step/play to advance

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        step10 = psim.Button("step+10")
        if (psim.Button("step") or state["playing"] or step10) \
                and state["i"] < steps:
            for _ in range(10 if step10 else 1):
                if state["i"] >= steps:
                    break
                world.step()
                state["i"] += 1
            ps_slab.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, slab).vertices))
            X = np.asarray(trusty.mpm.read_particles(world))
            pcloud.update_point_positions(X)
            pcloud.add_scalar_quantity("z", X[:, 2], enabled=True)

        rep = world.last_report()
        psim.Text(
            f"step {state['i']} / {steps}   "
            f"slab_z={slab_min_z(world, slab):+.4f}  "
            f"part_z={particle_min_z(world):+.4f}   "
            f"iters={rep.iterations}  res={rep.final_residual:.2e}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--backend", choices=["cpu", "accelerate", "cuda"],
                   default="cpu")
    p.add_argument("--no-viewer", action="store_true",
                   help="run the headless non-penetration self-check instead "
                        "of launching the polyscope viewer")
    args = p.parse_args()
    world, slab, floor_z = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, slab, floor_z, args.steps)
    else:
        run_polyscope(world, slab, floor_z, args.steps)


if __name__ == "__main__":
    main()
