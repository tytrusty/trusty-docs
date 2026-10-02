"""All five body types coupled in ONE simulation.

affine (near-rigid) + fem (deformable hex) + embedded (voxelized) + shell +
subspace (reduced) -- all in one world, stacked into a single centered tower on
a shared floor with contact on, so every layer presses on the one
below. Every pair of body types interacts through the same contact model, in
one implicit solve.

The soft solids (fem / embedded / subspace) share E = 1e5 and at least 100
degrees of freedom each; the affine cube stays stiff (1e9) as the base of the stack.


Usage:
    python examples/coupling/all_bodies.py                       # viewer (opens paused)
    python examples/coupling/all_bodies.py --no-viewer           # headless
    python examples/coupling/all_bodies.py --backend accelerate  # Apple solver
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import trusty

FRICTION_MU = 0.5   # Coulomb coefficient of every body and the floor


def box_tris(center, half, n=1):
    """Axis-aligned cube surface, each face split into n x n quads with welded
    shared edges/corners and outward winding. n=1 is the plain 8-vertex /
    12-triangle box; larger n gives more surface samples so an embedded body's
    coupled surface can follow the hex deformation instead of staying faceted.
    """
    h = float(half)
    verts: list = []
    index: dict = {}

    def vid(i, j, k):
        key = (i, j, k)
        if key not in index:
            index[key] = len(verts)
            verts.append([-h + 2.0 * h * i / n,
                          -h + 2.0 * h * j / n,
                          -h + 2.0 * h * k / n])
        return index[key]

    tris: list = []

    def cell(p0, p1, p2, p3, outward):
        a, b, d = np.asarray(verts[p0]), np.asarray(verts[p1]), np.asarray(verts[p2])
        if np.dot(np.cross(b - a, d - a), outward) < 0.0:
            p1, p3 = p3, p1            # flip winding so the normal points out
        tris.append((p0, p1, p2))
        tris.append((p0, p2, p3))

    for axis in range(3):
        for side in (0, n):
            outward = np.zeros(3)
            outward[axis] = 1.0 if side == n else -1.0
            u, v = (a for a in range(3) if a != axis)
            for s in range(n):
                for t in range(n):
                    def g(su, sv, _a=axis, _s=side, _u=u, _v=v):
                        c = [0, 0, 0]
                        c[_a] = _s
                        c[_u] = su
                        c[_v] = sv
                        return vid(*c)
                    cell(g(s, t), g(s + 1, t), g(s + 1, t + 1), g(s, t + 1), outward)

    V = np.asarray(verts, dtype=np.float64) + np.asarray(center, dtype=np.float64)
    F = np.asarray(tris, dtype=np.int32)
    return V, F


def shifted_hex(size, res, translate):
    m = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(m.vertices).copy()
    V = V - V.min(axis=0) + np.asarray(translate, dtype=np.float64)
    return trusty.make_hex_mesh(V, np.asarray(m.hexes))


def square_sheet(side, res, center):
    cx, cy, cz = center
    xs = np.linspace(-side / 2, side / 2, res)
    V  = np.array([[x + cx, y + cy, cz] for y in xs for x in xs], dtype=np.float64)
    F  = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            F.append((a, a + 1, a + res + 1))
            F.append((a, a + res + 1, a + res))
    return V, np.asarray(F, dtype=np.int32)


def build_subspace_basis(tmp: Path, size, res, num_handles):
    from trusty.subspace.precompute import build_basis as bb, pack
    mesh  = trusty.make_beam_hex_mesh(size=size, res=res)
    mat   = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.40)
    world = trusty.World()
    trusty.fem.add_hex_solid(world, mesh, mat, density=1000.0)
    rv    = np.asarray(mesh.vertices, dtype=np.float64)
    hexes = np.asarray(mesh.hexes)
    basis = bb.build_skinning_eigenmodes(
        world=world, num_nodes=rv.shape[0], rest_positions=rv,
        num_handles=num_handles, mesh_hash=bb.mesh_hash_sha256(rv, hexes))
    path = tmp / "sub.basis"
    pack.save(path, basis)
    return trusty.subspace.load(str(path))


def build_world(tmp: Path, backend: str = "auto"):
    trusty.check_capabilities("contact", "affine", "embedded", "subspace", "shells")
    soft  = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.40)
    world = trusty.World(backend=backend,
                         timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=80))
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-3, kappa=1e6))
    bodies = {}

    half      = 0.08      # half-extent of each stacked box (0.16 m footprint)
    gap       = 0.006     # rest gap between layers (and above the floor)
    n_handles = 3         # subspace: 12 reduced degrees of freedom per handle

    # Everything stacks into one centered tower, bottom -> top, so each layer
    # presses on the one below. The stiff affine cube anchors the base; the
    # three soft solids sit on it; the shell drapes over the top.
    z = gap

    # affine cube (stiff base)
    cz = z + half
    Vc, Fc = box_tris((0.0, 0.0, cz), half)
    bodies["affine"] = trusty.affine.add_affine_body(
        world, Vc, Fc, density=1000.0, stiffness=1e9)
    z = cz + half + gap

    # fem box (E=1e5, res 3 -> 4*4*4 = 64 nodes)
    fem_mesh = shifted_hex((2 * half, 2 * half, 2 * half), (3, 3, 3), (-half, -half, z))
    bodies["fem"] = trusty.fem.add_hex_solid(world, fem_mesh, soft, density=1000.0)
    z += 2 * half + gap

    # embedded box (E=1e5). The hex grid (voxel_size) carries the deformation;
    # the subdivided box surface (n=4) samples that field so the coupled
    # surface bulges instead of staying faceted between 8 corners.
    ez = z + half
    Ve, Fe = box_tris((0.0, 0.0, ez), half, n=4)
    bodies["embedded"] = trusty.embedded.add_embedded_solid(
        world, Ve, Fe, voxel_size=0.05, material=soft, density=1000.0)
    z = ez + half + gap

    # subspace (reduced) box (E=1e5, 3 handles)
    sub_size = (2 * half, 2 * half, 0.12)
    sub_res  = (4, 4, 3)            # 5*5*4 = 100 nodes
    basis    = build_subspace_basis(tmp, sub_size, sub_res, n_handles)
    sub_mesh = shifted_hex(sub_size, sub_res, (-half, -half, z))
    bodies["subspace"] = trusty.subspace.add_hex_body(
        world, sub_mesh, soft, density=1000.0, basis=basis)
    z += 0.12 + gap

    # Shell bodies: a closed thin-shell CUBE sitting on the stack, plus the
    # flat cloth sheet draped over it. A world holds one shell body, so
    # both are concatenated into ONE shell body with two disconnected components
    # (cube + sheet) -- cross-component contact lets the cloth drape over the
    # cube, and each component relaxes toward its own rest shape (the cube's
    # 90-degree edge dihedrals keep it boxy).
    Vcube, Fcube   = box_tris((0.0, 0.0, z + half), half, n=4)   # ~98 boundary verts
    z += 2 * half + gap
    Vsheet, Fsheet = square_sheet(4 * half, 7, (0.0, 0.0, z + 0.01))
    Vshell = np.vstack([Vcube, Vsheet])
    Fshell = np.vstack([Fcube, Fsheet + Vcube.shape[0]]).astype(np.int32)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 5.0e5
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 1.5e-3
    sh_cfg.density        = 1.0e3
    bodies["shell"] = trusty.shells.add_shell(world, Vshell, Fshell, sh_cfg)

    # Friction keeps the stack standing. It is set per body (a contact uses
    # the average of the two sides' coefficients), and the floor carries its own.
    for body in bodies.values():
        trusty.contact.set_body_friction(world, body, FRICTION_MU)
    trusty.add_floor_plane(world, 0.0, friction_mu=FRICTION_MU)

    n_fem = np.asarray(fem_mesh.vertices).shape[0]
    n_emb = np.asarray(trusty.fem.read_mesh(world, bodies["embedded"]).vertices).shape[0]
    print(f"degrees of freedom per body: affine=12  fem={3 * n_fem}  embedded={3 * n_emb}  "
          f"subspace={12 * n_handles} (reduced)  shell={3 * Vshell.shape[0]} "
          f"(cube+cloth)")


    return world, bodies, sub_mesh


def run_headless(world, bodies, steps: int):
    print("All five body types in one simulation "
          "(affine + fem + embedded + shell + subspace).")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}  converged={r.converged}")
    print(f"Done. {diverged} non-converged steps over {steps}.")
    assert diverged == 0, f"{diverged} steps did not converge"
    centres = {
        "affine": np.asarray(trusty.affine.surface(world, bodies["affine"])[0]),
        "fem": np.asarray(trusty.fem.read_positions(world, bodies["fem"])),
        "embedded": np.asarray(trusty.embedded.read_embedded_surface(world, bodies["embedded"])),
        "subspace": np.asarray(trusty.subspace.deformed_positions(world, body=bodies["subspace"])),
    }
    off = {n: float(np.linalg.norm(x[:, :2].mean(axis=0))) for n, x in centres.items()}
    print("sideways offset of each layer (m): "
          + "  ".join(f"{n}={d:.3f}" for n, d in off.items()))
    assert max(off.values()) < 0.02, "the tower did not stay stacked"
    print("OK: affine + fem + embedded + shell + subspace coupled in one solve.")


def run_polyscope(world, bodies, sub_mesh, steps: int):
    try:
        import polyscope as ps
    except ImportError:
        print("polyscope not installed; running headless.")
        run_headless(world, bodies, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Floor at z = 0 (matches trusty.add_floor_plane).
    fx, fy = 1.2, 0.6
    floor_v = np.array(
        [[-fx, -fy, 0.0], [fx, -fy, 0.0], [fx, fy, 0.0], [-fx, fy, 0.0]],
        dtype=np.float64)
    floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ps.register_surface_mesh("floor", floor_v, floor_f, color=(0.85, 0.85, 0.85))

    # One renderable per body type, each read back through its own surface/mesh API.
    Va, Fa = trusty.affine.surface(world, bodies["affine"])
    ps_aff = ps.register_surface_mesh(
        "affine", np.asarray(Va), np.asarray(Fa), color=(0.85, 0.35, 0.30))

    fem_mesh = trusty.fem.read_mesh(world, bodies["fem"])
    ps_fem = ps.register_volume_mesh(
        "fem", np.asarray(fem_mesh.vertices).copy(),
        hexes=np.asarray(fem_mesh.hexes), color=(0.30, 0.55, 0.85))

    emb = bodies["embedded"]
    ps_emb = ps.register_surface_mesh(
        "embedded", np.asarray(trusty.embedded.read_embedded_surface(world, emb)),
        np.asarray(trusty.embedded.surface_triangles(world, emb)),
        color=(0.40, 0.75, 0.45), smooth_shade=False)
    # The hex grid behind the embedded surface (what deforms), shown by
    # default like the other embedded examples.
    emb_hex = trusty.fem.read_mesh(world, emb)
    ps_emb_hex = ps.register_volume_mesh(
        "embedded hex", np.asarray(emb_hex.vertices).copy(),
        hexes=np.asarray(emb_hex.hexes), color=(0.20, 0.50, 0.30))

    shell = bodies["shell"]
    ps_shell = ps.register_surface_mesh(
        "shell", np.asarray(trusty.shells.read_positions(world, shell)),
        np.asarray(trusty.shells.read_triangles(world, shell)),
        color=(0.90, 0.75, 0.25), smooth_shade=False)

    ps_sub = ps.register_volume_mesh(
        "subspace", np.asarray(sub_mesh.vertices).copy(),
        hexes=np.asarray(sub_mesh.hexes), color=(0.60, 0.45, 0.80))

    state = {"i": 0, "playing": False}  # opens paused -- step/play to advance

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_aff.update_vertex_positions(np.asarray(trusty.affine.surface(world, bodies["affine"])[0]))
            ps_fem.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, bodies["fem"]).vertices))
            ps_emb.update_vertex_positions(
                np.asarray(trusty.embedded.read_embedded_surface(world, emb)))
            ps_emb_hex.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, emb).vertices))
            ps_shell.update_vertex_positions(
                np.asarray(trusty.shells.read_positions(world, shell)))
            ps_sub.update_vertex_positions(
                np.asarray(trusty.subspace.deformed_positions(world, body=bodies["subspace"])))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless instead of launching the polyscope viewer")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        world, bodies, sub_mesh = build_world(Path(td), args.backend)
        if args.no_viewer:
            run_headless(world, bodies, args.steps)
        else:
            run_polyscope(world, bodies, sub_mesh, args.steps)


if __name__ == "__main__":
    main()
