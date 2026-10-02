"""Watch contact as it happens, with a contact observer.

One beam falls crosswise onto another, which rests on a floor plane. An
observer receives a report from the contact solver at every Newton iteration
and keeps what the viewer draws: the contact pairs in red and the vertices
touching the floor in green.

The observer's three hooks:

- `on_prepare(report)`   -- counts of candidate and active contact pairs,
                            and the contact energy.
- `on_step_size(report)` -- the step the solver may take before anything
                            would pass through anything else.
- `wants_stencils()`     -- return True to also get `report.stencils`: the
                            geometry of every pair, for drawing. It costs a
                            copy, so it is empty otherwise; the viewer's
                            checkbox turns it on and off.

Usage:
    uv run examples/contact/observer.py                # polyscope
    uv run examples/contact/observer.py --no-viewer    # headless
    uv run examples/contact/observer.py --steps 200
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def _segments(groups):
    """One curve-network segment per pair, drawn from the centre of one
    primitive to the centre of the other. `groups` is a list of
    (pairs, ends_a, ends_b)."""
    ends = [np.stack([a[p[:, 0]], b[p[:, 1]]], axis=1)
            for p, a, b in groups if p.size]
    if not ends:
        return None, None
    V = np.concatenate(ends, axis=0).reshape(-1, 3)
    return V, np.arange(len(V), dtype=np.int32).reshape(-1, 2)


class ContactViewObserver(trusty.contact.ContactObserver):
    """Keeps the latest report from each hook, in a form the viewer can draw."""

    def __init__(self, want_stencils: bool = True):
        super().__init__()               # required
        self.want_stencils = want_stencils
        self.n_prepare = 0
        self.last_prepare = None
        self.last_step_size = None
        self.accepted = (None, None)     # contact pairs, as a curve network
        self.candidates = (None, None)   # pairs that are close, but not yet in contact
        self.plane_verts = None          # vertices touching a plane

    def wants_stencils(self):
        return self.want_stencils

    def on_step_size(self, report):
        self.last_step_size = report

    def on_prepare(self, report):
        self.n_prepare += 1
        self.last_prepare = report
        if np.asarray(report.stencils.X).size:
            self.keep_stencils(report.stencils)

    def keep_stencils(self, st):
        # The report is only valid during the call, so copy what we keep.
        X = np.asarray(st.X).copy()           # contact-surface vertices
        E, F = np.asarray(st.E), np.asarray(st.F)
        edge_mid = X[E].mean(axis=1) if E.size else np.zeros((0, 3))
        face_mid = X[F].mean(axis=1) if F.size else np.zeros((0, 3))
        self.accepted = _segments([
            (np.asarray(st.active_vv), X,        X),         # vertex-vertex
            (np.asarray(st.active_ev), edge_mid, X),         # edge-vertex
            (np.asarray(st.active_ee), edge_mid, edge_mid),  # edge-edge
            (np.asarray(st.active_fv), face_mid, X)])        # face-vertex
        pv = np.asarray(st.active_pv)                        # vertex indices
        self.plane_verts = X[pv] if pv.size else None
        self.candidates = _segments([
            (np.asarray(st.cand_ee), edge_mid, edge_mid),
            (np.asarray(st.cand_fv), face_mid, X)])


def _beam(size, res, offset, crosswise: bool):
    """Hex beam centred on the origin, turned a quarter turn about z if asked,
    then moved to `offset`. A rotation, not an axis swap: swapping axes would
    turn every element inside out."""
    mesh = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(mesh.vertices, dtype=np.float64).copy()
    V -= V.mean(axis=0)
    if crosswise:
        V[:, 0], V[:, 1] = -V[:, 1].copy(), V[:, 0].copy()
    V += np.asarray(offset, dtype=np.float64)
    return trusty.make_hex_mesh(V, np.asarray(mesh.hexes, dtype=np.int32))


def build_sim(backend: str, want_stencils: bool):
    trusty.check_capabilities("contact")

    size, res = (0.6, 0.2, 0.15), (6, 2, 2)
    world = trusty.World(backend=backend,
                         timestep=1.0 / 120.0,
                         newton=trusty.NewtonConfig(max_iters=50))

    obs = ContactViewObserver(want_stencils)
    trusty.contact.enable(world, trusty.contact.Config(dhat=4e-3), observer=obs)

    material = trusty.StableNeoHookean(youngs_modulus=5e5, poisson_ratio=0.3)
    lower = trusty.fem.add_hex_solid(
        world, _beam(size, res, (0.0, 0.0, 0.20), False), material, 1000.0)
    upper = trusty.fem.add_hex_solid(
        world, _beam(size, res, (0.0, 0.0, 0.50), True), material, 1000.0)
    trusty.add_floor_plane(world, 0.0)
    return world, obs, (lower, upper)


def _counts_line(obs) -> str:
    r = obs.last_prepare
    if r is None:
        return "no report yet"
    return (f"candidates pt={r.n_pt_broadphase} ee={r.n_ee_broadphase} "
            f"pv={r.n_pv_broadphase} | active vv={r.n_vv_active} "
            f"ev={r.n_ev_active} ee={r.n_ee_active} fv={r.n_fv_active} "
            f"pv={r.n_pv_active} | energy={r.barrier_energy:.3e}")


def _step_line(obs) -> str:
    s = obs.last_step_size
    if s is None:
        return "no step-size report yet"
    return f"largest safe step: planes {s.alpha_pv:.3e}, overall {s.alpha_min:.3e}"


def run_headless(world, obs, steps: int):
    print(f"Running {steps} steps headless...")
    for i in range(steps):
        world.step()
        if (i + 1) % 20 == 0:
            print(f"  step {i + 1:4d}  {_counts_line(obs)}")
            print(f"              {_step_line(obs)}")
    n_pairs = 0 if obs.accepted[0] is None else len(obs.accepted[0]) // 2
    print(f"Done. on_prepare ran {obs.n_prepare} times; {n_pairs} contact "
          f"pairs at the last iteration.")


def run_polyscope(world, obs, bodies, steps: int):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless. "
              "(`pip install polyscope`, or use --no-viewer)")
        run_headless(world, obs, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    volumes = []
    for name, body in zip(("lower", "upper"), bodies):
        mesh = trusty.fem.read_mesh(world, body)
        v = ps.register_volume_mesh(name,
                                    np.asarray(mesh.vertices).copy(),
                                    hexes=np.asarray(mesh.hexes))
        v.set_color((0.55, 0.7, 0.9) if name == "lower" else (0.95, 0.6, 0.35))
        v.set_transparency(0.6)
        volumes.append((body, v))

    state = {"i": 0, "playing": False, "candidates": False}

    def refresh():
        for body, v in volumes:
            v.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, body).vertices))

        V, E = obs.accepted
        if V is not None:
            ps.register_curve_network("contact pairs", V, E, radius=0.0025,
                                      color=(0.95, 0.2, 0.2), enabled=True)
        elif ps.has_curve_network("contact pairs"):
            ps.get_curve_network("contact pairs").set_enabled(False)

        V, E = obs.candidates
        if state["candidates"] and V is not None:
            ps.register_curve_network("candidates", V, E, radius=0.0008,
                                      color=(0.6, 0.6, 0.6), enabled=True)
        elif ps.has_curve_network("candidates"):
            ps.get_curve_network("candidates").set_enabled(False)

        if obs.plane_verts is not None:
            ps.register_point_cloud("on the floor", obs.plane_verts,
                                    radius=0.006, color=(0.2, 0.9, 0.3),
                                    enabled=True)
        elif ps.has_point_cloud("on the floor"):
            ps.get_point_cloud("on the floor").set_enabled(False)

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            refresh()

    def callback():
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance()
        elif state["playing"]:
            advance()

        # wants_stencils() is asked every iteration, so this takes effect on
        # the next step; it costs nothing while off.
        changed, obs.want_stencils = psim.Checkbox("fill report.stencils",
                                                   obs.want_stencils)
        if changed and not obs.want_stencils:
            obs.candidates = obs.accepted = (None, None)
            obs.plane_verts = None
            refresh()
        _, state["candidates"] = psim.Checkbox("show candidate pairs",
                                               state["candidates"])

        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'NOT converged'}")
        psim.Text(_counts_line(obs))
        psim.Text(_step_line(obs))
        psim.Text(f"on_prepare x{obs.n_prepare}")

    refresh()
    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"],
                   default="auto")
    p.add_argument("--no-stencils", action="store_true",
                   help="start with report.stencils turned off")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, obs, bodies = build_sim(args.backend, not args.no_stencils)
    if args.no_viewer:
        run_headless(world, obs, args.steps)
    else:
        run_polyscope(world, obs, bodies, args.steps)


if __name__ == "__main__":
    main()
