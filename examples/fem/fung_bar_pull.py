"""Two StableFung bars with the SAME small-strain stiffness, different nonlinearity.

Both bars share E (so the same mu0) and the same linear hardening product
gamma*mu1 -- which fixes the small-strain stiffness AND the rest state -- so they
respond identically at small stretch. The second bar's larger mu1 makes its
exponential term e^(mu1*(I2-3)/2) ramp faster, so it stiffens sooner as it is
pulled. Control it with --nonlinearity. Opens a polyscope viewer (press "play");
--no-viewer writes PNG frames instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402

BAR_SIZE = (0.3, 0.1, 0.1)
BAR_RES  = (8, 3, 3)
E, NU    = 1.0e5, 0.45         # soft, near-incompressible (tissue-ish)
GAMMA     = 400.0            # shared exp scale, small -> mu0 sets the small-strain stiffness
MU1_LINEAR = 2.0             # bar 1: barely nonlinear (near neo-Hookean)
GAP       = 0.18             # y offset between the two bars (render only)
PULL_AMP  = 0.15             # peak +x pull of the far face (~50% strain)
PULL_PERIOD = 120            # steps per stretch/release cycle


def build_world(backend: str, mu1_pair):
    """Two bars pulled identically; mu1_pair = (linear mu1, nonlinear mu1)."""
    mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
    V = np.asarray(mesh.vertices)
    right = np.where(V[:, 0] > BAR_SIZE[0] - 1e-9)[0]
    right_rest = V[right].copy()

    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         gravity=(0.0, 0.0, 0.0),  # isolate the tensile response
                         time_stepping="static",
                         newton=trusty.NewtonConfig(max_iters=80))
    bars = []
    for mu1 in mu1_pair:
        gamma = GAMMA   # shared: mu0 dominates small strain; mu1 sets the nonlinearity
        material = trusty.StableFung(
            youngs_modulus=E, poisson_ratio=NU, mu1=mu1, gamma=gamma)
        body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
        trusty.fem.pin_face(world, body, axis=0, coord=0.0)
        pin = trusty.boundary_conditions.pin_to_target(
            world, body, right.tolist(), right_rest, stiffness=1.0e8)
        bars.append((f"fung  mu1={mu1:g}", body, pin))

    return world, bars, right, right_rest


def pull_targets(step: int, right_rest: np.ndarray):
    dx = PULL_AMP * 0.5 * (1.0 - np.cos(2 * np.pi * step / PULL_PERIOD))
    targets = right_rest.copy()
    targets[:, 0] += dx
    return targets


def axial_reaction(world, pin) -> float:
    """Axial tension the pin carries (N) = -sum of k*(x - t) over the face."""
    f = np.asarray(trusty.boundary_conditions.pin_forces(world, pin))
    return -float(f[:, 0].sum())


def _register_visuals(ps, world, bars):
    ps_meshes = []
    for i, (label, body, _) in enumerate(bars):
        m = trusty.fem.read_mesh(world, body)
        verts = np.asarray(m.vertices).copy()
        verts[:, 1] += i * GAP
        pm = ps.register_volume_mesh(label, verts, hexes=np.asarray(m.hexes))
        pm.set_edge_width(1.0)
        ps_meshes.append(pm)
    return ps_meshes


def _update_visuals(ps_meshes, world, bars):
    fields = [np.asarray(trusty.fem.read_von_mises(world, b, True))
              for _, b, _ in bars]
    hi = max(1.0, max(float(f.max()) for f in fields))
    for i, (pm, (_, body, _)) in enumerate(zip(ps_meshes, bars)):
        verts = np.asarray(trusty.fem.read_positions(world, body)).copy()
        verts[:, 1] += i * GAP
        pm.update_vertex_positions(verts)
        pm.add_scalar_quantity("von Mises (Pa)", fields[i], defined_on="vertices",
                               vminmax=(0.0, hi), cmap="viridis", enabled=True)


def run_polyscope(backend: str, mu1_pair):
    ps = init_polyscope(headless=False)
    world, bars, _, right_rest = build_world(backend, mu1_pair)
    ps_meshes = _register_visuals(ps, world, bars)
    state = {"i": 0, "playing": False}   # play starts OFF

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step") or state["playing"]:
            for _, _, pin in bars:
                trusty.boundary_conditions.set_pin_targets(
                    world, pin, pull_targets(state["i"], right_rest))
            world.step()
            state["i"] += 1
            _update_visuals(ps_meshes, world, bars)
        psim.Text(f"step {state['i']}")
        for label, _, pin in bars:
            psim.Text(f"{label}: axial reaction = {axial_reaction(world, pin):8.2f} N")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(backend: str, n_steps: int, out_dir: Path, mu1_pair):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    world, bars, _, right_rest = build_world(backend, mu1_pair)
    ps_meshes = _register_visuals(ps, world, bars)
    ps.reset_camera_to_home_view()

    for i in range(n_steps):
        for _, _, pin in bars:
            trusty.boundary_conditions.set_pin_targets(
                world, pin, pull_targets(i, right_rest))
        world.step()
        _update_visuals(ps_meshes, world, bars)
        ps.screenshot(str(out_dir / f"fung_bar_pull_{i:04d}.png"), transparent_bg=False)
    print(f"Wrote {n_steps} screenshots to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    ap.add_argument("--nonlinearity", type=float, default=18.0,
                    help="exponential hardening rate mu1 of the second bar; higher "
                         "stiffens faster at large stretch (both bars keep the same "
                         "small-strain stiffness). Default 18.")
    ap.add_argument("--steps", type=int, default=PULL_PERIOD)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out_fung_bar_pull"))
    args = ap.parse_args()

    mu1_pair = (MU1_LINEAR, args.nonlinearity)
    if args.no_viewer:
        run_screenshots(args.backend, args.steps, args.out, mu1_pair)
    else:
        run_polyscope(args.backend, mu1_pair)


if __name__ == "__main__":
    main()
