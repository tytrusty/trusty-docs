"""Von Mises vs strain energy density on a block squeezed between two plates.

A soft, nearly incompressible block (E = 500 kPa, nu = 0.45) is clamped at its
bottom face and its top face is pushed down 30% by a moving pin. The clamped
faces cannot spread sideways, so the block bulges at mid-height and the stress
varies through it: highest in the bulging middle band, lowest under the plate
centres, where the material is squeezed almost equally from every side. The
two readouts measure different things:

  * **von Mises** is *deviatoric*: it discards the hydrostatic (pressure) part
    of the stress. It is the measure for yielding and failure.
  * **strain energy density** includes the volumetric term, so it also counts
    how hard the material is being squeezed.

Both take `nodal=` (True: a smooth per-node field; False: raw per-element
values) and `unit_material=` (evaluate at mu = lam = 1, a stiffness-independent
strain measure, so soft and stiff regions are directly comparable).

Usage:
    python examples/fem/visualize_stress.py              # live polyscope
    python examples/fem/visualize_stress.py --no-viewer  # print ranges
    python examples/fem/visualize_stress.py --squeeze 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

SIZE = 0.2                      # block edge (m)
RES = 10                        # hexes per edge
E, NU = 5.0e5, 0.45             # soft and nearly incompressible
PLATE_STIFFNESS = 1e10          # Pa; stiff enough to act as a rigid plate

FIELDS = ("von Mises (Pa)", "strain energy (J/m^3)",
          "von Mises (unit material)", "strain energy (unit material)")


def read_fields(world, body):
    """Every combination the two readouts offer, for one body."""
    vm, psi = trusty.fem.read_von_mises, trusty.fem.read_strain_energy_density
    return {
        FIELDS[0]: np.asarray(vm(world, body, nodal=True)),
        FIELDS[1]: np.asarray(psi(world, body, nodal=True)),
        FIELDS[2]: np.asarray(vm(world, body, nodal=True, unit_material=True)),
        FIELDS[3]: np.asarray(psi(world, body, nodal=True, unit_material=True)),
    }


class Press:
    """The block and the moving top plate that squeezes it."""

    def __init__(self, backend: str = "auto"):
        self.world = trusty.World(backend=backend,
                                  time_stepping="static",   # solve equilibrium each step
                                  gravity=(0.0, 0.0, 0.0),  # the plates are the only load
                                  newton=trusty.NewtonConfig(max_iters=60))

        mesh = trusty.make_beam_hex_mesh(size=(SIZE, SIZE, SIZE), res=(RES, RES, RES))
        self.mesh = mesh
        material = trusty.StableNeoHookean(youngs_modulus=E, poisson_ratio=NU)
        self.body = trusty.fem.add_hex_solid(self.world, mesh, material, density=1000.0)

        # Bottom face clamped; top face held by a pin whose targets move down.
        trusty.fem.pin_face(self.world, self.body, axis=2, coord=0.0)
        rest = np.asarray(mesh.vertices)
        self.top = np.flatnonzero(np.abs(rest[:, 2] - SIZE) < 1e-9)
        self.top_rest = rest[self.top].copy()
        self.pin = trusty.boundary_conditions.pin_to_target(
            self.world, self.body, self.top.tolist(), self.top_rest, PLATE_STIFFNESS)

    def squeeze_to(self, fraction: float):
        """Move the top plate down to `fraction` of the height and solve."""
        targets = self.top_rest.copy()
        targets[:, 2] -= fraction * SIZE
        trusty.boundary_conditions.set_pin_targets(self.world, self.pin, targets)
        self.world.step()

    def squeeze(self, fraction: float, steps: int):
        """Reach `fraction` in `steps` equal increments."""
        for k in range(1, steps + 1):
            self.squeeze_to(fraction * k / steps)


def report(press):
    width = max(len(f) for f in FIELDS)
    x = np.asarray(trusty.fem.read_positions(press.world, press.body))
    print(f"height {x[press.top, 2].mean():.3f} m (rest {SIZE} m), "
          f"width {x[:, 0].max() - x[:, 0].min():.3f} m (rest {SIZE} m)")
    for name, values in read_fields(press.world, press.body).items():
        print(f"  {name:<{width}}  min={values.min():10.4g}  max={values.max():10.4g}")


def run_viewer(press, fraction: float, steps: int):
    ps = init_polyscope(headless=False)
    m = ps.register_volume_mesh(
        "block", trusty.fem.read_positions(press.world, press.body),
        hexes=np.asarray(press.mesh.hexes))
    m.set_edge_width(1.0)

    def refresh():
        m.update_vertex_positions(trusty.fem.read_positions(press.world, press.body))
        for name, values in read_fields(press.world, press.body).items():
            m.add_scalar_quantity(name, values, defined_on="vertices",
                                  cmap="viridis", enabled=(name == FIELDS[0]))

    state = {"i": 0, "playing": False}
    refresh()
    ps.reset_camera_to_home_view()

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play (squeeze)", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["i"] += 1
            press.squeeze_to(fraction * state["i"] / steps)
            refresh()
        psim.Text(f"squeeze {100 * fraction * state['i'] / steps:.0f}% "
                  f"(step {state['i']} / {steps})")
        psim.Text("Switch fields under the block's Scalar Quantities.")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--squeeze", type=float, default=0.3,
                    help="fraction of the height the top plate moves down")
    ap.add_argument("--steps", type=int, default=10,
                    help="load increments to reach the full squeeze")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: squeeze fully, then print the field ranges")
    args = ap.parse_args()

    press = Press(args.backend)
    if args.no_viewer:
        press.squeeze(args.squeeze, args.steps)
        report(press)
        return
    run_viewer(press, args.squeeze, args.steps)


if __name__ == "__main__":
    main()
