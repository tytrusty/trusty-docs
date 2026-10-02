"""Convergence + runtime benchmark for the FEM element families.

For each discretization (Q1/P1 linear, P2 tet, Q2 serendipity/Lagrange hex),
sweep the corner-mesh resolution and report:
  - ACCURACY: the static tip deflection under gravity (a single static
    Newton solve -- the true equilibrium, mesh-convergent), and its absolute
    error in metres vs a fine higher-order ground truth (Q2 Lagrange).
  - RUNTIME: the mean dynamic Newton-solve wall time per step (ms), measured
    separately under the chosen mass model.

Higher-order families reach the converged deflection at far fewer DOFs than the
linear ones (which lock), at a higher per-step cost -- the accuracy/cost
trade-off. The mass model does NOT affect the static equilibrium; it only
changes the dynamic per-step cost, so `--lumped` vs the default consistent mass
differs only in the runtime panels.

Usage:
    uv run examples/bench_cantilever.py
    uv run examples/bench_cantilever.py --levels 2 4 8 --linear-levels 4 8 16 32 48
    uv run examples/bench_cantilever.py --lumped
    uv run examples/bench_cantilever.py --plot out/bench.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import trusty

ORDER = trusty.fem.ElementOrder

# label -> (mesh kind, element order)
FAMILIES = {
    "hex_q1":             ("hex", ORDER.Linear),
    "tet_p1":             ("tet", ORDER.Linear),
    "tet_p2":             ("tet", ORDER.Quadratic),
    "hex_q2_serendipity": ("hex", ORDER.QuadraticSerendipity),
    "hex_q2":             ("hex", ORDER.Quadratic),
}

# Linear families need far more resolution to reach quadratic-level accuracy,
# so they get their own (finer) sweep -- otherwise their error/node curves
# never overlap the quadratic ones.
LINEAR = {"hex_q1", "tet_p1"}

SIZE = (1.0, 0.1, 0.1)


def _make_body(family, nx, world_kwargs, consistent_mass=True):
    kind, order = FAMILIES[family]
    res = (nx, max(1, nx // 8), max(1, nx // 8))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    world = trusty.World(**world_kwargs)
    if kind == "hex":
        mesh = trusty.make_beam_hex_mesh(size=SIZE, res=res)
        body = trusty.fem.add_hex_solid(world, mesh, material, 1000.0, order=order,
                                        consistent_mass=consistent_mass)
    else:
        mesh = trusty.make_beam_tet_mesh(size=SIZE, res=res)
        body = trusty.fem.add_tet_solid(world, mesh, material, 1000.0, order=order,
                                        consistent_mass=consistent_mass)
    trusty.fem.pin_face(world, body, axis=0, coord=0.0)
    return world, body


def run_one(family, nx, backend, consistent, timing_steps):
    # --- accuracy: static equilibrium (single static Newton solve) ---
    base = dict(backend=backend, timestep=1.0 / 60.0)
    world, body = _make_body(family, nx, dict(base, time_stepping="static"))
    world.step()
    converged = world.last_report().converged
    verts = np.asarray(trusty.fem.read_positions(world, body))
    tip = verts[:, 0] > verts[:, 0].max() - 1e-3
    deflection = float(-verts[tip, 2].max())

    # --- runtime: dynamic per-step cost under the chosen mass model ---
    world2, body2 = _make_body(family, nx, base, consistent_mass=consistent)
    world2.step()  # warm (first solve pays setup)
    t0 = time.perf_counter()
    for _ in range(timing_steps):
        world2.step()
    ms_per_step = 1e3 * (time.perf_counter() - t0) / timing_steps

    return {"nodes": verts.shape[0], "deflection": deflection,
            "converged": converged, "ms": ms_per_step}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="length-resolution sweep for the QUADRATIC families")
    parser.add_argument("--linear-levels", type=int, nargs="+", default=[4, 8, 16, 32, 48],
                        help="finer sweep for the LINEAR (Q1/P1) families so their error "
                             "curves reach the quadratic range")
    parser.add_argument("--timing-steps", type=int, default=15,
                        help="dynamic steps timed for the per-step runtime measurement")
    parser.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--lumped", action="store_true",
                        help="use lumped mass for the dynamic runtime (default: consistent). "
                             "Does not affect the static accuracy.")
    parser.add_argument("--ref-nx", type=int, default=None,
                        help="length-resolution of the fine hex_q2 static ground truth "
                             "(default: 3x the finest swept level)")
    parser.add_argument("--plot", type=str, default=None, metavar="PNG",
                        help="also write error/runtime plots vs #nodes and vs runtime to PNG")
    args = parser.parse_args()
    consistent = not args.lumped
    levels_for = lambda family: args.linear_levels if family in LINEAR else args.levels

    # Ground truth: a single high-resolution, higher-order (Q2 Lagrange) static
    # solve, finer than any swept level. Error is the ABSOLUTE tip-deflection
    # difference to it, in metres.
    ref_nx = args.ref_nx if args.ref_nx is not None else 3 * max(args.levels)
    print(f"Computing static ground truth: hex_q2 at nx={ref_nx} ...")
    ref = run_one("hex_q2", ref_nx, args.backend, consistent, args.timing_steps)
    ref_defl = ref["deflection"]

    results = {f: {} for f in FAMILIES}
    for family in FAMILIES:
        for nx in levels_for(family):
            results[family][nx] = run_one(family, nx, args.backend, consistent, args.timing_steps)

    mass = "consistent" if consistent else "lumped"
    print(f"\nCantilever {SIZE}: static accuracy, {mass}-mass dynamic runtime, "
          f"backend={args.backend}")
    print(f"static ground-truth tip deflection (hex_q2, nx={ref_nx}, {ref['nodes']} nodes): "
          f"{ref_defl:.6e} m\n")
    print(f"{'family':<20} {'nx':>4} {'nodes':>7} {'deflection':>13} "
          f"{'abs.err (m)':>12} {'ms/step':>9} {'conv':>5}")
    for family in FAMILIES:
        for nx in levels_for(family):
            r = results[family][nx]
            print(f"{family:<20} {nx:>4} {r['nodes']:>7} "
                  f"{r['deflection']:>13.5e} {abs(r['deflection'] - ref_defl):>12.3e} "
                  f"{r['ms']:>9.2f} {str(r['converged']):>5}")
        print()

    if args.plot:
        make_plot(results, ref_defl, args, consistent, levels_for)


def make_plot(results, ref_defl, args, consistent, levels_for):
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, (ax_err, ax_time, ax_pareto) = plt.subplots(1, 3, figsize=(18, 5.2))
    for family in FAMILIES:
        nodes, errs, times = [], [], []
        for nx in levels_for(family):
            r = results[family][nx]
            nodes.append(r["nodes"])
            errs.append(abs(r["deflection"] - ref_defl))
            times.append(r["ms"])
        ax_err.plot(nodes, errs, "o-", label=family)
        ax_time.plot(nodes, times, "o-", label=family)
        ax_pareto.plot(times, errs, "o-", label=family)

    ax_err.set(xscale="log", yscale="log", xlabel="# nodes",
               ylabel="static tip-deflection error (m)", title="Accuracy vs resolution")
    ax_time.set(xscale="log", yscale="log", xlabel="# nodes", ylabel="dynamic ms / step",
                title="Runtime vs resolution")
    ax_pareto.set(xscale="log", yscale="log", xlabel="dynamic ms / step",
                  ylabel="static tip-deflection error (m)",
                  title="Cost vs accuracy (lower-left is better)")
    for ax in (ax_err, ax_time, ax_pareto):
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()

    mass = "consistent" if consistent else "lumped"
    fig.suptitle(f"Cantilever: static accuracy + {mass}-mass dynamic runtime "
                 f"(backend={args.backend})")
    fig.tight_layout()
    if os.path.dirname(args.plot):
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    fig.savefig(args.plot, dpi=120)
    print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
