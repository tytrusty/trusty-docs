"""Headless MPM dam-break benchmark.

Profiles world.step() in isolation (no viewer, no screenshotting). Prints a
per-step wall time summary so we can compare runs.

Usage:
    python examples/bench_dam_break.py --column 10 10 10 --steps 60
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import numpy as np

import trusty
from utils import make_column, make_open_box


BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([1.0, 1.0, 1.0])


def parse_dt(s: str) -> float:
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


_BACKENDS = {
    "cpu":        trusty.Backend.CPU,
    "accelerate": trusty.Backend.ACCELERATE,
    "cuda":       trusty.Backend.CUDA,
}


def build_sim(args):
    caps = ("mpm", "contact")
    trusty.check_capabilities(*(("cuda",) + caps if args.backend == "cuda"
                                else caps))

    box_V, box_F = make_open_box(BOX_LO, BOX_HI)
    positions = make_column(
        *args.column,
        spacing=args.spacing,
        origin=tuple(args.origin),
        mode=args.sample,
        seed=args.seed,
        poisson_r_factor=args.poisson_r_factor,
    )
    print(f"[bench] sampled {len(positions)} particles "
          f"(mode={args.sample}, spacing={args.spacing})")
    particle_volume = args.spacing ** 3

    cfg = trusty.SimulatorConfig()
    cfg.backend           = _BACKENDS[args.backend]
    cfg.timestep          = args.dt
    cfg.newton.tolerance  = args.newton_tol
    cfg.newton.max_iters  = args.newton_iters

    if args.linear_solver == "pcg":
        cfg.newton.linear_solver = trusty.LinearSolverType.Pcg
        cfg.newton.pcg_max_iters = args.pcg_iters
        cfg.newton.pcg_tolerance = args.pcg_tol
    else:
        cfg.newton.linear_solver = trusty.LinearSolverType.Direct
    cfg.newton.sparsity_cache_enabled = args.sparsity_cache

    cfg.contact.enabled = True
    cfg.contact.kappa = 1e4

    mpm_cfg = trusty.mpm.MpmConfig()
    mpm_cfg.cell_size         = args.cell_size
    mpm_cfg.cdpi_domain_scale = 0.5
    mpm_cfg.reset_cdpi_domain = True
    cfg.mpm = mpm_cfg

    world = trusty.World(cfg)
    mat = trusty.mpm.MpmMaterial()
    mat.density = args.density
    mat.lam     = args.lam
    mat.mu      = args.mu
    trusty.mpm.add_mpm_particles(world, positions, particle_volume, mat)

    trusty.contact.add_wall(world, "dam_break_walls", box_V, box_F)
    if args.noslip:
        wcfg = trusty.mpm.WallNoSlipConfig()
        wcfg.beta       = args.beta
        wcfg.quad_order = args.quad_order
        trusty.mpm.attach_wall_noslip(world, wcfg)

    if args.observer:
        world.set_observer(trusty.LoggingObserver())
    return world


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "accelerate", "cuda"),
                   default="cpu")
    p.add_argument("--column", type=int, nargs=3, default=(10, 10, 10))
    p.add_argument("--origin", type=float, nargs=3, default=(0.05, 0.05, 0.10))
    p.add_argument("--spacing", type=float, default=0.05)
    p.add_argument("--sample", choices=("poisson", "uniform"), default="poisson")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--poisson-r-factor", type=float, default=1.0)
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--lam",     type=float, default=1e5)
    p.add_argument("--mu",      type=float, default=0.0)
    p.add_argument("--cell-size", type=float, default=0.1)
    p.add_argument("--dt", type=parse_dt, default=1.0 / 240.0)
    p.add_argument("--newton-iters", type=int,   default=25)
    p.add_argument("--newton-tol",   type=float, default=1e-4)
    p.add_argument("--no-noslip", dest="noslip", default=True, action="store_false")
    p.add_argument("--beta", type=float, default=1e7)
    p.add_argument("--quad-order", type=int, default=2, choices=(1, 2))
    p.add_argument("--linear-solver", choices=("direct", "pcg"), default="direct",
                   help="inner linear solver (default direct=Cholesky; "
                        "pcg=Jacobi-preconditioned CG)")
    p.add_argument("--pcg-iters", type=int, default=500,
                   help="max PCG iterations per Newton step")
    p.add_argument("--pcg-tol", type=float, default=1e-8,
                   help="PCG relative residual tolerance")
    p.add_argument("--no-sparsity-cache", dest="sparsity_cache",
                   default=True, action="store_false",
                   help="disable the linear-system sparsity caches")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=5,
                   help="steps to discard before timing")
    p.add_argument("--observer", action="store_true",
                   help="attach LoggingObserver (verbose, perturbs timing)")
    args = p.parse_args()

    world = build_sim(args)
    n_particles = len(np.asarray(trusty.mpm.read_particles(world)))

    print(f"[bench] warmup {args.warmup} steps, measuring {args.steps} steps")
    for _ in range(args.warmup):
        world.step()

    times: List[float] = []
    iters: List[int] = []
    for s in range(args.steps):
        t0 = time.perf_counter()
        world.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        iters.append(world.last_report().iterations)

    total_ms = sum(times)
    mean_ms  = statistics.mean(times)
    med_ms   = statistics.median(times)
    p95_ms   = sorted(times)[int(0.95 * (len(times) - 1))]
    print()
    print(f"[bench] backend={args.backend}  steps={args.steps}  "
          f"particles={n_particles}")
    print(f"[bench] total   {total_ms:>9.1f} ms")
    print(f"[bench] mean    {mean_ms:>9.2f} ms/step  ({1e3 / mean_ms:.2f} step/s)")
    print(f"[bench] median  {med_ms:>9.2f} ms/step")
    print(f"[bench] p95     {p95_ms:>9.2f} ms/step")
    print(f"[bench] iters   mean={statistics.mean(iters):.2f} "
          f"min={min(iters)} max={max(iters)}")


if __name__ == "__main__":
    main()
