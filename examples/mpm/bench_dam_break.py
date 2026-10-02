"""Time world.step() on the dam break, with no viewer.

Builds the world from dam_break.py (every one of its flags works here too),
runs a few warm-up steps, then times each step and prints a summary.

Usage:
    uv run examples/mpm/bench_dam_break.py
    uv run examples/mpm/bench_dam_break.py --steps 60 --spacing 0.025 --cell-size 0.05
    uv run examples/mpm/bench_dam_break.py --linear-solver pcg
    uv run examples/mpm/bench_dam_break.py --backend accelerate
    uv run examples/mpm/bench_dam_break.py --backend cuda
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dam_break  # noqa: E402


def main():
    p = dam_break.make_parser()
    p.description = __doc__.split("\n")[0]
    p.set_defaults(steps=60)
    p.add_argument("--warmup", type=int, default=5,
                   help="steps to run before timing (default %(default)s)")
    p.add_argument("--linear-solver", choices=("direct", "pcg"), default="direct",
                   help="linear solver inside each Newton iteration: direct "
                        "(Cholesky) or pcg (Jacobi-preconditioned CG)")
    p.add_argument("--pcg-iters", type=int, default=500,
                   help="PCG iteration cap (default %(default)s)")
    p.add_argument("--pcg-tol", type=float, default=1e-8,
                   help="PCG relative tolerance (default %(default)g)")
    args = p.parse_args()

    newton = trusty.NewtonConfig(
        tolerance=1e-4, max_iters=25,   # dam_break.py's own settings
        linear_solver=(trusty.LinearSolverType.Pcg if args.linear_solver == "pcg"
                       else trusty.LinearSolverType.Direct),
        pcg_max_iters=args.pcg_iters,
        pcg_tolerance=args.pcg_tol)

    world = dam_break.build_world(args, newton)
    n_particles = len(np.asarray(trusty.mpm.read_particles(world)))

    print(f"[bench] {args.warmup} warm-up steps, then timing {args.steps}")
    for _ in range(args.warmup):
        world.step()

    times, iters = [], []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        world.step()
        times.append((time.perf_counter() - t0) * 1e3)
        iters.append(world.last_report().iterations)

    mean_ms = statistics.mean(times)
    print(f"[bench] backend={args.backend}  particles={n_particles}  "
          f"steps={args.steps}")
    print(f"[bench] total   {sum(times):9.1f} ms")
    print(f"[bench] mean    {mean_ms:9.2f} ms/step  ({1e3 / mean_ms:.2f} steps/s)")
    print(f"[bench] median  {statistics.median(times):9.2f} ms/step")
    print(f"[bench] p95     {sorted(times)[int(0.95 * (len(times) - 1))]:9.2f} ms/step")
    print(f"[bench] Newton iterations: mean {statistics.mean(iters):.2f}, "
          f"min {min(iters)}, max {max(iters)}")


if __name__ == "__main__":
    main()
