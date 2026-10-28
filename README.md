# trusty-sim

Examples and documentation for the `trusty-sim` Python package.

A physics simulation library for deformable solids, shells, rods, articulated
affine bodies, reduced-order (subspace) bodies and MPM, with IPC-style frictional
contact between them. You drive it from Python and step it with numpy arrays.

> **Pre-release test build (0.1.0a9).** The API will change between releases.
> macOS on Apple Silicon and Linux x86_64 are supported for now.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install trusty-sim==0.1.0a9
```

Optional extras:

- `pip install polyscope trimesh matplotlib` for the interactive viewer used
  by the examples.
- `pip install scipy` for building subspace bases.

With [uv](https://docs.astral.sh/uv/), use `uv venv` and replace `pip` with
`uv pip`. On Linux this installs the CPU build.

### GPU Installation

For the CUDA backend on Linux, install `trusty-sim-cuda12` in place of
`trusty-sim` (they replace each other: install one). You need an NVIDIA GPU and
a driver that supports CUDA 12.9 (version 575 or newer); the CUDA libraries come
from NVIDIA's wheels on PyPI, which pip installs for you:

```bash
python -m venv .venv && source .venv/bin/activate
pip install trusty-sim-cuda12==0.1.0a9
```

It contains GPU code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0,
10.0 and 12.0 (V100, T4 and RTX 20 series through H100, B200 and RTX 50 series).
Check it with `"cuda" in trusty.capabilities()`, then pass `backend="cuda"` to
`trusty.World`. The examples in this repository run on either build; `uv run`
installs the CPU build, so use the pip route for the GPU one.

## Requirements

| | |
|---|---|
| OS | macOS 11 or later, Apple Silicon (arm64); Linux x86_64, glibc 2.28 or later |
| Python | 3.11 or later (3.12 for the CUDA build) |
| numpy | 1.26 or later |

## Quick start

Drop a cantilever beam under gravity:

```python
import numpy as np
import trusty

world = trusty.World()
mesh = trusty.make_beam_hex_mesh(size=(1.0, 0.1, 0.1), res=(10, 2, 2))
beam = trusty.fem.add_hex_solid(
    world, mesh,
    trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3),
    density=1000.0,
)
trusty.fem.pin_face(world, beam, axis=0, coord=0.0)   # clamp the x = 0 end

for _ in range(60):
    world.step()

lowest_z = np.asarray(trusty.fem.read_positions(world, beam))[:, 2].min()
print(f"lowest point after 1 s: {lowest_z:.3f} m")   # 60 steps of the default 1/60 s
```

To see which modules this build includes:

```python
trusty.capabilities()   # {'fem', 'contact', 'affine', 'rods', 'shells', ...}
```

## Modules

| Module | What it simulates |
|---|---|
| `trusty.fem` | volumetric solids (hex and tet, linear and quadratic elements) |
| `trusty.contact` | frictional contact between everything, plus planes and walls |
| `trusty.boundary_conditions` | pins, stitches, attachments and surface loads |
| `trusty.affine` | articulated affine bodies, with joints, limits and actuators |
| `trusty.rods` | discrete elastic rods, including rod muscles |
| `trusty.shells` | thin shells and cloth |
| `trusty.embedded` | voxelized (embedded) elastic bodies |
| `trusty.subspace` | reduced-order bodies (modal and skinning eigenmodes) |
| `trusty.mpm` | material point method |

## Running the examples

Clone this repository, install trusty-sim as above, add the packages the
examples use, and run any of them from the repository root:

```bash
git clone https://github.com/tytrusty/trusty-docs
cd trusty-docs
pip install trimesh matplotlib scipy
python examples/contact/beam_drop.py
```

That opens the polyscope viewer. Most examples also run headless with
`--no-viewer`:

```bash
python examples/contact/beam_drop.py --no-viewer
```

With [uv](https://docs.astral.sh/uv/), skip the installs: `uv run
examples/contact/beam_drop.py` sets up an environment with trusty-sim, the
viewer and everything else the examples need the first time you use it.

| Folder | Examples |
|---|---|
| `examples/` | the cantilever beam, element-locking demo and a benchmark |
| `examples/fem/` | mesh types, material models, graded materials and stress visualization |
| `examples/contact/` | dropping onto a floor, friction on a ramp, contact exclusion, resting gaps and the contact observer |
| `examples/boundary_conditions/` | pins, stitches, attachments and surface loads |
| `examples/affine/` | articulated bodies, joints, limits, actuators and URDF loading |
| `examples/rods/` | elastic rods and rod muscles |
| `examples/shells/` | shells and cloth |
| `examples/embedded/` | voxelized (embedded) solids |
| `examples/subspace/` | reduced-order bodies and muscles |
| `examples/mpm/` | MPM dam break (fluid or elastic) and its benchmark |
| `examples/coupling/` | different body types interacting in one simulation, including the Trusty logo from the docs home page |

Each script's docstring describes the scene and its command-line options.

## Citing

If you use trusty-sim in your research, please cite it. The details are in
[CITATION.cff](CITATION.cff), and GitHub's "Cite this repository" button reads
it. As BibTeX:

```bibtex
@software{trusty_sim,
  author  = {Trusty, Ty},
  title   = {trusty-sim},
  version = {0.1.0a9},
  year    = {2026},
  url     = {https://tytrusty.github.io/trusty-docs/}
}
```

## License

### Examples

The examples in this repository are MIT-licensed. See [LICENSE](LICENSE).

### trusty-sim library

The `trusty-sim` library that the examples run on is available under dual
licensing. The full terms are on the
[License page](https://tytrusty.github.io/trusty-docs/license/) of the docs.

- **Research license:** free for research, academic and non-commercial use. The
  terms are in `LICENSE-BINARY.txt`, inside the installed package's
  `.dist-info/licenses/`.
- **Commercial license:** for proprietary or commercial deployment, please refer
  to [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).
