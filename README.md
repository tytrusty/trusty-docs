# trusty-sim

Examples and documentation for the `trusty-sim` Python package.

A physics simulation library for deformable solids, shells, rods, articulated
affine bodies, reduced-order (subspace) bodies and MPM, with IPC-style frictional
contact between them. You drive it from Python and step it with numpy arrays.

> **Pre-release test build (0.1.0a2).** The API will change between releases, and
> only macOS on Apple Silicon is supported for now.

## Install

This build is published on TestPyPI only. Install numpy from PyPI first, then
trusty-sim from TestPyPI:

```bash
pip install numpy
pip install --no-deps --index-url https://test.pypi.org/simple/ trusty-sim==0.1.0a2
```

Optional extras:

- `pip install polyscope trimesh matplotlib` for the interactive viewer used by
  the examples.
- `pip install scipy` for building subspace bases.

## Requirements

| | |
|---|---|
| OS | macOS 11 or later, Apple Silicon (arm64) |
| Python | 3.11 or later |
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

The examples use the optional viewer, and the subspace examples need scipy:

```bash
pip install polyscope trimesh matplotlib scipy
python examples/beam_hex_drop.py              # opens the polyscope viewer
python examples/beam_hex_drop.py --no-viewer  # most examples also run headless
```

| Folder | Examples |
|---|---|
| `examples/` | FEM beams, dam break (MPM), benchmarks |
| `examples/fem/` | material models, graded materials and stress visualization |
| `examples/contact/` | inspecting contact through the contact observer |
| `examples/boundary_conditions/` | pins, stitches, attachments and surface loads |
| `examples/affine/` | articulated bodies, joints, limits, actuators and URDF loading |
| `examples/rods/` | elastic rods and rod muscles |
| `examples/shells/` | shells and cloth |
| `examples/embedded/` | voxelized (embedded) solids |
| `examples/subspace/` | reduced-order bodies and muscles |
| `examples/coupling/` | different body types interacting in one simulation |

Each script's docstring describes the scene and its command-line options.

## License

The examples and documentation in this repository, and the `trusty-sim`
binaries, are provided as a pre-release preview: all rights reserved. Licence
terms will be published with a later release. See `LICENSE`.
