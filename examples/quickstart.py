"""Quickstart: a soft cantilever beam sagging under gravity.

The smallest complete TrustySim script: a hex-mesh beam, clamped at one
end, stepped for one second, then shown in the polyscope viewer.

Usage:
    python examples/quickstart.py
"""
import polyscope as ps
import trusty

mesh = trusty.make_beam_hex_mesh(size=(1.0, 0.1, 0.1), res=(20, 2, 2))
material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)

world = trusty.World(timestep=1.0 / 60.0)
beam = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
trusty.fem.pin_face(world, beam, axis=0, coord=0.0)  # clamp the x = 0 end

for _ in range(60):  # one second
    world.step()

ps.init()
ps.set_up_dir("z_up")
sagged = trusty.fem.read_mesh(world, beam)
ps.register_volume_mesh("beam", sagged.vertices, hexes=sagged.hexes)
ps.show()
