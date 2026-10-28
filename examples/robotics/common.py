"""Shared run harness for the Franka examples: command-line flags, step statistics, viewer loops."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402


def add_standard_args(p: argparse.ArgumentParser, *, steps: int | None, dt: float) -> None:
    p.add_argument("--steps", type=int, default=steps,
                   help="steps to run (default: the whole script)")
    p.add_argument("--dt", type=float, default=dt, help="timestep (s)")
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--no-viewer", action="store_true", help="run headless and print a summary")
    p.add_argument("--record", type=Path, default=None,
                   help="render offscreen to this .mp4 / .gif (needs ffmpeg)")
    p.add_argument("--fps", type=int, default=30, help="--record frame rate")
    p.add_argument("--camera", type=float, nargs=6, default=None,
                   metavar=("EX", "EY", "EZ", "TX", "TY", "TZ"),
                   help="camera eye and target (m), overriding the example's")


class Recorder:
    """Offscreen polyscope frames written to an .mp4 (h264) or .gif with ffmpeg."""

    def __init__(self, path: Path, fps: int = 30):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--record needs ffmpeg on PATH")
        self.path, self.fps = Path(path), fps
        self.dir = tempfile.mkdtemp(prefix="trusty_frames_")
        self.n = 0

    def frame(self, ps) -> None:
        ps.screenshot(os.path.join(self.dir, f"f{self.n:05d}.png"), transparent_bg=False)
        self.n += 1

    def finish(self) -> None:
        src = os.path.join(self.dir, "f%05d.png")
        if self.path.suffix == ".gif":
            vf = "fps={},scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i", src,
                   "-vf", vf.format(self.fps), str(self.path)]
        else:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i", src,
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", str(self.path)]
        subprocess.run(cmd, check=True)
        shutil.rmtree(self.dir, ignore_errors=True)
        print(f"wrote {self.path} ({self.n} frames)")


def follow_tcp(controller, offset, anchor, *, gain=0.5, z_max=None, alpha=0.05):
    """`run(follow=...)` camera aiming at a low-passed `anchor + gain * (tcp - anchor)`.

    @param offset (3,) eye minus aim point (m)
    @param anchor (3,) fixed point the aim is pulled toward (m)
    """
    offset, anchor = np.asarray(offset, float), np.asarray(anchor, float)
    state = {"aim": None}

    def follow():
        p = controller.pose()[:3, 3].copy()
        if z_max is not None:
            p[2] = min(p[2], z_max)
        p = anchor + gain * (p - anchor)
        a = state["aim"]
        state["aim"] = p if a is None else (1.0 - alpha) * a + alpha * p
        return tuple(state["aim"] + offset), tuple(state["aim"])

    return follow


def register(ps, args, meshes, setup_scene=None, camera=None, follow=None):
    """Put the scene in polyscope; returns (`update()` for the bodies' new positions, `follow`).

    @param meshes   callable -> [(name, V, F, rgb)] current render geometry
    @param follow   callable -> ((eye), (target)) per recorded frame; None keeps `camera`
    """
    handles = {name: ps.register_surface_mesh(name, V, F, color=rgb, smooth_shade=False)
               for name, V, F, rgb in meshes()}
    if setup_scene is not None:
        setup_scene(ps)
    if args.camera is not None:
        camera, follow = (args.camera[:3], args.camera[3:]), None
    if camera is not None:
        ps.look_at(*camera)

    def update():
        for name, V, _F, _rgb in meshes():
            handles[name].update_vertex_positions(V)

    return update, follow


def run(args, *, step, meshes, gui_text=None, camera=None, setup_scene=None, follow=None,
        interact=None, play=False) -> None:
    """Drive `step()` `args.steps` times, headless, recorded, or in the viewer.

    `args.steps` None runs the viewer until it is closed.

    @param meshes    callable -> [(name, V, F, rgb)] current render geometry
    @param follow    callable -> ((eye), (target)) per recorded frame; None keeps `camera`
    @param interact  callable(ps, psim) run every viewer frame, for mouse and extra widgets
    @param play      whether the viewer starts running instead of paused
    """
    if args.no_viewer and args.record is None:
        for _ in range(args.steps):
            step()
        return

    ps = init_polyscope(headless=args.record is not None)
    ps.set_ground_plane_mode("shadow_only")
    update, follow = register(ps, args, meshes, setup_scene, camera, follow)

    if args.record is not None:
        rec = Recorder(args.record, args.fps)
        every = max(1, round(1.0 / (args.fps * args.dt)))

        def frame():
            if follow is not None:
                ps.look_at(*follow())
            rec.frame(ps)

        frame()
        for i in range(args.steps):
            step()
            if (i + 1) % every == 0:
                update()
                frame()
        rec.finish()
        return

    import polyscope.imgui as psim
    state = {"i": 0, "play": play}

    def callback():
        if interact is not None:
            interact(ps, psim)
        _, state["play"] = psim.Checkbox("play", state["play"])
        psim.SameLine()
        if (psim.Button("step") or state["play"]) and (args.steps is None or state["i"] < args.steps):
            step()
            state["i"] += 1
            update()
        psim.Text(f"step {state['i']}" + ("" if args.steps is None else f" / {args.steps}"))
        for line in (gui_text() if gui_text else []):
            psim.Text(line)

    ps.set_user_callback(callback)
    ps.show()
