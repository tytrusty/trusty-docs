"""Moving a servoed arm's hand along a pose trajectory through the sensitivity of its own steps.

Shared by the Franka examples. Three points on the hand are tracked
(`trusty.control`): where they are after a step (y), where the next step takes
them with the servo targets held (the free response, d), and how they move with
the servo targets (J = dy/du of the converged step). Each step the controller
solves a small convex problem for the next change of targets,

    min_du  |r + J du|^2 + lam |D du|^2 + w |u + du - u_posture|^2 + s |du - du_prev|^2,
                                                                        r = y + d - y_ref

so it needs no kinematic model of the arm: contact, sag and the servos' own
softness are already in J and d. `D` is Marquardt scaling. The step is limited to a
trust box, a fraction of what each joint moves at its speed limit that grows while the
prediction holds and shrinks when it misses, each joint's target stays inside its limits, and
no target gets more than `max_lead` ahead of its joint.
The last term keeps one step's change of targets near the previous one, which stops the
targets flipping between their extremes when the tip's goal conflicts with a constraint.

With a `keep_out` obstacle, points on every link are tracked too, and each must stay `margin`
from the obstacle after the step: with p = y + d + J du and distance field phi, the linearised
constraint is phi(y + d) + grad phi . (J du) >= margin, one row per point. A slack keeps the
problem feasible when a point is already inside the margin.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.spatial.transform import Rotation

import trusty

import franka as fr

LEVER = 0.1   # m; the two off-origin tracked points sit this far along the hand's x and y
SCALE = 1e3   # the problem is posed in millimetres so the solver's tolerances suit its size
GUARD_POINTS = 12          # tracked points per link that keep it out of an obstacle
SLACK_WEIGHT = 1e3        # cost of a mm of keep-out violation, in mm of tip error squared


def make_T(R=None, t=None):
    """(4, 4) rigid transform from a (3, 3) rotation and a (3,) translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def rot_z(angle):
    return Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()


def pose_error(T, T_goal):
    """(6,) world-frame error (position; rotation vector) taking T to T_goal."""
    return np.r_[T_goal[:3, 3] - T[:3, 3], Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec()]


def interpolate(T0, T1, s):
    """Straight-line position and geodesic rotation blend from T0 to T1 at fraction s."""
    dR = Rotation.from_matrix(T0[:3, :3].T @ T1[:3, :3])
    return make_T(T0[:3, :3] @ Rotation.from_rotvec(dR.as_rotvec() * s).as_matrix(),
                  (1.0 - s) * T0[:3, 3] + s * T1[:3, 3])


def min_jerk(s):
    """Time scaling 0 to 1 with zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def frame_points(T):
    """(3, 3) points that carry a frame T: its origin and LEVER along its x and y.

    Track them (`trusty.control.track`) on the body the frame rides on, and `frame_pose` of
    their positions is the frame."""
    return T[:3, 3] + LEVER * np.vstack([np.zeros(3), T[:3, 0], T[:3, 1]])


def frame_pose(points):
    """(4, 4) frame whose `frame_points` are `points` (3, 3)."""
    x, y = points[1] - points[0], points[2] - points[0]
    x /= np.linalg.norm(x)
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    return make_T(np.c_[x, y, np.cross(x, y)], points[0])


def spread_points(V, k):
    """`k` of the vertices `V` (n, 3), each the farthest from those chosen before it."""
    chosen = [int(np.argmax(np.linalg.norm(V - V.mean(axis=0), axis=1)))]
    gap = np.linalg.norm(V - V[chosen[0]], axis=1)
    for _ in range(k - 1):
        chosen.append(int(np.argmax(gap)))
        gap = np.minimum(gap, np.linalg.norm(V - V[chosen[-1]], axis=1))
    return V[chosen]


class HandController:
    """Drives the hand's tool centre point to a commanded pose, one `command` per step.

    Each step: `command(T_next)`, `world.step()`, `observe()`.

    @param keep_out  callable (n, 3) points -> (n,) signed distance (m) to an obstacle, negative
                     inside; every link stays `margin` (m) outside it
    """

    def __init__(self, world, franka, *, keep_out=None, margin=0.06, damping=1e-3,
                 posture_weight=1e-6, smoothing=0.01, max_lead=0.05, trust_min=0.01):
        self.world, self.franka = world, franka
        self.keep_out, self.margin = keep_out, margin
        self.damping, self.posture_weight, self.smoothing = damping, posture_weight, smoothing
        self.du = np.zeros(len(franka.arm_actuators))   # the last change of targets
        self.trust, self.trust_min = 1.0, trust_min
        self.max_step = franka.arm_speeds * world.timestep   # rad per step at the speed limit
        self.max_lead = max_lead   # rad a servo target may lead its joint
        self.u = np.zeros(len(franka.arm_actuators))     # servo targets, offsets from the build pose
        self.steps = 0
        self.floor = None      # running mean of the model miss; below it the gain ratio is noise
        self._pending = None
        # The hand's frame points come first among its tracked points.
        self.guarded = []
        for name, body in franka.bodies.items():
            points = frame_points(franka.tcp) if body == franka.hand else np.empty((0, 3))
            if keep_out is not None and name != "base":
                V = np.asarray(trusty.affine.surface(world, body)[0])
                points = np.vstack([points, spread_points(V, GUARD_POINTS)])
            if len(points):
                trusty.control.track(world, body, points, inputs=franka.arm_actuators)
                if keep_out is not None:
                    self.guarded.append(body)
        self.limits = franka.arm_limits - franka.q0[:, None]
        self.posture = np.zeros_like(self.u)
        self._build_problem()

    def _build_problem(self):
        m = len(self.u)
        n_guard = sum(len(trusty.control.tracked_points(self.world, b)) for b in self.guarded)
        self._J, self._r = cp.Parameter((9, m)), cp.Parameter(9)
        self._D, self._shift = cp.Parameter(m, nonneg=True), cp.Parameter(m)
        self._last = cp.Parameter(m)
        self._lo, self._hi = cp.Parameter(m), cp.Parameter(m)
        du = cp.Variable(m)
        cost = (cp.sum_squares(self._J @ du + self._r)
                + self.damping * SCALE**2 * cp.sum_squares(cp.multiply(self._D, du))
                + self.posture_weight * SCALE**2 * cp.sum_squares(du + self._shift)
                + self.smoothing * SCALE**2 * cp.sum_squares(du - self._last))
        constraints = [du >= self._lo, du <= self._hi]
        if n_guard:
            self._G, self._g = cp.Parameter((n_guard, m)), cp.Parameter(n_guard)
            slack = cp.Variable(n_guard, nonneg=True)
            constraints.append(self._G @ du + slack >= self._g)
            cost = cost + SLACK_WEIGHT * SCALE**2 * cp.sum_squares(slack)
        self._du, self._problem = du, cp.Problem(cp.Minimize(cost), constraints)

    def pose(self):
        """(4, 4) tool centre point frame after the last step."""
        return frame_pose(trusty.control.tracked_points(self.world, self.franka.hand)[:3])

    def clearance(self):
        """Distance (m) from the obstacle of the closest guarded point; inf with no obstacle."""
        points = [trusty.control.tracked_points(self.world, b) for b in self.guarded]
        return min((self.keep_out(p).min() for p in points), default=math.inf)

    def _keep_out_rows(self):
        """Rows (G, g) of the linearised keep-out constraint G du >= g, one per guarded point."""
        h = 1e-4
        rows, bounds = [], []
        for body in self.guarded:
            y = trusty.control.tracked_points(self.world, body)
            p = y + trusty.control.free_response(self.world, body)
            J = trusty.control.sensitivity(self.world, body).reshape(len(y), 3, -1)
            normal = np.stack([(self.keep_out(p + h * e) - self.keep_out(p - h * e)) / (2 * h)
                               for e in np.eye(3)], axis=1)
            rows.append(np.einsum("kc,kcm->km", normal, J))
            bounds.append(self.margin - self.keep_out(p))
        return np.vstack(rows), np.concatenate(bounds)

    def bounds(self):
        """Trust box, joint limits and the lead cap, intersected, as bounds on du.

        The trust box is a fraction of what each joint may move in a step at its speed limit. The
        lead cap keeps a blocked servo from winding its target up without bound, which would press
        the arm into whatever blocks it with ever more torque."""
        pad = fr.ARM_LIMIT_MARGIN
        box = self.trust * self.max_step
        angle = np.array([trusty.affine.actuator_value(self.world, a)
                          for a in self.franka.arm_actuators])
        lo = np.clip(self.limits[:, 0] + pad - self.u, -box, box)
        hi = np.clip(self.limits[:, 1] - pad - self.u, -box, box)
        lead_lo, lead_hi = angle - self.max_lead - self.u, angle + self.max_lead - self.u
        # The lead cap wins over the trust box, so a joint that recoils pulls its target back at once.
        nearest = np.where(lead_hi < lo, lead_hi, lead_lo)
        outside = (lead_hi < lo) | (lead_lo > hi)
        lo, hi = np.where(outside, nearest, np.maximum(lo, lead_lo)), np.where(
            outside, nearest, np.minimum(hi, lead_hi))
        return lo, np.maximum(hi, np.nextafter(lo, np.inf))

    def command(self, T_next):
        """Set the servo targets that put the tool centre point at `T_next` after the next step."""
        if self.steps == 0:
            self._pending = None     # no sensitivity before the first step
            return
        w, hand = self.world, self.franka.hand
        y = trusty.control.tracked_points(w, hand)[:3].ravel()
        y_model = y + trusty.control.free_response(w, hand)[:3].ravel()
        J = trusty.control.sensitivity(w, hand)[:9]
        JtJ = J.T @ J
        self._J.value = SCALE * J
        self._r.value = SCALE * (y_model - frame_points(T_next).ravel())
        self._D.value = np.sqrt(np.maximum(np.diag(JtJ), 1e-3 * np.trace(JtJ) / len(self.u)))
        self._shift.value = self.u - self.posture
        self._last.value = self.du
        self._lo.value, self._hi.value = self.bounds()
        if self.guarded:
            self._G.value, self._g.value = self._keep_out_rows()
        self._problem.solve(solver=cp.CLARABEL)
        du = self._du.value
        self.du = du
        at_trust = bool(np.any(np.abs(du) >= self.trust * self.max_step * (1 - 1e-6)))
        self._pending = (y_model, J @ du, at_trust)
        self.u = self.u + du
        for actuator, v in zip(self.franka.arm_actuators, self.u):
            trusty.affine.set_actuator_target(w, actuator, float(v))

    def observe(self):
        """Adapt the trust box from the gain ratio of the step just taken."""
        self.steps += 1
        if self._pending is None:
            return
        y_model, p, at_trust = self._pending
        a = trusty.control.tracked_points(self.world, self.franka.hand)[:3].ravel() - y_model
        miss = np.linalg.norm(a - p)
        self.floor = miss if self.floor is None else 0.9 * self.floor + 0.1 * miss
        pp = p @ p
        if math.sqrt(pp) <= 2 * self.floor:
            return
        rho = (a @ p) / pp
        if rho > 0.75 and at_trust:
            self.trust = min(2 * self.trust, 1.0)
        elif rho < 0.25:
            self.trust = max(self.trust / 4, self.trust_min)


@dataclass
class Waypoint:
    """A min-jerk move of the tool centre point to `pose` (4, 4) and the gripper to `gripper` (m).

    Either left None is held.
    """

    duration: float
    pose: np.ndarray | None = None
    gripper: float | None = None
    name: str = ""


class Trajectory:
    """A queue of waypoints played back one timestep at a time."""

    def __init__(self, pose, gripper):
        self.pose, self.gripper = np.asarray(pose, float), float(gripper)
        self.queue, self.t, self.name = [], 0.0, "done"
        self._from = None

    def add(self, waypoints):
        self.queue.extend(waypoints)
        return self

    def go(self, waypoints):
        """Drop what is left of the current move and play `waypoints` from the commanded pose."""
        self.queue, self._from = [], None
        return self.add(waypoints)

    @property
    def done(self):
        return not self.queue

    def step(self, dt):
        """Advance `dt` (s); returns the commanded (pose (4, 4), gripper width (m))."""
        if self.queue:
            wp = self.queue[0]
            if self._from is None:
                self._from, self.t, self.name = (self.pose, self.gripper), 0.0, wp.name
            self.t += dt
            s = min_jerk(self.t / wp.duration) if wp.duration > 0 else 1.0
            T1 = self._from[0] if wp.pose is None else wp.pose
            g1 = self._from[1] if wp.gripper is None else wp.gripper
            self.pose = interpolate(self._from[0], T1, s)
            self.gripper = (1.0 - s) * self._from[1] + s * g1
            if self.t >= wp.duration - 1e-12:
                self.queue.pop(0)
                self._from = None
        return self.pose.copy(), self.gripper


class Driver:
    """Plays a trajectory through the hand controller: each `step()` commands the arm and the
    gripper, advances the world one timestep and records the solver's work."""

    def __init__(self, world, franka, trajectory, dt, controller=None):
        self.world, self.franka, self.trajectory, self.dt = world, franka, trajectory, dt
        self.controller = HandController(world, franka) if controller is None else controller
        self.steps, self.wall, self.max_iters, self.unconverged = 0, 0.0, 0, 0

    def step(self):
        T, width = self.trajectory.step(self.dt)
        self.controller.command(T)
        self.franka.set_gripper(self.world, width)
        t0 = time.perf_counter()
        self.world.step()
        self.wall += time.perf_counter() - t0
        self.controller.observe()
        report = self.world.last_report()
        self.steps += 1
        self.max_iters = max(self.max_iters, report.iterations)
        self.unconverged += not report.converged
        return T

    def summary(self):
        """One line: steps, mean step time, the most Newton iterations and unconverged steps."""
        return (f"{self.steps} steps, {1e3 * self.wall / max(self.steps, 1):.2f} ms/step, "
                f"newton max {self.max_iters}, unconverged {self.unconverged}")
