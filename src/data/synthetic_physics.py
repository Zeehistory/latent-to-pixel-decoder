"""Controlled 2D synthetic-physics video generator with exact ground-truth state.

This module is the backbone of the probing / manifold experiments: because we control the simulator we
have *exact* labels for every physical variable, which lets us validate probes and metrics (an oracle
on these labels must be ~perfect; shuffled latents must be ~chance).

Each generated clip yields:

* ``frames``: ``(T, C, H, W)`` float tensor in ``[0, 1]`` (rendered with anti-aliased disks).
* ``state``:  ``(T, state_dim)`` float tensor — the per-frame ground-truth physical state.
* ``state_keys``: names for each column of ``state``.
* ``meta``: scenario, gravity, seed, per-object static attributes, collision frames.

Scenarios implemented: ``bouncing_ball``, ``projectile``, ``collision``, ``pendulum``,
``two_body``, ``occlusion`` (object-permanence probe).

The renderer is intentionally simple and deterministic (no external physics engine) so the dataset is
reproducible anywhere and the ground truth is exact. Heavier engines (PyBullet/Box2D, soft-body,
fluids) can be added behind the same return contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Per-object dynamic state columns, recorded for object 0 (primary object) and, when present, object 1.
_BASE_KEYS = [
    "pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y", "radius", "mass", "visible",
]


def _state_keys(num_objects: int) -> list[str]:
    keys: list[str] = []
    for o in range(num_objects):
        keys.extend(f"obj{o}_{k}" for k in _BASE_KEYS)
    keys.extend(["gravity", "collision_event"])
    return keys


@dataclass
class Body:
    pos: np.ndarray  # (2,) in normalized [0,1] image coords (x right, y down)
    vel: np.ndarray  # (2,) per-frame velocity in normalized units
    radius: float
    mass: float
    color: tuple[float, float, float]
    visible: bool = True


@dataclass
class Clip:
    frames: torch.Tensor          # (T, C, H, W)
    state: torch.Tensor           # (T, state_dim)
    state_keys: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def _render(bodies: list[Body], image_size: int, occluder: tuple | None = None) -> np.ndarray:
    """Anti-aliased render of disks on a dark background -> (H, W, 3) float [0,1]."""
    h = w = image_size
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs / (w - 1)
    ys = ys / (h - 1)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = 0.05  # faint background

    for b in bodies:
        if not b.visible:
            continue
        d = np.sqrt((xs - b.pos[0]) ** 2 + (ys - b.pos[1]) ** 2)
        # soft edge over ~1.5 pixels
        edge = 1.5 / image_size
        alpha = np.clip((b.radius - d) / edge + 0.5, 0.0, 1.0)
        for c in range(3):
            img[..., c] = img[..., c] * (1 - alpha) + b.color[c] * alpha

    if occluder is not None:
        x0, y0, x1, y1 = occluder
        xi0, xi1 = int(x0 * w), int(x1 * w)
        yi0, yi1 = int(y0 * h), int(y1 * h)
        img[yi0:yi1, xi0:xi1] = 0.35
    return np.clip(img, 0.0, 1.0)


def _pack_state(
    per_obj: list[dict[str, float]],
    num_objects: int,
    gravity: float,
    collision: float,
) -> list[float]:
    row: list[float] = []
    for o in range(num_objects):
        d = per_obj[o]
        row.extend([d[k] for k in _BASE_KEYS])
    row.extend([gravity, collision])
    return row


class SyntheticPhysics:
    """Deterministic generator of physics clips with exact ground-truth state."""

    def __init__(
        self,
        image_size: int = 64,
        num_frames: int = 16,
        scenarios: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.num_frames = num_frames
        self.scenarios = scenarios or ["bouncing_ball", "projectile"]
        self.seed = seed

    # -- public API -----------------------------------------------------------------------------
    def generate(self, index: int) -> Clip:
        """Generate clip ``index`` deterministically (same index -> identical clip)."""
        rng = np.random.default_rng(self.seed * 100_003 + index)
        scenario = self.scenarios[index % len(self.scenarios)]
        method = getattr(self, f"_scenario_{scenario}")
        return method(rng)

    def dataset(self, num_clips: int) -> list[Clip]:
        return [self.generate(i) for i in range(num_clips)]

    # -- scenarios ------------------------------------------------------------------------------
    def _scenario_bouncing_ball(self, rng: np.random.Generator) -> Clip:
        gravity = float(rng.uniform(0.004, 0.012))
        radius = float(rng.uniform(0.05, 0.10))
        b = Body(
            pos=np.array([rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.4)]),
            vel=np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.01, 0.01)]),
            radius=radius,
            mass=radius**2,
            color=(0.9, 0.3, 0.3),
        )
        return self._simulate([b], gravity, restitution=0.9, scenario="bouncing_ball")

    def _scenario_projectile(self, rng: np.random.Generator) -> Clip:
        gravity = float(rng.uniform(0.004, 0.010))
        radius = float(rng.uniform(0.04, 0.08))
        b = Body(
            pos=np.array([rng.uniform(0.05, 0.2), rng.uniform(0.6, 0.85)]),
            vel=np.array([rng.uniform(0.02, 0.05), rng.uniform(-0.06, -0.03)]),
            radius=radius,
            mass=radius**2,
            color=(0.3, 0.8, 0.4),
        )
        return self._simulate([b], gravity, restitution=0.6, scenario="projectile")

    def _scenario_collision(self, rng: np.random.Generator) -> Clip:
        gravity = 0.0
        r0 = float(rng.uniform(0.05, 0.08))
        r1 = float(rng.uniform(0.05, 0.08))
        b0 = Body(np.array([0.2, 0.5]), np.array([0.04, 0.0]), r0, r0**2, (0.9, 0.3, 0.3))
        b1 = Body(np.array([0.8, 0.5]), np.array([-0.04, 0.0]), r1, r1**2, (0.3, 0.5, 0.9))
        return self._simulate([b0, b1], gravity, restitution=float(rng.uniform(0.7, 1.0)),
                              scenario="collision")

    def _scenario_two_body(self, rng: np.random.Generator) -> Clip:
        gravity = float(rng.uniform(0.0, 0.006))
        r0 = float(rng.uniform(0.05, 0.08))
        r1 = float(rng.uniform(0.05, 0.08))
        b0 = Body(np.array([0.3, 0.3]), np.array([0.02, 0.01]), r0, r0**2, (0.9, 0.6, 0.2))
        b1 = Body(np.array([0.7, 0.7]), np.array([-0.02, -0.01]), r1, r1**2, (0.4, 0.7, 0.9))
        return self._simulate([b0, b1], gravity, restitution=0.95, scenario="two_body")

    def _scenario_pendulum(self, rng: np.random.Generator) -> Clip:
        gravity = float(rng.uniform(0.006, 0.012))
        pivot = np.array([0.5, 0.2])
        length = float(rng.uniform(0.3, 0.45))
        theta = float(rng.uniform(0.6, 1.2))
        omega = 0.0
        radius = float(rng.uniform(0.05, 0.08))
        frames, states = [], []
        prev_vel = np.zeros(2)
        keys = _state_keys(1)
        for _t in range(self.num_frames):
            pos = pivot + length * np.array([np.sin(theta), np.cos(theta)])
            vel = length * omega * np.array([np.cos(theta), -np.sin(theta)])
            acc = vel - prev_vel
            prev_vel = vel
            body = Body(pos, vel, radius, radius**2, (0.8, 0.5, 0.9))
            frames.append(_render([body], self.image_size))
            per_obj = [self._obj_dict(pos, vel, acc, radius, radius**2, True)]
            states.append(_pack_state(per_obj, 1, gravity, 0.0))
            # pendulum update (small-step Euler)
            alpha = -gravity / length * np.sin(theta) * 60.0
            omega += alpha
            omega *= 0.999
            theta += omega
        return self._finish(frames, states, keys, scenario="pendulum", gravity=gravity)

    def _scenario_occlusion(self, rng: np.random.Generator) -> Clip:
        """Object permanence probe: a ball passes behind a static occluder (becomes invisible)."""
        gravity = 0.0
        radius = float(rng.uniform(0.05, 0.07))
        b = Body(np.array([0.1, 0.5]), np.array([0.05, 0.0]), radius, radius**2, (0.95, 0.8, 0.2))
        occ = (0.45, 0.0, 0.55, 1.0)  # vertical bar
        frames, states = [], []
        prev_vel = b.vel.copy()
        keys = _state_keys(1)
        for _t in range(self.num_frames):
            visible = not (occ[0] <= b.pos[0] <= occ[2])
            b.visible = visible
            acc = b.vel - prev_vel
            prev_vel = b.vel.copy()
            frames.append(_render([b], self.image_size, occluder=occ))
            per_obj = [self._obj_dict(b.pos, b.vel, acc, b.radius, b.mass, visible)]
            states.append(_pack_state(per_obj, 1, gravity, 0.0))
            b.pos = b.pos + b.vel
        return self._finish(frames, states, keys, scenario="occlusion", gravity=gravity,
                            extra={"occluder": occ})

    # -- core integrator ------------------------------------------------------------------------
    def _simulate(
        self,
        bodies: list[Body],
        gravity: float,
        restitution: float,
        scenario: str,
    ) -> Clip:
        n = len(bodies)
        keys = _state_keys(n)
        frames: list[np.ndarray] = []
        states: list[list[float]] = []
        prev_vel = [b.vel.copy() for b in bodies]
        for _t in range(self.num_frames):
            collision = 0.0
            # gravity (acts on +y / downward)
            for b in bodies:
                b.vel = b.vel + np.array([0.0, gravity])
            # integrate + wall bounces
            for b in bodies:
                b.pos = b.pos + b.vel
                for ax in range(2):
                    lo, hi = b.radius, 1.0 - b.radius
                    if b.pos[ax] < lo:
                        b.pos[ax] = lo
                        b.vel[ax] = -b.vel[ax] * restitution
                        collision = 1.0
                    elif b.pos[ax] > hi:
                        b.pos[ax] = hi
                        b.vel[ax] = -b.vel[ax] * restitution
                        collision = 1.0
            # pairwise elastic-ish collisions
            for i in range(n):
                for j in range(i + 1, n):
                    if self._resolve_pair(bodies[i], bodies[j], restitution):
                        collision = 1.0
            frames.append(_render(bodies, self.image_size))
            per_obj = []
            for k, b in enumerate(bodies):
                acc = b.vel - prev_vel[k]
                prev_vel[k] = b.vel.copy()
                per_obj.append(self._obj_dict(b.pos, b.vel, acc, b.radius, b.mass, b.visible))
            states.append(_pack_state(per_obj, n, gravity, collision))
        return self._finish(frames, states, keys, scenario=scenario, gravity=gravity)

    @staticmethod
    def _resolve_pair(a: Body, b: Body, restitution: float) -> bool:
        delta = a.pos - b.pos
        dist = float(np.linalg.norm(delta))
        min_dist = a.radius + b.radius
        if dist >= min_dist or dist == 0:
            return False
        normal = delta / dist
        # separate
        overlap = min_dist - dist
        total_mass = a.mass + b.mass
        a.pos = a.pos + normal * overlap * (b.mass / total_mass)
        b.pos = b.pos - normal * overlap * (a.mass / total_mass)
        # 1D elastic collision along the normal
        va = float(np.dot(a.vel, normal))
        vb = float(np.dot(b.vel, normal))
        a_new = (va * (a.mass - b.mass) + 2 * b.mass * vb) / total_mass
        b_new = (vb * (b.mass - a.mass) + 2 * a.mass * va) / total_mass
        a.vel = a.vel + (a_new - va) * normal * restitution
        b.vel = b.vel + (b_new - vb) * normal * restitution
        return True

    @staticmethod
    def _obj_dict(pos, vel, acc, radius, mass, visible) -> dict[str, float]:
        return {
            "pos_x": float(pos[0]), "pos_y": float(pos[1]),
            "vel_x": float(vel[0]), "vel_y": float(vel[1]),
            "acc_x": float(acc[0]), "acc_y": float(acc[1]),
            "radius": float(radius), "mass": float(mass),
            "visible": float(bool(visible)),
        }

    def _finish(self, frames, states, keys, scenario, gravity, extra=None) -> Clip:
        frame_arr = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
        frame_t = torch.from_numpy(frame_arr).float()
        state_t = torch.tensor(states, dtype=torch.float32)
        meta = {"scenario": scenario, "gravity": gravity, "num_objects": (state_t.shape[1] - 2) // len(_BASE_KEYS)}
        if extra:
            meta.update(extra)
        return Clip(frames=frame_t, state=state_t, state_keys=keys, meta=meta)


def state_dim_for(scenarios: list[str]) -> int:
    """Max state-vector length across scenarios (multi-object scenarios are wider)."""
    multi = {"collision", "two_body"}
    n = 2 if any(s in multi for s in scenarios) else 1
    return len(_state_keys(n))
