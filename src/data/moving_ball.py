"""Clean 2D one-ball *velocity* dataset with exact ground-truth state (Step 2, velocity-first).

This is the deliberately-minimal controlled dataset the velocity experiments are built on. The design
follows the supervisor's spec to the letter so that the only thing varying between clips is the physics
we want to probe/steer — nothing else:

* **One ball, clean white background.** No second object, no texture, no shading, no gravity by default
  — so a "velocity subspace" can't be confounded with object type, lighting, or scene appearance (the
  exact confound that made the Step-1 *category* directions un-steerable).
* **Constant velocity, object stays fully in frame the whole clip.** The ball never bounces or leaves
  the frame, so every frame is a clean linear-motion sample and the velocity label is constant over the
  clip. Initial position + velocity are sampled so the straight-line trajectory is guaranteed to keep
  the whole disk inside ``[radius, 1-radius]`` for all ``T`` frames (rejection-sampled).
* **32 frames, 128x128, fixed FPS.** Matches the spec; ``fps`` is recorded in meta so every comparison
  is at a fixed frame rate (velocity in *pixels/frame* is only meaningful at fixed FPS).
* **Exact metadata.** Per-frame center position, exact velocity vector, speed, direction angle, radius,
  plus a per-frame ``visible`` flag (used by the occlusion variant). Stored in the same packed
  ``state`` / ``state_keys`` contract as :mod:`synthetic_physics` so all the existing probe/steer/decode
  machinery (``LatentDataset``, ``train_probe``, steering) works unchanged.

Variants (selected by ``scenario``):

* ``constant_velocity`` — the core probe/steer dataset.
* ``scene_velocity`` — the **difference-vector steering** dataset (supervisor's 2026-06-26 proposal).
  Clips are grouped into *scenes*; within a scene there are ``clips_per_scene`` clips that share an
  **identical first frame and motion direction** and differ **only in speed**. This is exactly the
  controlled pair the supervisor wants for latent arithmetic: encode two same-scene clips
  ``H_a = E(v_a)``, ``H_b = E(v_b)`` and ``H_b - H_a`` isolates the *speed* factor (everything else —
  start position, direction, radius, colour, background — is held fixed). Decoding
  ``H_a + alpha*(H_b - H_a)`` while sweeping ``alpha`` should make the ball visibly speed up / slow down
  in pixel space, *on-manifold* (alpha=0 -> v_a, alpha=1 -> v_b), instead of pushing an off-manifold
  probe direction. Scenes are seeded by scene-id so every clip of a scene reproduces the same scene
  parameters; the per-clip ``rank`` (0=slowest .. K-1=fastest) selects the speed.
* ``occlusion`` — a static vertical wall in the middle of the frame; the ball passes *behind* it and is
  invisible for the middle frames. Tests whether velocity is still decodable while the ball is hidden
  (object-permanence / physical-state evidence).
* ``rotated`` — same *speed*, sampled over a full range of *directions* (and an optional global frame
  rotation), for the equivariance experiment: a true velocity subspace should be equivariant under a
  rotation of the velocity (up to a rotation of the subspace).

The renderer is deterministic and dependency-light (anti-aliased disk on white), so the dataset
reproduces bit-for-bit anywhere (cluster, laptop, CI) and the ground truth is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Per-frame ground-truth columns for the single ball. Kept compatible with synthetic_physics'
# ``obj0_*`` naming so VARIABLE_GROUPS in train_probe (pos / vel / acc / radius / visible) match.
_BALL_KEYS = [
    "pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y", "radius", "speed", "angle", "visible",
]


def _state_keys() -> list[str]:
    return [f"obj0_{k}" for k in _BALL_KEYS] + ["gravity", "collision_event"]


def state_dim() -> int:
    return len(_state_keys())


@dataclass
class BallClip:
    frames: torch.Tensor          # (T, C, H, W) float in [0, 1]
    state: torch.Tensor           # (T, state_dim)
    state_keys: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def _render(
    pos: np.ndarray,
    radius: float,
    image_size: int,
    color: tuple[float, float, float],
    visible: bool,
    occluder: tuple[float, float, float, float] | None,
    rotation: float,
) -> np.ndarray:
    """Anti-aliased disk on a clean **white** background -> (H, W, 3) float in [0, 1].

    ``rotation`` (radians) rotates the rendered image about its center — used only by the equivariance
    ``rotated`` scenario to mimic a camera roll. ``pos`` is in normalized [0, 1] image coords.
    """
    h = w = image_size
    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs / (w - 1)
    yn = ys / (h - 1)

    if rotation != 0.0:
        # rotate the sampling grid about the image center (equivalently rotate the camera)
        cx = cy = 0.5
        ca, sa = np.cos(rotation), np.sin(rotation)
        dx, dy = xn - cx, yn - cy
        xn = cx + ca * dx - sa * dy
        yn = cy + sa * dx + ca * dy

    img = np.ones((h, w, 3), dtype=np.float32)  # clean white background

    if visible:
        d = np.sqrt((xn - pos[0]) ** 2 + (yn - pos[1]) ** 2)
        edge = 1.5 / image_size  # ~1.5px soft edge
        alpha = np.clip((radius - d) / edge + 0.5, 0.0, 1.0)
        for c in range(3):
            img[..., c] = img[..., c] * (1 - alpha) + color[c] * alpha

    if occluder is not None:
        x0, y0, x1, y1 = occluder
        in_x = (xn >= x0) & (xn <= x1)
        in_y = (yn >= y0) & (yn <= y1)
        wall = in_x & in_y
        for c in range(3):
            img[..., c] = np.where(wall, _OCCLUDER_COLOR[c], img[..., c])

    return np.clip(img, 0.0, 1.0)


_OCCLUDER_COLOR = (0.55, 0.55, 0.55)  # neutral grey wall (clearly distinct from white bg + ball)


class MovingBall:
    """Deterministic generator of clean single-ball velocity clips with exact ground-truth state."""

    def __init__(
        self,
        image_size: int = 128,
        num_frames: int = 32,
        fps: int = 8,
        scenario: str = "constant_velocity",
        speed_range: tuple[float, float] = (0.010, 0.035),
        radius_range: tuple[float, float] = (0.07, 0.10),
        fixed_speed: float = 0.022,
        camera_rotation: bool = False,
        ball_color: tuple[float, float, float] = (0.15, 0.15, 0.15),
        clips_per_scene: int = 4,
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.num_frames = num_frames
        self.fps = fps
        self.scenario = scenario
        self.speed_range = speed_range
        self.radius_range = radius_range
        self.fixed_speed = fixed_speed
        self.camera_rotation = camera_rotation
        self.ball_color = ball_color
        self.clips_per_scene = int(clips_per_scene)
        self.seed = seed

    # -- public API -----------------------------------------------------------------------------
    def generate(self, index: int) -> BallClip:
        """Generate clip ``index`` deterministically (same index + seed -> identical clip)."""
        if self.scenario == "scene_velocity":
            return self._scene_velocity(index)
        rng = np.random.default_rng(self.seed * 100_003 + index)
        if self.scenario == "occlusion":
            return self._occlusion(rng, index)
        if self.scenario == "rotated":
            return self._rotated(rng, index)
        return self._constant_velocity(rng, index)

    def dataset(self, num_clips: int) -> list[BallClip]:
        return [self.generate(i) for i in range(num_clips)]

    # -- trajectory sampling --------------------------------------------------------------------
    def _sample_trajectory(
        self, rng: np.random.Generator, radius: float, speed: float, angle: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample (start position, velocity) so the whole disk stays in frame for all T frames.

        Rejection-samples a start position + direction until the straight-line path keeps the ball
        center within ``[radius, 1-radius]`` (so the full disk is inside the image) at every frame.
        Returns ``(pos0 (2,), vel (2,))`` in normalized units (per-frame velocity).
        """
        lo, hi = radius, 1.0 - radius
        for _ in range(2000):
            ang = float(rng.uniform(0, 2 * np.pi)) if angle is None else float(angle)
            vel = speed * np.array([np.cos(ang), np.sin(ang)])
            total = vel * (self.num_frames - 1)
            # feasible start box so that pos0 and pos0+total both stay within [lo, hi]
            box_lo = np.maximum(lo, lo - total)
            box_hi = np.minimum(hi, hi - total)
            if np.all(box_hi > box_lo):
                pos0 = box_lo + rng.uniform(0, 1, size=2) * (box_hi - box_lo)
                return pos0, vel
        # fallback: center, small horizontal motion (always feasible)
        return np.array([0.5, 0.5]), np.array([min(speed, (hi - lo) / (2 * self.num_frames)), 0.0])

    # -- scenarios ------------------------------------------------------------------------------
    def _constant_velocity(self, rng: np.random.Generator, index: int) -> BallClip:
        radius = float(rng.uniform(*self.radius_range))
        speed = float(rng.uniform(*self.speed_range))
        pos0, vel = self._sample_trajectory(rng, radius, speed, angle=None)
        return self._roll_out(pos0, vel, radius, occluder=None, rotation=0.0,
                              scenario="constant_velocity", index=index)

    def _scene_velocity(self, index: int) -> BallClip:
        """One clip of a *scene*: identical first frame + direction within a scene, only speed varies.

        Determinism is keyed on the **scene id**, not the clip index, so every clip of a scene replays
        the exact same scene parameters (start position, direction, radius, and the set of speeds). The
        per-clip ``rank`` (``index % clips_per_scene``) only selects which of the scene's speeds this
        clip uses — so clip ``rank=0`` is the slowest and ``rank=K-1`` the fastest, and frame 0 is
        bit-identical across the whole scene. This is the controlled same-scene pair the supervisor's
        latent-arithmetic steering (``H_b - H_a``) needs.
        """
        K = self.clips_per_scene
        scene = index // K
        rank = index % K
        # Scene-level RNG: identical for every clip in the scene -> shared scene geometry.
        srng = np.random.default_rng(self.seed * 100_003 + 7919 * (scene + 1))
        radius = float(srng.uniform(*self.radius_range))
        angle = float(srng.uniform(0, 2 * np.pi))
        # K distinct speeds spanning the range, sorted slow->fast (rank indexes into this).
        lo, hi = self.speed_range
        speeds = np.sort(srng.uniform(lo, hi, size=K))
        # nudge apart if two speeds collide so every rank is visibly distinct
        for j in range(1, K):
            min_gap = 0.4 * (hi - lo) / K
            if speeds[j] - speeds[j - 1] < min_gap:
                speeds[j] = min(hi, speeds[j - 1] + min_gap)
        # Start position is feasible for the FASTEST speed (so all slower ranks also stay in frame),
        # sampled from srng so it is identical across ranks. Reuse the same fixed direction.
        pos0, _ = self._sample_trajectory(srng, radius, float(speeds[-1]), angle=angle)
        speed = float(speeds[rank])
        vel = speed * np.array([np.cos(angle), np.sin(angle)])
        return self._roll_out(pos0, vel, radius, occluder=None, rotation=0.0,
                              scenario="scene_velocity", index=index, scene=scene, rank=rank,
                              scene_speeds=[float(s) for s in speeds])

    def _rotated(self, rng: np.random.Generator, index: int) -> BallClip:
        """Same *speed*, swept *direction* (and optional global camera roll) — equivariance probe."""
        radius = float(rng.uniform(*self.radius_range))
        speed = self.fixed_speed
        # sweep direction roughly uniformly over the circle as index grows (plus jitter)
        base = 2 * np.pi * (index * 0.61803398875)  # golden-ratio low-discrepancy coverage
        angle = (base + float(rng.uniform(-0.15, 0.15))) % (2 * np.pi)
        pos0, vel = self._sample_trajectory(rng, radius, speed, angle=angle)
        rotation = float(rng.uniform(0, 2 * np.pi)) if self.camera_rotation else 0.0
        return self._roll_out(pos0, vel, radius, occluder=None, rotation=rotation,
                              scenario="rotated", index=index)

    def _occlusion(self, rng: np.random.Generator, index: int) -> BallClip:
        """Ball crosses behind a static central vertical wall (invisible for the middle frames).

        Velocity is *horizontal* (so the ball reliably passes behind the wall) but the start side and
        radius/speed still vary. The ``visible`` flag is 0 while the ball center is within the wall
        band; the velocity label stays exact (constant) even on hidden frames — that is the whole point.
        """
        radius = float(rng.uniform(*self.radius_range))
        speed = float(rng.uniform(*self.speed_range))
        wall_w = 0.10
        wall = (0.5 - wall_w / 2, 0.0, 0.5 + wall_w / 2, 1.0)
        # horizontal crossing, random left->right or right->left, y kept clear of frame edges
        direction = 1.0 if rng.uniform() < 0.5 else -1.0
        vel = np.array([direction * speed, 0.0])
        total = vel * (self.num_frames - 1)
        lo, hi = radius, 1.0 - radius
        # start so the full horizontal sweep stays in frame and crosses the wall
        if direction > 0:
            x0 = lo
        else:
            x0 = hi
        x0 = float(np.clip(x0, max(lo, lo - total[0]), min(hi, hi - total[0])))
        y0 = float(rng.uniform(lo + 0.05, hi - 0.05))
        pos0 = np.array([x0, y0])
        return self._roll_out(pos0, vel, radius, occluder=wall, rotation=0.0,
                              scenario="occlusion", index=index)

    # -- rollout --------------------------------------------------------------------------------
    def _roll_out(
        self,
        pos0: np.ndarray,
        vel: np.ndarray,
        radius: float,
        occluder: tuple[float, float, float, float] | None,
        rotation: float,
        scenario: str,
        index: int,
        scene: int | None = None,
        rank: int | None = None,
        scene_speeds: list[float] | None = None,
    ) -> BallClip:
        speed = float(np.linalg.norm(vel))
        angle = float(np.arctan2(vel[1], vel[0]))
        frames: list[np.ndarray] = []
        states: list[list[float]] = []
        keys = _state_keys()
        pos = pos0.astype(np.float64).copy()
        for _t in range(self.num_frames):
            visible = True
            if occluder is not None:
                x0, y0, x1, y1 = occluder
                # hidden while the ball center lies within the wall band
                visible = not (x0 <= pos[0] <= x1)
            frames.append(_render(pos, radius, self.image_size, self.ball_color, visible,
                                  occluder, rotation))
            row = [
                float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]),
                0.0, 0.0,  # acceleration is exactly zero (constant velocity)
                float(radius), speed, angle, float(visible),
            ]
            row.extend([0.0, 0.0])  # gravity, collision_event (none here)
            states.append(row)
            pos = pos + vel

        frame_arr = np.stack(frames, 0).transpose(0, 3, 1, 2)  # (T, C, H, W)
        meta = {
            "scenario": scenario, "index": index, "fps": self.fps, "radius": radius,
            "speed": speed, "angle": angle, "vel_x": float(vel[0]), "vel_y": float(vel[1]),
            "rotation": rotation, "image_size": self.image_size, "num_frames": self.num_frames,
            "num_objects": 1,
        }
        if scene is not None:
            meta["scene"] = int(scene)
            meta["rank"] = int(rank)
            meta["clips_per_scene"] = self.clips_per_scene
            if scene_speeds is not None:
                meta["scene_speeds"] = scene_speeds
        if occluder is not None:
            meta["occluder"] = list(occluder)
            n_hidden = int(sum(1 for r in states if r[_BALL_KEYS.index("visible")] == 0.0))
            meta["n_hidden_frames"] = n_hidden
        return BallClip(
            frames=torch.from_numpy(frame_arr).float(),
            state=torch.tensor(states, dtype=torch.float32),
            state_keys=keys,
            meta=meta,
        )
