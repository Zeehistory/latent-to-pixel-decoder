"""MuJoCo-backed solid-mechanics dataset with exact simulator ground truth (Step 2).

Replaces the lightweight 2D renderer for solid mechanics with a real physics engine (MuJoCo), giving
shaded 3D renders that are closer to natural video (smaller domain gap to Physics-IQ) while still
providing *exact* per-frame physical state read straight from the simulator:

* position  -> ``data.qpos[:3]``
* velocity  -> ``data.qvel[:3]``
* acceleration -> ``data.qacc[:3]`` (equals the gravity vector in free flight; spikes on contact)
* gravity   -> ``model.opt.gravity`` (varied per clip so the gravity probe is meaningful)
* collision -> ``data.ncon > 0``

Scenarios (single free body, so the state vector is homogeneous across clips):
``free_fall`` (drop from rest), ``projectile`` (launched arc), ``bounce`` (drops and rebounds off the
floor). All three share one MJCF model + one renderer; only the initial conditions and per-clip gravity
differ, which keeps generation fast and deterministic.

MuJoCo is an optional dependency: ``pip install mujoco``. Off-screen rendering needs a GL backend
(``MUJOCO_GL=egl`` on headless Linux/cluster nodes; on macOS it works out of the box).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_registry import register_dataset
from .video_transforms import VideoTransform

# Per-frame state columns. Substrings (pos/vel/acc/radius/gravity/collision_event) match the quantity
# probe's VARIABLE_GROUPS so the same probe works unchanged.
STATE_KEYS = [
    "obj0_pos_x", "obj0_pos_y", "obj0_pos_z",
    "obj0_vel_x", "obj0_vel_y", "obj0_vel_z",
    "obj0_acc_x", "obj0_acc_y", "obj0_acc_z",
    "obj0_radius", "gravity", "collision_event",
]

_SCENARIOS = ["free_fall", "projectile", "bounce"]

# One reusable scene: floor + free ball + fixed camera + light. Bouncy contact via a stiff, low-damping
# solref so `bounce` visibly rebounds.
_MJCF = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <worldbody>
    <light pos="0 -1 4" dir="0 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.30 0.32 0.38 1"/>
    <camera name="cam" pos="0 -4.2 1.5" xyaxes="1 0 0 0 0.35 0.94"/>
    <body name="ball" pos="0 0 1.5">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{radius}" rgba="0.90 0.30 0.28 1"
            solref="-12000 -40" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoPhysicsDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        try:
            import mujoco  # noqa: F401
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "MuJoCo is required for the 'mujoco_physics' dataset. Install with `pip install mujoco`. "
                "On headless cluster nodes set MUJOCO_GL=egl for off-screen rendering."
            ) from e

        self.cfg = cfg
        self.num_frames = cfg.num_frames
        self.image_size = cfg.image_size
        self.num_clips = cfg.num_clips
        self.seed = cfg.seed
        self.radius = float(getattr(cfg, "ball_radius", 0.14))
        self.scenarios = list(cfg.scenarios) if getattr(cfg, "scenarios", "all") != "all" else _SCENARIOS
        self.fps = cfg.fps
        self.transform = VideoTransform(
            image_size=encoder_image_size, num_frames=encoder_frames, do_normalize=True)
        self._cache: dict[int, dict[str, Any]] = {}
        self._mj = None  # lazy (model, data, renderer)

    def __len__(self) -> int:
        return self.num_clips

    def _engine(self):
        if self._mj is None:
            import mujoco
            model = mujoco.MjModel.from_xml_string(_MJCF.format(radius=self.radius))
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, self.image_size, self.image_size)
            self._mj = (mujoco, model, data, renderer)
        return self._mj

    def _init_conditions(self, scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Return (init_pos xyz, init_linvel xyz, gravity magnitude, steps_per_frame)."""
        gravity = float(rng.uniform(6.0, 13.0))
        if scenario == "free_fall":
            pos = np.array([rng.uniform(-0.3, 0.3), 0.0, rng.uniform(1.6, 2.2)])
            vel = np.zeros(3)
        elif scenario == "projectile":
            pos = np.array([rng.uniform(-1.2, -0.6), 0.0, rng.uniform(0.4, 0.9)])
            vel = np.array([rng.uniform(1.6, 3.2), 0.0, rng.uniform(1.5, 3.0)])
        else:  # bounce
            pos = np.array([rng.uniform(-0.3, 0.3), 0.0, rng.uniform(1.2, 1.8)])
            vel = np.array([rng.uniform(-0.4, 0.4), 0.0, 0.0])
        frame_dt = 1.0 / float(self.fps)
        steps_per_frame = max(1, int(round(frame_dt / 0.004)))
        return pos, vel, gravity, steps_per_frame

    def _generate(self, idx: int) -> dict[str, Any]:
        mujoco, model, data, renderer = self._engine()
        scenario = self.scenarios[idx % len(self.scenarios)]
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        pos, vel, gravity, steps = self._init_conditions(scenario, rng)

        mujoco.mj_resetData(model, data)
        model.opt.gravity[:] = [0.0, 0.0, -gravity]
        data.qpos[:3] = pos
        data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion
        data.qvel[:3] = vel
        mujoco.mj_forward(model, data)

        frames, states = [], []
        for _t in range(self.num_frames):
            renderer.update_scene(data, camera="cam")
            img = renderer.render().astype(np.float32) / 255.0  # (H, W, 3)
            frames.append(img.transpose(2, 0, 1))               # (3, H, W)
            collision = 1.0 if data.ncon > 0 else 0.0
            states.append([
                *data.qpos[:3], *data.qvel[:3], *data.qacc[:3], self.radius, gravity, collision
            ])
            for _ in range(steps):
                mujoco.mj_step(model, data)

        frame_t = torch.from_numpy(np.stack(frames, 0)).float()
        return {
            "id": f"{scenario}_{idx:05d}",
            "frames": frame_t,
            "state": torch.tensor(states, dtype=torch.float32),
            "state_mask": torch.ones(len(STATE_KEYS)),
            "state_keys": list(STATE_KEYS),
            "category": scenario,
            "meta": {"scenario": scenario, "gravity": gravity, "engine": "mujoco"},
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._cache.get(idx)
        if sample is None:
            sample = self._generate(idx)
            self._cache[idx] = sample
        out = dict(sample)
        out["encoder_input"] = self.transform(sample["frames"])
        return out


@register_dataset("mujoco_physics")
def _build_mujoco(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return MuJoCoPhysicsDataset(cfg, encoder_image_size, encoder_frames)
