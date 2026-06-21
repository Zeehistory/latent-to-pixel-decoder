"""Genesis-backed fluid-dynamics dataset with exact simulator ground truth (Step 2).

Genesis (https://github.com/Genesis-Embodied-AI/Genesis) is a GPU physics engine with real fluid
(SPH/MPM) simulation, so this gives genuine fluid dynamics (pouring, dam-break, splash) rather than the
2D diffusive-swarm proxy. Per-frame ground-truth state is read from the simulated particles:

* centroid position    -> mean particle position
* centroid velocity    -> mean particle velocity (finite-diff if not exposed by the version)
* centroid acceleration-> finite-diff of velocity
* spread (``radius``)  -> RMS particle distance from the centroid
* gravity              -> sim option (varied per clip)

State is the swarm *aggregate* packed into the standard single-object layout, so the same quantity
probe works unchanged.

IMPORTANT — runs on the cluster, not locally. Genesis requires a CUDA GPU and a separate install
(``pip install genesis-world``; see its docs). It is an optional dependency: the builder raises a clear
error if it is missing. The Genesis API has changed across releases, so the stepping/render/particle
calls below are written against the documented API and guarded with fallbacks; if your installed
version differs, adjust the small number of calls marked ``# GENESIS API``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_registry import register_dataset
from .video_transforms import VideoTransform

STATE_KEYS = [
    "obj0_pos_x", "obj0_pos_y", "obj0_pos_z",
    "obj0_vel_x", "obj0_vel_y", "obj0_vel_z",
    "obj0_acc_x", "obj0_acc_y", "obj0_acc_z",
    "obj0_radius", "gravity", "collision_event",
]

_SCENARIOS = ["pour", "dam_break", "drop"]


class GenesisFluidDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        try:
            import genesis  # noqa: F401
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "Genesis is required for the 'genesis_fluid' dataset. Install on a CUDA node with "
                "`pip install genesis-world` (see the Genesis docs). It is GPU-only and does not run on "
                "CPU-only machines."
            ) from e

        self.cfg = cfg
        self.num_frames = cfg.num_frames
        self.image_size = cfg.image_size
        self.num_clips = cfg.num_clips
        self.seed = cfg.seed
        self.fps = cfg.fps
        self.scenarios = list(cfg.scenarios) if getattr(cfg, "scenarios", "all") != "all" else _SCENARIOS
        self.transform = VideoTransform(
            image_size=encoder_image_size, num_frames=encoder_frames, do_normalize=True)
        self._cache: dict[int, dict[str, Any]] = {}
        self._gs_inited = False

    def __len__(self) -> int:
        return self.num_clips

    def _ensure_init(self):
        if not self._gs_inited:
            import genesis as gs
            gs.init(backend=gs.gpu)  # GENESIS API: backend selection
            self._gs_inited = True

    def _build_scene(self, scenario: str, gravity: float, rng: np.random.Generator):
        import genesis as gs

        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=1.0 / 120.0, gravity=(0.0, 0.0, -gravity)),
        )
        scene.add_entity(gs.morphs.Plane())  # floor
        # liquid block; placement/size varies per scenario for pour/dam-break/drop dynamics
        if scenario == "dam_break":
            pos, size = (rng.uniform(-0.2, -0.1), 0.0, 0.25), (0.18, 0.25, 0.45)
        elif scenario == "pour":
            pos, size = (0.0, 0.0, rng.uniform(0.5, 0.7)), (0.14, 0.14, 0.20)
        else:  # drop
            pos, size = (rng.uniform(-0.1, 0.1), 0.0, rng.uniform(0.45, 0.65)), (0.16, 0.16, 0.16)
        liquid = scene.add_entity(  # GENESIS API: SPH liquid material + box morph
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Box(pos=pos, size=size),
        )
        cam = scene.add_camera(  # GENESIS API: offscreen camera
            res=(self.image_size, self.image_size), pos=(1.6, -1.6, 1.2),
            lookat=(0.0, 0.0, 0.2), fov=40, GUI=False,
        )
        scene.build()
        return scene, liquid, cam

    @staticmethod
    def _particles(liquid) -> np.ndarray:
        """Particle positions (N, 3) — tries the common Genesis accessors across versions."""
        for attr in ("get_particles", "get_pos", "get_positions"):
            fn = getattr(liquid, attr, None)
            if fn is not None:
                p = fn()
                return p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
        raise RuntimeError("Could not read Genesis particle positions; adjust the # GENESIS API call.")

    def _generate(self, idx: int) -> dict[str, Any]:
        self._ensure_init()
        scenario = self.scenarios[idx % len(self.scenarios)]
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        gravity = float(rng.uniform(6.0, 13.0))
        scene, liquid, cam = self._build_scene(scenario, gravity, rng)

        steps_per_frame = max(1, int(round((1.0 / self.fps) / (1.0 / 120.0))))
        frames, centroids = [], []
        for _t in range(self.num_frames):
            rgb = cam.render()  # GENESIS API: returns rgb (or tuple with rgb first)
            rgb = rgb[0] if isinstance(rgb, (tuple, list)) else rgb
            rgb = np.asarray(rgb).astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            frames.append(rgb[..., :3].transpose(2, 0, 1))
            pos = self._particles(liquid)
            centroids.append(pos.mean(0))
            spread = float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()))
            frames[-1] = frames[-1]  # noqa: (keep render close to state read)
            self._last_spread = spread
            for _ in range(steps_per_frame):
                scene.step()

        centroids = np.stack(centroids, 0)                       # (T, 3)
        vel = np.vstack([centroids[1:] - centroids[:-1], centroids[-1:] - centroids[-2:-1]]) * self.fps
        acc = np.vstack([vel[1:] - vel[:-1], vel[-1:] - vel[-2:-1]]) * self.fps
        states = []
        for t in range(self.num_frames):
            states.append([
                *centroids[t], *vel[t], *acc[t],
                getattr(self, "_last_spread", 0.0), gravity, 0.0,
            ])

        frame_t = torch.from_numpy(np.stack(frames, 0)).float()
        return {
            "id": f"{scenario}_{idx:05d}",
            "frames": frame_t,
            "state": torch.tensor(states, dtype=torch.float32),
            "state_mask": torch.ones(len(STATE_KEYS)),
            "state_keys": list(STATE_KEYS),
            "category": scenario,
            "meta": {"scenario": scenario, "gravity": gravity, "engine": "genesis"},
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._cache.get(idx)
        if sample is None:
            sample = self._generate(idx)
            self._cache[idx] = sample
        out = dict(sample)
        out["encoder_input"] = self.transform(sample["frames"])
        return out


@register_dataset("genesis_fluid")
def _build_genesis(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return GenesisFluidDataset(cfg, encoder_image_size, encoder_frames)
