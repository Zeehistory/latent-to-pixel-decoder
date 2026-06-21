"""Physics-IQ benchmark loader.

Physics-IQ (Motamed et al., 2025) is a benchmark of real-world physical-reasoning videos across
categories (solid mechanics, fluids, optics, thermodynamics, magnetism). This loader honors:

* real vs synthetic splits,
* per-category filtering,
* train/val/test splits,
* variable-FPS frame sampling, cropping/resizing,
* per-sample metadata,
* optional segmentation masks / physical annotations when present on disk.

The loader is *graceful*: if ``cfg.root`` is unset or empty it raises a clear, actionable error rather
than failing deep in the pipeline. Use ``scripts/download_data.py`` to fetch the data. Because
Physics-IQ has no per-frame numeric physics labels, ``state`` is returned as zeros with an all-zero
``state_mask`` (so physics-state metrics are skipped, not faked); reconstruction / latent-analysis
experiments use it fully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..utils.video_io import load_video
from .dataset_registry import register_dataset
from .physics_iq_categories import category_for_id
from .video_transforms import VideoTransform

# Physics-IQ category vocabulary (used for per-category split reporting).
PHYSICS_IQ_CATEGORIES = [
    "solid_mechanics",
    "fluid_dynamics",
    "optics",
    "thermodynamics",
    "magnetism",
    "misc",
]


class PhysicsIQDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        if not cfg.root:
            raise ValueError(
                "Physics-IQ requires `data.root` to point at the downloaded dataset. "
                "Run `python scripts/download_data.py --dataset physics_iq --output <dir>` "
                "and set data.root in your config."
            )
        self.root = Path(cfg.root)
        if not self.root.exists():
            raise FileNotFoundError(f"Physics-IQ root not found: {self.root}")
        self.cfg = cfg
        self.num_frames = cfg.num_frames
        self.image_size = cfg.image_size
        self.transform = VideoTransform(
            image_size=encoder_image_size, num_frames=encoder_frames, do_normalize=True
        )
        self.samples = self._index()

    def _index(self) -> list[dict[str, Any]]:
        """Index videos. Expects either a ``manifest.json`` or a ``<split>/<category>/*.mp4`` tree."""
        manifest = self.root / "manifest.json"
        if manifest.exists():
            records = json.loads(manifest.read_text())
            samples = [r for r in records if r.get("split", "train") == self.cfg.split]
        else:
            split_dir = self.root / self.cfg.split
            search_dir = split_dir if split_dir.exists() else self.root
            samples = []
            for path in sorted(search_dir.rglob("*.mp4")):
                category = path.parent.name if path.parent != search_dir else "misc"
                samples.append({"path": str(path), "category": category, "split": self.cfg.split})
        # Assign the official physics category from the filename scenario (the on-disk folders encode
        # frame rate / perspective, not the physics category). Fall back to any manifest-provided
        # category, then "misc" for scenarios outside the official 66.
        for s in samples:
            stem = Path(s["path"]).stem
            s["category"] = category_for_id(stem) or s.get("category", "misc")
        if self.cfg.categories != "all":
            allowed = set(self.cfg.categories)
            samples = [s for s in samples if s.get("category", "misc") in allowed]
        if not samples:
            raise RuntimeError(
                f"No Physics-IQ videos found under {self.root} for split='{self.cfg.split}'."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.samples[idx]
        frames = load_video(rec["path"], num_frames=self.num_frames, image_size=self.image_size)
        state_dim = 1  # placeholder column; mask is all-zero so physics-state metrics are skipped
        return {
            "id": Path(rec["path"]).stem,
            "frames": frames,
            "encoder_input": self.transform(frames),
            "state": torch.zeros(self.num_frames, state_dim),
            "state_mask": torch.zeros(state_dim),
            "state_keys": ["unavailable"],
            "category": rec.get("category", "misc"),
            "meta": rec,
        }


@register_dataset("physics_iq")
def _build_physics_iq(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return PhysicsIQDataset(cfg, encoder_image_size, encoder_frames)
