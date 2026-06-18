"""Dataset registry + torch ``Dataset`` wrappers.

Every sample is a dict with a stable schema so downstream code (extraction, decoding, probing) is
dataset-agnostic:

    {
        "id":           str,
        "frames":       (T, C, H, W) float in [0, 1]   # un-normalized; raw pixels for decoder targets
        "encoder_input":(T, C, H, W) float              # encoder-preprocessed (resized + normalized)
        "state":        (T, state_dim) float            # GT physical state (zeros if unavailable)
        "state_mask":   (state_dim,) float              # 1 where state is valid for this sample
        "state_keys":   list[str]
        "category":     str
        "meta":         dict
    }
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from .synthetic_physics import SyntheticPhysics, state_dim_for
from .video_transforms import VideoTransform

_REGISTRY: dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str) -> Callable[[Callable[..., Dataset]], Callable[..., Dataset]]:
    def deco(fn: Callable[..., Dataset]) -> Callable[..., Dataset]:
        _REGISTRY[name] = fn
        return fn

    return deco


def build_dataset(cfg: Any, encoder_image_size: int = 256, encoder_frames: int | None = None) -> Dataset:
    """Instantiate a dataset from a data config (``cfg.name`` resolves the builder)."""
    name = cfg.name
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, encoder_image_size=encoder_image_size, encoder_frames=encoder_frames)


def _pad_state(state: torch.Tensor, target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``(T, d)`` state to ``(T, target_dim)`` and return a validity mask over columns."""
    t, d = state.shape
    mask = torch.zeros(target_dim)
    mask[:d] = 1.0
    if d == target_dim:
        return state, mask
    out = torch.zeros(t, target_dim)
    out[:, :d] = state
    return out, mask


class SyntheticPhysicsDataset(Dataset):
    """Wraps :class:`SyntheticPhysics`, padding state to a fixed width across scenarios."""

    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        scenarios = list(cfg.scenarios) if cfg.scenarios != "all" else [
            "bouncing_ball", "projectile", "collision", "pendulum", "two_body", "occlusion"
        ]
        self.gen = SyntheticPhysics(
            image_size=cfg.image_size,
            num_frames=cfg.num_frames,
            scenarios=scenarios,
            seed=cfg.seed,
        )
        self.num_clips = cfg.num_clips
        self.state_dim = state_dim_for(scenarios)
        self.transform = VideoTransform(
            image_size=encoder_image_size,
            num_frames=encoder_frames,
            do_normalize=True,
        )
        # Cache clips (tiny dataset) for determinism + speed.
        self._cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip = self._cache.get(idx)
        if clip is None:
            clip = self.gen.generate(idx)
            self._cache[idx] = clip
        state, mask = _pad_state(clip.state, self.state_dim)
        return {
            "id": f"{clip.meta['scenario']}_{idx:05d}",
            "frames": clip.frames,  # raw pixels (decoder target)
            "encoder_input": self.transform(clip.frames),
            "state": state,
            "state_mask": mask,
            "state_keys": clip.state_keys,
            "category": clip.meta["scenario"],
            "meta": clip.meta,
        }


@register_dataset("synthetic_physics")
def _build_synthetic(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return SyntheticPhysicsDataset(cfg, encoder_image_size, encoder_frames)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that stacks tensors and keeps python-side metadata as lists."""
    out: dict[str, Any] = {}
    tensor_keys = ["frames", "encoder_input", "state", "state_mask"]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["id"] = [b["id"] for b in batch]
    out["category"] = [b["category"] for b in batch]
    out["state_keys"] = batch[0]["state_keys"]
    out["meta"] = [b["meta"] for b in batch]
    return out
