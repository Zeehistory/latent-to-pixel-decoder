"""Data pipeline: synthetic physics generator, Physics-IQ / DROID loaders, transforms, registry."""

from __future__ import annotations

# Import submodules for their registry side effects so `build_dataset` resolves all names.
from . import droid, physics_iq  # noqa: F401
from .dataset_registry import build_dataset, collate, register_dataset
from .synthetic_physics import Clip, SyntheticPhysics, state_dim_for
from .video_transforms import VideoTransform

__all__ = [
    "build_dataset",
    "collate",
    "register_dataset",
    "SyntheticPhysics",
    "Clip",
    "state_dim_for",
    "VideoTransform",
]
