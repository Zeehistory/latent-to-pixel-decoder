"""DROID robotics dataset loader (deferred — Stage 2 extension).

DROID is a large-scale robot-manipulation dataset. In this project robotics is a *second-stage*
extension used to test whether latent failures can be decoded and eventually steered toward successful
task outcomes. The loader is intentionally a typed stub: it documents the intended return contract
(identical to the other datasets, plus action / success-label fields) so the rest of the pipeline can
target it without change.

Contract (planned):
    {
        ...standard sample keys...,
        "actions":       (T, action_dim) float,
        "success":       float (0/1),
        "language_goal": str,
    }
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .dataset_registry import register_dataset


class DroidDataset(Dataset):
    def __init__(self, cfg: Any, encoder_image_size: int, encoder_frames: int | None) -> None:
        raise NotImplementedError(
            "DROID support is a Stage-2 robotics extension and is not implemented yet. "
            "See docs/method.md (roadmap) and src/analysis/steering.py for the intended use."
        )

    def __len__(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:  # pragma: no cover - stub
        raise NotImplementedError


@register_dataset("droid")
def _build_droid(cfg, encoder_image_size, encoder_frames) -> Dataset:
    return DroidDataset(cfg, encoder_image_size, encoder_frames)
