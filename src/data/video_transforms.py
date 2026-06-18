"""Video preprocessing transforms operating on ``(T, C, H, W)`` float tensors in ``[0, 1]``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ImageNet statistics — V-JEPA preprocessing normalizes with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize(frames: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


def center_crop(frames: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = frames.shape
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return frames[:, :, top : top + size, left : left + size]


def normalize(frames: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    m = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    return (frames - m) / s


def denormalize(frames: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    m = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    return frames * s + m


def temporal_subsample(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly sample ``num_frames`` along time (handles variable FPS sources)."""
    t = frames.shape[0]
    idx = torch.linspace(0, max(t - 1, 0), num_frames).round().long().clamp(0, t - 1)
    return frames[idx]


@dataclass
class VideoTransform:
    """Composable preprocessing for encoder input.

    ``image_size`` resize, optional center crop, optional ImageNet normalization, temporal subsample.
    The decoder targets typically use the *un-normalized* frames (``normalize=False``).
    """

    image_size: int = 256
    num_frames: int | None = None
    do_normalize: bool = True
    do_center_crop: bool = False

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if self.num_frames is not None:
            frames = temporal_subsample(frames, self.num_frames)
        frames = resize(frames, self.image_size)
        if self.do_center_crop:
            frames = center_crop(frames, self.image_size)
        if self.do_normalize:
            frames = normalize(frames)
        return frames
