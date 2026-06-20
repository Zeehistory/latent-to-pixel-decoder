"""Loss functions for decoder training.

A weighted mixture of pixel, perceptual, temporal, and physics-aware terms. Each term is its own
function so it can be unit-tested; :class:`DecoderLoss` combines them per the loss config. Terms whose
weight is zero are skipped (no wasted compute). Optional perceptual loss (LPIPS) is imported lazily and
skipped with a warning if the dependency is absent — never silently faked.

Frame tensors are ``(B, T, C, H, W)`` in ``[0, 1]``; state tensors are ``(B, T, state_dim)``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1 variant: ``sqrt((x-y)^2 + eps^2)``."""
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Mean SSIM over frames. Inputs ``(B, T, C, H, W)`` in ``[0, 1]``. Returns SSIM in ``[0, 1]``."""
    b, t, c, h, w = pred.shape
    p = pred.reshape(b * t, c, h, w)
    g = target.reshape(b * t, c, h, w)
    win = _gaussian_window(window_size, 1.5, c, pred.device)
    p, g = p.float(), g.float()
    pad = window_size // 2
    mu_p = F.conv2d(p, win, padding=pad, groups=c)
    mu_g = F.conv2d(g, win, padding=pad, groups=c)
    mu_p2, mu_g2, mu_pg = mu_p**2, mu_g**2, mu_p * mu_g
    sig_p = F.conv2d(p * p, win, padding=pad, groups=c) - mu_p2
    sig_g = F.conv2d(g * g, win, padding=pad, groups=c) - mu_g2
    sig_pg = F.conv2d(p * g, win, padding=pad, groups=c) - mu_pg
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_pg + c1) * (2 * sig_pg + c2)) / ((mu_p2 + mu_g2 + c1) * (sig_p + sig_g + c2))
    return s.mean().clamp(0, 1)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, scales: int = 3) -> torch.Tensor:
    total = 0.0
    p, g = pred, target
    b, t = pred.shape[:2]
    for i in range(scales):
        total = total + (1.0 - ssim(p, g))
        if i < scales - 1:
            p = F.avg_pool2d(p.flatten(0, 1), 2).reshape(b, t, p.shape[2], p.shape[3] // 2, p.shape[4] // 2)
            g = F.avg_pool2d(g.flatten(0, 1), 2).reshape(b, t, g.shape[2], g.shape[3] // 2, g.shape[4] // 2)
    return total / scales


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match frame-to-frame deltas, encouraging consistent motion rather than per-frame independence."""
    dp = pred[:, 1:] - pred[:, :-1]
    dg = target[:, 1:] - target[:, :-1]
    return F.l1_loss(dp, dg)


def masked_state_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """MSE over per-frame state, honoring a per-column validity ``mask`` ``(B, state_dim)``."""
    err = (pred - target) ** 2
    if mask is not None:
        m = mask.unsqueeze(1)  # (B, 1, state_dim)
        denom = m.sum().clamp(min=1.0)
        return (err * m).sum() / denom / pred.shape[1]
    return err.mean()


def _state_cols(state_keys: list[str], substr: str) -> list[int]:
    return [i for i, k in enumerate(state_keys) if substr in k]


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """L2 on position columns across time (object trajectories)."""
    cols = _state_cols(state_keys, "pos_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def velocity_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    cols = _state_cols(state_keys, "vel_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def acceleration_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    cols = _state_cols(state_keys, "acc_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def collision_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """Binary cross-entropy on the collision-event column."""
    cols = _state_cols(state_keys, "collision_event")
    if not cols:
        return pred.new_zeros(())
    logits = pred[..., cols]
    tgt = target[..., cols].clamp(0, 1)
    return F.binary_cross_entropy_with_logits(logits, tgt)


class _LPIPS:
    """Lazy LPIPS holder; returns None if the optional dependency is missing."""

    _net: Any = None
    _warned = False

    @classmethod
    def get(cls):
        if cls._net is not None:
            return cls._net
        try:
            import lpips

            cls._net = lpips.LPIPS(net="alex")
            cls._net.eval()
            for p in cls._net.parameters():
                p.requires_grad_(False)
            return cls._net
        except Exception:
            if not cls._warned:
                warnings.warn("LPIPS unavailable (pip install -e .[extras]); perceptual loss skipped.", stacklevel=2)
                cls._warned = True
            return None


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    net = _LPIPS.get()
    if net is None:
        return pred.new_zeros(())
    net = net.to(pred.device)
    b, t, c, h, w = pred.shape
    p = pred.reshape(b * t, c, h, w).float() * 2 - 1
    g = target.reshape(b * t, c, h, w).float() * 2 - 1
    return net(p, g).mean()


class DecoderLoss(torch.nn.Module):
    """Weighted combination of the above terms, driven by a loss config."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred_frames: torch.Tensor | None,
        target_frames: torch.Tensor | None,
        pred_state: torch.Tensor | None = None,
        target_state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        state_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        c = self.cfg
        device = (pred_frames if pred_frames is not None else pred_state).device
        total = torch.zeros((), device=device)
        logs: dict[str, float] = {}

        def add(name: str, weight: float, value: torch.Tensor) -> None:
            nonlocal total
            if weight != 0.0:
                total = total + weight * value
                logs[name] = float(value.detach())

        if pred_frames is not None and target_frames is not None:
            if target_frames.shape != pred_frames.shape:
                target_frames = F.interpolate(
                    target_frames.flatten(0, 1), size=pred_frames.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).reshape(pred_frames.shape)
            add("charbonnier", c.charbonnier, charbonnier_loss(pred_frames, target_frames))
            add("ssim", c.ssim, ssim_loss(pred_frames, target_frames))
            add("ms_ssim", c.ms_ssim, ms_ssim_loss(pred_frames, target_frames))
            if c.lpips != 0.0:
                add("lpips", c.lpips, lpips_loss(pred_frames, target_frames))
            add("temporal", c.temporal_consistency, temporal_consistency_loss(pred_frames, target_frames))

        if pred_state is not None and target_state is not None and state_keys is not None:
            add("state", c.state, masked_state_loss(pred_state, target_state, state_mask))
            add("trajectory", c.trajectory, trajectory_loss(pred_state, target_state, state_keys))
            add("velocity", c.velocity, velocity_loss(pred_state, target_state, state_keys))
            add("acceleration", c.acceleration, acceleration_loss(pred_state, target_state, state_keys))
            add("collision", c.collision, collision_loss(pred_state, target_state, state_keys))

        logs["total"] = float(total.detach())
        return total, logs
