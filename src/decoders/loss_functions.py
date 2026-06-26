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


def foreground_weighted_charbonnier(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 10.0, eps: float = 1e-3,
    fg_thresh: float = 0.5,
) -> torch.Tensor:
    """Charbonnier reconstruction that scores the (dark) foreground separately from the background.

    On the clean moving-ball scene the object is a small dark disk (~2% of pixels) on a white
    background, so an *area-averaged* pixel loss is minimized by predicting a blank frame and the
    decoder collapses (ball dropped entirely). We instead compute the charbonnier **mean over the
    foreground** and **mean over the background** independently and return ``bg + gamma * fg`` so the
    tiny ball contributes on equal footing with the whole background regardless of its area.

    Foreground membership uses a **hard darkness threshold** (``1-target > fg_thresh``). An earlier
    *soft* darkness weight (``w = 1-target``) failed: the background's residual darkness (~0.05) leaks
    into the foreground denominator (98% of pixels × 0.05 >> 2% of pixels × 0.8), diluting the ball
    gradient ~4x, so the decoder still collapsed to a uniform frame (observed: fg4 step_300 rel-darkness
    decayed 0.039→0.01). With a hard mask the foreground mean is taken over ball pixels ONLY, so a
    uniform-collapse prediction incurs the full ``gamma * ~0.77`` foreground penalty (vs a diluted
    ~0.19) and the only way down is to actually localize and darken the ball.
    """
    err = torch.sqrt((pred - target) ** 2 + eps**2)                    # (B,T,C,H,W)
    dark = (1.0 - target.mean(dim=2, keepdim=True)).clamp(min=0.0)     # (B,T,1,H,W) target darkness
    fg = (dark > fg_thresh).to(err.dtype)                              # hard ball mask (no bg leak)
    bg = 1.0 - fg
    fg_err = (err * fg).sum() / fg.expand_as(err).sum().clamp(min=1.0)  # mean error on the ball ONLY
    bg_err = (err * bg).sum() / bg.expand_as(err).sum().clamp(min=1.0)  # mean error on the background
    return bg_err + gamma * fg_err


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


def _soft_ball_centroid(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable per-frame centroid of the dark ball. Returns (centroid (B,T,2), mass (B,T), dark²-weighted spread (B,T)).

    Weights each pixel by ``darkness²`` (= ``(1-gray)²``) so the dark ball (darkness ~0.8 -> w ~0.66)
    dominates the faint white background (darkness ~0.05 -> w ~0.003) by ~250x per pixel — the centroid
    tracks the ball, not the frame center. Fully differentiable (soft mass-weighted mean), so it can
    supervise WHERE the rendered ball is, frame by frame.
    """
    gray = frames.mean(dim=2)                                   # (B,T,H,W) ~1 bg, ~0 ball
    w = (1.0 - gray).clamp(min=0.0) ** 2                        # dark² weighting
    b, t, h, wd = w.shape
    xs = torch.linspace(0, 1, wd, device=w.device).view(1, 1, 1, wd)
    ys = torch.linspace(0, 1, h, device=w.device).view(1, 1, h, 1)
    mass = w.sum(dim=(2, 3))                                    # (B,T)
    denom = mass.clamp(min=1e-4)
    cx = (w * xs).sum(dim=(2, 3)) / denom
    cy = (w * ys).sum(dim=(2, 3)) / denom
    cen = torch.stack([cx, cy], dim=-1)                        # (B,T,2)
    # second moment about the centroid = how spread-out the dark mass is (a path-covering smear is large)
    var = (w * ((xs - cx[..., None, None]) ** 2 + (ys - cy[..., None, None]) ** 2)).sum(dim=(2, 3)) / denom
    return cen, mass, var


def frame_position_loss(pred: torch.Tensor, target_state: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """MSE between the rendered ball's per-frame centroid and the GT per-frame position.

    THE fix for temporal-average blur: an L1/SSIM pixel loss is happy to render the ball as a static
    smear covering its whole path (low average pixel error), whose centroid barely moves -> the decoded
    speed collapses ~4x. This loss ties the *rendered* ball's centroid to the exact GT position at EVERY
    frame, so the only way down is to render the ball translating at the true speed. Uses GT position
    (normalized [0,1], same convention as the soft centroid). The moving_ball dataset names the columns
    ``pos_x``/``pos_y``; multi-object synthetic_physics uses ``obj0_pos_x``/``obj0_pos_y`` — accept both.
    """
    def _find(*names: str) -> int | None:
        for n in names:
            if n in state_keys:
                return state_keys.index(n)
        return None

    xi = _find("pos_x", "obj0_pos_x")
    yi = _find("pos_y", "obj0_pos_y")
    if xi is None or yi is None:
        return pred.new_zeros(())
    gt = torch.stack([target_state[..., xi], target_state[..., yi]], dim=-1)  # (B,T,2)
    cen, _, _ = _soft_ball_centroid(pred)
    return F.mse_loss(cen, gt)


def frame_spread_loss(pred: torch.Tensor, max_var: float = 0.01) -> torch.Tensor:
    """Penalize a rendered dark mass that is more SPREAD OUT than a compact disk (anti path-smear).

    Only penalizes spread in EXCESS of ``max_var`` (a disk of radius ~0.11 has 2nd moment ~0.006), so a
    correct compact ball is free while a path-covering smear is pushed down. Complements the centroid
    loss: centroid says *where*, spread says *don't smear across the trajectory*.
    """
    _, _, var = _soft_ball_centroid(pred)
    return (var - max_var).clamp(min=0.0).mean()


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
            lo, hi = getattr(c, "target_lo", 0.0), getattr(c, "target_hi", 1.0)
            if lo != 0.0 or hi != 1.0:
                # compress targets off the sigmoid boundary so the decoder optimum sits at a finite,
                # non-saturated logit (keeps gradients alive -> escapes uniform-collapse on white scenes)
                target_frames = lo + (hi - lo) * target_frames
            add("charbonnier", c.charbonnier, charbonnier_loss(pred_frames, target_frames))
            if getattr(c, "foreground", 0.0) != 0.0:
                add("foreground", c.foreground,
                    foreground_weighted_charbonnier(pred_frames, target_frames,
                                                    getattr(c, "foreground_gamma", 50.0)))
            add("ssim", c.ssim, ssim_loss(pred_frames, target_frames))
            add("ms_ssim", c.ms_ssim, ms_ssim_loss(pred_frames, target_frames))
            if c.lpips != 0.0:
                add("lpips", c.lpips, lpips_loss(pred_frames, target_frames))
            add("temporal", c.temporal_consistency, temporal_consistency_loss(pred_frames, target_frames))
            # Per-frame rendered-ball position supervision (kills temporal-average blur). Uses the raw
            # rendered frames + GT position; independent of the target compression above.
            if target_state is not None and state_keys is not None:
                add("frame_position", getattr(c, "frame_position", 0.0),
                    frame_position_loss(pred_frames, target_state, state_keys))
            add("frame_spread", getattr(c, "frame_spread", 0.0),
                frame_spread_loss(pred_frames, getattr(c, "frame_spread_max_var", 0.01)))

        if pred_state is not None and target_state is not None and state_keys is not None:
            add("state", c.state, masked_state_loss(pred_state, target_state, state_mask))
            add("trajectory", c.trajectory, trajectory_loss(pred_state, target_state, state_keys))
            add("velocity", c.velocity, velocity_loss(pred_state, target_state, state_keys))
            add("acceleration", c.acceleration, acceleration_loss(pred_state, target_state, state_keys))
            add("collision", c.collision, collision_loss(pred_state, target_state, state_keys))

        logs["total"] = float(total.detach())
        return total, logs
