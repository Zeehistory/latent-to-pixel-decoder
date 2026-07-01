"""Measure the ball's motion directly from rendered/decoded pixels (Step 2 velocity verification).

The clean moving-ball dataset is a dark disk on a white background, so we can recover the ball's
per-frame center by an intensity-weighted centroid of the *dark* pixels — no learned tracker needed.
From the centroid track we read off the empirical **velocity** (mean inter-frame displacement) and
**speed**. This is the objective, pixel-level evidence for velocity steering: after we add ``alpha*d_v``
to the latents and decode, does the *decoded ball actually move faster/slower/in a new direction*?

These functions take ``(T, C, H, W)`` float frames in ``[0, 1]`` (the decoder's output) and return
normalized-coordinate quantities (units of image width per frame), matching the dataset's state units.
"""

from __future__ import annotations

import numpy as np
import torch


def ball_centroids(frames: torch.Tensor, darkness_thresh: float = 0.5) -> np.ndarray:
    """Per-frame ball center ``(T, 2)`` in normalized [0,1] coords (x, y), NaN if the ball is absent.

    Works on a dark-ball / light-background scene: weight each pixel by how *dark* it is relative to the
    background and take the weighted centroid. Frames with negligible dark mass (e.g. the ball fully
    occluded) yield NaN so callers can skip them.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    t, _c, h, w = x.shape
    gray = x.mean(dim=1)  # (T, H, W), ~1 background, ~0 ball
    dark = (1.0 - gray).clamp(min=0.0)
    dark = torch.where(dark > (1.0 - darkness_thresh), dark, torch.zeros_like(dark))
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    mass = dark.sum(dim=(1, 2))  # (T,)
    cx = (dark * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    cy = (dark * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    out = torch.stack([cx, cy], dim=1).numpy()
    out[mass.numpy() < 1e-3] = np.nan
    return out


def measured_velocity(frames: torch.Tensor, darkness_thresh: float = 0.5) -> dict[str, float]:
    """Empirical velocity from the decoded frames.

    Returns ``{vel_x, vel_y, speed, n_valid}`` where velocity is the mean inter-frame displacement of
    the centroid (normalized units per frame) over frames where the ball is visible. ``speed`` is its
    magnitude. ``n_valid`` is how many consecutive-visible frame pairs contributed.
    """
    c = ball_centroids(frames, darkness_thresh)
    disp = np.diff(c, axis=0)  # (T-1, 2)
    valid = ~np.isnan(disp).any(axis=1)
    if valid.sum() == 0:
        return {"vel_x": float("nan"), "vel_y": float("nan"), "speed": float("nan"), "n_valid": 0}
    v = disp[valid].mean(axis=0)
    return {
        "vel_x": float(v[0]), "vel_y": float(v[1]),
        "speed": float(np.linalg.norm(v)), "n_valid": int(valid.sum()),
    }


def measured_bounce(
    frames: torch.Tensor,
    darkness_thresh: float = 0.5,
    incoming_hint: dict[str, float] | None = None,
) -> dict[str, float]:
    """Empirical bounce metrics from decoded frames.

    Finds the first downward-then-upward vertical velocity sign change as the bounce, then reports:
    ``speed_ratio`` (post-bounce speed / pre-bounce speed), ``rebound_peak_y`` (minimum y after bounce,
    i.e. highest point on screen), and ``post_bounce_speed``. ``incoming_hint`` may supply
    ``bounce_frame`` from GT to align timing when the tracker is noisy.
    """
    c = ball_centroids(frames, darkness_thresh)
    if np.isnan(c).all():
        return _empty_bounce()

    disp = np.diff(c, axis=0)
    valid = ~np.isnan(disp).any(axis=1)
    if valid.sum() < 3:
        return _empty_bounce()

    vy = disp[:, 1]
    vx = disp[:, 0]
    speeds = np.linalg.norm(disp, axis=1)

    bounce_idx = -1
    hint = int(incoming_hint.get("bounce_frame", -1)) if incoming_hint else -1
    if 0 <= hint < len(vy) - 1:
        bounce_idx = hint
    else:
        for i in range(1, len(vy)):
            if not valid[i - 1] or not valid[i]:
                continue
            if vy[i - 1] < -1e-5 and vy[i] > 1e-5:
                bounce_idx = i
                break

    if bounce_idx < 1:
        return _empty_bounce(n_valid=int(valid.sum()))

    pre = speeds[max(0, bounce_idx - 2):bounce_idx]
    post = speeds[bounce_idx:min(len(speeds), bounce_idx + 3)]
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    pre_sp = float(np.mean(pre)) if len(pre) else float("nan")
    post_sp = float(np.mean(post)) if len(post) else float("nan")
    ratio = post_sp / (pre_sp + 1e-9) if np.isfinite(pre_sp) and pre_sp > 1e-6 else float("nan")

    post_y = c[bounce_idx + 1 :]
    post_y = post_y[~np.isnan(post_y[:, 1]), 1] if len(post_y) else np.array([])
    peak_y = float(np.min(post_y)) if len(post_y) else float("nan")

    post_v = disp[bounce_idx: bounce_idx + 2]
    post_v = post_v[~np.isnan(post_v).any(axis=1)]
    pv = post_v.mean(axis=0) if len(post_v) else np.array([float("nan"), float("nan")])

    return {
        "speed_ratio": float(ratio),
        "post_bounce_speed": post_sp,
        "incoming_speed": pre_sp,
        "rebound_peak_y": peak_y,
        "bounce_frame": int(bounce_idx),
        "post_vel_x": float(pv[0]) if len(pv) else float("nan"),
        "post_vel_y": float(pv[1]) if len(pv) else float("nan"),
        "n_valid": int(valid.sum()),
    }


def _empty_bounce(n_valid: int = 0) -> dict[str, float]:
    nan = float("nan")
    return {
        "speed_ratio": nan, "post_bounce_speed": nan, "incoming_speed": nan,
        "rebound_peak_y": nan, "bounce_frame": -1,
        "post_vel_x": nan, "post_vel_y": nan, "n_valid": n_valid,
    }
