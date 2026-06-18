"""Publication-quality visualizations, saved automatically to an organized output directory.

All plotting uses a non-interactive Matplotlib backend so it runs headless (CI / clusters). Functions
return the path written. Tensors are ``(T, C, H, W)`` in ``[0, 1]`` unless noted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..utils.video_io import save_video  # noqa: E402


def _to_hwc(frames: torch.Tensor) -> np.ndarray:
    return frames.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()


def reconstruction_grid(
    original: torch.Tensor, recon: torch.Tensor, path: str | Path, max_frames: int = 8
) -> Path:
    """Two-row grid: original (top) vs reconstruction (bottom)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o, r = _to_hwc(original), _to_hwc(recon)
    t = min(max_frames, o.shape[0], r.shape[0])
    fig, axes = plt.subplots(2, t, figsize=(1.6 * t, 3.4))
    if t == 1:
        axes = axes.reshape(2, 1)
    for i in range(t):
        axes[0, i].imshow(o[i]); axes[0, i].axis("off")
        axes[1, i].imshow(r[i]); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", rotation=0, ha="right", labelpad=30)
    axes[1, 0].set_ylabel("recon", rotation=0, ha="right", labelpad=30)
    fig.suptitle("Reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def error_map(original: torch.Tensor, recon: torch.Tensor, path: str | Path, max_frames: int = 8) -> Path:
    """Per-pixel absolute error heatmap over frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if recon.shape[-1] != original.shape[-1]:
        recon = torch.nn.functional.interpolate(recon, size=original.shape[-2:], mode="bilinear",
                                                align_corners=False)
    err = (original - recon).abs().mean(1).detach().cpu().numpy()  # (T, H, W)
    t = min(max_frames, err.shape[0])
    fig, axes = plt.subplots(1, t, figsize=(1.6 * t, 1.9))
    if t == 1:
        axes = [axes]
    for i in range(t):
        axes[i].imshow(err[i], cmap="magma", vmin=0, vmax=max(err.max(), 1e-3))
        axes[i].axis("off")
    fig.suptitle("Absolute error")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def trajectory_overlay(
    frames: torch.Tensor,
    state: torch.Tensor,
    state_keys: list[str],
    path: str | Path,
    draw_velocity: bool = True,
) -> Path:
    """Overlay object trajectories (and optional velocity arrows) on the last frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_hwc(frames)[-1]
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    keys = state_keys
    n_obj = sum(1 for k in keys if k.endswith("pos_x"))
    s = state.detach().cpu().numpy()
    for o in range(n_obj):
        try:
            xi = keys.index(f"obj{o}_pos_x"); yi = keys.index(f"obj{o}_pos_y")
        except ValueError:
            continue
        xs, ys = s[:, xi] * w, s[:, yi] * h
        ax.plot(xs, ys, "-o", ms=3, lw=1.5, label=f"obj{o}")
        if draw_velocity:
            vxi, vyi = keys.index(f"obj{o}_vel_x"), keys.index(f"obj{o}_vel_y")
            ax.arrow(xs[-1], ys[-1], s[-1, vxi] * w * 4, s[-1, vyi] * h * 4,
                     color="cyan", head_width=3, length_includes_head=True)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Trajectory + velocity")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def layerwise_probe_plot(records: list[dict], path: str | Path) -> Path:
    """Line plot of probe R² by layer for each variable (linear vs MLP), with the control band."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variables = sorted({r["variable"] for r in records})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for var in variables:
        for probe in ("linear", "mlp"):
            pts = sorted([r for r in records if r["variable"] == var and r["probe"] == probe],
                         key=lambda r: r["layer"])
            if not pts:
                continue
            ls = "-" if probe == "linear" else "--"
            ax.plot([p["layer"] for p in pts], [p["r2"] for p in pts], ls, marker="o", ms=3,
                    label=f"{var} ({probe})")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("probe R²")
    ax.set_title("Layerwise physical decodability")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def cka_heatmap(matrix: np.ndarray, labels: list[str], path: str | Path, title: str = "CKA") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def panel_video(
    original: torch.Tensor, recon: torch.Tensor, path: str | Path, fps: int = 8
) -> Path:
    """Side-by-side original|reconstruction|error mp4."""
    if recon.shape[-1] != original.shape[-1]:
        recon = torch.nn.functional.interpolate(recon, size=original.shape[-2:], mode="bilinear",
                                                align_corners=False)
    err = (original - recon).abs().clamp(0, 1)
    panel = torch.cat([original, recon, err], dim=-1)  # concat along width
    return save_video(panel, path, fps=fps)


def save_reconstruction_video(recon: torch.Tensor, path: str | Path, fps: int = 8) -> Path:
    return save_video(recon, path, fps=fps)
