"""Restitution subspace / operator toolkit (Step 2 extension — bounce steering).

Mirrors :mod:`velocity_ops` for the ``scene_restitution`` dataset: within-scene pairs share incoming
trajectory and differ only in wall coefficient of restitution ``e``. The difference vector
``Delta H = H_b - H_a`` isolates the restitution factor for on-manifold edits and command operators.
"""
from __future__ import annotations

import numpy as np

from src.analysis import velocity_ops as vo

# Re-export shared latent utilities so callers need one import.
group_scenes = vo.group_scenes
scene_rank = vo.scene_rank
layer_flat = vo.layer_flat
roll_layer = vo.roll_layer
clip_start_pos = vo.clip_start_pos
canon_shift = vo.canon_shift
pca_gram = vo.pca_gram
explained_curve = vo.explained_curve
participation_ratio = vo.participation_ratio
project = vo.project
random_basis = vo.random_basis
cosine = vo.cosine
rel_error = vo.rel_error
LinearLS = vo.LinearLS


def clip_restitution(sample: dict) -> float:
    """Ground-truth restitution coefficient ``e`` for a clip (from dataset meta)."""
    meta = sample.get("meta") or {}
    if "restitution" in meta:
        return float(meta["restitution"])
    raise KeyError("sample meta missing 'restitution' (not a scene_restitution cache?)")


def clip_incoming_velocity(sample: dict) -> np.ndarray:
    """Incoming (pre-bounce) velocity vector from meta or frame-0 state."""
    meta = sample.get("meta") or {}
    if "incoming_vel_x" in meta and "incoming_vel_y" in meta:
        return np.array([float(meta["incoming_vel_x"]), float(meta["incoming_vel_y"])], dtype=np.float64)
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return np.array([
        float(state[0, keys.index("obj0_vel_x")]),
        float(state[0, keys.index("obj0_vel_y")]),
    ], dtype=np.float64)


def clip_bounce_metrics(sample: dict) -> dict[str, float]:
    """GT bounce summary from dataset meta (speed ratio, rebound height proxy)."""
    meta = sample.get("meta") or {}
    return {
        "restitution": float(meta.get("restitution", float("nan"))),
        "speed_ratio": float(meta.get("speed_ratio", float("nan"))),
        "post_bounce_speed": float(meta.get("post_bounce_speed", float("nan"))),
        "incoming_speed": float(meta.get("incoming_speed", float("nan"))),
        "bounce_frame": float(meta.get("bounce_frame", -1)),
        "rebound_peak_y": float(meta.get("rebound_peak_y", float("nan"))),
    }


def gt_rebound_peak_y(sample: dict) -> float:
    """Minimum y (highest screen position) after the bounce frame — from packed state."""
    meta = sample.get("meta") or {}
    bf = int(meta.get("bounce_frame", -1))
    if bf < 0:
        return float("nan")
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    yi = keys.index("obj0_pos_y")
    post = state[bf + 1 :, yi]
    return float(np.min(post)) if len(post) else float("nan")


COMMAND_FEATURE_DIM = 11


def command_features(
    ea: float,
    eb: float,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
) -> np.ndarray:
    """Rich command vector for synthesizing a restitution edit without ``H_b``.

    Columns: ``[1, e_b, e_a, de, e_b^2, e_a*e_b, |v_in|, v_in_x, v_in_y, de*|v_in|, de*v_in_y]``.
    Incoming velocity is shared within a scene but included as context for cross-scene generalization.
    """
    de = float(eb - ea)
    va = np.asarray(va if va is not None else [0.0, 0.0], dtype=np.float64).reshape(2)
    vb = np.asarray(vb if vb is not None else va, dtype=np.float64).reshape(2)
    vin = va  # identical across ranks in scene_restitution
    sp = float(np.linalg.norm(vin))
    return np.array([
        1.0, float(eb), float(ea), de, float(eb) ** 2, float(ea) * float(eb),
        sp, float(vin[0]), float(vin[1]), de * sp, de * float(vin[1]),
    ], dtype=np.float64)


class ScalarRidgeOperator:
    """Linear map ``B (D,)`` with ``Delta H ~= B * de``, fit by streaming normal equations."""

    def __init__(self, dim: int, ridge: float = 1.0):
        self.dim = dim
        self.ridge = ridge
        self.xtx = 0.0
        self.xty = np.zeros(dim)
        self.n = 0

    def add(self, de: float, dH: np.ndarray) -> None:
        de = float(de)
        self.xtx += de * de
        self.xty += de * dH
        self.n += 1

    def solve(self) -> np.ndarray:
        denom = self.xtx + self.ridge
        self.B = self.xty / denom  # (D,)
        return self.B

    def predict(self, de: float) -> np.ndarray:
        return float(de) * self.B
