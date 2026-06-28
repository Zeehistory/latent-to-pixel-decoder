"""Velocity subspace / operator toolkit (Step 2, PI direction 2026-06-27).

The speed-only result proved that a *per-pair, same-scene* difference vector ``Delta = H_b - H_a`` steers
faithfully, but a single *global* averaged vector does not generalize — the velocity factor is localized
to the spatial tokens the ball occupies, so it lands in different tokens per scene. This module builds the
machinery to go from a per-instance edit to a TRANSFERABLE velocity representation:

* **PCA of Delta H** (within-scene + global): is the local velocity subspace low-rank (~2D = velocity's
  two degrees of freedom)? how much higher is the global rank (the token-misalignment cost)?
* **subspace projection** ``P_U(Delta)``: does projecting the difference onto a learned velocity subspace
  ``U`` preserve the steer, while a random same-rank subspace destroys it?
* **ridge operator** ``F_U: Delta v -> Delta H`` fit at the full flattened dimension by streaming normal
  equations (only a 2x2 inverse), so we can steer straight from a velocity command.
* **canonicalization** ``A_s``: roll the token grid so the ball's start cell is canonical, regress in
  canonical coordinates, map back — to recover cross-scene generalization.

Latents come from :class:`encoders.feature_extractor.LatentDataset`: each sample's ``layers[L]`` is an
``(L_tok, D)`` tensor with ``L_tok = T' * Hp * Wp`` tokens (grid ``(T', Hp, Wp) = (8, 16, 16)``) and
``D = 1024``. Velocity ground truth is read from the packed ``obj0_vel_x/obj0_vel_y`` state columns.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_SCENE_RE = re.compile(r"^scene(\d+)_v(\d+)$")


def scene_rank(sample_id: str) -> tuple[int, int] | None:
    m = _SCENE_RE.match(sample_id)
    return (int(m.group(1)), int(m.group(2))) if m else None


def clip_velocity(sample: dict) -> np.ndarray:
    """Ground-truth 2D velocity of a clip = mean of its (obj0_vel_x, obj0_vel_y) state columns."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    vx = state[:, keys.index("obj0_vel_x")].mean()
    vy = state[:, keys.index("obj0_vel_y")].mean()
    return np.array([float(vx), float(vy)])


def clip_start_pos(sample: dict) -> np.ndarray:
    """Frame-0 ball center (normalized [0,1]) from (obj0_pos_x, obj0_pos_y)."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return np.array([float(state[0, keys.index("obj0_pos_x")]),
                     float(state[0, keys.index("obj0_pos_y")])])


@dataclass
class Scene:
    scene_id: int
    rank_to_idx: dict[int, int] = field(default_factory=dict)


def group_scenes(ds) -> dict[int, dict[int, int]]:
    """Group a LatentDataset into ``{scene_id: {rank: dataset_index}}`` (>=2 ranks only)."""
    scenes: dict[int, dict[int, int]] = defaultdict(dict)
    for i in range(len(ds)):
        sr = scene_rank(ds._ids[i])
        if sr is not None:
            scenes[sr[0]][sr[1]] = i
    return {s: r for s, r in scenes.items() if len(r) >= 2}


# ----------------------------------------------------------------------------------------------------
# flatten / grid helpers (per layer)
# ----------------------------------------------------------------------------------------------------
def layer_flat(sample_layer) -> np.ndarray:
    """(L_tok, D) -> flat (L_tok*D,) float64."""
    return np.asarray(sample_layer, dtype=np.float64).reshape(-1)


def roll_layer(arr_flat: np.ndarray, grid: tuple[int, int, int], shift_hw: tuple[int, int]) -> np.ndarray:
    """Roll a flattened layer's SPATIAL token grid by ``shift_hw = (dh, dw)`` (with wraparound).

    Assumes token order is temporal-major then row-major spatial: index = t*(Hp*Wp) + h*Wp + w. This is
    the standard V-JEPA flattening; Phase-5 validation confirms it empirically (roll -> decode -> the
    ball should appear shifted by the same grid offset). ``A_s`` and ``A_s^{-1}`` are roll/-roll.
    """
    T, H, W = grid
    D = arr_flat.size // (T * H * W)
    x = arr_flat.reshape(T, H, W, D)
    x = np.roll(x, shift=shift_hw, axis=(1, 2))
    return x.reshape(-1)


def pos_to_cell(pos: np.ndarray, grid: tuple[int, int, int]) -> tuple[int, int]:
    """Map a normalized [0,1] (x,y) ball center to a (row=h, col=w) cell in the HpxWp spatial grid.

    Image coords: x is horizontal (-> column w), y is vertical (-> row h). Clamped to the grid.
    """
    _, H, W = grid
    w = int(np.clip(round(pos[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(pos[1] * (H - 1)), 0, H - 1))
    return h, w


def canon_shift(pos: np.ndarray, grid: tuple[int, int, int]) -> tuple[int, int]:
    """Shift (dh, dw) that moves the ball's start cell to the grid CENTER (canonical position)."""
    _, H, W = grid
    h, w = pos_to_cell(pos, grid)
    return (H // 2 - h, W // 2 - w)


# ----------------------------------------------------------------------------------------------------
# PCA of Delta H (per layer)
# ----------------------------------------------------------------------------------------------------
def pca_gram(deltas: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """PCA via the Gram trick for tall-skinny data (N samples, D >> N features).

    ``deltas`` is (N, D). Returns ``(basis (k, D) orthonormal rows, explained_variance (k,))`` where
    basis rows are the top-k right singular vectors of the mean-centered data.
    """
    X = deltas - deltas.mean(0, keepdims=True)
    G = X @ X.T  # (N, N)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0, None)
    V = V[:, order]
    n = int((w > 1e-12 * (w[0] + 1e-30)).sum())
    if k is not None:
        n = min(n, k)
    sv = np.sqrt(w[:n])
    basis = (X.T @ V[:, :n]) / (sv + 1e-12)  # (D, n)
    return basis.T.copy(), w  # (n, D), full eigvals (=variance*N)


def explained_curve(eigvals: np.ndarray, upto: int = 8) -> list[float]:
    """Cumulative fraction of variance captured by the top-1..upto components."""
    pos = np.clip(eigvals, 0, None)
    tot = pos.sum() + 1e-30
    cum = np.cumsum(pos) / tot
    return [float(cum[min(i, len(cum) - 1)]) for i in range(upto)]


def participation_ratio(eigvals: np.ndarray) -> float:
    """Effective dimensionality = (sum lambda)^2 / sum(lambda^2). ~2 means a 2D subspace dominates."""
    pos = np.clip(eigvals, 0, None)
    return float((pos.sum() ** 2) / ((pos ** 2).sum() + 1e-30))


def project(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project ``delta`` (D,) onto the subspace spanned by orthonormal ``basis`` (k, D): P_U delta."""
    c = basis @ delta            # (k,)
    return basis.T @ c           # (D,)


def random_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthonormal (k, D) basis — the same-rank control subspace."""
    A = rng.standard_normal((k, dim))
    Q, _ = np.linalg.qr(A.T)     # (D, k)
    return Q.T.copy()


# ----------------------------------------------------------------------------------------------------
# ridge operator  F_U : Delta v (2,) -> Delta H (D,)    fit by streaming normal equations
# ----------------------------------------------------------------------------------------------------
class RidgeOperator:
    """Linear map B (D x 2) with ``Delta H ~= B Delta v``, fit by accumulating X^T X (2x2) and X^T Y (2xD).

    Solve once: ``B^T = (X^T X + lambda I)^{-1} X^T Y``  (a single 2x2 inverse), so the full flattened
    dimension D = T'*Hp*Wp*1024 is no obstacle — we never materialize Y. ``predict(dv)`` returns B dv.
    """

    def __init__(self, dim: int, ridge: float = 1.0):
        self.dim = dim
        self.ridge = ridge
        self.XtX = np.zeros((2, 2))
        self.XtY = np.zeros((2, dim))
        self.n = 0

    def add(self, dv: np.ndarray, dH: np.ndarray) -> None:
        dv = dv.reshape(2)
        self.XtX += np.outer(dv, dv)
        self.XtY += np.outer(dv, dH)
        self.n += 1

    def solve(self) -> np.ndarray:
        A = self.XtX + self.ridge * np.eye(2)
        self.Bt = np.linalg.solve(A, self.XtY)  # (2, D)
        return self.Bt

    def predict(self, dv: np.ndarray) -> np.ndarray:
        return dv.reshape(2) @ self.Bt          # (D,)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-30))


def rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-30))
