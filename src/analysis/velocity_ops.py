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


def direction_bin(v: np.ndarray, n_bins: int) -> int:
    """Bin a 2D velocity's heading atan2(vy, vx) into ``n_bins`` equal wedges over [0, 2*pi).

    The velocity edit's spatial footprint depends on the ball's heading (different directions traverse
    different tokens), so a single global operator cannot place it correctly. Conditioning the operator
    on the heading bin is the equivariance-free way to make F_U direction-aware: within a wedge the path
    orientation is roughly fixed, so a position-canonicalized ridge per bin can place the edit.
    """
    ang = np.arctan2(float(v[1]), float(v[0])) % (2 * np.pi)
    return int(min(n_bins - 1, int(ang / (2 * np.pi) * n_bins)))


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


def principal_angles_bases(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Principal angles (degrees) between two orthonormal subspaces ``A`` (k,D) and ``B`` (m,D).

    Singular values of ``A @ B.T`` are the cosines of the principal angles. Small angles -> the subspaces
    overlap (share directions); angles near 90 deg -> orthogonal / disentangled. Returns the mean and the
    smallest principal angle (the most-aligned direction pair).
    """
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return {"mean_deg": float(ang.mean()), "min_deg": float(ang.min())}


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Re-orthonormalize rows of ``basis`` (k,D) via QR (defensive; PCA bases are already ~orthonormal)."""
    Q, _ = np.linalg.qr(basis.T)
    return Q.T.copy()


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


# ----------------------------------------------------------------------------------------------------
# masked TRAJECTORY-TRANSPORT operator  (PI direction 2026-06-29)
#
# The command-only ridge F_U: Delta v -> Delta H plateaus at ~34 deg (latent cos 0.39) because Delta v
# says WHAT velocity to write but not WHERE in the token grid to write it; the missing ~63% is the
# scene-local PLACEMENT (which tokens the ball's path occupies). We hand the operator that geometry as
# soft trajectory masks built from ground-truth ball centers and let it learn only velocity->channel:
#
#   Delta H_hat[t,i,j,:] = M_b*(c+_t + v_b @ B+_t)     # target mask: write a ball here (+ motion-specific)
#                        + M_a*(c-_t + v_a @ B-_t)     # source mask: remove the old ball (+ motion-specific)
#                        + M_U*((v_b-v_a) @ Bd_t)      # union mask: correction in the changed tube
#
# Each mask carries a velocity-INDEPENDENT BIAS (c+_t writes "a ball is here", c-_t removes it): the
# dominant part of Delta H near the ball is its PRESENCE -- a dark disk, the same at any speed; velocity
# only sets WHERE, which the mask already encodes. A bias-free mask*(v @ B) form cannot emit that constant
# (it must express "ball present" as a linear function of v, which partly cancels across +/- velocities
# and collapses; empirically gave cos~0.13 / shared cos~-0.95). The v-terms are then the secondary
# motion-specific modulation. LINEAR in the params, so per temporal token t it is a ridge of the 8-dim
# per-token feature phi=[M_b, M_b*v_b, M_a, M_a*v_a, M_U*dv] -> Delta H token (D,); B_t is (8, D). See
# ``LinearLS`` + ``transport_features``.
# ----------------------------------------------------------------------------------------------------
def clip_positions(sample: dict) -> np.ndarray:
    """Per-frame ball centers ``(T_frames, 2)`` (normalized [0,1] x,y) from the packed state columns."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    xi, yi = keys.index("obj0_pos_x"), keys.index("obj0_pos_y")
    return np.stack([state[:, xi], state[:, yi]], axis=1).astype(np.float64)


def temporal_token_centers(positions: np.ndarray, n_t: int) -> np.ndarray:
    """Collapse ``(T_frames, 2)`` per-frame centers to ``(n_t, 2)`` per-temporal-token centers.

    V-JEPA's tubelet size is 2, so temporal token ``t`` aggregates frames ``2t`` and ``2t+1``; its center
    is the mean of those two frame centers. Requires ``T_frames == 2 * n_t`` (16 frames -> 8 tokens).
    """
    T = positions.shape[0]
    if T != 2 * n_t:
        raise ValueError(f"expected T_frames={2 * n_t} for n_t={n_t}, got {T}")
    return positions.reshape(n_t, 2, 2).mean(axis=1)


def forward_sim_positions(start: np.ndarray, vel: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear constant-velocity roll-out ``pos[f] = start + vel*f`` (``(n_frames, 2)``), clamped to [0,1].

    Exact for the v2d dataset (constant velocity, no bounces, ball stays in frame), so the TARGET
    trajectory mask is reconstructable at test time from the command alone — no H_b. The clamp guards
    against tiny numerical drift past the frame edge; feasible scenes never actually leave [0,1].
    """
    f = np.arange(n_frames, dtype=np.float64).reshape(-1, 1)
    return np.clip(start.reshape(1, 2) + f * vel.reshape(1, 2), 0.0, 1.0)


def gaussian_mask(centers: np.ndarray, grid: tuple[int, int, int], sigma: float) -> np.ndarray:
    """Peak-normalized Gaussian soft masks ``(T, H, W)`` around per-temporal-token centers.

    ``centers`` is ``(T, 2)`` normalized (x,y). Uses the same image->cell convention as ``pos_to_cell``
    (x -> continuous column ``w_c = x*(W-1)``, y -> continuous row ``h_c = y*(H-1)``), so the masks align
    with ``roll_layer``'s ``(T,H,W,D)`` token layout and the decoder's grid orientation. Max value 1 at
    the center cell; ``sigma`` is in cell units.
    """
    T, H, W = grid
    if centers.shape[0] != T:
        raise ValueError(f"expected {T} centers, got {centers.shape[0]}")
    hh = np.arange(H).reshape(1, H, 1)
    ww = np.arange(W).reshape(1, 1, W)
    w_c = (centers[:, 0] * (W - 1)).reshape(T, 1, 1)
    h_c = (centers[:, 1] * (H - 1)).reshape(T, 1, 1)
    d2 = (hh - h_c) ** 2 + (ww - w_c) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float64)


COMMAND_FEATURE_DIM = 13


def command_features(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Rich command feature vector ``(13,)`` for synthesizing the velocity edit WITHOUT H_b.

    The pixel proof showed velocity lives in a global low-rank subspace, not the ball tokens, so the edit
    should be synthesized from the COMMAND (v_a, v_b) — richer than the bare Delta v the plain ridge uses.
    Columns: ``[1, v_b(2), v_a(2), dv(2), |v_b|, |v_a|, u_b(2), u_a(2)]`` where ``u`` are unit headings.
    The bias + magnitudes + unit directions let a linear map capture direction/speed-dependent structure
    that ``dv`` alone cannot (e.g. the edit's magnitude scaling with speed, sign with heading).
    """
    va = np.asarray(va, dtype=np.float64).reshape(2)
    vb = np.asarray(vb, dtype=np.float64).reshape(2)
    dv = vb - va
    sa = float(np.linalg.norm(va)); sb = float(np.linalg.norm(vb))
    ua = va / (sa + 1e-9); ub = vb / (sb + 1e-9)
    return np.array([1.0, vb[0], vb[1], va[0], va[1], dv[0], dv[1], sb, sa,
                     ub[0], ub[1], ua[0], ua[1]], dtype=np.float64)


TRANSPORT_FEATURE_DIM = 8


def transport_features(M_a: np.ndarray, M_b: np.ndarray, va: np.ndarray, vb: np.ndarray,
                       grid: tuple[int, int, int]) -> np.ndarray:
    """Assemble the per-token transport feature matrix ``phi`` of shape ``(T*H*W, 8)``.

    Row order is ``roll_layer``'s flatten ``index = t*(H*W) + h*W + w`` so that ``phi @ B_t`` reshapes
    straight back to a flat per-layer edit consumable by ``steer_velocity2d._apply_edit``. The 8 columns
    are ``[M_b, M_b*v_bx, M_b*v_by, M_a, M_a*v_ax, M_a*v_ay, M_U*dvx, M_U*dvy]`` (``M_U = max(M_a, M_b)``):
    the bare ``M_b``/``M_a`` columns are the velocity-INDEPENDENT presence write/remove biases, the
    ``M_*v_*`` columns the motion-specific modulation, and ``M_U*dv`` the changed-tube correction.
    """
    Mu = np.maximum(M_a, M_b)
    mb = M_b.reshape(-1, 1); ma = M_a.reshape(-1, 1); mu = Mu.reshape(-1, 1)
    dv = (vb - va).reshape(1, 2)
    return np.concatenate([mb, mb * vb.reshape(1, 2), ma, ma * va.reshape(1, 2), mu * dv],
                          axis=1).astype(np.float64)  # (T*H*W, 8)


class LinearLS:
    """Streaming ridge least-squares ``Y ~= X B`` for arbitrary input dim ``p`` (generalizes RidgeOperator).

    Accumulates ``X^T X`` (``p x p``) and ``X^T Y`` (``p x out``) over batches of rows, then solves
    ``B = (X^T X + lambda I)^{-1} X^T Y`` (a single ``p x p`` inverse). The transport operator uses
    ``p = 8`` (two presence biases + four mask*velocity + two union*dv) and ``out = 1024`` per
    (layer, temporal token).
    """

    def __init__(self, in_dim: int, out_dim: int, ridge: float = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ridge = ridge
        self.XtX = np.zeros((in_dim, in_dim))
        self.XtY = np.zeros((in_dim, out_dim))
        self.n = 0

    def add(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Accumulate a batch: ``X`` is ``(n, in_dim)``, ``Y`` is ``(n, out_dim)``."""
        self.XtX += X.T @ X
        self.XtY += X.T @ Y
        self.n += X.shape[0]

    def solve(self) -> np.ndarray:
        A = self.XtX + self.ridge * np.eye(self.in_dim)
        self.B = np.linalg.solve(A, self.XtY)  # (in_dim, out_dim)
        return self.B

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.B
