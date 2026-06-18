"""Latent-space analysis: geometry, manifolds, direction codes, intervention/steering, visualization."""

from __future__ import annotations

from . import visualization
from .direction_codes import circular_code_score
from .intervention import apply_intervention, discover_direction, intervention_sweep
from .latent_geometry import layerwise_cka_matrix, pca, pooled_features
from .manifold_analysis import embed_2d, variable_axis_alignment

__all__ = [
    "visualization",
    "circular_code_score",
    "discover_direction",
    "apply_intervention",
    "intervention_sweep",
    "pooled_features",
    "pca",
    "layerwise_cka_matrix",
    "embed_2d",
    "variable_axis_alignment",
]
