"""Latent steering toward target physical behavior / robot success (deferred — Stage 2).

Builds on ``intervention`` to *optimize* a latent perturbation that drives decoded behavior toward a
target (e.g. rotate velocity by Δθ, increase elasticity, or interpolate a failed robot trajectory
toward a successful one). This is the bridge to the robotics goal and is intentionally a documented
stub: steering must be validated, not faked, and needs a trained decoder + robotics data (DROID) that
arrive in a later stage.
"""

from __future__ import annotations

from typing import Any


def steer_to_target(decoder: Any, latents: Any, grid: Any, target: Any, **kwargs: Any):
    """Optimize a latent edit so the decoded state matches ``target`` (planned)."""
    raise NotImplementedError(
        "Steering optimization is a Stage-2 component. Intended design: gradient-based or CMA search "
        "over a low-rank latent edit, constrained to discovered physical directions "
        "(see analysis/intervention.py), validated by decoded-state error + plausibility metrics."
    )


def interpolate_trajectories(latents_fail: Any, latents_success: Any, steps: int = 8):
    """Interpolate between failed and successful latent trajectories (planned robotics use)."""
    raise NotImplementedError("Trajectory interpolation requires DROID support (Stage 2).")
