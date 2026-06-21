"""Official Physics-IQ scenario -> physics-category mapping.

Physics-IQ (Motamed et al., 2025) groups its 66 scenarios into five physics categories. The category is
*not* in the video filename (which encodes id / fps / perspective / take / scenario), so we carry the
official mapping here. Source of truth: ``descriptions/descriptions_original.csv`` in
google-deepmind/physics-IQ-benchmark (columns ``scenario``, ``category``).

Filename convention (the ``id`` stored in the latent cache is the filename stem):
    ``0001_full-videos_16FPS_perspective-left_take-1_trimmed-ball-and-block-fall``
The scenario is the trailing token with an optional ``trimmed-`` prefix.

Note a few official assignments are deliberately non-obvious (kept verbatim, not second-guessed):
``match`` -> fluid_dynamics, ``match-blows-balloon`` -> thermodynamics,
``magnet-transparent-peakaboo`` -> solid_mechanics, ``blow-balloon`` -> fluid_dynamics.
"""

from __future__ import annotations

# Short category slugs (match the style of data.physics_iq.PHYSICS_IQ_CATEGORIES).
_SOLID = "solid_mechanics"
_FLUID = "fluid_dynamics"
_OPTICS = "optics"
_THERMO = "thermodynamics"
_MAGNET = "magnetism"

SCENARIO_TO_CATEGORY: dict[str, str] = {
    # --- Solid Mechanics (38) ---
    "ball-and-block-fall": _SOLID, "ball-behind-rotating-paper": _SOLID, "ball-hits-duck": _SOLID,
    "ball-hits-nothing": _SOLID, "ball-in-basket": _SOLID, "ball-in-sand": _SOLID, "ball-ramp": _SOLID,
    "ball-rolls-off": _SOLID, "ball-rolls-on-glass": _SOLID, "ball-train": _SOLID, "balls-collide": _SOLID,
    "block-domino": _SOLID, "cut-orange": _SOLID, "cut-paper": _SOLID, "dominos-with-space": _SOLID,
    "double-cradle": _SOLID, "duck-and-dominos": _SOLID, "duck-falls-in-box": _SOLID, "duck-static": _SOLID,
    "magnet-transparent-peakaboo": _SOLID, "marble-run-x": _SOLID, "marble-run-y": _SOLID,
    "mug-breaks": _SOLID, "roll-behind-box": _SOLID, "roll-front-box": _SOLID, "roll-in-box": _SOLID,
    "silk-cover": _SOLID, "single-cradle": _SOLID, "smiley-ball-rotates": _SOLID,
    "solid-ball-peakaboo": _SOLID, "stable-blocks": _SOLID, "teapot-rotates": _SOLID,
    "two-balls-pass": _SOLID, "unstable-block-stack": _SOLID, "weight-on-ceramic": _SOLID,
    "weight-on-paper": _SOLID, "weight-on-pillow": _SOLID, "weight-protects-duck": _SOLID,
    # --- Fluid Dynamics (15) ---
    "blow-balloon": _FLUID, "domino-in-juice": _FLUID, "fill-glass-red-drink": _FLUID,
    "glass-stays-same": _FLUID, "juice-in-water": _FLUID, "liquid-on-duck": _FLUID,
    "liquid-overfill": _FLUID, "match": _FLUID, "napkin-soak": _FLUID, "paint-on-glass": _FLUID,
    "paper-fall-water": _FLUID, "paper-in-water": _FLUID, "potato-in-water": _FLUID, "siphon": _FLUID,
    "water-in-juice": _FLUID,
    # --- Optics (8) ---
    "light-on-block": _OPTICS, "light-on-mug": _OPTICS, "light-on-mug-block": _OPTICS,
    "light-on-statue": _OPTICS, "mirror-ball-fall": _OPTICS, "mirror-ball-rotate": _OPTICS,
    "mirror-teapot-rotate": _OPTICS, "rolling-reflection": _OPTICS,
    # --- Thermodynamics (3) ---
    "lit-candle": _THERMO, "match-blows-balloon": _THERMO, "paper-smoke": _THERMO,
    # --- Magnetism (2) ---
    "magnet-domino": _MAGNET, "magnet-wrench": _MAGNET,
}


def scenario_for_id(sample_id: str) -> str:
    """Extract the scenario slug from a Physics-IQ filename stem / sample id.

    ``0001_full-videos_16FPS_perspective-left_take-1_trimmed-ball-and-block-fall`` -> ``ball-and-block-fall``
    Falls back to the trailing token (minus ``trimmed-``) for unrecognised formats.
    """
    token = sample_id.split("_")[-1]
    if token.startswith("trimmed-"):
        token = token[len("trimmed-"):]
    return token


def category_for_id(sample_id: str) -> str | None:
    """Physics category for a sample id, or ``None`` if the scenario is not in the official mapping."""
    return SCENARIO_TO_CATEGORY.get(scenario_for_id(sample_id))
