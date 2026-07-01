"""Regression tests for the restitution scene variant (scene_restitution)."""
from __future__ import annotations

import numpy as np

from src.data.moving_ball import MovingBall

K = 8


def _scene(scene_idx: int = 0, image_size: int = 128, radius: float = 0.11):
    g = MovingBall(
        image_size=image_size, num_frames=16, fps=4, scenario="scene_restitution",
        clips_per_scene=K, speed_range=(0.018, 0.032), restitution_range=(0.35, 0.95),
        radius_range=(radius, radius), seed=0,
    )
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def test_shared_frame_zero():
    """Frame-0 pixels and state are identical across ranks (only e differs post-bounce)."""
    clips, _ = _scene()
    f0 = [c.frames[0].numpy() for c in clips]
    assert all(np.array_equal(f0[0], f) for f in f0[1:])
    px = {round(float(c.state[0, c.state_keys.index("obj0_pos_x")]), 9) for c in clips}
    py = {round(float(c.state[0, c.state_keys.index("obj0_pos_y")]), 9) for c in clips}
    assert len(px) == 1 and len(py) == 1


def test_distinct_restitution_ranks():
    clips, _ = _scene()
    es = {round(c.meta["restitution"], 5) for c in clips}
    assert len(es) == K
    sv = [c.meta["scene_restitutions"] for c in clips]
    assert all(np.allclose(s, sv[0]) for s in sv)
    for r, c in enumerate(clips):
        assert np.isclose(c.meta["restitution"], c.meta["scene_restitutions"][r])


def test_bounce_occurs_mid_clip():
    clips, _ = _scene()
    for c in clips:
        bf = int(c.meta["bounce_frame"])
        assert 3 <= bf <= 12, f"bounce_frame={bf} out of expected mid-clip range"
        assert c.meta["post_bounce_speed"] > 0


def test_higher_restitution_higher_speed_ratio():
    """Within a scene, larger e should yield larger post/pre speed ratio (monotone trend)."""
    clips, _ = _scene()
    pairs = sorted((c.meta["restitution"], c.meta["speed_ratio"]) for c in clips)
    ratios = [p[1] for p in pairs]
    assert float(np.corrcoef([p[0] for p in pairs], ratios)[0, 1]) > 0.85


def test_ball_stays_trackable():
    clips, _ = _scene()
    for c in clips:
        lum = c.frames[0].numpy().mean(0)
        assert lum.min() < 0.5


def test_determinism():
    a, _ = _scene(scene_idx=2)
    b, _ = _scene(scene_idx=2)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.frames.numpy(), cb.frames.numpy())
        assert np.array_equal(ca.state.numpy(), cb.state.numpy())
