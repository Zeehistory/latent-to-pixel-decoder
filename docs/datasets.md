# Datasets

## Synthetic physics (built-in, no download)
Deterministic 2D simulator (`src/data/synthetic_physics.py`) renders anti-aliased disks and records
*exact* ground-truth state per frame. Scenarios: `bouncing_ball`, `projectile`, `collision`,
`pendulum`, `two_body`, `occlusion` (object permanence). Ground-truth columns per object:
`pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, radius, mass, visible`, plus global `gravity` and
`collision_event`. This is what makes probe/metric validation possible (oracle ≈ perfect).

## Physics-IQ (benchmark)
Real-world physical-reasoning videos. See `scripts/download_data.py --dataset physics_iq` for the
official source and expected layout (`manifest.json` or `<split>/<category>/*.mp4`). Set
`data.root` in `configs/data/physics_iq.yaml`. No per-frame numeric labels, so physics-state metrics
are skipped (mask = 0) rather than faked; reconstruction and latent analysis use it fully.

## DROID (robotics, Stage 2)
Typed stub (`src/data/droid.py`). Used later to test decoding/steering of latent failures toward task
success. See the roadmap in `docs/method.md`.

## Sample schema (dataset-agnostic)
Every dataset yields the same dict (`id, frames, encoder_input, state, state_mask, state_keys,
category, meta`), so extraction/decoding/probing code is dataset-independent.
