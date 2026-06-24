# Research roadmap

This project investigates how **frozen** V-JEPA2 video representations encode physics, and whether that
structure can be **steered**. The work proceeds in four stages. Each stage ships with always-on controls
(shuffled-latent, randomized-label, oracle-state, copy/mean/random-frame) so that "the latent contains
the information" is never confused with "a probe memorized the labels."

> **North star.** (1) *Understand* how physical quantities and laws are encoded in the latent space of
> frozen V-JEPA-style video models, and (2) *steer* them to produce more physically-consistent video on
> both physics and robotics datasets.

---

## Step 1 — Category probe on real video (Physics-IQ) — ✅ done (positive)

Can a linear probe separate physics categories in frozen V-JEPA2-large latents?

- Linear probe on clip-pooled latents, 3 categories (`fluid_dynamics`, `optics`, `solid_mechanics`),
  scenario-grouped cross-validation, majority baseline 0.592.
- **Linear accuracy 0.908 (z-scored) / 0.898 (raw).** Raw ≈ z-scored (~1 pt gap) ⇒ the category
  structure is genuinely linear in the native latent space, not manufactured by normalization.
- Controls collapse far below majority ⇒ no scenario leakage.
- Geometry: the steering-relevant direction is the *mean-centered* contrast (fluid↔solid cosine ≈ −0.70),
  not raw class means (which share a large common offset and look near-identical).

**Open:** MLP probe numbers, temporal `[z_1..z_T]` / GRU probes, and the dropped thermo/magnetism categories.

## Step 2 — Quantity probe + steering on clean synthetic data — 🔬 in progress

Real benchmark categories are *compound* (they mix lighting, camera, object type, and physics), so Step 2
isolates a **single quantity — velocity — using a maximally clean 2D dataset**: a single dark ball on a
white background, with exact ground-truth position/velocity/speed/angle. This removes visual confounds so
that any decodable or steerable structure is about motion, not appearance.

**Datasets** (`src/data/moving_ball.py`, generated on the fly):
- `constant_velocity` — one ball, white bg, 32 frames @ 128×128, constant velocity.
- `occlusion` — a static wall the ball passes behind; tests object permanence.
- `rotated` — same speed, swept direction; tests equivariance.

**Probes & evidence:**
- Temporal velocity probe (`src/training/velocity_probe.py`) compares `clip_pool` vs `temporal (8×1024)`
  vs `temporal_diff` representations for `vel` / `speed` / `angle`.
- A pixel-level ball tracker (`src/analysis/ball_tracking.py`, intensity-weighted centroid, verified
  to <1e-3) provides **objective visual evidence**: after decoding a steered latent it re-measures the
  ball's speed, so steering is judged by what the pixels actually do, not just by a latent readout.

**Milestone:** find the velocity subspace, steer it, and visually confirm the decoded ball moves
faster/slower. (Prior work decoded velocity but never demonstrated successful *steering* — that is the
new contribution.)

Run end-to-end on SLURM with `source scripts/slurm_step2_velocity_pipeline.sh` (demos → latent
extraction → velocity/occlusion/equivariance probes → decoder training → steering).

## Step 3 — Transfer learned directions to real video — ◻️ quantity version TODO

Test whether quantity directions learned on synthetic data transfer to real Physics-IQ video: steer a
latent along the direction, decode with the transformer decoder, and check whether the result is *more
physically consistent*. Steering rule: `z'_t = z_t + α · d_v`.

The *category-centroid* version of this is done and is a clean **negative** — steering along a
solid−fluid category direction repaints the target category's visual statistics (a global appearance
shift) rather than producing a physics counterfactual on a fixed object. This motivates using
**quantity** directions (from Step 2) instead of category centroids.

## Step 4 — Robotics — ◻️ not started

Apply the same detect-and-steer machinery to robotics. Ideal outcome: identify the main latent-space
differences between **achievable vs non-achievable actions**, then steer failed cases toward success.
(DROID is stubbed in the repo.)

---

## Cross-cutting — generalization

Currently on V-JEPA2-L. Validate on larger V-JEPA2 models (and potentially other video models) to show
the principle transfers rather than overfitting a single backbone.
