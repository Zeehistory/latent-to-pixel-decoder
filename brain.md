# brain.md — project memory + roadmap

> **Living doc.** Update this on every milestone, however small. Append to the changelog at the bottom
> and flip the status tags (`TODO` / `RUNNING` / `DONE` / `BLOCKED`) as things move.
> Last updated: 2026-06-23 (Step 2 velocity pipeline built).

---

## North star

1. **Understand** how different physical quantities / physical laws are encoded in the latent space of
   frozen V-JEPA-style video models.
2. **Steer** them — use those observations to intervene in the latent space and produce more
   physically-consistent video, on both **physics** and **robotics** datasets.

**Where we are on it:** the Transformer decoder results show V-JEPA latents *do* encode dynamics — but
*which* physical variables, and *how*, are still unknown. That "which/how" is the whole point of
Steps 1–3 below; Step 4 is the robotics payoff.

---

## What we've accomplished so far

### Infrastructure (DONE)
- Frozen **V-JEPA2-large** encoder (`facebook/vjepa2-vitl-fpc64-256`, ViT-L, 24 layers), never
  fine-tuned. Multi-layer token latents cached to disk (WebDataset shards + Parquet meta + sha256) so
  probes/decoder never rerun the encoder.
- **Transformer video decoder**: learned query tokens cross-attend the frozen latents. Modes:
  latent→frame, context→future, latent→state. **Result: latents encode dynamics** (this is what
  motivates the rest).
- Full pipeline runs offline on CPU with a deterministic **mock encoder** → real weights are a config
  swap. Every run records git state + hardware + checksums. Controls are always on: shuffled-latent,
  randomized-label, oracle-state, copy/mean/random-frame.

### Step 1 — category probe on Physics-IQ (DONE, positive)
Linear probe on clip-pooled V-JEPA2-large latents, 3 categories (`fluid_dynamics`, `optics`,
`solid_mechanics`; thermo+magnetism dropped, too few scenarios for valid grouped CV), scenario-grouped
CV, majority rate **0.592**.

| probe | best lin. acc | macro-F1 | best layer | shuffled-label ctrl |
|-------|---------------|----------|------------|---------------------|
| standardized (z-score) | **0.908** | 0.897 | 18 | 0.392 |
| raw (no z-score)       | **0.898** | 0.887 | 23 | 0.462 |

- **Raw ≈ z-scored (~1 pt gap)** → z-scoring is *not* manufacturing the separability; category
  structure is genuinely linear in the native latent space. (Directly answers the supervisor's z-score
  worry.) Use **raw** directions for steering.
- Controls collapse far below majority → no scenario leakage.
- Geometry (deepest layer): fluid↔solid = **−0.702** (most anti-aligned), optics↔solid = −0.482,
  fluid↔optics = −0.287. But classifier-free separation is *low* (Fisher ≈ 0.11, silhouette ≈ 0.09):
  thin discriminative direction, not separated blobs.
  - ⚠️ **Figure caveat:** the original `category_cosine_layer23.png` plotted *raw class-mean* cosines
    (all ≈ +0.96–0.98, because the means share a large common offset) — misleading, reads as "all
    categories identical." The −0.702 comes from the **mean-centered contrast** directions, which is
    the steering-relevant geometry. Patched `probe_categories.py` to plot the centered version as the
    primary figure (raw kept as `*_rawmean_*`). **Re-run needed to regenerate.**
- Peak layer differs by run (std=18, raw=23) → pick steering layer from the full curve, not the argmax.

### Step 3 — category-direction steering (DONE, clean negative)
Steered real fluid Physics-IQ clips along learned solid−fluid direction, decoded, swept α (fraction of
per-token norm, norms ~150–240):
- Probe-weight dir: readout moves (ρ=1.0), **pixels don't** until decode breaks into mush.
- Diff-of-means dir: pixels move, but it's a **global blue/cool appearance shift**, not object physics;
  readout saturates at smallest real α (step function, not a controllability curve).
- Fine low-α over 6 distinct fluids: every object/liquid **preserved**, only a mild blue tint. Nothing
  freezes or solidifies.
- **Conclusion:** solid/fluid axis is decodable as a *label* and renderable as *appearance*, but
  steering repaints toward the target category's visual statistics, **not** a physics counterfactual on
  a fixed object. The crude category direction conflates appearance with physics; the decoder paints
  appearance. → **Don't invest more in category-centroid steering.** This sharpens what Step 2 must do.

### Writeup (DONE)
- `paper/main.tex` — ICLR-format **raw notes** (not a paper draft yet) of the above.
- `paper/figures/DOWNLOAD.md` — what to scp from the cluster (3 figures).

---

## Roadmap

> **Target dates** anchor to today (2026-06-23) assuming roughly sequential execution; they cascade,
> so a slip in one step pushes the rest. Step 2 data-gen can overlap Step 1 (start now), which buys
> back time. Step 4 is "if time allows." Adjust these as reality lands.
>
> Key insight driving the order: Physics-IQ categories are **compound** — a "category direction" mixes
> lighting, camera angle, object type, *and* physics. Step 1 tests whether we can even tell rigid-body
> from fluid in latent space (expect overlap but distinct subspaces). Step 2 gets **clean labels** from
> synthetic data to isolate *which subspace encodes which physical quantity*. Step 3 tests transfer of
> those quantity directions to real video via the decoder. Step 4 is robotics.

### STEP 1 — probe Physics-IQ category structure (~1 week · target by 2026-06-30) — DONE for linear, temporal variants open
Physics-IQ has broad categories (optics, fluid dynamics, …) but **no detailed labels**, so a category
direction is likely compound (lighting / camera / object type / physics). Still valuable: can probes
tell apart e.g. rigid-body motion from fluid dynamics? Expect some overlap, but distinct subspaces.

How: start with **linear + MLP** probes.
- MLP is nonlinear → reveals *richer* info, tells you *what physics is there*.
- For *directions*, **linear** probes are better.
- Plan: train a linear classifier on the 5–6 Physics-IQ categories. Best case → distinct subspaces per
  category. Otherwise try MLP and hope it does better.
- **Temporal**: videos have dynamics, so instead of a single latent `z_t` + label, try the full
  sequence `[z_1, …, z_T]` + label. If that doesn't work, try a small **GRU** temporal probe.

**Status:** linear (raw+z) DONE on 3 categories. **TODO:** (a) MLP probe numbers, (b) temporal probe
`[z_1..z_T]` and/or GRU — clip-pooling threw away the temporal axis where dynamics live (`8×1024`
temporal-preserving probe is the natural next rep), (c) revisit the dropped thermo/magnetism categories.

### STEP 2 — synthetic datasets with exact labels (~2 weeks · target by 2026-07-14; start data gen now) — VELOCITY-FIRST probes DONE, steering BLOCKED
Physics-IQ has no specific labels, so generate synthetic datasets specializing in e.g. solid mechanics
(freefall, projectile, bounce) and fluid dynamics, with **exact** ground-truth state. Then train probes
for **velocity, acceleration, gravity, …** → answers precisely *which subspace encodes which quantity*.

#### Velocity-first controlled experiments (2026-06-23, BUILT — run on cluster)

Per supervisor direction: start with **velocity only**, use a maximally clean 2D dataset that removes
all visual confounds (object type, lighting, camera angle). Interim milestone: find the velocity subspace,
steer it, and **visually verify** the decoded ball actually moves faster/slower.

**Dataset**: `src/data/moving_ball.py` — `MovingBall` generator. Three variants:
- `constant_velocity` (512 clips): one ball, clean **white background**, 32 frames, 128×128, fixed FPS=8,
  ball stays fully in frame (constant velocity, no bounce/gravity). Exact GT: per-frame center position,
  velocity vector, speed, angle. Config: `configs/data/moving_ball_velocity.yaml`.
- `occlusion` (512 clips): static central vertical wall; ball passes behind it (hidden mid-frames).
  Tests object-permanence: is velocity decodable even when the ball is invisible?
  Config: `configs/data/moving_ball_occlusion.yaml`.
- `rotated` (512 clips): **same speed**, swept direction (golden-ratio). Tests equivariance.
  Config: `configs/data/moving_ball_equivariance.yaml`.

**Ball pixel tracker** (`src/analysis/ball_tracking.py`): intensity-weighted centroid recovers exact GT
velocity from rendered frames (verified: matches to <1e-3). This is the *objective visual evidence* for
steering — after decoding a steered latent, we re-track the ball and confirm speed changed.

**New scripts (all cluster-ready):**

| Script | What it does |
|--------|-------------|
| `scripts/generate_ball_demos.py` | CPU-only demo gen, inspect before cluster |
| `scripts/probe_velocity.py` | Temporal velocity probe: **clip_pool** vs **temporal (8×1024)** vs **temporal_diff** |
| `scripts/probe_occlusion.py` | Velocity decodable from hidden-frame tokens? (train on visible, eval on hidden) |
| `scripts/probe_equivariance.py` | Is the velocity subspace equivariant under direction rotation? |
| `scripts/steer_velocity.py` | Steer speed/vel_x/vel_y, decode, measure pixel-level speed change |

**Temporal probe design**: `src/training/velocity_probe.py` keeps the `(T'=8, D=1024)` sequence,
not just the `(1×1024)` clip mean. Three representations compared: `clip_pool` (baseline, old approach),
`temporal` (full 8×1024 flattened — the main improvement), `temporal_diff` (consecutive differences,
natural feature for a rate-of-change quantity). Targets: `vel` (signed), `speed` (magnitude),
`angle` (as cos/sin).

**Equivariance metric**: subspace circularity (ratio of PCA eigenvalues; 1=circle=equivariant),
direction-R² (recover angle from 2D subspace), and per-pair equivariance error `||R_obs − R_θ||_F / 2`.

**SLURM pipeline** (all on bouchet):
1. `sbatch slurm_ball_demos.sh` → inspect demos first
2. `for D in moving_ball_{velocity,occlusion,equivariance}; do DATASET=$D sbatch slurm_extract_ball.sh; done`
3. `sbatch slurm_probe_velocity.sh` → fills R² table (clip_pool vs temporal)
4. `sbatch slurm_probe_occlusion.sh` → visible vs hidden R²
5. `sbatch slurm_probe_equivariance.sh` → circularity + equivariance error
6. (needs decoder first) `TARGET=speed sbatch slurm_steer_velocity.sh`
Or in one go: `source scripts/slurm_step2_velocity_pipeline.sh`

**Status:** RUN ON CLUSTER (2026-06-24). Velocity/occlusion/equivariance probes done; steering did NOT complete (see digest). See `outputs/analysis/STEP2_RESULTS_DIGEST.md`.

| layer | clip_pool vel R² | temporal vel R² | temporal_diff vel R² |
|-------|------------------|-----------------|----------------------|
| 6    | 0.992            | 0.991           | 0.977                |
| 9    | 0.995            | 0.996           | 0.989                |
| 12    | 0.996            | 0.997           | 0.994                |
| 15    | 0.994            | 0.997           | 0.994                |
| 18    | 0.995            | 0.997           | 0.994                |
| 21    | 0.995            | 0.998           | 0.995                |
| 23    | 0.995            | 0.998           | 0.995                |

**Older MuJoCo/Genesis engines:** `mujoco_solid`, `genesis_fluid` still in repo (use `slurm_train_probe.sh`
for the broader quantity-probe table). Genesis API may need version-specific tweaks (`# GENESIS API` lines).

### STEP 3 — transfer synthetic directions to real video (~1–2 weeks · target by 2026-07-28) — category version DONE (negative); quantity version TODO
Test whether directions learned from **synthetic** data transfer to **real** Physics-IQ video. Learn a
subspace for e.g. "velocity," steer it in Physics-IQ latents, decode with the Transformer decoder, and
check if the decoded video is **more physically consistent**.

Steering rule: `z'_t = z_t + α · d_v` where `d_v` is the quantity direction.

**Status:** the *category-centroid* version of this is DONE and is a clean **negative** (appearance, not
physics — see above). **TODO:** redo with **quantity** directions from Step 2 (a direction tied to a
measured physical quantity is far likelier to be a renderable physics axis than a category centroid).
Optional: appearance-orthogonalized direction — only worth it once a quantity/temporal direction shows
object-level effects.

### STEP 4 — robotics (~2 weeks · target by 2026-08-11, if time allows) — NOT STARTED
Tackle the robotics case and answer John's "we can detect and steer — so what?". No concrete plan yet;
figure it out as we go. **Ideal outcome:** identify the main latent-space differences between
**achievable vs non-achievable actions**, then steer failed cases → success. (DROID is stubbed in the
repo for this.) This would be huge.

### Cross-cutting — model generalization (ongoing)
Currently on **V-JEPA2-L**. Also try **larger V-JEPA2** models and potentially **other models**, to
show the principle transfers/generalizes rather than overfitting one model.

---

## Open questions / risks
- Category directions are appearance-dominated (Step 3 negative). Does a *quantity* direction avoid this?
- Does clip-pooling vs temporal-preserving latent change which quantities are decodable?
- Will synthetic→real transfer survive the domain gap (synthetic renders aren't photorealistic)?
- Robotics "so what?" — needs a concrete success metric before Step 4.

---

## Changelog

- **2026-06-24** — Step 2 velocity-first RUN on cluster (bouchet, gpu_rtx6000). Velocity probe: vel linear R²: clip_pool 0.996@L10, temporal 0.998@L23 (controls collapse: shuf -0.738, rand -0.582). Occlusion: hidden-token vel R² 0.9975@L9 (visible 0.9986). Equivariance: max direction-R² 0.985@L8, circ 0.954; best equivErr 0.571@L11. Steering: speed: readout ρ=1.000, decoded-measured ρ=—; vel_x: readout ρ=1.000, decoded-measured ρ=—; vel_y: readout ρ=1.000, decoded-measured ρ=—. Fixes applied this run: probe/decoder/steer mem 64G→192G, shard-cache eviction between layers, LatentDataset cache pruned to selected layers (decoder OOM), re-extracted equivariance latents (prior cache had no metadata.parquet). Full numbers in `outputs/analysis/STEP2_RESULTS_DIGEST.md`.

- **2026-06-24** — Step 2 velocity-first RUN on cluster (bouchet, gpu_rtx6000). Velocity probe: vel linear R²: clip_pool 0.996@L10, temporal 0.998@L23 (controls collapse: shuf -0.738, rand -0.582). Occlusion: hidden-token vel R² 0.9975@L9 (visible 0.9986). Equivariance: max direction-R² 0.985@L8, circ 0.954; best equivErr 0.571@L11. Steering: speed: missing; vel_x: missing; vel_y: missing. Fixes applied this run: probe/decoder/steer mem 64G→192G, shard-cache eviction between layers, LatentDataset cache pruned to selected layers (decoder OOM), re-extracted equivariance latents (prior cache had no metadata.parquet). Full numbers in `outputs/analysis/STEP2_RESULTS_DIGEST.md`.
- **2026-06-23** — Step 2 velocity-first: built the full controlled experiment pipeline. New files:
  `src/data/moving_ball.py` (clean 2D generator, 3 scenarios), `src/analysis/ball_tracking.py`
  (pixel-tracker for visual evidence), `src/training/velocity_probe.py` (temporal 8×1024 probe),
  `src/analysis/steering.py` extended with `discover_velocity_direction`. New scripts:
  `generate_ball_demos.py`, `probe_velocity.py`, `probe_occlusion.py`, `probe_equivariance.py`,
  `steer_velocity.py`. SLURM scripts: `slurm_ball_demos.sh`, `slurm_extract_ball.sh`,
  `slurm_probe_velocity.sh`, `slurm_probe_occlusion.sh`, `slurm_probe_equivariance.sh`,
  `slurm_steer_velocity.sh`, `slurm_step2_velocity_pipeline.sh` (one-shot orchestrator).
  Tests: `tests/test_moving_ball.py`. All ready to run on bouchet; results pending.
- **2026-06-23** — Caught a misleading figure: `category_cosine_layer23.png` plotted raw class-mean
  cosines (all ~+0.97), not the steering-relevant centered contrasts. Added
  `subspace.contrast_directions`, patched `probe_categories.py` to plot centered cosine as primary
  (raw kept as `*_rawmean_*`) and to persist both matrices in the summary JSON. Verified locally that
  centering recovers the negative cosine. **Cluster re-run needed to regenerate Fig. 2.**
- **2026-06-23** — Added absolute target dates to the roadmap steps (anchored to 2026-06-23).
- **2026-06-23** — Created `brain.md`. Wrote `paper/main.tex` (ICLR-format raw notes) +
  `paper/figures/DOWNLOAD.md`.
- **2026-06-22** — Step 3 category steering: clean negative (appearance shift, not physics). Decided to
  stop investing in category-centroid steering.
- **2026-06-22** — Step 1 category probe (SLURM 15764874): linear acc 0.908 (z) / 0.898 (raw), raw ≈
  z-score → separability is real, not a normalization artifact.
- **(earlier)** — Transformer decoder shows V-JEPA2 latents encode dynamics. Infra built: frozen
  encoder, latent caching, decoder, probes, controls, mock-encoder CPU path.
