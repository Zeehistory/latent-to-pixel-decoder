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

### STEP 2 — synthetic datasets with exact labels (~2 weeks · target by 2026-07-14; start data gen now) — DONE: faithful pixel-level velocity steering achieved (held-out ρ=0.90, decoded-vs-GT r=0.95) via difference-vector edit + per-frame frame_position decoder loss; the magnitude gap that blocked Step 1/2 is CLOSED
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

**Status:** RUN ON CLUSTER (2026-06-24). Probes done (velocity/occlusion/equivariance). Decoder background-collapse FIXED this session (target compression + hard-mask foreground + state co-training → ball renders, decoder partially conditions on per-clip latent). **Steering ran but FAILED the milestone:** readout perfectly steerable (ρ=1.000 all targets) but decoded pixel-tracked velocity is essentially flat across α (speed ρ=0.64 / vel_x 0.32 / vel_y 0.25 are rank-corr on near-flat noisy 6-clip curves; decoded speed sits ~0.0087 for all α vs GT range 0.008–0.038). The decoder renders a roughly-fixed slow ball and does NOT translate a steered velocity subspace into changed motion — the SAME readout-moves-pixels-don't pattern as Step 1 category steering (see line ~62). Decoder faithfulness is the bottleneck, not subspace decodability. See `outputs/analysis/moving_ball_velocity/steer_{speed,vel_x,vel_y}/`.

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

#### Difference-vector steering — supervisor proposal (2026-06-26, BUILT)

**Why:** probe-direction steering hit a wall (readout ρ=1.0 but pixels don't move — same as Step 1). The
probe direction is an *abstract* axis fit to predict a number; pushing the latent along it goes
**off-manifold**, and the decoder won't render an off-manifold perturbation. The PI's fix (LLM
activation-steering analogy): derive the steering vector from **two real encoded videos** of the *same
scene* that differ only in speed, so the difference is the velocity factor in the model's own
coordinates, and interpolating between them stays **on-manifold** (the endpoints are clips the decoder
was trained to reconstruct).

    H_a = E(slow), H_b = E(fast)   [identical first frame + direction, only speed differs]
    Δ = H_b − H_a                  → decode H_a + α·Δ ; α=0→v_a, α=1→v_b, α∉[0,1] extrapolates

**Dataset** (`scene_velocity` scenario in `src/data/moving_ball.py`; `configs/data/moving_ball_scene.yaml`):
16 frames, FPS=4, **256×256** (→ VJEPA2-L grid 8×16×16 = 8×256 tokens, H∈ℝ^{8×256×1024} per layer, matching
the spec — the "196" in the PI's note is 14², a 224² model; we lock 256²→256). White bg, **one ball,
FIXED radius (0.11, ~4% of frame — deliberately NOT small, to dodge the background-collapse trap) and
colour**. Organized into **scenes**: `clips_per_scene=4` clips share a bit-identical first frame + motion
direction and differ only in speed (rank 0=slowest..3=fastest). Across scenes: start position, direction,
and the 4 speeds randomize. Scene-disjoint splits via distinct seeds (train=0/val=1/test=2). IDs encode
scene+rank (`scene00007_v2`) so steering can pair same-scene clips. Validated locally: first-frame max
deviation = 0.0, monotone speeds, same direction, ball stays in-frame, train/test scenes disjoint.

**Decoder** (`configs/train/moving_ball_scene_decoder.yaml`): the same transformer decoder but **smaller**
(depth 12 not 24 — this data is far simpler than Physics-IQ, per the PI), 256px/16fr output, decoding the
full multi-scale latent state `[6,12,18,23]`. Carries forward **all three hard-won loss fixes** (target
compression 0.05/0.95, hard-mask foreground, **state/trajectory/velocity co-training** — the co-training
is what forces the decoder to actually use the per-clip latent instead of rendering an average ball).

**Steering** (`scripts/steer_velocity_diff.py`): pairs same-scene test clips, forms per-layer Δ=H_b−H_a,
decodes H_a+α·Δ, re-tracks the decoded ball, and reports **decoded-speed-vs-α** (monotonicity ρ +
decoded-vs-GT-interpolated-speed linearity r). Two variants: **per-pair** (each scene's own Δ, the
on-manifold edit — primary) and **mean-Δ** (one vector averaged across scenes, LLM-style, applied to
held-out H_a — tests global generalization). Why this should beat probe-steering: α∈[0,1] interpolates
two reconstructable latents, so endpoints are guaranteed faithful and the path stays near the manifold.

**SLURM** (scavenge_gpu + `--requeue` + NFS paths, per scheduling memory):
`slurm_extract_scene.sh` (SPLIT=train/val/test → seed+size), `slurm_train_scene.sh` (resume-safe),
`slurm_steer_scene.sh`. Pilot-first: 40 train scenes → 1500 steps → steer 10 test scenes, validate the
ball renders + α-sweep moves it, THEN scale to 300/–/100 (disk-limited; project vol at 97%, val skipped).

**Status (2026-06-26): BUILT + PILOT DONE (encouraging) + full run RUNNING.**

**PILOT RESULTS** (40 train scenes, depth-12 decoder @ 1500 steps, 10 held-out test scenes; encoder grid
confirmed (8,16,16)=256 tokens). Decoder trained cleanly — loss descended to ~0.3 (NO collapse), and
trajectory/velocity co-training losses fell to ~1e-3, i.e. the decoder **conditions on the per-clip
latent** (localizes the ball; no average-blob). Difference-vector steering, per-pair (each scene's own
Δ=H_b−H_a), decoded ball speed vs α:

| α | −0.5 | 0.0 (=H_a) | 0.5 | 1.0 (=H_b) | 1.5 |
|---|------|------|------|------|------|
| decoded speed | 0.0116 | 0.0116 | 0.0161 | 0.0185 | 0.0168 |
| GT-interp speed | 0.0084 | 0.0193 | 0.0301 | 0.0410 | 0.0518 |

- **Per-pair decoded-speed monotonicity ρ = 0.80** (decoded ball speeds UP +60% from α=0→1). **This is the
  first time pixel motion tracks the steering knob** — the probe approach was FLAT (~0.0087 for all α,
  see above). The PI's on-manifold hypothesis (interpolate two real latents, not push a probe axis) is
  **supported in direction**. Filmstrips show the decoded ball visibly larger/more-displaced at higher α.
- **Magnitude gap (the remaining issue): decoded-vs-GT r = 0.33.** The decoder **compresses the speed
  range** ~2–3× and renders a rough/smeared ball (under-trained 1500-step pilot) — centroid of a blurry
  blob under-moves vs the crisp GT. Endpoint fidelity is off too (at α=0=real slow clip, decoded 0.0116 vs
  GT 0.019). Expect the full run (300 scenes × 6000 steps) to sharpen the ball and widen the rendered range.
- **mean-Δ variant FAILS (ρ = −0.90)** — a single Δ averaged across scenes anti-correlates. Confirms the
  spatial-misalignment caveat: per-token Δ is localized to each scene's ball path, so averaging across
  scenes (different positions/directions) blurs it to noise. **Per-pair, same-scene Δ is the right method;
  the LLM-style global vector does NOT transfer here.** (Informative negative for the PI.)
- Artifacts: `outputs/analysis/moving_ball_scene/diff_steer_pilot/` (controllability plot, per-scene
  filmstrips + α=0→1.5 mp4s, summary json).

**FULL RUN DONE (2026-06-26): ρ holds, but the magnitude gap did NOT close — mixed/partial.**
300 train scenes (1200 clips) + 100 test (400), depth-12 decoder @ 6000 steps (rtx_5000_ada, ~2.4h).
Steered 20 held-out test scenes (per-pair Δ).

| α | −0.5 | 0.0 (=H_a) | 0.5 | 1.0 (=H_b) | 1.5 |
|---|------|------|------|------|------|
| decoded speed | 0.0076 | 0.0073 | 0.0077 | 0.0101 | 0.0092 |
| GT-interp | 0.0095 | 0.0197 | 0.0299 | 0.0401 | 0.0503 |

- **Monotonicity ρ = 0.80** (same as pilot) — the direction holds on held-out scenes. **decoded-vs-GT r =
  0.34** (pilot 0.33) — **scaling 7.5× data + 4× steps did NOT close the magnitude gap.** The aggregate
  decoded ball barely speeds up (0.0073→0.0101) and is ~4× compressed vs GT.
- **Heterogeneous per-scene:** ~6–7/20 scenes show genuine 2–3× decoded speed-up at α=1 (e.g. scene2
  0.0056→0.0178, scene3 0.0069→0.0189) — *real pixel-level velocity steering on a subset*. But ~8/20 are
  flat and **2/20 render an untrackable (too-faint) ball (NaN)**. Filmstrips show a crude dark *blob*, not
  a clean disk; at high α it renders a bigger/more-displaced smear (which is what the centroid tracker
  reads as "faster"), not a faithfully-moving disk.
- **mean-Δ ρ = 0.00** (pilot −0.90) — global averaged vector still useless. Per-pair same-scene Δ is essential.
- **HONEST conclusion:** the difference-vector method is **directionally better than probe steering** (it
  *does* move pixels monotonically — probes were flat — and works clearly on a subset), but it does **NOT**
  achieve faithful velocity-*magnitude* rendering, and **the bottleneck is decoder faithfulness, not the
  steering vector** (same deep limiter as Step 1/2 probe steering). More data/steps did not help → the next
  lever is decoder *rendering* (higher capacity, sharper-disk inductive bias, or an explicit
  position/flow decode target), NOT more scale. One confound to rule out: the centroid tracker's hard
  darkness>0.5 threshold drops faint decoded balls to NaN and may under-read blurry-blob speed — worth a
  softer tracker before over-concluding. Artifacts: `outputs/analysis/moving_ball_scene/diff_steer/`.

**MAGNITUDE GAP CLOSED (2026-06-26 eve): faithful pixel-level velocity steering achieved.**
The bottleneck was the decoder's *objective*, not its capacity or the steering vector. The pixel loss was
minimized by rendering a static path-covering SMEAR (its centroid barely moves → tracker reads it slow);
even reconstructing the real fast clip H_b was ~4× too slow. Fix: a per-frame **`frame_position` loss**
that ties the decoded ball's differentiable soft centroid to the GT `obj0_pos` every frame (a smear's
centroid is pinned at the path midpoint → large per-frame error), plus a light `frame_spread` anti-smear
term (`src/decoders/loss_functions.py`; weights in `moving_ball_scene_decoder.yaml`: frame_position=20,
frame_spread=2). `trajectory`/`velocity` only constrained the auxiliary STATE HEAD, never the rendered
pixels — this closes that gap. CPU-validated first (`tests/test_frame_position_loss.py`, 6 green: soft
centroid recovers GT pos <0.01px; smear total loss 1.88 vs faithful 0.09). Retrained
`moving_ball_scene_decoder_fp` to step 6000 (b200, ~45 min, same 208M decoder). Re-measured on the 20
held-out test scenes:

| α | −0.5 | 0.0 (=H_a) | 0.5 | 1.0 (=H_b) | 1.5 |
|---|------|------|------|------|------|
| decoded speed | 0.0207 | 0.0199 | 0.0300 | 0.0395 | 0.0431 |
| GT-interp | 0.0095 | 0.0197 | 0.0299 | 0.0401 | 0.0503 |

- **Monotonicity ρ = 0.90** (was 0.80) and **decoded-vs-GT r = 0.952** (was 0.34) — gap CLOSED in the
  on-manifold α∈[0,1] regime. α=0 reconstructs v_a to ~3.6%, α=1 reconstructs v_b to ~1.9% — the decoder
  renders the *right speed*, not just the right ordering. **0/100 NaN** decoded points (was 2 + ~8 flat) —
  clean disks, the tracker confound is moot.
- **Honest caveats:** extrapolation (α<0, α>1) **saturates** — decoded doesn't reach GT's extrapolated
  speeds (expected; off-manifold beyond the trained velocity range). **mean-Δ ρ = 0.30** — the single
  global averaged "speed-up vector" still does NOT generalize; per-pair same-scene Δ remains essential
  (the velocity edit is scene-local in token space). Capacity was NOT the limiter (same decoder) → good
  sign for scaling. Artifacts: `outputs/analysis/moving_ball_scene/diff_steer_fp_last/` (summary json,
  controllability plot, per-scene filmstrips + α0→α1.5 mp4s); decoder
  `outputs/runs/moving_ball_scene_decoder_fp/checkpoints/last.pt`.

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

- **2026-06-28** — Step 2 VELOCITY SUBSPACE / OPERATOR (PI direction 2026-06-27): moved from speed-only
  to TRUE 2D velocity (direction+speed) and from a per-instance edit toward a transferable representation.
  Built `scene_velocity2d` (8 clips/scene share ONE start position, each a distinct velocity VECTOR;
  `configs/data/moving_ball_scene_velocity2d.yaml`, `tests/test_scene_velocity2d.py` 8 green). Encoded
  train 4000 / test 800 (`moving_ball_scene_v2d`); retrained the frame_position decoder
  (`moving_ball_scene_v2d_decoder_fp`, 8000 steps — direction-agnostic loss transfers). New analysis:
  `src/analysis/velocity_ops.py` (Gram-trick PCA, streaming-normal-equations ridge F_U at full
  D=8·256·1024, token-roll canonicalization A_s), `scripts/velocity_subspace.py` (Phase 2 + fits
  ridge/canon/U artifacts), `scripts/steer_velocity2d.py` (Phases 3-5 decode+track velocity VECTOR).
  **Phase-2 LATENT result (held-out test):** velocity ΔH is NOT a clean 2D subspace — *within* a scene
  (shared start) effective rank ≈5.5 (top-2 only ~46%) because varying DIRECTION moves the ball through
  different tokens; *across* scenes the per-scene velocity 2D-subspaces are near-orthogonal (mean
  principal angle ~71°) and global effective rank explodes to 36-58. A single global linear operator /
  top-2 PCA subspace recovers only ~37% of held-out ΔH direction (ridge cos≈0.39 @ L12, U2 retain≈0.39)
  vs **~0.001 for a random same-rank subspace** — so U is real but partial (captures the scene-shared
  component; the majority is scene-local placement). This QUANTIFIES why the global "speed-up vector"
  failed and motivates canonicalization. Best layer L12. Artifacts +
  `outputs/analysis/moving_ball_v2d/subspace/subspace_summary.json`.
  **PIXEL proof DONE (40 held-out test scenes, decoder final loss 0.055; headline metric = decoded-vs-TARGET
  velocity-vector angle error; the random same-rank subspace is the "no-op stay-at-v_a" floor ≈44° because
  rank0→rank7 differ ~45° in heading):**
    - **full_delta (per-pair Δ): angle err 1.66°, speed_ratio 0.997, ρvx/ρvy 0.999** — faithful 2D velocity
      steering WORKS per-pair; the speed-only win extends cleanly to direction+magnitude.
    - **subspace_U (Phase 3): U beats random at every rank, saturates ~k=8** — U2 27°, U4 23°, U8 21°
      (U16=U8, only 8 comps saved) vs **random 44°**. A low-rank (~8D) GLOBAL velocity subspace is REAL and
      preserves steering while a random same-rank subspace does not (doesn't reach 1.7° because within-scene
      ΔH is ~5.5D; U captures the scene-shared part).
    - **ridge_global (Phase 4, steer straight from Δv): 34.4°** (vs 44° no-op) — a learned linear F_U
      transfers to held-out scenes but only PARTIALLY (matches latent cos 0.39), as predicted.
    - **canon_ridge (Phase 5): 34.8° — NO lift over raw ridge.** Honest negative: GT-informed START-roll
      canonicalization doesn't help, because within a scene the start is ALREADY shared and the ΔH spread is
      DIRECTION-induced (different headings → different path tokens), which a start-roll can't align. → next
      lever is DIRECTION-aware canonicalization (rotation) / learned register, not start-roll.
  Artifacts: `outputs/analysis/moving_ball_v2d/steer_last/{steer2d_summary.json,scene*_methods.png}`; decoder
  `.../runs/moving_ball_scene_v2d_decoder_fp/checkpoints/last.pt`.
  Scheduling lesson (user directive): always pick the least-queued partition — gpu_devel
  (decoder, instant, 6h QOS) + bigmem (subspace, needs ≥256G mem) started in ~30s vs scavenge_gpu (370
  deep) / day (ETA hours). See memory [[cluster-partition-playbook]], [[velocity-subspace-operator-plan]].

- **2026-06-26 (eve)** — Step 2 MAGNITUDE GAP CLOSED → faithful pixel-level velocity steering. The
  remaining gap (decoded ball ~4× too slow, decoded-vs-GT r=0.34) was the decoder's OBJECTIVE, not its
  capacity or the steering vector: the pixel loss was minimized by a static path-covering smear (centroid
  barely moves), and `trajectory`/`velocity` only constrained the auxiliary state head, never the rendered
  pixels. Added `frame_position_loss` (ties the decoded ball's differentiable soft centroid to GT
  `obj0_pos` every frame) + `frame_spread_loss` (anti-smear) in `src/decoders/loss_functions.py`; schema in
  `src/utils/config.py`; weights (frame_position=20, frame_spread=2, frame_spread_max_var=0.004) in both
  `moving_ball_scene_decoder.yaml` and `moving_ball_decoder_large.yaml`. CPU-validated before any GPU
  (`tests/test_frame_position_loss.py`, 6 green: soft centroid matches the renderer's GT pos to <0.01px;
  static smear scores total loss 1.88 vs 0.09 for a faithful render). NB: the prior `obj0_pos_x` vs
  `pos_x` "bug" was a non-issue — `_state_keys()` prefixes `obj0_`, so the original lookup was correct; the
  term was simply inert because its config weight defaulted to 0. Retrained `moving_ball_scene_decoder_fp`
  to step 6000 (b200 via gpu_devel-for-measure / scavenge-for-train, ~45 min, same 208M decoder). Held-out
  (20 test scenes): **monotonicity ρ=0.90** (was 0.80), **decoded-vs-GT r=0.952** (was 0.34); α=0/α=1
  reconstruct v_a/v_b to ~3.6%/1.9%; aggregate decoded 0.0199/0.0300/0.0395 vs GT 0.0197/0.0299/0.0401 at
  α=0/0.5/1.0; **0 NaN** (was 2 untrackable + ~8 flat). Honest caveats: α∉[0,1] extrapolation saturates
  (off-manifold); mean-Δ global vector still weak (ρ=0.30) — per-pair same-scene Δ essential. Scheduling
  lesson: the measurement only loads the tiny TEST cache (~80 clips) so it fits gpu_devel's 60G cap and
  schedules instantly — don't wait ~17h on scavenge backfill for a 5-min decode. Artifacts:
  `outputs/analysis/moving_ball_scene/diff_steer_fp_last/`; decoder
  `outputs/runs/moving_ball_scene_decoder_fp/checkpoints/last.pt`.

- **2026-06-26** — Step 2 DIFFERENCE-VECTOR STEERING built + pilot (supervisor's on-manifold proposal; first pixel-level win). New `scene_velocity` dataset (`src/data/moving_ball.py`): scenes of 4 clips with bit-identical first frame + direction, only speed varies; 256²/16fr → encoder grid confirmed **(8,16,16)=256 tokens, H∈ℝ^{8×256×1024}** (resolves the "196" — it's 256). New `scripts/steer_velocity_diff.py`: Δ=H_b−H_a between two real same-scene clips, decode H_a+α·Δ. New configs (`moving_ball_scene.yaml`, `moving_ball_scene_decoder.yaml` — smaller depth-12 decoder, carries the 3 loss fixes + co-training) and SLURM (`slurm_{extract,train,steer}_scene.sh`). **Pilot (40 scenes, 1500 steps, 10 test):** per-pair decoded-speed monotonicity **ρ=0.80** (decoded ball speeds up +60% from α=0→1) — FIRST time pixels track the steering knob (probe approach was flat ~0.0087). Remaining gap: decoder compresses the speed range ~2–3× (decoded-vs-GT r=0.33, rough/blurry ball at 1500 steps). mean-Δ global vector FAILS (ρ=−0.90 — per-token Δ is scene-local; averaging across scenes blurs it). Full run (300/100, 6000 steps) RUNNING to close the magnitude gap. Bugs fixed en route: `clips_per_scene` missing from `DataConfig` schema; SLURM scripts now `exit $STATUS` (a Python failure had passed `afterok`); `--alphas=` form (leading-dash value). Scheduling: train on **gpu_devel** (instant, fast GPU; 1-job/user cap) + dependents on scavenge_gpu — internalized as a FAST-TURNAROUND PLAYBOOK in memory. **FULL RUN (300/100, 6000 steps) DONE — honest result: ρ=0.80 HOLDS on 20 held-out scenes but the magnitude gap did NOT close (r=0.34, unchanged; aggregate decoded ball barely speeds 0.0073→0.0101 vs GT 0.020→0.040). Heterogeneous: ~6–7/20 scenes show real 2–3× decoded speed-up, ~8 flat, 2 untrackable (NaN). Method is directionally better than probes (pixels DO move monotonically) but does NOT render faithful velocity magnitude — bottleneck is decoder faithfulness, not the steering vector; more scale didn't help. Next lever = decoder rendering, not data. Also OOM-fixed num_workers 4→0 mid-run.** Artifacts: `outputs/analysis/moving_ball_scene/diff_steer{,_pilot}/`.

- **2026-06-24 (eve)** — Step 2 DECODER COLLAPSE FIXED + steering re-run (honest result: milestone NOT met). The moving-ball decoder had collapsed to a blank frame (loss pinned flat ~0.84), blocking steering. Root-caused & fixed THREE stacked issues: (1) **sigmoid saturation** — pure-white (1.0) targets drive the output logit to +∞ where grad dies; fixed via target compression `loss.target_lo/hi=0.05/0.95` (finite optimum, live grads). (2) **foreground-gradient dilution** — soft darkness mask leaked bg into the fg denominator (~4× weaker ball grad); fixed via HARD mask in `foreground_weighted_charbonnier`. After (1)+(2) the ball renders but the decoder learned a dataset-AVERAGE ball (latent-blind, loss plateaued ~0.65). (3) **latent-blindness** — fixed via STATE CO-TRAINING (`loss.state=1, trajectory=5, velocity=5`; state head already built in reconstruct mode) → decoder now partially conditions on per-clip latent (decoded vel_y tracks GT sign by step 600). Trained `moving_ball_decoder_fg6` to step 3000 (LR decayed; effectively converged). **Steering (step_3000):** readout ρ=1.000 for speed/vel_x/vel_y, but decoded-measured ρ=0.64/0.32/0.25 are rank-corr on NEAR-FLAT noisy 6-clip curves — decoded speed sits ~0.0087 for ALL α (GT range 0.008–0.038). I.e. the velocity subspace is perfectly steerable in the readout but the decoder does NOT render the steered velocity as pixel motion — same "readout moves, pixels don't" bottleneck as Step 1. **The PI's milestone (visually confirm the ball changes speed under steering) is NOT achieved; the decoder's faithfulness is the limiter, not subspace decodability.** Infra fixes this session: decoder training needs `--mem=192G` + `train.num_workers=0` (per-worker unbounded `LatentDataset._shard_cache` OOMs; ~100GB cache); OOM mid-`torch.save` corrupts that checkpoint (resume from prior); cleaned ~300G of dead-run checkpoints (project quota was full → training died on Errno 122). Artifacts: `outputs/analysis/moving_ball_velocity/steer_{speed,vel_x,vel_y}/` (filmstrips, mp4s, summary json); decoder `outputs/runs/moving_ball_decoder_fg6/checkpoints/step_3000.pt`. Code: `src/decoders/loss_functions.py`, `src/utils/config.py`, `configs/train/moving_ball_decoder_large.yaml` (all uncommitted).

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
