# Method

## Research question

Can frozen V-JEPA-style video representations be decoded into physically meaningful visual predictions,
and do their latent spaces contain structured representations of physical variables (position,
velocity, acceleration, direction, gravity, collisions, object permanence)?

## Pipeline

1. **Ingest** physics videos — controlled synthetic clips with exact ground-truth state
   (`src/data/synthetic_physics.py`) and the Physics-IQ benchmark (`src/data/physics_iq.py`).
2. **Encode** with a *frozen* V-JEPA/V-JEPA2 backbone (`src/encoders/`). We never fine-tune the encoder
   in phase 1 — the question is what already exists in the representation.
3. **Extract & cache** multi-layer token latents (and optionally attention) to disk
   (`src/encoders/feature_extractor.py`) so decoder/probe training never reruns the encoder.
4. **Decode** with a transformer video decoder whose learned query tokens cross-attend the frozen
   latents (`src/decoders/transformer_video_decoder.py`). Four modes:
   - **A** latent → frame reconstruction,
   - **B** context latents → future frames,
   - **C** latent → physical state,
   - **D** latent → physical diagram (overlays).
5. **Probe** each layer with linear + MLP probes to measure decodability of each physical variable
   (`src/training/train_probe.py`).
6. **Evaluate** reconstruction, physics, and latent-geometry metrics against baselines and controls.
7. **Analyze** latent geometry — CKA across layers/datasets, intrinsic dimension, circular direction
   codes, gravity axes, and latent interventions.

## Claim taxonomy (read before reporting results)

These are *different* claims and the code/report keep them separate:

| Claim | Evidence in this repo |
|-------|-----------------------|
| The decoder reconstructs pixels well | reconstruction metrics > frame baselines |
| The latent contains physical information | MLP probe R² ≫ control |
| …linearly accessible | linear probe R² ≫ control |
| …nonlinear only | MLP ≫ linear, linear ≈ control |
| The model internally uses physical laws | **not** claimed from decodability alone |
| The model can be steered | monotonic, controlled intervention sweeps (Exp. 6) |

## Controls (always on)

- **Shuffled-latent** and **randomized-label** probe controls collapse to ~chance
  (`train_probe.py`); a test asserts the real probe beats them.
- **Oracle-state** is the physics-metric upper bound; **copy-first-frame / mean-frame / random-frame**
  are reconstruction baselines (`src/eval/baselines.py`).

## Roadmap / deferred

- Latent-diffusion refinement head (`decoders/latent_diffusion_decoder.py`).
- Optical-flow & FVD metrics (`eval/reconstruction_metrics.py`).
- DROID robotics + latent steering toward task success
  (`data/droid.py`, `analysis/steering.py`) — the bridge to robotics.
- Multi-node SLURM launch, HF Hub release utilities.
