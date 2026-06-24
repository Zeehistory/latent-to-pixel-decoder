<div align="center">

# vjepa-latent-physics

**Probing and steering physical quantities in the latent space of frozen V-JEPA2 video models.**

[![ci](https://img.shields.io/badge/ci-github--actions-blue)](.github/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

</div>

A research-grade, reproducible framework for two questions about **frozen** V-JEPA2 video
representations: (1) **which latent subspace encodes which physical quantity** — position, velocity,
acceleration, direction, gravity, collisions, object permanence — and (2) **can we steer that
subspace** and decode the result back into video to visually confirm the intervention changed the
physics, not just the appearance.

The methodology is deliberately staged (see [ROADMAP.md](ROADMAP.md)): probe real benchmark video
(Step 1), isolate single quantities with maximally clean synthetic data (Step 2), transfer the learned
quantity directions back to real video (Step 3), and apply the same machinery to robotics (Step 4).

> **Research discipline.** This repo is careful to separate distinct claims:
> *"the decoder reconstructs pixels"* ≠ *"the latent contains physical information"* ≠
> *"that information is **linearly** accessible"* ≠ *"the model internally uses physical laws"* ≠
> *"the model can be steered."* Every experiment ships with baselines and controls
> (oracle, random/frozen, **shuffled-latent**, **randomized-label**) so the conclusions stay honest.

---

## Why this exists

V-JEPA-style models learn predictive video representations in a latent space. The open question this
repo operationalizes:

> *Can frozen V-JEPA latents be decoded into physically meaningful predictions, and do they contain
> structured representations of physical variables?*

Concretely the pipeline answers: **which layers** are most decodable for each physical variable,
whether variables are **linear / nonlinear / curved-manifold** structured, how far physics-benchmark
data sits from natural video in latent space, and whether **latent interventions** produce predictable
decoded physical changes.

## Architecture at a glance

```
 physics video ──► [frozen V-JEPA encoder] ──► multi-layer latent tokens ──► [decoder]
                            │                          │                         ├─ A: latent → frames (reconstruction)
                            │                          │                         ├─ B: context latents → future frames
                            │                          ├─ probes (linear/MLP) ─► C: latent → physical state
                            │                          └─ geometry/manifold     └─ D: latent → physical diagram (overlays)
                            └─ caching: WebDataset shards + Parquet metadata + sha256
```

## Install

```bash
git clone <repo> && cd vjepa-latent-physics
python -m pip install -e .[dev]                 # offline mock pipeline + tooling
# optional extras:
python -m pip install -e .[encoders]            # real V-JEPA2 weights (transformers + hub)
python -m pip install -e .[extras,demo,logging] # LPIPS/UMAP/WebDataset, Gradio demo, WandB/TB
```

CPU/macOS-MPS works out of the box for the **mock encoder** and the smoke pipeline. CUDA is only
needed for real V-JEPA2 weights and large-scale training.

## Quickstart (runs offline, CPU)

The smoke pipeline generates a tiny synthetic-physics dataset, extracts latents with a deterministic
**mock encoder** (V-JEPA2-shaped), trains a small transformer decoder, fits layerwise probes, evaluates
with controls, and writes visualizations:

```bash
python scripts/run_full_pipeline.py \
  --config configs/train/smoke_synthetic.yaml \
  --output_dir outputs/smoke
```

Outputs under `outputs/smoke/`:

```
latents/            WebDataset shards + metadata.parquet + checksums.json
runs/decoder/       checkpoints/ (EMA + resume), resolved_config.yaml, run_meta.json (git + hardware)
probes/             layerwise_decodability.csv
eval/metrics.json   reconstruction + physics metrics + baselines/controls
viz/                recon_grid.png, error_map.png, trajectory_overlay.png, layerwise_probe.png, *.mp4
```

## Using real V-JEPA2 weights

It's a **config swap, not a code change** — the wrappers share the same `LatentBundle` contract as the
mock encoder:

```bash
python scripts/extract_latents.py \
  --dataset physics_iq \
  --encoder vjepa2_large \
  --layers all \
  --output_dir outputs/latents/physics_iq/vjepa2_large
```

`configs/encoder/vjepa2_large.yaml` points at `facebook/vjepa2-vitl-fpc64-256` (HuggingFace), i.e.
VJEPA2 ViT-L with 24 layers. Run on a CUDA box; the encoder stays **frozen** by default. Treat ViT-H /
32-layer validation as a separate config and latent cache, not a drop-in reuse of the ViT-L run.

## Full training

```bash
accelerate launch scripts/train_decoder.py \
  --config configs/train/physics_iq_transformer_large.yaml \
  --encoder vjepa2_large \
  --latent_dir outputs/latents/physics_iq/vjepa2_large \
  --output_dir outputs/runs/physics_iq_decoder_large
```

Physics-IQ evaluation reports global metrics plus `reconstruction.by_category` / `physics.by_category`
entries, so fluid dynamics, optics, thermodynamics, solid mechanics, magnetism, and misc clips stay
separated in `metrics.json`.

Decoder sizes (`configs/decoder/`): `decoder_small` (512/8/8), `decoder_base` (768/12/12),
`decoder_large` (1024/24/16), `decoder_huge` (1536/32/24).

## Experiments

| # | Experiment | Script / config |
|---|------------|-----------------|
| 1 | Layerwise physical decodability | `scripts/train_probe.py` · `configs/analysis/layerwise_probe.yaml` |
| 2 | Transformer decoder reconstruction | `scripts/train_decoder.py` · `configs/train/*` |
| 3 | Future prediction (mode B) | `configs/train/*` `decoder.mode: future` |
| 4 | Dataset shift in latent space | `scripts/analyze_latents.py` · `configs/analysis/manifold.yaml` |
| 5 | Latent manifold structure (circular direction codes, gravity axis) | `src/analysis/{manifold_analysis,direction_codes}.py` |
| 6 | Latent intervention / steering prototype | `src/analysis/{intervention,steering}.py` · `configs/analysis/steering.yaml` |

See [docs/experiments.md](docs/experiments.md) for the full protocol and expected outputs.

## Repo layout

```
configs/   encoder · decoder · data · train · eval · analysis  (OmegaConf YAML)
src/       data · encoders · decoders · training · eval · analysis · utils
scripts/   download_data · extract_latents · train_decoder · train_probe · eval_decoder ·
           analyze_latents · visualize_reconstructions · run_full_pipeline
tests/     CPU-only shape / training-step / metric-correctness / control-sanity tests
docs/      method · datasets · reproducibility · open_source_release · model_card · dataset_card · experiments
demo/      Gradio app
```

## Research roadmap

The scientific programme runs in four stages; full detail, status, and results live in
[ROADMAP.md](ROADMAP.md).

| Step | Question | Status |
|------|----------|--------|
| **1 — Category probe (real video)** | Can linear probes separate physics categories in Physics-IQ latents? | ✅ **Done (positive).** Linear acc 0.91 (z) / 0.90 (raw); raw≈z-score ⇒ separability is real, not a normalization artifact. |
| **2 — Quantity probe + steering (synthetic)** | Which subspace encodes *velocity*, and can we steer it so the decoded ball visibly speeds up/slows down? | 🔬 **In progress.** Clean moving-ball dataset; velocity/occlusion/equivariance probes done, decoder + steering running on SLURM. |
| **3 — Transfer to real video** | Do quantity directions learned on synthetic data transfer to real Physics-IQ video? | ◻️ Category-centroid version done (clean **negative**: appearance shift, not physics). Quantity-direction version **TODO**. |
| **4 — Robotics** | Can we identify and steer the latent difference between achievable vs non-achievable actions? | ◻️ Not started (DROID stubbed). |

**Engineering status.** Implemented and CPU-tested: synthetic physics generators with exact
ground-truth state, mock + real-V-JEPA2 encoder wrappers, multi-layer latent extraction & caching,
transformer video decoder (modes A/C), linear/MLP/temporal probes, reconstruction + physics + latent
metrics, always-on baselines/controls, and the visualization suite. Deferred (typed stubs with
documented contracts): latent-diffusion refinement head, optical-flow & FVD metrics, DROID/robotics,
multi-node SLURM launch, HF Hub upload, polished demo. See
[docs/open_source_release.md](docs/open_source_release.md).

## Troubleshooting

- **`transformers` not installed** — only needed for real weights: `pip install -e .[encoders]`. The
  mock pipeline runs without it.
- **No CUDA / on macOS** — everything except real-weight extraction and large training runs on CPU/MPS.
- **LPIPS / UMAP missing** — optional; install `.[extras]`. Metrics that need them are skipped with a
  logged warning, never silently faked.

## Citing

See [CITATION.cff](CITATION.cff). This project builds on Meta AI's V-JEPA and V-JEPA 2. Respect the
upstream model and dataset licenses (see [docs/open_source_release.md](docs/open_source_release.md)).

## License

Apache-2.0 — see [LICENSE](LICENSE).
