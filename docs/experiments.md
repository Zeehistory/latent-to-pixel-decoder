# Experiments

| # | Name | How to run | Key outputs |
|---|------|------------|-------------|
| 1 | Layerwise physical decodability | `scripts/train_probe.py --latent_dir <L> --output_dir <O>` | `layerwise_decodability.csv`, `layerwise_probe.png` |
| 2 | Transformer decoder reconstruction | `scripts/train_decoder.py --config configs/train/...` then `eval_decoder.py` | `metrics.json`, recon panels |
| 3 | Future prediction | set `decoder.mode: future`, `decoder.context_frames` | future-frame metrics |
| 4 | Dataset shift | `analyze_latents.py` + `eval.generalization` (MMD²/Fréchet/CKA) | shift report |
| 5 | Manifold structure | `analysis/manifold_analysis.py`, `direction_codes.py` | intrinsic dim, circular-code score |
| 6 | Latent intervention/steering | `analysis/intervention.py` + `configs/analysis/steering.yaml` | controllability sweep |

Each experiment reports **baselines and controls** alongside the model. See `docs/method.md` for the
claim taxonomy. Expected smoke-run sanity: reconstruction PSNR above the copy-first-frame baseline;
probe R² for position/velocity above shuffled-latent/randomized-label controls.
