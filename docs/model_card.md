# Model card — V-JEPA Physics Decoder (template)

- **Model:** transformer video decoder over frozen V-JEPA/V-JEPA2 latents.
- **Inputs:** cached multi-layer latent tokens `{layer: (B, L, D)}` + token grid.
- **Outputs:** reconstructed/future frames (modes A/B/D) and/or physical state (mode C).
- **Encoder:** frozen; not modified. Real weights: `facebook/vjepa2-*` (their license applies).
- **Training data:** synthetic physics (exact GT) and/or Physics-IQ.
- **Intended use:** research on physical-representation decoding & interpretability.
- **Limitations:** decodability ≠ the model "using physical laws" (see claim taxonomy). Synthetic
  results may not transfer to real video; report controls.
- **Metrics:** PSNR/SSIM/MS-SSIM/LPIPS/temporal-consistency; physics RMSE/F1; layerwise probe R².
- **Ethical considerations:** trained on physics/robotics video; not for surveillance or generation of
  deceptive media.
