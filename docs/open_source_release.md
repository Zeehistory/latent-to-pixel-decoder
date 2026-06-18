# Open-source release notes

## License compatibility
This repo is Apache-2.0. **Upstream models/datasets carry their own licenses** — V-JEPA / V-JEPA 2
weights (Meta AI) and Physics-IQ are governed by their respective terms; review them before
redistributing weights or data. We ship no third-party weights or datasets in this repo.

## Release checklist
- [ ] `pytest` green on CPU; CI passing.
- [ ] `ruff check` / `mypy src` clean.
- [ ] Smoke pipeline reproduces (`run_full_pipeline.py`).
- [ ] Model card (`docs/model_card.md`) + dataset card (`docs/dataset_card.md`) filled in.
- [ ] Pretrained decoder checkpoint on a small public subset uploaded; download script added.
- [ ] Hugging Face Hub upload utility wired (deferred — see roadmap).

## Deferred for release
Latent-diffusion head, FVD/optical-flow metrics, DROID, full steering, multi-node SLURM auto-requeue,
HF Hub upload utilities, polished hosted demo.
