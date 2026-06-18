# Reproducibility

Every run snapshots, into its `output_dir`:
- `resolved_config.yaml` — the fully-merged OmegaConf config.
- `run_meta.json` — UTC timestamp, **git commit**, dirty flag, and hardware summary
  (`src/utils/reproducibility.py`, `src/utils/gpu.py`).
- `metrics.jsonl` — per-step training metrics.

Determinism: `set_seed(seed, deterministic=True)` seeds Python/NumPy/Torch and sets cuDNN
deterministic flags. Latent caches store a `checksums.json` (sha256 per shard) and `extract_meta.json`
(encoder id, config, preprocessing, layer indices, provenance).

Distributed/scale: native PyTorch + HuggingFace Accelerate (`accelerate launch ...`); multi-node via
`src/utils/slurm.py` (`write_sbatch_script`). Mixed precision (`fp16`/`bf16`), gradient accumulation,
gradient checkpointing, and EMA are all config-driven.

CPU reproduction: the smoke pipeline (`configs/train/smoke_synthetic.yaml`) reproduces the full
loop offline; CI runs it on every push.
