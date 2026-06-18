"""Config loading and schema definitions.

Configs are OmegaConf YAML files. We keep lightweight dataclass schemas so that defaults are explicit
and discoverable in code, while still allowing free-form experiment overrides from YAML and the CLI.

The canonical entry point is :func:`load_config`, which merges (in order):

1. dataclass defaults (:class:`ExperimentConfig`),
2. the YAML file,
3. any ``key=value`` dotlist overrides (e.g. from ``argparse`` remainder args).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class EncoderConfig:
    name: str = "mock"
    # HuggingFace id used by the real wrappers, e.g. "facebook/vjepa2-vitl-fpc64-256".
    hf_model_id: str | None = None
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    patch_size: int = 16
    tubelet_size: int = 2
    image_size: int = 256
    num_frames: int = 16
    frozen: bool = True
    # Which transformer block outputs to extract. "all" or a list of ints.
    layers: Any = "all"
    extract_attention: bool = False
    device: str = "auto"
    dtype: str = "float32"


@dataclass
class DecoderConfig:
    name: str = "transformer"
    mode: str = "reconstruct"  # reconstruct | future | state | diagram
    hidden_dim: int = 512
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_query_tokens_per_frame: int = 64  # learned video query tokens per output frame
    use_layer_embedding: bool = True
    gradient_checkpointing: bool = False
    out_image_size: int = 64
    out_num_frames: int = 16
    out_channels: int = 3
    # mode == "state" head configuration
    state_dim: int = 0  # filled in from the dataset's state vector length
    # mode == "future": number of context frames whose latents are visible to the decoder
    context_frames: int = 8


@dataclass
class DataConfig:
    name: str = "synthetic_physics"
    root: str | None = None
    image_size: int = 64
    num_frames: int = 16
    fps: int = 8
    categories: Any = "all"
    split: str = "train"
    # synthetic generator knobs
    num_clips: int = 64
    scenarios: Any = field(default_factory=lambda: ["bouncing_ball", "projectile"])
    seed: int = 0


@dataclass
class LossConfig:
    charbonnier: float = 1.0
    ssim: float = 0.0
    ms_ssim: float = 0.0
    lpips: float = 0.0
    temporal_consistency: float = 0.0
    state: float = 0.0
    trajectory: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    collision: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Any = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    warmup_steps: int = 50
    max_steps: int = 200
    scheduler: str = "cosine"  # cosine | constant | linear
    grad_accum: int = 1
    ema_decay: float = 0.999


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_workers: int = 0
    mixed_precision: str = "no"  # no | fp16 | bf16
    log_every: int = 10
    ckpt_every: int = 100
    eval_every: int = 100
    resume: str | None = None
    seed: int = 0
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/run"
    latent_dir: str | None = None
    wandb: bool = False
    tags: Any = field(default_factory=list)


def _defaults() -> DictConfig:
    cfg = OmegaConf.structured(ExperimentConfig())
    assert isinstance(cfg, DictConfig)
    return cfg


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge dataclass defaults, an optional YAML file, and CLI dotlist overrides."""
    cfg = _defaults()
    if path is not None:
        file_cfg = OmegaConf.load(str(path))
        cfg = OmegaConf.merge(cfg, file_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Snapshot a resolved config to disk (used for every run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))


def to_container(cfg: DictConfig) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out
