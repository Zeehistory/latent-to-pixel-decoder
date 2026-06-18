"""Dataset registry, sample schema, and collation."""

from __future__ import annotations

import torch

from src.data import build_dataset
from src.data.dataset_registry import collate


def test_synthetic_dataset_schema(tiny_cfg):
    ds = build_dataset(tiny_cfg.data, encoder_image_size=tiny_cfg.encoder.image_size,
                       encoder_frames=tiny_cfg.encoder.num_frames)
    assert len(ds) == tiny_cfg.data.num_clips
    s = ds[0]
    for key in ("id", "frames", "encoder_input", "state", "state_mask", "state_keys", "category"):
        assert key in s
    assert s["frames"].shape[0] == tiny_cfg.data.num_frames
    assert s["encoder_input"].shape[-1] == tiny_cfg.encoder.image_size
    assert s["state"].shape[-1] == len(s["state_keys"])


def test_state_padding_consistent_across_scenarios(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.data.scenarios = ["bouncing_ball", "collision"]  # 1- and 2-object scenarios
    ds = build_dataset(cfg.data, encoder_image_size=cfg.encoder.image_size,
                       encoder_frames=cfg.encoder.num_frames)
    dims = {ds[i]["state"].shape[-1] for i in range(len(ds))}
    assert len(dims) == 1  # padded to a common width


def test_collate(tiny_cfg):
    ds = build_dataset(tiny_cfg.data, encoder_image_size=tiny_cfg.encoder.image_size,
                       encoder_frames=tiny_cfg.encoder.num_frames)
    batch = collate([ds[0], ds[1]])
    assert batch["frames"].shape[0] == 2
    assert isinstance(batch["id"], list) and len(batch["id"]) == 2
    assert torch.is_tensor(batch["state"])


def test_unknown_dataset_raises(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.data.name = "does_not_exist"
    try:
        build_dataset(cfg.data)
        raised = False
    except KeyError:
        raised = True
    assert raised
