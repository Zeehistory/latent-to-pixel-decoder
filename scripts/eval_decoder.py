#!/usr/bin/env python
"""Evaluate a trained decoder: reconstruction + physics metrics, with baselines and controls.

Writes ``<output_dir>/metrics.json``. The JSON separates the trained model from baselines
(copy-first-frame / mean-frame / random-frame) and the oracle-state upper bound, and tags physics
metrics with whether ground-truth state was available — so claims stay honest.

Example
-------
    python scripts/eval_decoder.py --config configs/train/smoke_synthetic.yaml \
        --latent_dir outputs/smoke/latents --checkpoint outputs/smoke/runs/decoder/checkpoints/last.pt \
        --output_dir outputs/smoke/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.eval.baselines import frame_baselines, oracle_state
from src.eval.physics_metrics import physics_metrics
from src.eval.reconstruction_metrics import reconstruction_metrics
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _avg(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = dicts[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in dicts if d[k] is not None]
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


@torch.no_grad()
def evaluate_decoder(cfg, latent_dir: str, checkpoint: str, output_dir: str | Path,
                     device: str = "cpu") -> dict[str, Any]:
    dataset = LatentDataset(latent_dir, layers=cfg.encoder.layers)
    rec0 = dataset.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames

    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(dataset.available_layers())
    load_checkpoint(checkpoint, decoder, map_location=device)

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=latent_collate)
    recon_model, recon_base = [], {k: [] for k in ("copy_first_frame", "mean_frame", "random_frame")}
    phys_model, phys_oracle = [], []
    state_keys = dataset[0]["state_keys"]

    for batch in loader:
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}
        out = decoder(latents, grid)
        target = batch["frames"].to(device)
        if out.frames is not None:
            recon_model.append(reconstruction_metrics(out.frames, target))
            for name, base in frame_baselines(target).items():
                recon_base[name].append(reconstruction_metrics(base, target))
        if out.state is not None and batch["state_mask"].sum() > 0:
            tgt_state, mask = batch["state"].to(device), batch["state_mask"][0]
            phys_model.append(physics_metrics(out.state, tgt_state, state_keys, mask))
            phys_oracle.append(physics_metrics(oracle_state(tgt_state), tgt_state, state_keys, mask))

    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "num_clips": len(dataset),
        "reconstruction": {
            "model": _avg(recon_model) if recon_model else None,
            "baselines": {k: _avg(v) for k, v in recon_base.items() if v and v[0]},
        },
        "physics": {
            "model": _avg(phys_model) if phys_model else None,
            "oracle_upper_bound": _avg(phys_oracle) if phys_oracle else None,
            "state_available": bool(phys_model),
        },
        "claim_taxonomy_note": (
            "reconstruction.model beating baselines => pixels are decodable; physics.model near "
            "oracle => state is decodable. These are distinct claims; see docs/method.md."
        ),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    res = evaluate_decoder(cfg, args.latent_dir, args.checkpoint, args.output_dir, args.device)
    print(f"[eval_decoder] metrics -> {args.output_dir}/metrics.json")
    print(json.dumps(res["reconstruction"]["model"], indent=2))


if __name__ == "__main__":
    main()
