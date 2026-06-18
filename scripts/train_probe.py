#!/usr/bin/env python
"""Experiment 1: layerwise physical decodability probes (linear + MLP, with controls).

Example
-------
    python scripts/train_probe.py --latent_dir outputs/smoke/latents \
        --output_dir outputs/smoke/probes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.analysis.visualization import layerwise_probe_plot
from src.training import probe_layers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    records = probe_layers(args.latent_dir, layers=layers, seed=args.seed,
                           output_csv=out / "layerwise_decodability.csv")
    if records:
        layerwise_probe_plot(records, out / "layerwise_probe.png")
    print(f"[train_probe] {len(records)} probe results -> {out}")


if __name__ == "__main__":
    main()
