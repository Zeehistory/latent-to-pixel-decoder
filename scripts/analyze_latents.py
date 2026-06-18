#!/usr/bin/env python
"""Latent-space analysis: layerwise CKA, intrinsic dimension, PCA embedding, dataset-shift, manifolds.

Example
-------
    python scripts/analyze_latents.py --latent_dir outputs/smoke/latents --output_dir outputs/smoke/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.analysis.latent_geometry import layerwise_cka_matrix, pca, pooled_features
from src.analysis.manifold_analysis import participation_ratio, twonn_intrinsic_dim
from src.analysis.visualization import cka_heatmap
from src.encoders.feature_extractor import LatentDataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = LatentDataset(args.latent_dir).available_layers()

    summary = {"layers": layers, "intrinsic_dim": {}}
    for li in layers:
        feats, _ = pooled_features(args.latent_dir, li)
        summary["intrinsic_dim"][str(li)] = {
            "participation_ratio": round(participation_ratio(feats), 3),
            "twonn": round(float(twonn_intrinsic_dim(feats)), 3),
        }

    mat = layerwise_cka_matrix(args.latent_dir, layers)
    cka_heatmap(mat, [f"L{li}" for li in layers], out / "layerwise_cka.png", "Layerwise CKA")

    # PCA scatter of the last layer, colored by category
    feats, cats = pooled_features(args.latent_dir, layers[-1])
    coords, var = pca(feats, 2)
    summary["last_layer_pca_explained_var"] = [round(float(v), 3) for v in var]
    np.save(out / "last_layer_pca_coords.npy", coords)

    (out / "latent_analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"[analyze_latents] wrote analysis -> {out}")


if __name__ == "__main__":
    main()
