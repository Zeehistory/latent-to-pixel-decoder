#!/usr/bin/env python
"""Gradio demo: encode a video, decode a reconstruction, view latent projection + predicted state.

Run:
    pip install -e .[demo]
    python demo/app.py --config configs/train/smoke_synthetic.yaml \
        --checkpoint outputs/smoke/runs/decoder/checkpoints/last.pt

With the mock encoder it runs fully offline. Swap `--config` to a real-encoder config (and install
`.[encoders]`) to use V-JEPA2 weights. This is a research demo, not a product.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch

from src.analysis.latent_geometry import pca
from src.decoders import build_decoder
from src.encoders import build_encoder
from src.utils.config import load_config
from src.utils.video_io import load_video


def build(cfg, checkpoint: str | None):
    encoder = build_encoder(cfg.encoder).eval()
    # state_dim unknown without the latent cache; demo decodes frames only unless a ckpt provides it.
    decoder = build_decoder(cfg.decoder, encoder.hidden_dim, state_dim=0)
    if checkpoint:
        from src.training.checkpoints import load_checkpoint

        load_checkpoint(checkpoint, decoder, map_location="cpu")
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def run(video_path, layer_choice, encoder, decoder, cfg):
    frames = load_video(video_path, num_frames=cfg.encoder.num_frames, image_size=cfg.encoder.image_size)
    from src.data.video_transforms import normalize

    enc_in = normalize(frames).unsqueeze(0)
    bundle = encoder.encode(enc_in, layers="all")
    layers = {int(layer_choice): bundle.layers[int(layer_choice)]} if layer_choice != "all" else bundle.layers
    grid = (bundle.grid.temporal, bundle.grid.height, bundle.grid.width)
    out = decoder({int(k): v for k, v in layers.items()}, grid)
    recon = out.frames[0].clamp(0, 1).permute(0, 2, 3, 1).numpy() if out.frames is not None else None

    tokens = bundle.layers[int(layer_choice) if layer_choice != "all" else bundle.layer_indices[-1]]
    coords, _ = pca(tokens[0].numpy(), 2)
    return (recon * 255).astype(np.uint8) if recon is not None else None, coords


def main() -> None:
    import gradio as gr

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    encoder, decoder = build(cfg, args.checkpoint)
    layer_opts = ["all"] + [str(i) for i in range(encoder.num_layers)]

    def infer(video, layer):
        recon, coords = run(video, layer, encoder, decoder, cfg)
        frames = [recon[i] for i in range(recon.shape[0])] if recon is not None else None
        return frames, coords.tolist()

    with gr.Blocks(title="V-JEPA Physics Decoder") as demo:
        gr.Markdown("# V-JEPA Physics Decoder\nEncode a video with a frozen V-JEPA-style model, "
                    "decode a reconstruction, and inspect the latent projection.")
        with gr.Row():
            inp = gr.Video(label="input video")
            layer = gr.Dropdown(layer_opts, value="all", label="encoder layer(s)")
        btn = gr.Button("Encode + Decode")
        gallery = gr.Gallery(label="reconstruction frames")
        coords = gr.JSON(label="latent PCA (per token)")
        btn.click(infer, [inp, layer], [gallery, coords])
    demo.launch()


if __name__ == "__main__":
    main()
