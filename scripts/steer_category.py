#!/usr/bin/env python
"""Step 3 (category steering): turn a fluid clip "solid-like" along a category direction, then decode.

This is the wild experiment: the Step-1 category probe gives us a per-class linear direction
``w_solid`` and ``w_fluid`` (saved as ``category_directions.npz``). We take a real **fluid** Physics-IQ
clip, extract its latents ``z``, and steer along the *contrast* direction

    z' = z + alpha * (w_solid - w_fluid)

then **decode** ``z'`` to pixels with the trained transformer decoder. The question: does the same
scene start to behave/look solid-like as ``alpha`` grows? Our understanding of the physics subspace is
still crude, so there is no guarantee — this is an exploratory probe, and the decoded filmstrips are the
evidence to eyeball.

Two honesty notes baked in:

* The probe directions live in the probe's feature space (z-scored if the probe was standardized). We
  map them back to **raw latent space** via ``coef / std`` (see ``category_steering_direction``) so the
  edit is applied in the space the decoder actually consumes.
* For a *non-circular* controllability signal we fit an **independent** solid-vs-fluid logistic readout
  on the target cache (excluding the steered clips) and track ``P(solid)`` along the sweep — a margin
  read straight off ``(w_solid - w_fluid)`` would be monotonic by construction and prove nothing.

Example
-------
    python scripts/steer_category.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --target_latent_dir outputs/latents/physics_iq/vjepa2_large \
        --directions outputs/analysis/physics_iq/category_probe/category_directions.npz \
        --checkpoint outputs/runs/physics_iq_decoder_large/checkpoints/last.pt \
        --from_category fluid_dynamics --to_category solid_mechanics \
        --layer -1 --output_dir outputs/analysis/steer_category/fluid_to_solid --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from scipy.stats import spearmanr

from src.analysis import steering
from src.analysis import visualization as viz
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--target_latent_dir", required=True, help="cache to steer + decode (e.g. physics_iq)")
    p.add_argument("--directions", required=True, help="category_directions.npz from probe_categories.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--from_category", default="fluid_dynamics", help="category of the clips we steer")
    p.add_argument("--to_category", default="solid_mechanics", help="category we steer toward")
    p.add_argument("--layer", type=int, default=-1, help="intervention layer; -1 = deepest available")
    p.add_argument("--alphas", default="0,2,4,6,8")
    p.add_argument("--num_samples", type=int, default=4, help="how many from_category clips to steer")
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    device = args.device

    target = LatentDataset(args.target_latent_dir, layers=cfg.encoder.layers)
    layer = max(target.available_layers()) if args.layer < 0 else args.layer

    # 1. build the raw-latent steering direction w_to - w_from (mapped out of the probe's z-scored space).
    direction_np, dinfo = steering.category_steering_direction(
        args.directions, layer, args.from_category, args.to_category)
    direction = torch.from_numpy(direction_np).to(device)

    # which clips are from_category — these are the ones we steer.
    steer_idx = [i for i in range(len(target)) if target[i]["category"] == args.from_category]
    if not steer_idx:
        raise SystemExit(f"No clips with category '{args.from_category}' in {args.target_latent_dir}.")
    steer_idx = steer_idx[: args.num_samples]
    steer_ids = {target[i]["id"] for i in steer_idx}

    # 2. independent readout: P(to_category) fit on the rest of the cache (excludes the steered clips).
    readout = steering.category_readout(args.target_latent_dir, layer, args.to_category,
                                        exclude_ids=steer_ids)

    # 3. build decoder + load checkpoint.
    rec0 = target.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(target.available_layers())
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # 4. steer, decode, verify per clip.
    per_sample_curves: dict[str, list[dict]] = {}
    all_pred = {a: [] for a in alphas}
    for n, i in enumerate(steer_idx):
        batch = latent_collate([target[i]])
        sid = batch["id"][0]
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}

        if n == 0:
            # Magnitude diagnostic: the unit direction is added per token, so the per-token edit size
            # is |alpha|. Compare against the latent's own per-token norm to judge whether the sweep is
            # a gentle nudge or a hard push (helps distinguish "alpha too small" from "decoder ignores
            # the edit" when decoded pixels do not move).
            tok_norm = float(latents[layer].norm(dim=-1).mean())
            print(f"[steer_category] layer{layer} per-token L2 norm ~{tok_norm:.2f}; "
                  f"|alpha| range {min(abs(a) for a in alphas):g}..{max(abs(a) for a in alphas):g} "
                  f"=> per-token edit {min(abs(a) for a in alphas):g}..{max(abs(a) for a in alphas):g} "
                  f"({100*max(abs(a) for a in alphas)/max(tok_norm,1e-6):.1f}% of token norm at max)")

        curve = steering.readout_along_direction(latents, direction, layer, alphas, readout)
        per_sample_curves[sid] = curve
        for r in curve:
            all_pred[r["alpha"]].append(r["readout"])

        decoded = steering.decode_intervention(decoder, latents, grid, direction, layer, alphas)
        frames_by_alpha = {a: decoded[a].frames[0].cpu() for a in alphas if decoded[a].frames is not None}
        if frames_by_alpha:
            viz.steering_filmstrip(frames_by_alpha, out / f"{sid}_filmstrip.png")
            if n == 0:
                base = frames_by_alpha[min(alphas, key=abs)]
                for a in (min(alphas), max(alphas)):
                    if a in frames_by_alpha:
                        viz.panel_video(base, frames_by_alpha[a], out / f"{sid}_alpha{a:g}.mp4",
                                        fps=cfg.data.fps)

    # aggregate controllability curve: does P(to_category) rise with alpha?
    mean_curve = [{"alpha": a, "readout": float(np.mean(v))} for a, v in sorted(all_pred.items())]
    rho = float(spearmanr([r["alpha"] for r in mean_curve], [r["readout"] for r in mean_curve]).statistic)
    viz.steering_sweep_plot(
        {"mean (steered clips)": mean_curve, **{k: v for k, v in list(per_sample_curves.items())[:6]}},
        out / "controllability.png",
        title=f"Steering {args.from_category} -> {args.to_category} @layer{layer}",
        ylabel=f"P({args.to_category}) (independent readout)")

    summary = {
        "from_category": args.from_category, "to_category": args.to_category, "layer": int(layer),
        "alphas": alphas, "n_steered": len(steer_idx), "direction_info": dinfo,
        "readout_monotonicity_spearman": round(rho, 4),
        "mean_readout_curve": [{"alpha": r["alpha"], "readout": round(r["readout"], 5)} for r in mean_curve],
        "target_latent_dir": args.target_latent_dir, "directions": args.directions,
    }
    (out / "category_steering_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / f"direction_{args.from_category}_to_{args.to_category}_layer{layer}.npy", direction_np)
    print(f"[steer_category] {args.from_category}->{args.to_category} @layer{layer}: "
          f"readout monotonicity rho={rho:.3f} over {len(steer_idx)} clips -> {out}")


if __name__ == "__main__":
    main()
