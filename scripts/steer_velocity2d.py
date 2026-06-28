#!/usr/bin/env python
"""Pixel-level proof for the 2D-velocity subspace / operator (Phases 3-5).

Loads the held-out TEST scene cache + the faithful decoder + the artifacts fit on TRAIN by
``scripts/velocity_subspace.py`` (global PCA basis U, ridge operator B, canonicalized ridge B_canon).
For each test scene it forms the anchor->extreme pair (v_a -> v_b, Delta v = v_b - v_a, Delta H = H_b - H_a)
and steers H_a by several methods, then DECODES and re-tracks the ball's 2D velocity vector:

  full_delta      H_a + Delta H                         per-pair on-manifold baseline (the r~0.95 analog)
  subspace_U[k]   H_a + P_U(Delta H)        U = top-k global PCA basis  (Phase 3: does U preserve it?)
  random[k]       H_a + P_R(Delta H)        random same-rank subspace   (Phase 3 control: should fail)
  ridge_global    H_a + B . Delta v         steer straight from velocity (Phase 4: does it transfer?)
  canon_ridge     H_a + roll^-1( B_canon . Delta v )   (Phase 5: canonicalize -> transfer recovered?)

Reports, per method, the decoded-vs-target velocity vector correlation (vx & vy), mean angle error (deg),
and speed ratio, aggregated over the steered scenes. Decoded velocity is compared to the TARGET v_b.

    python scripts/steer_velocity2d.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../moving_ball_scene_v2d/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_v2d/subspace \
        --checkpoint .../moving_ball_scene_v2d_decoder_fp/checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_v2d/steer --ks 2,4,8 --num_scenes 30 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis import visualization as viz
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, grid, device):
    """H_a (dict L->(1,Ltok,D) torch) + per-layer flat edit (numpy D,) -> new latent dict."""
    T, H, W = grid
    out = {}
    for L, t in Ha.items():
        Ltok = t.shape[1]
        Dd = t.shape[2]
        e = torch.from_numpy(edit_flat[L].reshape(Ltok, Dd).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


@torch.no_grad()
def _decode_vel(decoder, latents, grid, want_frames=False):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"speed": float("nan"), "vel_x": float("nan"), "vel_y": float("nan")}, None
    fr = out.frames[0].cpu()
    return measured_velocity(fr), (fr if want_frames else None)


def _agg(decoded, target):
    """decoded/target: lists of (vx,vy). Return correlation(vx,vy), angle err deg, speed ratio."""
    d = np.asarray(decoded); t = np.asarray(target)
    ok = np.isfinite(d).all(1)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "rho_vx": float("nan"), "rho_vy": float("nan"),
                "angle_err_deg": float("nan"), "speed_ratio": float("nan")}
    rho_vx = float(np.corrcoef(t[:, 0], d[:, 0])[0, 1])
    rho_vy = float(np.corrcoef(t[:, 1], d[:, 1])[0, 1])
    # per-sample angle between decoded and target velocity
    dot = (d * t).sum(1)
    cos = dot / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sr = np.linalg.norm(d, axis=1) / (np.linalg.norm(t, axis=1) + 1e-12)
    return {"n": int(len(d)), "rho_vx": round(rho_vx, 4), "rho_vy": round(rho_vy, 4),
            "angle_err_deg": round(float(np.nanmean(ang)), 2),
            "speed_ratio": round(float(np.nanmedian(sr)), 3)}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", default="2,4,8", help="subspace ranks to test")
    p.add_argument("--num_scenes", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    ks = [int(x) for x in args.ks.split(",")]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[steer2d] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}")

    # artifacts
    Bt = {L: np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64) for L in layers}
    Bt_canon = {L: np.load(art / f"ridge_canon_Bt_L{L}.npy").astype(np.float64) for L in layers}
    Ubasis = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}
    rng = np.random.default_rng(0)
    Rbasis = {L: {k: vo.random_basis(Bt[L].shape[1], k, rng) for k in ks} for L in layers}

    # decoder
    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    methods = ["full_delta", "ridge_global", "canon_ridge"] + \
              [f"subspace_U{k}" for k in ks] + [f"random{k}" for k in ks]
    decoded = {m: [] for m in methods}
    targets = []
    per_scene = {}

    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
        dv = vb - va
        pos = vo.clip_start_pos(sa)
        sh = vo.canon_shift(pos, grid)
        Ha = _to_dev(sa, layers, device)

        dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}
        edits = {
            "full_delta": dH,
            "ridge_global": {L: dv @ Bt[L] for L in layers},
            "canon_ridge": {L: vo.roll_layer(dv @ Bt_canon[L], grid, (-sh[0], -sh[1])) for L in layers},
        }
        for k in ks:
            edits[f"subspace_U{k}"] = {L: vo.project(dH[L], Ubasis[L][:k]) for L in layers}
            edits[f"random{k}"] = {L: vo.project(dH[L], Rbasis[L][k]) for L in layers}

        targets.append(vb)
        keep_frames = {}
        sc_row = {"v_a": va.tolist(), "v_b": vb.tolist()}
        for m in methods:
            Hstar = _apply_edit(Ha, edits[m], grid, device)
            want = m in ("full_delta", "ridge_global", "canon_ridge") and n < 6
            meas, fr = _decode_vel(decoder, Hstar, grid, want_frames=want)
            decoded[m].append([meas["vel_x"], meas["vel_y"]])
            sc_row[m] = [round(meas["vel_x"], 5), round(meas["vel_y"], 5)]
            if fr is not None:
                keep_frames[m] = fr
        per_scene[f"scene{s:05d}"] = sc_row
        if keep_frames:
            try:  # never let a plotting hiccup lose the numeric results
                viz.steering_filmstrip(
                    {i: keep_frames[m] for i, m in enumerate(keep_frames)},
                    out / f"scene{s:05d}_methods.png")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] filmstrip scene{s:05d} failed: {e}")
        print(f"  scene{s:05d}: v_b=({vb[0]:.4f},{vb[1]:.4f}) "
              f"full={tuple(round(x,4) for x in decoded['full_delta'][-1])} "
              f"ridge={tuple(round(x,4) for x in decoded['ridge_global'][-1])} "
              f"canon={tuple(round(x,4) for x in decoded['canon_ridge'][-1])}")

    results = {m: _agg(decoded[m], targets) for m in methods}
    summary = {"test_dir": args.test_dir, "checkpoint": args.checkpoint, "layers": layers,
               "n_scenes": len(scene_ids), "ks": ks, "target": "v_b (decoded vs target velocity)",
               "results": results, "per_scene": per_scene}
    (out / "steer2d_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[steer2d] decoded-vs-target velocity (aggregate):")
    for m in methods:
        r = results[m]
        print(f"  {m:16s} rho_vx={r['rho_vx']!s:>7} rho_vy={r['rho_vy']!s:>7} "
              f"angle_err={r['angle_err_deg']!s:>6}deg speed_ratio={r['speed_ratio']} n={r['n']}")
    print(f"[steer2d] -> {out}/steer2d_summary.json")


if __name__ == "__main__":
    main()
