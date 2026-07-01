#!/usr/bin/env python
"""Pixel-level proof for restitution subspace / command operators.

Steers anchor->extreme pairs (low e -> high e) with the same method suite as velocity v2d, then
decodes and measures bounce physics from tracked pixels.

Methods: full_delta, subspace_U[k], random[k], ridge_global, canon_ridge, ridge_projU8,
cmd_U8 (gain sweep), ridge_rich.

Metrics (aggregate + per-scene):
  - speed_ratio: decoded post/pre bounce speed vs GT target
  - restitution_err: |decoded_ratio - target_ratio|
  - rebound_peak_err: |decoded_peak_y - target_peak_y| (lower y = higher bounce)
  - rho_speed_ratio: correlation of decoded vs target speed ratios across scenes

    python scripts/steer_restitution.py --config configs/train/moving_ball_scene_restitution_decoder.yaml \
        --test_dir .../moving_ball_scene_restitution/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_restitution/subspace \
        --checkpoint .../checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_restitution/steer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.analysis import restitution_ops as ro
from src.analysis import visualization as viz
from src.analysis.ball_tracking import measured_bounce
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, grid, device):
    T, H, W = grid
    out = {}
    for L, t in Ha.items():
        Ltok = t.shape[1]
        Dd = t.shape[2]
        e = torch.from_numpy(edit_flat[L].reshape(Ltok, Dd).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


def _agg(decoded: list[dict], targets: list[dict]) -> dict:
    """Aggregate bounce metrics across steered scenes."""
    dr = np.array([d["speed_ratio"] for d in decoded], float)
    tr = np.array([t["speed_ratio"] for t in targets], float)
    de = np.array([d["restitution"] for d in decoded], float)
    te = np.array([t["restitution"] for t in targets], float)
    dpeak = np.array([d["rebound_peak_y"] for d in decoded], float)
    tpeak = np.array([t["rebound_peak_y"] for t in targets], float)
    ok = np.isfinite(dr) & np.isfinite(tr)
    n = int(ok.sum())
    if n < 2:
        return {"n": n, "rho_speed_ratio": float("nan"), "rho_restitution": float("nan"),
                "restitution_ratio_mae": float("nan"), "rebound_peak_mae": float("nan")}
    rho_sr = float(np.corrcoef(tr[ok], dr[ok])[0, 1])
    ok_e = ok & np.isfinite(de) & np.isfinite(te)
    rho_e = float(np.corrcoef(te[ok_e], dr[ok_e])[0, 1]) if ok_e.sum() >= 2 else float("nan")
    return {
        "n": n,
        "rho_speed_ratio": round(rho_sr, 4),
        "rho_restitution": round(rho_e, 4),
        "restitution_ratio_mae": round(float(np.mean(np.abs(dr[ok] - tr[ok]))), 4),
        "rebound_peak_mae": round(float(np.nanmean(np.abs(dpeak[ok] - tpeak[ok]))), 4),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", default="2,4,8,16")
    p.add_argument("--num_scenes", type=int, default=40)
    p.add_argument("--cmd_scales", default="1.0,1.5,2.0,2.5,3.0")
    p.add_argument("--cmd_ku", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    ks = [int(x) for x in args.ks.split(",")]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = ro.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[steer_rest] {len(scenes)} test scenes, steering {len(scene_ids)}")

    B = {L: np.load(art / f"ridge_B_L{L}.npy").astype(np.float64) for L in layers}
    B_canon = {L: np.load(art / f"ridge_canon_B_L{L}.npy").astype(np.float64) for L in layers}
    Ubasis = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}
    rng = np.random.default_rng(0)
    Rbasis = {L: {k: ro.random_basis(B[L].size, k, rng) for k in ks} for L in layers}

    Wu = {}
    Brich = {}
    have_cmd = False
    wu_tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    try:
        Wu = {L: np.load(art / f"cmd_Wu{wu_tag}_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        have_cmd = True
    except FileNotFoundError:
        print("[steer_rest] no command-operator artifacts; skipping cmd methods")

    cmd_scales = [float(x) for x in args.cmd_scales.split(",") if x.strip()]
    cmd_methods = ([f"cmd_U8_s{g:g}" for g in cmd_scales] + ["ridge_rich"]) if have_cmd else []
    methods = ["full_delta", "ridge_global", "canon_ridge", "ridge_projU8"] + cmd_methods + \
              [f"subspace_U{k}" for k in ks] + [f"random{k}" for k in ks]

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    decoded = {m: [] for m in methods}
    targets = []
    per_scene = {}

    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        ea, eb = ro.clip_restitution(sa), ro.clip_restitution(sb)
        de = eb - ea
        va = ro.clip_incoming_velocity(sa)
        pos = ro.clip_start_pos(sa)
        sh = ro.canon_shift(pos, grid)
        Ha = _to_dev(sa, layers, device)
        bounce_hint = {"bounce_frame": int((sa.get("meta") or {}).get("bounce_frame", -1))}

        dH = {L: ro.layer_flat(sb["layers"][L]) - ro.layer_flat(sa["layers"][L]) for L in layers}
        edits = {
            "full_delta": dH,
            "ridge_global": {L: de * B[L] for L in layers},
            "canon_ridge": {L: ro.roll_layer(de * B_canon[L], grid, (-sh[0], -sh[1])) for L in layers},
            "ridge_projU8": {L: ro.project(de * B[L], Ubasis[L][:8]) for L in layers},
        }
        if have_cmd:
            phi = ro.command_features(ea, eb, va, va)
            cU = {L: (phi @ Wu[L]) @ Ubasis[L][: Wu[L].shape[1]] for L in layers}
            for g in cmd_scales:
                edits[f"cmd_U8_s{g:g}"] = {L: g * cU[L] for L in layers}
            edits["ridge_rich"] = {L: phi @ Brich[L] for L in layers}
        for k in ks:
            edits[f"subspace_U{k}"] = {L: ro.project(dH[L], Ubasis[L][:k]) for L in layers}
            edits[f"random{k}"] = {L: ro.project(dH[L], Rbasis[L][k]) for L in layers}

        tgt = {
            "restitution": eb,
            "speed_ratio": float((sb.get("meta") or {}).get("speed_ratio", float("nan"))),
            "rebound_peak_y": ro.gt_rebound_peak_y(sb),
            "post_bounce_speed": float((sb.get("meta") or {}).get("post_bounce_speed", float("nan"))),
        }
        targets.append(tgt)
        sc_row = {"e_a": ea, "e_b": eb, "target": tgt}
        for m in methods:
            Hstar = _apply_edit(Ha, edits[m], grid, device)
            want = m in ("full_delta", "ridge_global") and n < 6
            out_dec = decoder(Hstar, grid)
            fr = out_dec.frames[0].cpu() if out_dec.frames is not None else None
            meas = measured_bounce(fr, incoming_hint=bounce_hint) if fr is not None else {
                "speed_ratio": float("nan"), "rebound_peak_y": float("nan"),
            }
            row = {
                "restitution": eb,  # target e (decoded uses speed_ratio as proxy)
                "speed_ratio": meas["speed_ratio"],
                "rebound_peak_y": meas["rebound_peak_y"],
                "post_bounce_speed": meas.get("post_bounce_speed", float("nan")),
            }
            decoded[m].append(row)
            sc_row[m] = row
            if want and fr is not None:
                try:
                    viz.steering_filmstrip({0: fr}, out / f"scene{s:05d}_{m}.png")
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] filmstrip {m} scene{s:05d}: {e}")
        per_scene[f"scene{s:05d}"] = sc_row
        print(f"  scene{s:05d}: e_b={eb:.3f} target_ratio={tgt['speed_ratio']:.3f} "
              f"full_ratio={decoded['full_delta'][-1]['speed_ratio']}")

    results = {m: _agg(decoded[m], targets) for m in methods}
    summary = {
        "test_dir": args.test_dir, "checkpoint": args.checkpoint, "layers": layers,
        "quantity": "restitution", "n_scenes": len(scene_ids), "ks": ks,
        "target": "bounce speed_ratio + rebound_peak_y vs GT clip-b",
        "results": results, "per_scene": per_scene,
    }
    (out / "steer_restitution_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[steer_rest] aggregate:")
    for m in methods:
        r = results[m]
        print(f"  {m:18s} rho_sr={r['rho_speed_ratio']!s:>7} ratio_mae={r['restitution_ratio_mae']!s:>7} "
              f"peak_mae={r['rebound_peak_mae']!s:>7} n={r['n']}")
    print(f"[steer_rest] -> {out}/steer_restitution_summary.json")


if __name__ == "__main__":
    main()
