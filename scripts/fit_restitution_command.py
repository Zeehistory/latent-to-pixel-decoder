#!/usr/bin/env python
"""Fit command-only restitution operators (W_U, B_rich) in the global PCA subspace.

    python scripts/fit_restitution_command.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_restitution/subspace
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.analysis import restitution_ops as ro
from src.encoders.feature_extractor import LatentDataset

P = ro.COMMAND_FEATURE_DIM


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=8)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    U = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64)[:KU] for L in layers}

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = ro.group_scenes(tr), ro.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[rest_cmd] train {len(tr_scenes)} scenes; P={P} KU={KU}", flush=True)

    rich = {L: None for L in layers}
    wu = {L: ro.LinearLS(P, KU, args.ridge) for L in layers}
    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s])
        a = ranks[0]
        sa = tr[tr_scenes[s][a]]
        ea = ro.clip_restitution(sa)
        va = ro.clip_incoming_velocity(sa)
        Ha = {L: ro.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            eb = ro.clip_restitution(sb)
            vb = ro.clip_incoming_velocity(sb)
            phi = ro.command_features(ea, eb, va, vb).reshape(1, P)
            for L in layers:
                dH = ro.layer_flat(sb["layers"][L]) - Ha[L]
                if rich[L] is None:
                    rich[L] = ro.LinearLS(P, dH.size, args.ridge)
                rich[L].add(phi, dH.reshape(1, -1))
                wu[L].add(phi, (U[L] @ dH).reshape(1, KU))
        del Ha
        gc.collect()
        n += 1

    B_rich = {L: rich[L].solve().astype(np.float32) for L in layers}
    W_U = {L: wu[L].solve().astype(np.float32) for L in layers}
    del rich, wu
    gc.collect()

    gate = {L: {"rich_cos": [], "cmdU_cos": [], "coord_cos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s])
        a = ranks[0]
        sa = te[te_scenes[s][a]]
        ea = ro.clip_restitution(sa)
        va = ro.clip_incoming_velocity(sa)
        Ha = {L: ro.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]
            eb = ro.clip_restitution(sb)
            vb = ro.clip_incoming_velocity(sb)
            phi = ro.command_features(ea, eb, va, vb)
            for L in layers:
                dH = ro.layer_flat(sb["layers"][L]) - Ha[L]
                ctrue = U[L] @ dH
                cpred = phi @ W_U[L].astype(np.float64)
                gate[L]["coord_cos"].append(ro.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(ro.cosine(cpred @ U[L], dH))
                gate[L]["rich_cos"].append(ro.cosine(phi @ B_rich[L].astype(np.float64), dH))
        del Ha
        gc.collect()

    wu_tag = "" if KU == 8 else f"_ku{KU}"
    summary = {"layers": layers, "P": P, "KU": KU, "quantity": "restitution",
               "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes), "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu{wu_tag}_L{L}.npy", W_U[L])
        np.save(art / f"cmd_Brich_L{L}.npy", B_rich[L])
        print(f"[rest_cmd] L{L}: cmd_U reconstr cos={row['cmdU_cos']:.3f}", flush=True)
    (art / "cmd_operator_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[rest_cmd] saved -> {art}")


if __name__ == "__main__":
    main()
