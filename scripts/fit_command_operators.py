#!/usr/bin/env python
"""Fit COMMAND-ONLY velocity operators that synthesize the edit inside the global velocity subspace U.

The pixel proof refuted spatial placement: ``transport`` (localized) ~= no-op, while ``subspace_U8``
(project the TRUE Delta H onto the global PCA subspace U) decodes at 21 deg vs the command-only ridge's
34 deg. So velocity lives in a global low-rank subspace, and the open lever is the SYNTHESIS gap:
predict U's coordinates from the command (no H_b). Ceiling = subspace_U8 = 21 deg.

Fits two operators per layer in one streaming pass over TRAIN (reusing the U8 basis saved by
velocity_subspace.py as ``global_basis_L*.npy``):

  * ``W_U`` : command_features (13) -> U8 coordinates (8).  Reconstruct edit = c @ U8  (cmd_U8 method).
              Directly targets the 21 deg ceiling; the placement-free, decoder-aligned synthesis.
  * ``B_rich``: command_features (13) -> full flattened Delta H (D).  A richer-feature global ridge
              (vs the bare-dv ``ridge_Bt`` = 34 deg) to test whether speed/heading nonlinearity helps.

Held-out TEST latent gate: cos(pred, true Delta H) for each, plus the cos of the U8-coordinate prediction
vs the true U8 coordinates (how close to the 21 deg ceiling the synthesis gets). Decode is the decisive
test (run scripts/steer_velocity2d.py after).

    python scripts/fit_command_operators.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --artifacts_dir outputs/analysis/moving_ball_v2d/subspace --ridge 1.0
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

P = vo.COMMAND_FEATURE_DIM  # 13
KU = 8                       # U-subspace rank to synthesize into (global_basis has 8 rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True, help="dir with global_basis_L*.npy; outputs land here")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    U = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64)[:KU] for L in layers}  # (KU, D)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[cmd] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; P={P} KU={KU}",
          flush=True)

    rich = {L: None for L in layers}   # LinearLS(P, D)  command -> full dH
    wu = {L: vo.LinearLS(P, KU, args.ridge) for L in layers}  # command -> U8 coords

    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = vo.clip_velocity(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = vo.clip_velocity(sb)
            phi = vo.command_features(va, vb).reshape(1, P)
            for L in layers:
                dH = (vo.layer_flat(sb["layers"][L]) - Ha[L])           # (D,)
                if rich[L] is None:
                    rich[L] = vo.LinearLS(P, dH.size, args.ridge)
                rich[L].add(phi, dH.reshape(1, -1))
                wu[L].add(phi, (U[L] @ dH).reshape(1, KU))              # U8 coords (KU,)
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[cmd]   trained {n}/{len(tr_scenes)} scenes", flush=True)

    B_rich = {L: rich[L].solve().astype(np.float32) for L in layers}   # (P, D)
    W_U = {L: wu[L].solve().astype(np.float32) for L in layers}        # (P, KU)
    del rich, wu; gc.collect()

    # ---- held-out gate: cos(pred dH, true dH) for both; cos of U8-coord prediction vs true coords ----
    gate = {L: {"rich_cos": [], "cmdU_cos": [], "coord_cos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = vo.clip_velocity(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = vo.clip_velocity(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                ctrue = U[L] @ dH                                   # true U8 coords
                cpred = phi @ W_U[L].astype(np.float64)             # predicted U8 coords
                gate[L]["coord_cos"].append(vo.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(vo.cosine(cpred @ U[L], dH))   # reconstructed edit vs true
                gate[L]["rich_cos"].append(vo.cosine(phi @ B_rich[L].astype(np.float64), dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "KU": KU, "ridge": args.ridge,
               "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes),
               "note": "ridge_global cos~0.39 @34deg; subspace_U8 decode=21deg is the cmd_U8 ceiling",
               "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu_L{L}.npy", W_U[L])          # (P, KU)
        np.save(art / f"cmd_Brich_L{L}.npy", B_rich[L])    # (P, D)
        print(f"[cmd] L{L}: cmd_U8 reconstr cos={row['cmdU_cos']:.3f} (U8-coord cos={row['coord_cos']:.3f}) "
              f"| ridge_rich cos={row['rich_cos']:.3f}  (ridge_dv baseline 0.39)", flush=True)
    summary["artifacts"] = {"W_U": "cmd_Wu_L*.npy (P,KU) -> coords; edit = coords @ U8",
                            "B_rich": "cmd_Brich_L*.npy (P,D)",
                            "command_features": "velocity_ops.command_features(va,vb) (13,)"}
    (art / "cmd_operator_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[cmd] saved W_U + B_rich + cmd_operator_meta.json -> {art}", flush=True)


if __name__ == "__main__":
    main()
