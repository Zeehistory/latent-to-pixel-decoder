#!/usr/bin/env python
"""Phase 2 + operator fitting for restitution: PCA of Delta H and ridge artifacts.

Latent-only (no decoder). Fits on TRAIN ``scene_restitution`` cache; saves artifacts for
``scripts/steer_restitution.py``.

    python scripts/restitution_subspace.py \
        --train_dir .../moving_ball_scene_restitution/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_restitution/test/vjepa2_large \
        --layers 6,12,18,23 --output_dir outputs/analysis/moving_ball_restitution/subspace
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--save_k", type=int, default=8)
    p.add_argument("--max_global_pairs", type=int, default=800)
    p.add_argument("--max_angle_scenes", type=int, default=60)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = ro.group_scenes(tr), ro.group_scenes(te)
    print(f"[rest_subspace] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}")

    summary = {
        "train_dir": args.train_dir, "test_dir": args.test_dir, "layers": layers,
        "quantity": "restitution", "n_train_scenes": len(tr_scenes),
        "n_test_scenes": len(te_scenes), "per_layer": {},
    }

    dim_by_layer: dict[int, int] = {}
    ridge: dict[int, ro.ScalarRidgeOperator] = {}
    ridge_canon: dict[int, ro.ScalarRidgeOperator] = {}
    within_curves = {L: [] for L in layers}
    within_pr = {L: [] for L in layers}
    global_buf = {L: [] for L in layers}
    global_de: list[float] = []
    scene_bases = {L: [] for L in layers}
    rng = np.random.default_rng(0)

    n_scene = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s])
        a = ranks[0]
        sa = tr[tr_scenes[s][a]]
        ea = ro.clip_restitution(sa)
        pos = ro.clip_start_pos(sa)
        grid = tuple(int(x) for x in sa["grid"])
        Ha = {L: ro.layer_flat(sa["layers"][L]) for L in layers}
        sh = ro.canon_shift(pos, grid)
        per_layer_deltas = {L: [] for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            eb = ro.clip_restitution(sb)
            de = eb - ea
            for L in layers:
                dH = ro.layer_flat(sb["layers"][L]) - Ha[L]
                dim_by_layer[L] = dH.size
                ridge.setdefault(L, ro.ScalarRidgeOperator(dH.size, args.ridge)).add(de, dH)
                ridge_canon.setdefault(L, ro.ScalarRidgeOperator(dH.size, args.ridge)).add(
                    de, ro.roll_layer(dH, grid, sh))
                per_layer_deltas[L].append(dH.astype(np.float32))
                if len(global_buf[L]) < args.max_global_pairs:
                    global_buf[L].append(dH.astype(np.float32))
            if len(global_de) < args.max_global_pairs:
                global_de.append(de)
        for L in layers:
            M = np.stack(per_layer_deltas[L], 0).astype(np.float64)
            basis, ev = ro.pca_gram(M)
            within_curves[L].append(ro.explained_curve(ev, 6))
            within_pr[L].append(ro.participation_ratio(ev))
            if n_scene < args.max_angle_scenes and basis.shape[0] >= 2:
                scene_bases[L].append(basis[:2])
        n_scene += 1
        if n_scene % 100 == 0:
            print(f"[rest_subspace]   processed {n_scene}/{len(tr_scenes)} train scenes")

    for L in layers:
        D = dim_by_layer[L]
        B = ridge[L].solve()
        B_canon = ridge_canon[L].solve()
        np.save(out / f"ridge_B_L{L}.npy", B.astype(np.float32))
        np.save(out / f"ridge_canon_B_L{L}.npy", B_canon.astype(np.float32))

        Xg = np.stack(global_buf[L], 0).astype(np.float32)
        gbasis, gev = ro.pca_gram(Xg, k=args.save_k)
        np.save(out / f"global_basis_L{L}.npy", gbasis.astype(np.float32))
        del Xg, global_buf[L]
        gc.collect()

        wc = np.asarray(within_curves[L]).mean(0).tolist()
        wpr = float(np.mean(within_pr[L]))
        angs = []
        sb = scene_bases[L]
        for i in range(len(sb)):
            for j in range(i + 1, len(sb)):
                sv = np.linalg.svd(sb[i] @ sb[j].T, compute_uv=False)
                angs.append(float(np.degrees(np.arccos(np.clip(sv, -1, 1)).mean())))
        mean_angle = float(np.mean(angs)) if angs else float("nan")

        summary["per_layer"][str(L)] = {
            "dim": int(D),
            "within_scene_explained_top1to6": [round(x, 4) for x in wc],
            "within_scene_participation_ratio": round(wpr, 3),
            "global_explained_top1to8": [round(x, 4) for x in ro.explained_curve(gev, 8)],
            "global_participation_ratio": round(ro.participation_ratio(gev), 3),
            "mean_principal_angle_deg_between_scene_subspaces": round(mean_angle, 2),
        }
        print(f"[rest_subspace] L{L}: within PR={wpr:.2f}, global PR="
              f"{summary['per_layer'][str(L)]['global_participation_ratio']:.1f}")

    B_mem = {L: ridge[L].B for L in layers}
    top2 = {L: np.load(out / f"global_basis_L{L}.npy")[:2].astype(np.float64) for L in layers}
    randb = {L: ro.random_basis(dim_by_layer[L], 2, rng) for L in layers}
    acc = {L: {"rcos": [], "rrel": [], "ucos": [], "xcos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s])
        a = ranks[0]
        sa = te[te_scenes[s][a]]
        ea = ro.clip_restitution(sa)
        Ha = {L: ro.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]
            de = ro.clip_restitution(sb) - ea
            for L in layers:
                dH = ro.layer_flat(sb["layers"][L]) - Ha[L]
                pred = B_mem[L] * de
                acc[L]["rcos"].append(ro.cosine(pred, dH))
                acc[L]["rrel"].append(ro.rel_error(pred, dH))
                acc[L]["ucos"].append(ro.cosine(ro.project(dH, top2[L]), dH))
                acc[L]["xcos"].append(ro.cosine(ro.project(dH, randb[L]), dH))
        del Ha
        gc.collect()

    gen = {}
    for L in layers:
        gen[str(L)] = {
            "ridge_predict_dH_cosine": round(float(np.mean(acc[L]["rcos"])), 4),
            "ridge_predict_dH_rel_error": round(float(np.mean(acc[L]["rrel"])), 4),
            "U2_projection_retains_cosine": round(float(np.mean(acc[L]["ucos"])), 4),
            "random2_projection_retains_cosine": round(float(np.mean(acc[L]["xcos"])), 4),
        }
    summary["test_latent_generalization"] = gen
    summary["artifacts"] = {
        "ridge": "ridge_B_L*.npy", "ridge_canon": "ridge_canon_B_L*.npy",
        "global_basis": "global_basis_L*.npy", "save_k": args.save_k,
    }
    (out / "subspace_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rest_subspace] wrote {out}/subspace_summary.json + artifacts")


if __name__ == "__main__":
    main()
