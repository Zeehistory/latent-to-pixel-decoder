#!/usr/bin/env python
"""Calibrate cmd_U8 gain for restitution steering (minimize speed-ratio MAE on val split).

    python scripts/calibrate_restitution_gain.py \
        --summary outputs/analysis/moving_ball_restitution/steer/steer_restitution_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def ratio_mae(ps, scenes, method, targets):
    errs = []
    for k in scenes:
        d = ps[k][method]["speed_ratio"]
        t = targets[k]["speed_ratio"]
        if np.isfinite(d) and np.isfinite(t):
            errs.append(abs(d - t))
    return float(np.mean(errs)) if errs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--val_frac", type=float, default=0.5)
    p.add_argument("--out", default="")
    args = p.parse_args()

    d = json.loads(Path(args.summary).read_text())
    ps = d["per_scene"]
    keys = sorted(ps)
    gains = sorted(
        (m for m in d["results"] if re.fullmatch(r"cmd_U8_s[0-9.]+", m)),
        key=lambda m: float(m.split("_s")[1]),
    )
    if not gains:
        raise SystemExit("no cmd_U8_s{gain} methods in summary")

    nval = max(1, int(round(len(keys) * args.val_frac)))
    val, test = keys[:nval], keys[nval:]
    val_err = {g: ratio_mae(ps, val, g, {k: ps[k]["target"] for k in val}) for g in gains}
    best = min(val_err, key=val_err.get)
    best_gain = float(best.split("_s")[1])

    refs = {m: round(ratio_mae(ps, test, m, {k: ps[k]["target"] for k in test}), 4)
            for m in ("full_delta", "ridge_global", "subspace_U8") if m in d["results"]}
    res = {
        "summary": args.summary,
        "metric": "speed_ratio_mae",
        "n_val": len(val), "n_test": len(test),
        "val_ratio_mae_by_gain": {g: round(v, 4) for g, v in val_err.items()},
        "selected_gain": best_gain,
        "val_ratio_mae_at_selected": round(val_err[best], 4),
        "HELDOUT_test_ratio_mae_at_selected": round(
            ratio_mae(ps, test, best, {k: ps[k]["target"] for k in test}), 4),
        "test_references": refs,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
