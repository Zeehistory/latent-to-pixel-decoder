#!/usr/bin/env python
"""Print a side-by-side comparison of ViT-L vs ViT-H ablation steer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

HEADLINE = [
    "full_delta",
    "subspace_U8",
    "ridge_global",
    "canon_ridge",
    "ridge_projU8",
    "cmd_U8_s2",
    "random8",
]


def load_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("results", d)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default=".", help="repo root")
    p.add_argument("--metric", default="angle_err_deg")
    args = p.parse_args()
    base = Path(args.base_dir)
    metric = args.metric

    encoders = {
        "vjepa2_large": base / "outputs/analysis/moving_ball_v2d_vjepa2_large/steer/steer2d_summary.json",
        "vjepa2_huge": base / "outputs/analysis/moving_ball_v2d_vjepa2_huge/steer/steer2d_summary.json",
    }

    rows = {}
    for enc, path in encoders.items():
        if not path.exists():
            print(f"[compare] missing {path} — run ENCODER={enc} pipeline first")
            continue
        rows[enc] = load_summary(path)

    if not rows:
        raise SystemExit("no summaries found")

    print(f"\n{'method':<20} " + " ".join(f"{e:>14}" for e in rows))
    print("-" * (20 + 15 * len(rows)))
    for m in HEADLINE:
        vals = []
        for enc in rows:
            r = rows[enc].get(m, {})
            v = r.get(metric, float("nan"))
            vals.append(f"{v:>14.2f}" if isinstance(v, (int, float)) else f"{'—':>14}")
        print(f"{m:<20} " + " ".join(vals))
    print()


if __name__ == "__main__":
    main()
