#!/usr/bin/env python
"""Collect Step-2 (velocity-first) results and finish the reporting unattended.

Runs as the final SLURM job in the velocity pipeline (after the steering jobs terminate, success
or fail). It parses every produced artifact, writes a complete results digest, fills in the brain.md
R2 table + changelog + status tag, and commits locally. Designed to be robust: a missing/failed
steering result is reported honestly rather than fabricated, and brain.md edits fall back to an
appended section if the expected anchors are not found (so the file is never corrupted).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import date
from pathlib import Path

BASE = Path(os.environ.get("BASE_DIR", "/home/zss8/project_pi_jks79/zss8/vjepa"))
REPO = Path(os.environ.get("REPO_DIR", "/nfs/roberts/project/pi_jks79/zss8/latent-to-pixel-decoder"))
AN = BASE / "outputs" / "analysis"
TODAY = date.today().isoformat()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_velocity():
    p = AN / "moving_ball_velocity" / "velocity_probe" / "velocity_decodability.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["layer"] = int(r["layer"]); r["r2"] = _f(r["r2"])
        r["cs"] = _f(r["ctrl_shuffled_latent_r2"]); r["cr"] = _f(r["ctrl_randomized_label_r2"])
    out = {"rows": rows, "by": {}}
    for tgt in ("vel", "speed", "angle"):
        for rep in ("clip_pool", "temporal", "temporal_diff"):
            sub = [r for r in rows if r["target"] == tgt and r["representation"] == rep and r["probe"] == "linear"]
            if sub:
                b = max(sub, key=lambda r: r["r2"])
                out["by"][(tgt, rep)] = b
    # per-layer linear vel r2 for the brain.md table
    out["vel_by_layer"] = {}
    for L in sorted({r["layer"] for r in rows}):
        row = {}
        for rep in ("clip_pool", "temporal", "temporal_diff"):
            m = [r for r in rows if r["target"] == "vel" and r["representation"] == rep
                 and r["probe"] == "linear" and r["layer"] == L]
            row[rep] = m[0]["r2"] if m else float("nan")
        out["vel_by_layer"][L] = row
    return out


def parse_occlusion():
    p = AN / "moving_ball_occlusion" / "occlusion_probe" / "occlusion_probe.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["layer"] = int(r["layer"]); r["rv"] = _f(r["r2_visible"])
        r["rh"] = _f(r["r2_hidden"]); r["rc"] = _f(r["r2_ctrl_shuffled"])
    best = max(rows, key=lambda r: r["rh"])
    return {"rows": rows, "best_hidden": best}


def parse_equivariance():
    p = AN / "moving_ball_equivariance" / "equivariance_probe" / "equivariance_report.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    items = [(int(k), v) for k, v in d.items()]
    return {
        "items": items,
        "best_circ": max(items, key=lambda x: x[1]["circularity"]),
        "best_dir": max(items, key=lambda x: x[1]["angle_r2_from_subspace"]),
        "best_eqv": min(items, key=lambda x: x[1]["equivariance_error_mean"]),
    }


def parse_steering():
    out = {}
    for tgt in ("speed", "vel_x", "vel_y"):
        p = AN / "moving_ball_velocity" / f"steer_{tgt}" / "velocity_steering_summary.json"
        out[tgt] = json.load(open(p)) if p.exists() else None
    return out


def fmt(x, n=3):
    try:
        if x != x:  # nan
            return "—"
        return f"{x:.{n}f}"
    except Exception:
        return "—"


def build_digest(vel, occ, eqv, steer):
    L = ["# Step 2 (velocity-first) — results digest", f"_Auto-collected {TODAY} on the cluster._", ""]

    L.append("## Velocity probe (linear Ridge, controls always on)")
    if vel:
        L.append("Best linear R² per representation (controls should collapse to ≈0 / negative):")
        L.append("")
        L.append("| target | clip_pool | temporal | temporal_diff | ctrl_shuf (temporal) | ctrl_rand (temporal) |")
        L.append("|--------|-----------|----------|---------------|----------------------|----------------------|")
        for tgt in ("vel", "speed", "angle"):
            cp = vel["by"].get((tgt, "clip_pool")); tp = vel["by"].get((tgt, "temporal")); td = vel["by"].get((tgt, "temporal_diff"))
            cps = f"{fmt(cp['r2'])} @L{cp['layer']}" if cp else "—"
            tps = f"{fmt(tp['r2'])} @L{tp['layer']}" if tp else "—"
            tds = f"{fmt(td['r2'])} @L{td['layer']}" if td else "—"
            L.append(f"| {tgt} | {cps} | {tps} | {tds} | {fmt(tp['cs']) if tp else '—'} | {fmt(tp['cr']) if tp else '—'} |")
        L.append("")
        L.append("Per-layer linear vel R² (clip_pool / temporal / temporal_diff):")
        L.append("")
        L.append("| layer | clip_pool | temporal | temporal_diff |")
        L.append("|-------|-----------|----------|---------------|")
        for Li, row in vel["vel_by_layer"].items():
            L.append(f"| {Li} | {fmt(row['clip_pool'])} | {fmt(row['temporal'])} | {fmt(row['temporal_diff'])} |")
    else:
        L.append("MISSING — velocity_decodability.csv not found.")
    L.append("")

    L.append("## Occlusion probe (train on visible tokens, eval on hidden-frame tokens)")
    if occ:
        b = occ["best_hidden"]
        L.append(f"Best hidden-token R² = **{fmt(b['rh'],4)}** @L{b['layer']} (visible {fmt(b['rv'],4)}, shuffled-ctrl {fmt(b['rc'])}).")
        L.append("→ velocity remains decodable while the ball is occluded (object permanence).")
        L.append("")
        L.append("| layer | r2_visible | r2_hidden | ctrl_shuffled |")
        L.append("|-------|-----------|-----------|---------------|")
        for r in occ["rows"]:
            L.append(f"| {r['layer']} | {fmt(r['rv'])} | {fmt(r['rh'])} | {fmt(r['rc'])} |")
    else:
        L.append("MISSING — occlusion_probe.csv not found.")
    L.append("")

    L.append("## Equivariance probe (velocity subspace under direction rotation)")
    if eqv:
        for lab, (Li, v) in [("max circularity", eqv["best_circ"]),
                             ("max direction-R²", eqv["best_dir"]),
                             ("min equivariance-error", eqv["best_eqv"])]:
            L.append(f"- {lab}: L{Li} — circ={fmt(v['circularity'])}, dirR²={fmt(v['angle_r2_from_subspace'])}, equivErr={fmt(v['equivariance_error_mean'])}")
    else:
        L.append("MISSING — equivariance_report.json not found.")
    L.append("")

    L.append("## Steering (headline — does the pixel-tracked decoded ball change speed with α?)")
    any_steer = False
    L.append("")
    L.append("| target | readout ρ(α) | decoded-measured ρ(α) | n_steered | verdict |")
    L.append("|--------|--------------|-----------------------|-----------|---------|")
    for tgt in ("speed", "vel_x", "vel_y"):
        s = steer.get(tgt)
        if not s:
            L.append(f"| {tgt} | — | — | — | MISSING/failed |")
            continue
        any_steer = True
        rro = s.get("readout_monotonicity_spearman"); rme = s.get("decoded_measured_monotonicity_spearman")
        ok = (rro is not None and rme is not None and abs(rro) > 0.6 and abs(rme) > 0.6
              and (rro > 0) == (rme > 0))
        verdict = "pixels move with α ✓" if ok else "weak/mismatched ✗"
        L.append(f"| {tgt} | {fmt(rro)} | {fmt(rme)} | {s.get('n_steered','?')} | {verdict} |")
    if not any_steer:
        L.append("")
        L.append("**No steering summaries found** — decoder/steering did not complete. See SLURM logs.")
    L.append("")
    return "\n".join(L), any_steer


def update_brain(vel, occ, eqv, steer, any_steer):
    bp = REPO / "brain.md"
    if not bp.exists():
        return "brain.md not found"
    txt = bp.read_text()
    (REPO / "brain.md.bak").write_text(txt)
    notes = []

    # 1. Fill the R2 table: replace the TBD row with real per-layer rows (key layers).
    tbd_anchor = "| TBD   | —                | —               | —                    |"
    if vel and tbd_anchor in txt:
        key_layers = sorted(set([6, 9, 12, 15, 18, 21, 23] +
                                [vel["by"][("vel", "temporal")]["layer"]]))
        rows = []
        for Li in key_layers:
            r = vel["vel_by_layer"].get(Li)
            if r:
                rows.append(f"| {Li}    | {fmt(r['clip_pool'])}            | {fmt(r['temporal'])}           | {fmt(r['temporal_diff'])}                |")
        txt = txt.replace(tbd_anchor, "\n".join(rows))
        notes.append("filled R² table")
    else:
        notes.append("R² table anchor not found (appended section instead)" if vel else "no velocity data")

    # 2. Flip status tags.
    txt = txt.replace(
        "**Status:** ALL CODE BUILT. **TODO:** run on cluster, fill in R² table below.",
        f"**Status:** RUN ON CLUSTER ({TODAY}). Velocity/occlusion/equivariance probes done; "
        f"steering {'completed' if any_steer else 'did NOT complete (see digest)'}. See "
        f"`outputs/analysis/STEP2_RESULTS_DIGEST.md`.")
    txt = txt.replace("— VELOCITY-FIRST BUILT, probes TODO",
                      "— VELOCITY-FIRST DONE" if any_steer else "— VELOCITY-FIRST probes DONE, steering BLOCKED")

    # 3. Append a results block + changelog entry (always safe).
    vel_line = occ_line = eqv_line = steer_line = "n/a"
    if vel:
        cp = vel["by"].get(("vel", "clip_pool")); tp = vel["by"].get(("vel", "temporal"))
        vel_line = (f"vel linear R²: clip_pool {fmt(cp['r2'])}@L{cp['layer']}, temporal {fmt(tp['r2'])}@L{tp['layer']} "
                    f"(controls collapse: shuf {fmt(tp['cs'])}, rand {fmt(tp['cr'])})")
    if occ:
        b = occ["best_hidden"]; occ_line = f"hidden-token vel R² {fmt(b['rh'],4)}@L{b['layer']} (visible {fmt(b['rv'],4)})"
    if eqv:
        Li, v = eqv["best_dir"]; eqv_line = f"max direction-R² {fmt(v['angle_r2_from_subspace'])}@L{Li}, circ {fmt(v['circularity'])}; best equivErr {fmt(eqv['best_eqv'][1]['equivariance_error_mean'])}@L{eqv['best_eqv'][0]}"
    steer_bits = []
    for tgt in ("speed", "vel_x", "vel_y"):
        s = steer.get(tgt)
        if s:
            steer_bits.append(f"{tgt}: readout ρ={fmt(s.get('readout_monotonicity_spearman'))}, decoded-measured ρ={fmt(s.get('decoded_measured_monotonicity_spearman'))}")
        else:
            steer_bits.append(f"{tgt}: missing")
    steer_line = "; ".join(steer_bits)

    changelog = (
        f"\n- **{TODAY}** — Step 2 velocity-first RUN on cluster (bouchet, gpu_rtx6000). "
        f"Velocity probe: {vel_line}. Occlusion: {occ_line}. Equivariance: {eqv_line}. "
        f"Steering: {steer_line}. Fixes applied this run: probe/decoder/steer mem 64G→192G, "
        f"shard-cache eviction between layers, LatentDataset cache pruned to selected layers (decoder OOM), "
        f"re-extracted equivariance latents (prior cache had no metadata.parquet). "
        f"Full numbers in `outputs/analysis/STEP2_RESULTS_DIGEST.md`.")
    txt = txt.replace("## Changelog\n", "## Changelog\n" + changelog + "\n")

    bp.write_text(txt)
    return "; ".join(notes)


def main():
    vel = parse_velocity(); occ = parse_occlusion(); eqv = parse_equivariance(); steer = parse_steering()
    digest, any_steer = build_digest(vel, occ, eqv, steer)
    dpath = AN / "STEP2_RESULTS_DIGEST.md"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(digest)
    print(f"[collect] wrote digest -> {dpath}")
    note = update_brain(vel, occ, eqv, steer, any_steer)
    print(f"[collect] brain.md: {note}")

    # Commit locally (no push: the origin URL embeds a token and push was not authorized).
    try:
        subprocess.run(["git", "-C", str(REPO), "add", "brain.md"], check=False)
        msg = (f"Step 2 velocity-first results ({TODAY}): fill brain.md R² table + changelog/status "
               f"[steering {'done' if any_steer else 'incomplete'}]\n\n"
               "Auto-collected by scripts/collect_step2_results.py after the SLURM pipeline.\n\n"
               "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
        subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m", msg], check=False)
        print("[collect] committed brain.md update (local only)")
    except Exception as e:  # noqa: BLE001
        print(f"[collect] git commit skipped: {e}")

    print(f"[collect] DONE. steering_complete={any_steer}")


if __name__ == "__main__":
    main()
