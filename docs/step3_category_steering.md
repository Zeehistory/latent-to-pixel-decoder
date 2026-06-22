# Step 3 — Category steering (fluid → solid): results

Can we take a real **fluid** Physics-IQ clip, push its V-JEPA2 latents along a learned solid−fluid
direction, decode, and watch the scene become **solid-like**? Short answer: **no** — but the negative
is clean and mechanistic, and it sharpens what Step 2 should do.

## Step 1 — is the category structure real (and not a z-score artifact)?

Linear probe on clip-pooled VJEPA2-large latents, scenario-grouped CV, 3 categories
(`solid_mechanics`, `fluid_dynamics`, `optics`; thermodynamics + magnetism dropped — too few scenarios
for valid grouped CV). Majority rate 0.592.

| probe | best linear acc | macro-F1 | best layer | shuffled-label ctrl |
|-------|-----------------|----------|------------|---------------------|
| **standardized (z-score)** | 0.908 | 0.897 | 18 | 0.392 |
| **raw (no z-score)**       | 0.898 | 0.887 | 23 | 0.462 |

**Raw ≈ standardized (~1 pt gap).** Z-scoring is *not* manufacturing the separability — the category
structure is genuinely linear in the native latent space. Controls collapse far below majority, so it
is not scenario leakage. (Addresses the z-score concern directly: the normalization is a fidelity knob,
not the source of the signal.)

## Step 1 geometry — which categories are distinct?

At the deepest layer (23), per-class raw-space direction cosines:

| pair | cosine |
|------|--------|
| fluid ↔ **solid** | **−0.702** |
| optics ↔ solid    | −0.482 |
| fluid ↔ optics    | −0.287 |

**Solid and fluid are the most anti-aligned pair** — the cleanest available contrast, and exactly the
steer we attempt. But classifier-free separability is *low* (Fisher ratio ≈ 0.11–0.12, silhouette ≈
0.09): the categories are separable along a thin discriminative direction, not as well-separated blobs.
This foreshadowed that steering would be subtle / fragile.

## Step 3 — steering + decode

Per-token latent norms are large (150–240 by layer), so α is expressed as a **fraction of per-token
norm**. Findings, in the order we discovered them:

1. **Probe-weight direction, deepest layer only.** The independent readout moved monotonically
   (ρ = 1.0) but **pixels did not change** through α = 0.43·norm, then **broke into mush** at 0.85·norm
   — no coherent solid intermediate. The discriminative max-margin weight vector is not a direction the
   decoder renders.
2. **Difference-of-class-means direction, all layers.** Pixels *do* move now — but the change is a
   **global blue/cool appearance shift + block artifacts**, dominated by dataset-level appearance
   (backgrounds, palette), not object physics. The independent readout **saturates** at the smallest
   real α (≥0.25 → 1.0), so it is a step function, not a controllability curve.
3. **Fine low-α (≤0.3), 6 distinct fluids** (balloon, domino-in-juice, fill-glass-red-drink,
   glass-stays-same, juice-in-water, liquid-on-duck). In the coherent regime, **every object and liquid
   is preserved**; the only systematic effect is a mild blue tint creeping into the background. No
   liquid freezes, holds an edge, or solidifies.

## Conclusion

The solid/fluid axis is **decodable as a label** and the class-centroid direction is **renderable as
appearance**, but steering it **repaints the scene toward the target category's visual statistics rather
than imposing a physics counterfactual on a fixed object**. The crude category direction conflates
appearance with physics, and appearance is what the decoder paints.

## What this motivates

- **Step 2 — temporal-preserving latent.** Clip-pooling collapses all `2048 = 8·16·16` tokens to one
  `1024`-vector, discarding the temporal axis where dynamics (the actual physics) live. A `8×1024`
  temporal-preserving probe is the natural next representation.
- **Quantity directions from labelled (synthetic) data.** Instead of a category label, learn directions
  for *measured* physical quantities (velocity, gravity, …) where ground truth exists — the Step-2/Step-3
  `steer_decode.py` path. A direction tied to a physical quantity is far more likely to be a renderable
  physics axis than a category centroid.
- **(Optional) appearance-orthogonalized direction.** Project the solid−fluid vector to remove its
  component along the global color/appearance axis and re-test. Worth it only if a quantity-based or
  temporal direction shows object-level effects; on current evidence the category direction is
  appearance-dominated and orthogonalizing may remove the only component that renders.

## Reproduce

```bash
# Step 1 (both probes + geometry, 3 categories):
sbatch scripts/slurm_probe_categories.sh
# Step 3 (diff_means + all layers, fine alpha, distinct scenarios):
NUM_SAMPLES=6 sbatch scripts/slurm_steer_category.sh
# variants: METHOD=probe / ALL_LAYERS=0 / ALPHAS=... / LAYER=...
```
