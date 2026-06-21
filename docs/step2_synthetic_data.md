# Step 2 — synthetic physics data (MuJoCo + Genesis)

Step 2 generates synthetic clips with **exact** physical labels (velocity / acceleration / gravity /
position) so we can probe *which latent subspace encodes which quantity*. The data is generated
**on the cluster, on the fly** during the latent-extraction job — nothing is generated locally and there
is no large dataset to download; you only install the engine packages once.

## Engines

| Dataset (`--dataset`) | Engine | Physics | Notes |
|---|---|---|---|
| `mujoco_solid` | MuJoCo | rigid body: free_fall, projectile, bounce | exact state from the simulator; recommended for solid mechanics |
| `genesis_fluid` | Genesis | fluids (SPH/MPM): pour, dam_break, drop | GPU-only; aggregate swarm state |
| `synthetic_solid` / `synthetic_fluid` | built-in 2D | lightweight fallback | no engine install; cartoonish visuals |

## One-time install (on a compute node, inside the conda env)

```bash
module load miniconda && conda activate vjepa-physics-decoder
cd /path/to/latent-to-pixel-decoder

pip install -e .[sim_mujoco]     # MuJoCo (rigid body)
pip install -e .[sim_genesis]    # Genesis (fluids) — run on a CUDA node
```

Headless rendering: the SLURM script exports `MUJOCO_GL=egl` automatically. If you run MuJoCo
interactively on a node, `export MUJOCO_GL=egl` first.

## Run (on the cluster)

```bash
# 1. extract VJEPA2-L latents for each synthetic dataset (GPU job; generates clips on the fly)
DATASET=mujoco_solid  sbatch scripts/slurm_extract_synthetic.sh
DATASET=genesis_fluid sbatch scripts/slurm_extract_synthetic.sh

# 2. quantity probes (which layer/subspace encodes velocity / acceleration / gravity)
LATENT_DIR=.../latents/mujoco_solid/vjepa2_large \
OUTPUT_DIR=.../analysis/mujoco_solid/quantity_probe sbatch scripts/slurm_train_probe.sh
```

The probe reports per-layer linear + MLP R² for `pos / vel / acc / radius / gravity / collision`, with
shuffled-latent and randomized-label controls (both must collapse to ~0). Because the labels are exact,
genuinely encoded quantities should be clearly decodable while controls stay near zero.

## Notes / caveats

- **Genesis API drift.** The Genesis stepping/render/particle calls are written against the documented
  API and guarded with fallbacks; lines marked `# GENESIS API` in `src/data/genesis_fluid.py` may need
  small version-specific tweaks on your install.
- **Domain gap.** MuJoCo/Genesis renders are shaded 3D (closer to natural video than the 2D fallback),
  which should help Step-3 transfer to real Physics-IQ, but they are still not photorealistic.
- **Exact state.** MuJoCo state is read straight from the simulator (`qpos/qvel/qacc`, `opt.gravity`);
  Genesis state is the particle aggregate (centroid pos/vel/acc, spread, gravity).
