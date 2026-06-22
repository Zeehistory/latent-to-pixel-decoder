#!/bin/bash
#SBATCH --job-name=vjepa_catsteer
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/catsteer_%j.out
#SBATCH --error=logs/catsteer_%j.err

# Step 3 (category steering): steer real fluid Physics-IQ clips toward "solid" along (w_solid - w_fluid)
# and decode the result. Requires:
#   * the category probe directions (slurm_probe_categories.sh -> category_directions.npz)
#   * Physics-IQ latents and a trained transformer decoder.
#
# Defaults reflect the Phase-A/B findings: use the RAW probe directions (raw separates as well as
# z-scored, so the directions live in the decoder's native latent space) at LAYER -1 = the deepest
# cached layer (23), where raw is strongest and solid<->fluid are most anti-aligned (cos=-0.70).
# ALPHAS includes negatives as a control: P(solid) should DROP for alpha<0 and RISE for alpha>0.
# Keep the sweep gentle — separability is thin (Fisher~0.12), so large alpha goes off-distribution fast.
TARGET_LATENT_DIR=${TARGET_LATENT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/latents/physics_iq/vjepa2_large"}
DIRECTIONS=${DIRECTIONS:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/analysis/physics_iq/category_probe/raw/category_directions.npz"}
CHECKPOINT=${CHECKPOINT:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
FROM_CATEGORY=${FROM_CATEGORY:-"fluid_dynamics"}
TO_CATEGORY=${TO_CATEGORY:-"solid_mechanics"}
LAYER=${LAYER:-"-1"}
ALPHAS=${ALPHAS:-"-4,-2,0,2,4,6,8"}
NUM_SAMPLES=${NUM_SAMPLES:-"4"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/analysis/steer_category/${FROM_CATEGORY}_to_${TO_CATEGORY}"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/steer_category.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --target_latent_dir "$TARGET_LATENT_DIR" \
    --directions "$DIRECTIONS" \
    --checkpoint "$CHECKPOINT" \
    --from_category "$FROM_CATEGORY" \
    --to_category "$TO_CATEGORY" \
    --layer="$LAYER" \
    --alphas="$ALPHAS" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
