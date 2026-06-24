#!/bin/bash
#SBATCH --job-name=vjepa_equiv_probe
#SBATCH --partition=gpu_rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=logs/equiv_probe_%j.out
#SBATCH --error=logs/equiv_probe_%j.err

# Step 2 (velocity-first): equivariance probe.
# Tests whether rotating the velocity direction rotates the latent subspace representation.
# Must run after slurm_extract_ball.sh for moving_ball_equivariance.
#
#   sbatch scripts/slurm_probe_equivariance.sh
#
# For the camera-rotation variant, re-extract with camera_rotation=true in the config and re-run.
BASE_DIR=${BASE_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa"}
LATENT_DIR=${LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_equivariance/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/analysis/moving_ball_equivariance/equivariance_probe"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/probe_equivariance.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all

echo "[equiv_probe] done -> $OUTPUT_DIR"
