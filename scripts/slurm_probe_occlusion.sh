#!/bin/bash
#SBATCH --job-name=vjepa_occ_probe
#SBATCH --partition=gpu_rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=logs/occ_probe_%j.out
#SBATCH --error=logs/occ_probe_%j.err

# Step 2 (velocity-first): occlusion probe — is velocity decodable during hidden frames?
# Trains on visible-frame tokens, evaluates on hidden-frame tokens per layer.
# Must run after slurm_extract_ball.sh for moving_ball_occlusion.
#
#   sbatch scripts/slurm_probe_occlusion.sh
BASE_DIR=${BASE_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa"}
LATENT_DIR=${LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_occlusion/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/analysis/moving_ball_occlusion/occlusion_probe"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/probe_occlusion.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all

echo "[occ_probe] done -> $OUTPUT_DIR"
