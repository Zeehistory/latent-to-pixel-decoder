#!/bin/bash
#SBATCH --job-name=rest_subspace
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rest_subspace_%j.out
#SBATCH --error=logs/rest_subspace_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/restitution/_job_init.sh"

python -u scripts/restitution_subspace.py \
  --train_dir "$LATENT_TRAIN" \
  --test_dir "$LATENT_TEST" \
  --layers "$ENCODER_LAYERS" \
  --output_dir "$SUBSPACE_DIR" \
  --save_k 8

echo "[rest_subspace] -> $SUBSPACE_DIR/subspace_summary.json"
