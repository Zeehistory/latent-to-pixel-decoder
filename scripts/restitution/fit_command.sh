#!/bin/bash
#SBATCH --job-name=rest_cmd
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rest_cmd_%j.out
#SBATCH --error=logs/rest_cmd_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/restitution/_job_init.sh"

python -u scripts/fit_restitution_command.py \
  --train_dir "$LATENT_TRAIN" \
  --test_dir "$LATENT_TEST" \
  --layers "$ENCODER_LAYERS" \
  --artifacts_dir "$SUBSPACE_DIR"

echo "[rest_cmd] -> $SUBSPACE_DIR/cmd_operator_meta.json"
