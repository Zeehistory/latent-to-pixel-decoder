#!/bin/bash
#SBATCH --job-name=rest_steer
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rest_steer_%j.out
#SBATCH --error=logs/rest_steer_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/restitution/_job_init.sh"

if [ ! -f "$CKPT" ]; then
  echo "[rest_steer] missing checkpoint: $CKPT" >&2
  exit 1
fi

python -u scripts/steer_restitution.py \
  --config "$TRAIN_CONFIG" \
  --test_dir "$LATENT_TEST" \
  --artifacts_dir "$SUBSPACE_DIR" \
  --checkpoint "$CKPT" \
  --output_dir "$STEER_DIR" \
  --ks 2,4,8,16 \
  --num_scenes "$NUM_SCENES" \
  --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
  --device cuda

python scripts/calibrate_restitution_gain.py \
  --summary "$STEER_DIR/steer_restitution_summary.json" \
  --val_frac 0.5 \
  --out "$STEER_DIR/cmd_gain_calibration.json" || true

echo "[rest_steer] -> $STEER_DIR/steer_restitution_summary.json"
