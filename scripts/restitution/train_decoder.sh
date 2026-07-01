#!/bin/bash
#SBATCH --job-name=rest_train
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --output=logs/rest_train_%j.out
#SBATCH --error=logs/rest_train_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/restitution/_job_init.sh"

RESUME=""
CKDIR="$RUN_ROOT/checkpoints"
if [ -d "$CKDIR" ]; then
  LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1 || true)
  if [ -n "$LATEST" ]; then
    RESUME="train.resume=$LATEST"
    echo "[rest_train] resuming from $LATEST"
  fi
fi

accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_decoder.py \
  --config "$TRAIN_CONFIG" \
  --latent_dir "$LATENT_TRAIN" \
  --output_dir "$RUN_ROOT" \
  optim.max_steps="$MAX_STEPS" \
  train.ckpt_every=500 \
  train.log_every=50 \
  $RESUME
