#!/bin/bash
#SBATCH --job-name=abl_train
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/abl_train_%j.out
#SBATCH --error=logs/abl_train_%j.err

# Train frame_position decoder on v2d latents (resume-safe).
#
#   ENCODER=vjepa2_huge MAX_STEPS=8000 sbatch scripts/ablation/train_decoder.sh

set -euo pipefail
source "$(dirname "$0")/_encoder_env.sh"

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$BASE_DIR"
mkdir -p logs

RESUME=""
CKDIR="$RUN_ROOT/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST" ]; then
        RESUME="train.resume=$LATEST"
        echo "[abl_train] resuming from $LATEST"
    fi
fi

echo "[abl_train] ENCODER=$ENCODER OUT=$RUN_ROOT MAX_STEPS=$MAX_STEPS"
accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_decoder.py \
    --config "$TRAIN_CONFIG" \
    --latent_dir "$LATENT_TRAIN" \
    --output_dir "$RUN_ROOT" \
    optim.max_steps="$MAX_STEPS" \
    train.ckpt_every=500 \
    train.log_every=50 \
    $RESUME
