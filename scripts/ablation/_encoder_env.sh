#!/bin/bash
# Shared paths and encoder-specific knobs for ablation pipelines.
# Source from other scripts:  source "$(dirname "$0")/_encoder_env.sh"
#
# Usage:
#   export BASE_DIR=$HOME/vjepa-latent-physics
#   export ENCODER=vjepa2_huge    # or vjepa2_large (default)
#   source scripts/ablation/_encoder_env.sh

ENCODER=${ENCODER:-vjepa2_large}
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}

case "$ENCODER" in
  vjepa2_large)
    ENCODER_LAYERS=${ENCODER_LAYERS:-6,12,18,23}
    TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train/moving_ball_scene_decoder.yaml}
    EXTRACT_BATCH=${EXTRACT_BATCH:-8}
  ;;
  vjepa2_huge)
    ENCODER_LAYERS=${ENCODER_LAYERS:-8,16,24,31}
    TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train/moving_ball_scene_decoder_v2h.yaml}
    EXTRACT_BATCH=${EXTRACT_BATCH:-4}
  ;;
  *)
    echo "unknown ENCODER=$ENCODER (vjepa2_large|vjepa2_huge)" >&2
    return 1 2>/dev/null || exit 1
  ;;
esac

LATENT_ROOT="$BASE_DIR/outputs/latents/moving_ball_scene_v2d"
LATENT_TRAIN="$LATENT_ROOT/train/$ENCODER"
LATENT_TEST="$LATENT_ROOT/test/$ENCODER"

ANALYSIS_ROOT="$BASE_DIR/outputs/analysis/moving_ball_v2d_${ENCODER}"
SUBSPACE_DIR="$ANALYSIS_ROOT/subspace"
STEER_DIR=${STEER_DIR:-$ANALYSIS_ROOT/steer}

RUN_ROOT=${RUN_ROOT:-$BASE_DIR/outputs/runs/moving_ball_scene_v2d_decoder_fp_${ENCODER}}
CKPT="$RUN_ROOT/checkpoints/last.pt"

# Decoder train RAM: 1024-d ViT-H fit at 320G; 1280-d needs more (pass via sbatch --mem=$TRAIN_MEM).
case "${TRAIN_CONFIG}" in
  *d1280*) TRAIN_MEM=${TRAIN_MEM:-384G} ;;
  *)       TRAIN_MEM=${TRAIN_MEM:-320G} ;;
esac

MAX_STEPS=${MAX_STEPS:-8000}
NUM_SCENES=${NUM_SCENES:-40}
