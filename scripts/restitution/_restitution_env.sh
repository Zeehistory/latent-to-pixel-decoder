#!/bin/bash
# Shared paths for the restitution (bounce) pipeline.
#   export BASE_DIR=$HOME/vjepa-latent-physics
#   export ENCODER=vjepa2_large   # or vjepa2_huge
#   source scripts/restitution/_restitution_env.sh

ENCODER=${ENCODER:-vjepa2_large}
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}

case "$ENCODER" in
  vjepa2_large)
    ENCODER_LAYERS=${ENCODER_LAYERS:-6,12,18,23}
    EXTRACT_BATCH=${EXTRACT_BATCH:-8}
  ;;
  vjepa2_huge)
    ENCODER_LAYERS=${ENCODER_LAYERS:-8,16,24,31}
    EXTRACT_BATCH=${EXTRACT_BATCH:-4}
    TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train/moving_ball_scene_restitution_decoder_v2h.yaml}
  ;;
  *)
    echo "unknown ENCODER=$ENCODER" >&2
    return 1 2>/dev/null || exit 1
  ;;
esac

TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train/moving_ball_scene_restitution_decoder.yaml}

LATENT_ROOT="$BASE_DIR/outputs/latents/moving_ball_scene_restitution"
LATENT_TRAIN="$LATENT_ROOT/train/$ENCODER"
LATENT_TEST="$LATENT_ROOT/test/$ENCODER"

ANALYSIS_ROOT="$BASE_DIR/outputs/analysis/moving_ball_restitution_${ENCODER}"
SUBSPACE_DIR="$ANALYSIS_ROOT/subspace"
STEER_DIR=${STEER_DIR:-$ANALYSIS_ROOT/steer}

RUN_ROOT=${RUN_ROOT:-$BASE_DIR/outputs/runs/moving_ball_scene_restitution_decoder_fp_${ENCODER}}
CKPT="$RUN_ROOT/checkpoints/last.pt"
TRAIN_MEM=${TRAIN_MEM:-320G}
MAX_STEPS=${MAX_STEPS:-8000}
NUM_SCENES=${NUM_SCENES:-40}
