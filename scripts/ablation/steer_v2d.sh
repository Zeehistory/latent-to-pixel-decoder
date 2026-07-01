#!/bin/bash
#SBATCH --job-name=abl_steer
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/abl_steer_%j.out
#SBATCH --error=logs/abl_steer_%j.err

# Pixel-level steering proof: full_delta, subspace_U, ridge, cmd_U8, etc.
#
#   PART=$(scripts/_pick_partition.sh 0 gpu_devel scavenge_gpu)
#   ENCODER=vjepa2_huge sbatch --partition="$PART" scripts/ablation/steer_v2d.sh

set -euo pipefail
source "$(dirname "$0")/_encoder_env.sh"

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$BASE_DIR"
mkdir -p logs

if [ ! -f "$CKPT" ]; then
    echo "[abl_steer] missing checkpoint: $CKPT" >&2
    exit 1
fi

python -u scripts/steer_velocity2d.py \
    --config "$TRAIN_CONFIG" \
    --test_dir "$LATENT_TEST" \
    --artifacts_dir "$SUBSPACE_DIR" \
    --checkpoint "$CKPT" \
    --output_dir "$STEER_DIR" \
    --ks 2,4,8,16 \
    --num_scenes "$NUM_SCENES" \
    --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --device cuda

echo "[abl_steer] -> $STEER_DIR/steer2d_summary.json"

python scripts/calibrate_cmd_gain.py \
    --summary "$STEER_DIR/steer2d_summary.json" \
    --val_frac 0.5 \
    --out "$STEER_DIR/cmd_gain_calibration.json" || true
