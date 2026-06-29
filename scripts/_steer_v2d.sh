#!/bin/bash
#SBATCH --job-name=v2d_steer
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2d_steer_%j.out
#SBATCH --error=logs/v2d_steer_%j.err
# Phases 3-5 pixel proof: decode + track the 2D-velocity steer for full_delta / transport{,_oracle,
# _shuffle} / subspace_U[k] / random[k] / ridge_global / canon_ridge on held-out test scenes.
# The decode loads only the small TEST cache (~80 clips), so gpu_devel (instant, 6h, 60G cap) fits and
# beats the backlogged scavenge_gpu. ALWAYS submit via the queue-aware picker (user directive):
#   PART=$(scripts/_pick_partition.sh 0 gpu_devel scavenge_gpu)
#   sbatch --partition="$PART" scripts/_steer_v2d.sh [CKPT]
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=/home/zss8/project_pi_jks79/zss8/vjepa
CK="${1:-last}"
OUTTAG="${2:-$CK}"   # output dir suffix (lets parallel decodes not clobber each other)
CKPT=$BASE/outputs/runs/moving_ball_scene_v2d_decoder_fp/checkpoints/${CK}.pt
python -u scripts/steer_velocity2d.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_v2d/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/steer_${OUTTAG} \
    --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-40} --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --device cuda
echo "[v2d_steer] exit=$? ckpt=$CKPT out=steer_${OUTTAG}"
