#!/bin/bash
#SBATCH --job-name=v2d_steer
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2d_steer_%j.out
#SBATCH --error=logs/v2d_steer_%j.err
# Phases 3-5 pixel proof: decode + track the 2D-velocity steer for full_delta / subspace_U[k] /
# random[k] / ridge_global / canon_ridge on held-out test scenes. Usage: sbatch scripts/_steer_v2d.sh [CKPT]
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=/home/zss8/project_pi_jks79/zss8/vjepa
CK="${1:-last}"
CKPT=$BASE/outputs/runs/moving_ball_scene_v2d_decoder_fp/checkpoints/${CK}.pt
python scripts/steer_velocity2d.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_v2d/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/steer_${CK} \
    --ks 2,4,8,16 --num_scenes 40 --device cuda
echo "[v2d_steer] exit=$? ckpt=$CKPT"
