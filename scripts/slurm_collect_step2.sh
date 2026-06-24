#!/bin/bash
#SBATCH --job-name=vjepa_collect
#SBATCH --partition=gpu_rtx6000
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

# Final step of the Step-2 velocity pipeline: parse every result, write a digest, fill in brain.md
# (R2 table + changelog + status) and commit locally. Submit gated on the steering jobs so it runs
# unattended after the pipeline finishes, regardless of whether this shell is still alive:
#   sbatch --dependency=afterany:<decoder>:<steer_speed>:<steer_vx>:<steer_vy> scripts/slurm_collect_step2.sh
BASE_DIR=${BASE_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa"}
REPO_DIR=${REPO_DIR:-"/nfs/roberts/project/pi_jks79/zss8/latent-to-pixel-decoder"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

BASE_DIR="$BASE_DIR" REPO_DIR="$REPO_DIR" python scripts/collect_step2_results.py
echo "[collect] slurm wrapper done"
