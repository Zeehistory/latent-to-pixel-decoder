#!/bin/bash
#SBATCH --job-name=vjepa_train
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Usage:
#   sbatch scripts/slurm_train_decoder.sh
#
# Run extract first:  sbatch scripts/slurm_extract_latents.sh
# Or chain them:      sbatch --dependency=afterok:<extract_job_id> scripts/slurm_train_decoder.sh

LATENT_DIR=${LATENT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/latents/physics_iq/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/runs/physics_iq_decoder_large"}
CONFIG=${CONFIG:-"configs/train/physics_iq_transformer_large.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# accelerate launch uses torchrun under the hood; one process per GPU
accelerate launch \
    --num_processes $SLURM_GPUS_ON_NODE \
    --mixed_precision bf16 \
    scripts/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR"
