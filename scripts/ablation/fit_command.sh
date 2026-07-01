#!/bin/bash

#SBATCH --job-name=abl_cmdop

#SBATCH --partition=bigmem

#SBATCH --requeue

#SBATCH --cpus-per-task=8

#SBATCH --mem=256G

#SBATCH --time=02:00:00

#SBATCH --output=logs/abl_cmdop_%j.out

#SBATCH --error=logs/abl_cmdop_%j.err



source "${SLURM_SUBMIT_DIR}/scripts/ablation/_job_init.sh"



python -u scripts/fit_command_operators.py \

    --train_dir "$LATENT_TRAIN" \

    --test_dir "$LATENT_TEST" \

    --layers "$ENCODER_LAYERS" \

    --ridge 1.0 \

    --artifacts_dir "$SUBSPACE_DIR"



echo "[abl_cmdop] artifacts in $SUBSPACE_DIR"

