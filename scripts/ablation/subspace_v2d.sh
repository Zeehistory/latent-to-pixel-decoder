#!/bin/bash

#SBATCH --job-name=abl_subspace

#SBATCH --partition=bigmem

#SBATCH --requeue

#SBATCH --cpus-per-task=8

#SBATCH --mem=256G

#SBATCH --time=03:00:00

#SBATCH --output=logs/abl_subspace_%j.out

#SBATCH --error=logs/abl_subspace_%j.err



source "${SLURM_SUBMIT_DIR}/scripts/ablation/_job_init.sh"



python -u scripts/velocity_subspace.py \

    --train_dir "$LATENT_TRAIN" \

    --test_dir "$LATENT_TEST" \

    --layers "$ENCODER_LAYERS" \

    --output_dir "$SUBSPACE_DIR" \

    --ridge 1.0 \

    --save_k 8 \

    --max_global_pairs 800



echo "[abl_subspace] -> $SUBSPACE_DIR/subspace_summary.json"

