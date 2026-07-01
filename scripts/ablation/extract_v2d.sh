#!/bin/bash

#SBATCH --job-name=abl_extract

#SBATCH --partition=scavenge_gpu

#SBATCH --requeue

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=04:00:00

#SBATCH --output=logs/abl_extract_%j.out

#SBATCH --error=logs/abl_extract_%j.err



# Extract scene_velocity2d latents for ablation (ViT-L or ViT-H).

#

#   export BASE_DIR=$PWD

#   ENCODER=vjepa2_huge SPLIT=train sbatch scripts/ablation/extract_v2d.sh



source "${SLURM_SUBMIT_DIR}/scripts/ablation/_job_init.sh"



SPLIT=${SPLIT:-train}

case "$SPLIT" in

  train) NUM_CLIPS=${NUM_CLIPS:-4000}; SEED=${SEED:-0} ;;

  test)  NUM_CLIPS=${NUM_CLIPS:-800};  SEED=${SEED:-2} ;;

  *) echo "unknown SPLIT=$SPLIT"; exit 1 ;;

esac



OUTPUT_DIR="$LATENT_ROOT/$SPLIT/$ENCODER"



echo "[abl_extract] ENCODER=$ENCODER SPLIT=$SPLIT layers=$ENCODER_LAYERS -> $OUTPUT_DIR"

python scripts/extract_latents.py \

    --config "$TRAIN_CONFIG" \

    --encoder "$ENCODER" \

    --layers "$ENCODER_LAYERS" \

    --output_dir "$OUTPUT_DIR" \

    --batch_size "$EXTRACT_BATCH" \

    --shard_size 128 \

    data.scenario=scene_velocity2d \

    data.clips_per_scene=8 \

    data.num_clips="$NUM_CLIPS" \

    data.seed="$SEED" \

    "data.speed_range=[0.012,0.026]" \

    "data.radius_range=[0.11,0.11]"

